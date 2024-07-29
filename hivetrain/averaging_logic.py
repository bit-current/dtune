import copy
import hashlib
import os
import time
import torch
import random
import math
import mlflow
import mlflow.pytorch
from huggingface_hub import hf_hub_download
from hivetrain.config.mlflow_config import MLFLOW_UI_URL, CURRENT_MODEL_NAME
from hivetrain.dataset import SubsetFineWebEdu2Loader
from hivetrain.hf_manager import model_hash
from hivetrain.config.mlflow_config import (
    MLFLOW_UI_URL,
    CURRENT_MODEL_NAME,
    MLFLOW_ACTIVE,
)
from hivetrain.utils.mlflow_utils import VERSION, initialize_mlflow, log_model_metrics
import math
from copy import deepcopy
from hivetrain.btt_connector import BittensorNetwork, sync
from bittensor import logging
from transformers import TrainingArguments, Trainer, AdamW
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from huggingface_hub import HfApi

class Averager:
    def __init__(
        self,
        model,
        tokenizer,
        hf_manager,
        chain_manager,
        bittensor_network,
        hf_token=os.environ.get("HF_TOKEN"),
        device="cuda",
        batch_size=1,
        data_loader = None
    ):
        self.model = model.to(device)
        self.hf_token = hf_token
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager
        self.device=device
        self.hf_api = api = HfApi()
        self.validator_hashes = {}
        self.data_loader = data_loader

        self.base_loss, self.base_perplexity = self.evaluate_model()

        # initialize mlflow
        wandb.init(project="distributed-training-v4-1-1-1",entity="alizawahry1", name=f"averager-{str(time.time())}")


    def evaluate_model(self, metric="perplexity"):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch_num, batch in enumerate(
                self.data_loader
            ):  # FIXME turn me into a generator?
                try:
                    outputs = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device),
                    )
                    loss = outputs.loss
                    total_loss += loss.item() * batch["input_ids"].size(0)
                    total_samples += batch["input_ids"].size(0)
                except Exception as e:
                    continue

        average_loss = total_loss / total_samples
        perplexity = math.exp(average_loss) if metric == "perplexity" else None
        print(f"Average Loss for evaluated model: {average_loss}")
        
        return average_loss, perplexity

    def get_miner_and_validator_uids(self):
        validator_uids = self.bittensor_network.get_validator_uids()
        miner_uids = [miner for miner in range(len(self.bittensor_network.metagraph.hotkeys)) if miner not in validator_uids]
        return miner_uids, validator_uids

    def create_uid_to_hotkey_map(self):
        return {uid: self.bittensor_network.metagraph.hotkeys[uid] for uid in range(len(self.bittensor_network.metagraph.hotkeys))}

    @staticmethod
    def calculate_miner_distribution(miner_uids, validator_uids):
        num_of_overlapping_miners = len(miner_uids) // (2 * len(validator_uids))
        overlapping_miners = miner_uids[:num_of_overlapping_miners]
        remaining_miners = miner_uids[num_of_overlapping_miners:]
        miners_per_validator = len(remaining_miners) // len(validator_uids)
        additional_miners = len(remaining_miners) % len(validator_uids)
        return overlapping_miners, remaining_miners, miners_per_validator, additional_miners

    # @staticmethod
    # def assign_miners_to_validators(validator_uids, overlapping_miners, remaining_miners, miners_per_validator, additional_miners):
    #     miner_assignments = {vali_uid: list(overlapping_miners) for vali_uid in validator_uids}
    #     current_index = 0
    #     for i, vali_uid in enumerate(validator_uids):
    #         num_miners = miners_per_validator + (1 if i < additional_miners else 0)
    #         miner_assignments[vali_uid].extend(remaining_miners[current_index:current_index + num_miners])
    #         current_index += num_miners
    #     return miner_assignments

    def save_assignments(self,hf_manager, miner_assignments, uid_to_hotkey):
        base_dir = hf_manager.get_averaged_miner_assignment_directory()
        assignment_path = os.path.join(base_dir, "validator_miner_assignment.pt")
        hotkey_path = os.path.join(base_dir, "uid_hotkey.pt")
        counter_path = os.path.join(base_dir, "counter.pt")
                
        torch.save(miner_assignments, assignment_path)
        torch.save(uid_to_hotkey, hotkey_path)
        torch.save(self.counter_pushed, counter_path)
        hf_manager.push_miner_assignemnts(["validator_miner_assignment.pt", "uid_hotkey.pt", "counter.pt"])

    @staticmethod
    def get_total_runs(num_miners, miners_per_validator):
        return math.ceil(num_miners / miners_per_validator)

    @staticmethod
    def get_current_run(counter, total_runs):
        return counter % total_runs

    def assign_miners_to_validators(self):
        try:
            miner_uids, validator_uids = self.get_miner_and_validator_uids()
            uid_to_hotkey = self.create_uid_to_hotkey_map()
            
            miner_uids = self.bittensor_network.check_valis(miner_uids, uid_to_hotkey, self.chain_manager, self.hf_api)
            validator_uids = self.bittensor_network.check_valis(validator_uids, uid_to_hotkey, self.chain_manager, self.hf_api)
            
            if len(miner_uids) == 0 or len(validator_uids) == 0:
                print("No miners or validators available. Setting empty assignments")
                self.save_assignments(self.hf_manager, {}, uid_to_hotkey)
                return False

            current_block = self.bittensor_network.subtensor.block
            
            # Set the fixed number of miners per validator
            if len(validator_uids)%len(miner_uids) != 0:
                miners_per_validator = len(validator_uids)//len(miner_uids) +1  # This can be adjusted or made into a class attribute if needed
            else:
                miners_per_validator = len(validator_uids)//len(miner_uids)

            total_runs = self.get_total_runs(len(miner_uids), miners_per_validator)
            current_run = self.get_current_run(self.counter_pushed, total_runs)
            
            # Calculate the ratio based on the fixed number of miners per validator
            ratio = miners_per_validator / len(miner_uids)
            
            miner_assignments = {vali_uid: [] for vali_uid in validator_uids}

            random.shuffle(miner_uids)
            for i, vali_uid in enumerate(validator_uids):
                start_index = (i + current_run * len(validator_uids)) % len(miner_uids)
                assigned_miners = [miner_uids[(start_index + j) % len(miner_uids)] for j in range(miners_per_validator)]
                miner_assignments[vali_uid] = assigned_miners

            self.save_assignments(self.hf_manager, miner_assignments, uid_to_hotkey)
            
            # Log the current assignment details
            print(f"Block: {current_block}, Run: {current_run}, Total Runs: {total_runs}")
            print(f"Miners per validator: {miners_per_validator}")
            print(f"Total miners: {len(miner_uids)}, Total validators: {len(validator_uids)}")
            print(f"Ratio: {ratio:.2f}, Miners assigned this run: {miners_per_validator * len(validator_uids)}")
            
            return True
        
        except Exception as e:
            print(f"An error occurred while assigning miners to validators deterministically: {str(e)}")
            return False

    
    def receive_weights(
        self, repo_id="your_username/your_repo_name", gradient_file_name="gradients.pt", loss_file_name="loss.pt"
    ):
        try:
            print(f"Receiving weights from: {repo_id}")
            # Download the gradients file from Hugging Face Hub
            gradient_file_path = hf_hub_download(
                repo_id=repo_id, filename=gradient_file_name, use_auth_token=True
            )

            loss_file_path = hf_hub_download(
                repo_id=repo_id, filename=loss_file_name, use_auth_token=True
            )
            
            with open(gradient_file_path, "rb") as file:
                gradient_hash = hashlib.sha256(file.read()).hexdigest()
    
            # Load the gradients directly using torch.load
            aggregated_gradients = torch.load(gradient_file_path)
            loss = torch.load(loss_file_path)

            if self.have_nans(aggregated_gradients):
                return None, gradient_hash, loss

            return aggregated_gradients, gradient_hash, loss
        except Exception as e:
            print(f"Error receiving gradients from Hugging Face: {e}")
            return None, None, None

    def receive_and_score_weights(self):
        # Get validators uids
        self.bittensor_network.sync(lite=False) 

        validator_uids = self.bittensor_network.get_validator_uids()

        # n = len(self.bittensor_network.metagraph.hotkeys) #FIXME I am only for testing NOPROD
        # self.validator_combined_weights = torch.full((n,1), 1/n, dtype=torch.float32) #FIXME I am only for testing NOPROD
        # Get average of validator weights weighted by their stake?
        self.validator_weights = []
        self.validator_hotkeys = []
        self.validator_losses = []

        for uid, hotkey in tqdm(enumerate(self.bittensor_network.metagraph.hotkeys)):
            if uid in validator_uids:
                try:
                    print(f"Receiving from uid: {uid}")
                    repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                    weight, weight_hash, loss = self.receive_weights(repo_id=repo_id)

                    try:
                        last_hash = self.validator_hashes[hotkey]
                    except KeyError:
                        last_hash = None

                    if last_hash == weight_hash:
                        raise ValueError("Weight has not been updated since last averaging run !")

                    self.validator_weights.append(weight)
                    self.validator_hotkeys.append(hotkey)
                    self.validator_losses.append(loss)

                    self.validator_hashes[hotkey] = weight_hash

                except Exception as e:
                    print(f"Receiving gradients failed due to: {e}")
                    # self.validator_weights.append(None)
                    # self.validator_hotkeys.append(hotkey)
                    # self.validator_losses.append({"loss":99999,"perplexity":9e99})

    @staticmethod
    def have_nans(aggregated_weights):
        for tensor in aggregated_weights.values():
            if torch.isnan(tensor).any():
                print("NaN values detected in the aggregated gradients.")
                return True
        return False

    def average_weights(self, beta=1.0):
        # self.validator_weights = [
        #     weight for weight in self.validator_weights if weight is not None
        # ]
        #FIXME Add error handling here
        try:
            assert len(self.validator_weights) > 0
        except:
            print("No weights ! No Averaging")
            return

        print(f"Number of weights being averaged: {len(self.validator_weights)}")

        validator_scores = []
        for losses in self.validator_losses:
            perplexity = losses["perplexity"]
            perplexity_score = max(0, self.base_perplexity - perplexity)
            perplexity_score = perplexity_score
            validator_scores.append(perplexity_score)
        perplexity_normalization = sum(validator_scores)

        averaged_weights = {
            name: torch.zeros_like(grad)
            for name, grad in self.validator_weights[0].items()
        }

        if perplexity_normalization == 0:
            print("PPX scores all zero")
            return 
            
        for weight, ppx_score in zip(self.validator_weights,validator_scores):
            print("Averaging Gradient")
            ppx_weight = ppx_score/perplexity_normalization
            for name, weight_value in weight.items():
                averaged_weights[name] += weight_value * ppx_weight
            print(f"dtype: {weight_value.dtype}")


        self.old_state_dict = deepcopy(self.model.state_dict())
        # for name, weight in averaged_weights.items():
        #     if name in self.old_state_dict:
        #         averaged_weights[name] = averaged_weights[name].cpu()

        return averaged_weights

    # def push_to_hf_hub(self, commit_message="Pushing model to Hub"):
    #     training_args = TrainingArguments(
    #         output_dir=self.local_dir,  # Local directory to save the model
    #         per_device_train_batch_size=1,  # Dummy argument, won't actually be used for training here
    #         per_device_eval_batch_size=1,  # Dummy argument, necessary to specify but won't be used
    #         push_to_hub=True,  # Enable pushing to hub
    #         push_to_hub_model_id=self.repo_id,  # Repository ID on the Hugging Face Hub
    #         push_to_hub_organization=None,  # Specify organization name here if applicable
    #         push_to_hub_token=self.hf_token,  # Hugging Face authentication token
    #     )

    #     # Initialize the Trainer
    #     trainer = Trainer(
    #         model=self.model,  # Your PyTorch model
    #         args=training_args,
    #     )

    #     # Push the model to the Hugging Face Hub
    #     trainer.push_to_hub(commit_message=commit_message)

    # def assign_miners_to_validators(self):
    #     validator_uids = self.bittensor_network.get_validator_uids()
    #     miner_uids = [uid for uids in range(len(self.bittensor_network.metagraph.hotkeys)) if uid not in validator_uids]

    
    def run_periodic_averaging(self, t):
        self.counter_pushed = 0
        while True:
            print("Averaging Beggining")
            start_time = time.time()
            
            self.data_loader._fetch_data_to_buffer(18) 

            self.receive_and_score_weights()
            averaged_weights = self.average_weights()
            self.counter_pushed += 1
            if averaged_weights is not None:
                temp_state_dict = self.model.state_dict()
                temp_state_dict.update(averaged_weights)
                self.model.load_state_dict(temp_state_dict)
                # for name, param in self.model.named_parameters():
                #     if name in averaged_weights:
                #         param.data.copy_(averaged_weights[name])
            
                #self.model.load_state_dict(averaged_weights)
                averaged_loss, averaged_perplexity = self.evaluate_model()
                if averaged_perplexity < self.base_perplexity:
                    print("Averaged Model improves loss")
                    averaged_model_path = os.path.abspath(os.path.join(
                            self.hf_manager.get_averaged_model_directory(), "averaged_model.pt"
                        ))

                    lora_weights = {k: v for k, v in self.model.state_dict().items() if 'lora' in k}
                    torch.save(lora_weights, averaged_model_path) #FIXME validator should only send Dora weights
                    self.base_perplexity = averaged_perplexity
                    self.base_loss = averaged_loss
                else:
                    print("Averaged Model Failed to improve loss")
                    self.model.load_state_dict(self.old_state_dict)

                print("Logging Results")
                wandb.log({
                    "pushes": self.counter_pushed,
                    "averaged_model_loss":averaged_loss,
                    "averaged_model_perplexity":averaged_perplexity
                })

            try:
                result_assign = self.assign_miners_to_validators()
                if result_assign:
                    print("Miner assignments successfully set")
                else:
                    print("Miner assignments failed")
            except Exception as e:
                print(f"Miner assignments failed: {e}")
            
            self.hf_manager.push_averaged_model("averaged_model.pt")
            self.hf_manager.clear_hf_cache()
            print("Averaging round Done.")

            #self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            print(f"Sleeping for {time_to_wait}")
            time.sleep(time_to_wait)