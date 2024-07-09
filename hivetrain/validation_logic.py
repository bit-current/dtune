import os
import random
import torch
import time
import math
from bittensor import logging
import logging
from copy import deepcopy
from torch.optim import AdamW
import wandb
import math


class ModelValidator:
    def __init__(
        self,
        device,
        model,
        tokenizer,
        optimizer,
        check_update_interval=300,
        bittensor_network=None,
        chain_manager=None,
        hf_manager=None,
        interval=300,
        assignment_interval=10,  # Interval for changing block number
        batch_size=1,
        data_loader=None

    ):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.interval = interval  # Validation interval in seconds
        self.assignment_interval = assignment_interval  # Interval for changing block number
        #self.base_loss, self.base_perplexity = self.evaluate_model()
        self.bittensor_network = bittensor_network
        self.scores = {}
        self.normalized_scores = {}
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager
        self.last_pull_time = 0
        self.check_update_interval = check_update_interval
        self.base_losses = []
        self.best_perplexities = []
        self.data_loader = data_loader
        
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.scores = {}
        self.normalized_scores = {}


    def update_model_weights(self, weights):
        temp_state_dict = self.model.state_dict()
        temp_state_dict.update(weights)
        self.model.load_state_dict(temp_state_dict)
        

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

    def validate_and_score(self):
        print("Receiving Gradients from chain")
        self.bittensor_network.sync(lite=True)

        
        #Could synchronize valis well
        while True:
            if self.hf_manager.check_for_new_submissions(self.hf_manager.averaged_miner_assignment_repo_id):
                print("Found new assignments")
                self.hf_manager.pull_latest_assignments()
                assignment_uids = torch.load(os.path.join(self.hf_manager.get_averaged_miner_assignment_directory(), "validator_miner_assignment.pt"))
                uid_to_hotkey = torch.load(os.path.join(self.hf_manager.get_averaged_miner_assignment_directory(), "uid_hotkey.pt"))
                my_hotkey = self.bittensor_network.wallet.hotkey.ss58_address
                my_uid = self.bittensor_network.metagraph.hotkeys.index(my_hotkey)
                try:
                    selected_miner_uids = list(set(assignment_uids[my_uid]))
                except KeyError:
                    print("Validator not recognized in New Assignment")
                    if len(assignment_uids.keys()) > 0:
                        random_key = random.choice(list(assignment_uids.keys()))
                        selected_miner_uids = list(set(assignment_uids[random_key][:5]))
                        print(f"Copying assignments of UID:{random_key}")
                    else: ## In case of a new run with no assigned valis
                        validator_uids = self.bittensor_network.get_validator_uids() 
                        miner_uids = [miner for miner in range(len(self.bittensor_network.metagraph.hotkeys)) if miner not in validator_uids]
                        miner_vali_ratio = 5 #to one
                        selected_miner_uids = random.sample(miner_uids, miner_vali_ratio)

                        print(f"No assignments found to copy. Generating random sample of size:{miner_vali_ratio}")
                break
            else:
                print("No new assignments found. Sleeping")
                time.sleep(60)

        if self.hf_manager.check_for_new_submissions(self.hf_manager.averaged_model_repo_id):
            print(
                "Averaged model updated on Hugging Face. Pulling latest model..."
            )
            self.hf_manager.pull_latest_model()
            time.sleep(10)  # Give enough time for pull. If you get an update error, get better internet or increase sleep time
            self.model = self.hf_manager.update_model(self.model)
            self.model = self.model.to(self.device)
            self.optimizer = AdamW(
                self.model.parameters(), lr=5e-5
            )  # Reinitialize the optimizer
            print("Evaluating new averager model")
            self.base_loss, self.base_perplexity = self.evaluate_model()
            self.base_weights = {
                name: param.clone() for name, param in self.model.named_parameters()
            }
            self.last_pull_time = time.time()

        self.original_state_dict = deepcopy(self.model.state_dict())

        valid_gradients = []

        self.best_losses = []
        self.best_perplexities = []
        self.averaging_weights = []
        print(f"Validating over: {selected_miner_uids}")
        for miner_id in selected_miner_uids:
            
            miner_hotkey = self.bittensor_network.metagraph.hotkeys[miner_id]
            try:
                miner_hotkey_2 = uid_to_hotkey[miner_id]
                if miner_hotkey !=  miner_hotkey_2:
                    print("Hotkey changed, prior miner deregistered. Skipping.")
                    continue
            except KeyError:
                    print("Hotkey Added before validation round. Skipping.")
                    continue

            hf_repo = self.chain_manager.retrieve_hf_repo(miner_hotkey)
            print(f"UID: {miner_id} Repo: {hf_repo} Hotkey:{miner_hotkey}")
            gradients = self.hf_manager.receive_gradients(hf_repo)
            if gradients is not None:
                print(f"Gradients: {miner_id} Exist !")
                try:
                    self.update_model_weights(gradients)
                except:
                    print("Improperly Shaped Gradients. Skipping.")
                    continue
                print(f"Evaluating model")
                loss, perplexity = self.evaluate_model()
                metrics = {
                        f"loss_{miner_id}": loss,
                        f"perplexity_{miner_id}": perplexity
                    }
                wandb.log(metrics)
                
                if loss < self.base_loss or perplexity < self.base_perplexity:
                    loss_score = max(0, self.base_loss - loss)
                    perplexity_score = max(0, self.base_perplexity - perplexity)
                    valid_gradients.append((miner_id, perplexity_score, gradients))
                    self.scores[miner_id] = perplexity_score**2 # We should exponentially reward better performing miners
                    self.best_losses.append(loss)
                    self.best_perplexities.append(perplexity)
                    self.averaging_weights.append(perplexity_score)
                    print(f"Gradients from {miner_id} good to average")

                else:
                    loss_score = 9999999999
                    perplexity_score = 0
                    self.scores[miner_id] = 0
                    print(f"Gradients from {miner_id} are invalid. Excluding from updates.")
                self.model.load_state_dict(self.original_state_dict)
            else:
                print(f"No gradients received from {miner_id}")

        normalization_factor = sum(self.averaging_weights)
        # Create gradient updates
        accumulated_gradients = {}
        for _, ppx_score, gradients in valid_gradients:
            ppx_weight = ppx_score/normalization_factor
            for name, grad in gradients.items():
                if name not in accumulated_gradients:
                    accumulated_gradients[name] = grad * ppx_weight
                else:
                    accumulated_gradients[name] += grad * ppx_weight


        # Update the model
        self.update_model_weights(accumulated_gradients)
        print("Evaluating Averaged Loss + Perplexity")
        loss, perplexity = self.evaluate_model()
        
        wandb.log({
                "loss_validator_avg":loss,
                "ppx_validator_avg":perplexity
            })

        # if loss > self.base_loss:
        #     if best_
        #     best_idx = self.best_losses.index(self.base_loss)
        #     self.update_model_weights(valid_gradients[best_idx][2])            
        #     print("Averaged not optimal. Uploading best individual model")
        # else:
        #     print("Averaged optimal. Uploading.")
        #     self.base_loss = loss
        #     self.base_perplexity = perplexity

        final_averaged_loss = {
            "loss":loss,
            "perplexity":perplexity
        }
        # Push the updated model to Hugging Face
        #try:
        model_gradients_path = os.path.abspath(os.path.join(
                self.hf_manager.get_local_gradient_directory(), "gradients.pt"
            ))
        
        loss_path = os.path.abspath(os.path.join(
                self.hf_manager.get_local_gradient_directory(), "loss.pt"
            ))
            
        lora_weights = {k: v for k, v in self.model.state_dict().items() if 'lora' in k}
        torch.save(lora_weights, model_gradients_path) #FIXME validator should only send Dora weights
        torch.save(final_averaged_loss, loss_path)

        self.hf_manager.push_gradients(["gradients.pt", "loss.pt"])
        print("Successfully pushed the updated model to Hugging Face.")
        
        if self.bittensor_network.should_set_weights():
            self.bittensor_network.set_weights(self.scores)
                
    def start_periodic_validation(self):

        wandb.init(project="distributed-training-v2-10-2-1",entity="alizawahry1", name=f"validator-{str(time.time())}")
        while True:

            start_time = time.time()            
            self.validate_and_score()
            self.hf_manager.clear_hf_cache()
            
            self.data_loader._fetch_data_to_buffer(18) 
            
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, self.interval - elapsed_time)
            print(f"One round done sleeping for: {time_to_wait}")
            time.sleep(time_to_wait)


class DeltaValidator(ModelValidator):
    def update_model_weights(self, weight_deltas, alpha=5e-4):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weight_deltas:
                    try:
                        param.data = weight_deltas[name] + param.data
                    except Exception as e:
                        logging.warning(f"Error loading gradients: {e}")
