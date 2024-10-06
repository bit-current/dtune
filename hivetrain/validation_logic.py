import hashlib
import os
import random
import torch
import time
import math
from copy import deepcopy
from torch.optim import AdamW
import wandb
import math
from tqdm import tqdm
from huggingface_hub import HfApi


class ModelValidator:
    def __init__(
        self,
        device,
        model,
        tokenizer,
        optimizer,
        check_update_interval=300,
        commune_network=None,
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
        self.commune_network = commune_network
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
        self.gradient_hashes = {}


    def update_model_weights(self, weights):
        temp_state_dict = self.model.state_dict()
        temp_state_dict.update(weights)
        self.model.load_state_dict(temp_state_dict)
        

    def evaluate_model(self, metric="perplexity"):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch_num, batch in tqdm(enumerate(
                self.data_loader
            )):  # FIXME turn me into a generator?
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
                    print(e)
                    continue

        average_loss = total_loss / total_samples
        perplexity = math.exp(average_loss) if metric == "perplexity" else None
        print(f"Average Loss for evaluated model: {average_loss}")
        
        return average_loss, perplexity
    
    def fetch_and_process_assignments(self):
        while True:
            if self.hf_manager.check_for_new_submissions(self.hf_manager.averaged_miner_assignment_repo_id):
                print("Found new assignments")
                self.hf_manager.pull_latest_assignments()
                return torch.load(os.path.join(self.hf_manager.get_averaged_miner_assignment_directory(), "validator_miner_assignment.pt"))
            else:
                print("No new assignments found. Skipping")
                return None

    def get_selected_miner_uids(self, assignments):
        uid_to_hotkey = torch.load(os.path.join(self.hf_manager.get_averaged_miner_assignment_directory(), "uid_hotkey.pt"))
        my_hotkey = self.commune_network.my_hotkey
        my_uid = self.commune_network.my_uid
        
        try:
            selected_miner_uids = list(set(assignments[my_uid]))
        except KeyError:
            print("Validator not recognized in New Assignment")
            if len(assignments.keys()) > 0:
                random_key = random.choice(list(assignments.keys()))
                selected_miner_uids = list(set(assignments[random_key][:5]))
                print(f"Copying assignments of UID:{random_key}")
            else:
                validator_uids = self.commune_network.get_validator_uids() 
                miner_uids = [miner for miner in range(len(self.commune_network.hotkeys)) if miner not in validator_uids]
                miner_vali_ratio = 5
                selected_miner_uids = random.sample(miner_uids, miner_vali_ratio)
                print(f"No assignments found to copy. Generating random sample of size:{miner_vali_ratio}")
        
        return selected_miner_uids

    def check_and_update_model(self):
        if self.hf_manager.check_for_new_submissions(self.hf_manager.averaged_model_repo_id):
            print("Averaged model updated on Hugging Face. Pulling latest model...")
            self.hf_manager.pull_latest_model()
            time.sleep(10)  # Give enough time for pull
            new_model = self.hf_manager.update_model(self.model) #TODO add try except failsafe here
            if new_model is not None:
                self.model = new_model
                self.model = self.model.to(self.device)
                self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
                print("Evaluating new averager model")
                self.base_loss, self.base_perplexity = self.evaluate_model()
                self.base_weights = {name: param.clone() for name, param in self.model.named_parameters()}
                self.last_pull_time = time.time()

        self.original_state_dict = deepcopy(self.model.state_dict())

    def validate_and_score(self):
        print("Receiving Gradients from chain")
        self.commune_network.sync(lite=True)
        

        # Fetch and process assignments
        #try:
         #   assignments = self.fetch_and_process_assignments()
        # except:
            # assignments = None
        
        all_uids = [i for i in range(len(self.commune_network.hotkeys))]
        miner_uids, valid_miner_names = self.commune_network.check_valis_or_miners(all_uids, repo_type="miner")
        # miner_uids = [miner for miner in range(len(self.commune_network.hotkeys)) if miner not in validator_uids]

        # if assignments is not None:
        #     selected_miner_uids = self.get_selected_miner_uids(assignments)
        # else:
        #     if len(miner_uids) > 5:
        #         selected_miner_uids = random.sample(miner_uids, 5)
        #     else:
        selected_miner_uids = miner_uids
        print(selected_miner_uids)
        print(valid_miner_names)

        # Check for model updates
        try:
            self.check_and_update_model()
        except Exception:
            print("Loading averager model failed. Possible corrupt averager weights.")

        # Create a dictionary to store UID:path mappings
        uid_gradient_paths = {}

        # Fetch all gradients at once
        print("Fetching all gradients...")
        for miner_id in selected_miner_uids:
            
            #miner_hotkey = self.commune_network.hotkeys[miner_id]
            hf_repo = self.commune_network.names[miner_id]
            
            print(f"Fetching repo: {hf_repo} for UID {miner_id}")
            gradient_path = self.hf_manager.receive_gradients(hf_repo, path_only=True)
            if gradient_path is None:
                print("Skipping UID as no registered repo")
                continue
            with open(gradient_path, "rb") as file:
                gradient_hash = hashlib.sha256(file.read()).hexdigest()

            try:
                if gradient_hash == self.gradient_hashes[miner_id]:
                    print("Skipping UID as no new weights")
                    continue
                else:
                    self.gradient_hashes[miner_id] = gradient_hash
            except KeyError:
                self.gradient_hashes[miner_id] = gradient_hash

            uid_gradient_paths[miner_id] = gradient_path

        if len(uid_gradient_paths) == 0:
            print("Skipping Validation Round. No new weights")
            return
        
        valid_gradients = []
        self.best_losses = []
        self.best_perplexities = []
        self.averaging_weights = []

        print(f"Validating over: {selected_miner_uids}")
        for miner_id, gradient_path in uid_gradient_paths.items():
            print(f"Processing UID: {miner_id}")
            try:
                gradients = torch.load(gradient_path)
            except:
                gradients = None
            
            if gradients is not None:
                print(f"Gradients for UID {miner_id} exist!")
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
                    perplexity_score = max(0, self.base_perplexity - perplexity)
                    valid_gradients.append((miner_id, perplexity_score, gradients))
                    self.scores[miner_id] = perplexity_score**2  # Exponentially reward better performing miners
                    self.best_losses.append(loss)
                    self.best_perplexities.append(perplexity)
                    self.averaging_weights.append(perplexity_score)
                    print(f"Gradients from {miner_id} good to average")
                else:
                    self.scores[miner_id] = 0
                    print(f"Gradients from {miner_id} are invalid. Excluding from updates.")

                self.model.load_state_dict(self.original_state_dict)
                os.remove(gradient_path)
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

        final_averaged_loss = {
            "loss":loss,
            "perplexity":perplexity
        }
        # Push the updated model to Hugging Face
        #try:
        model_gradients_path = os.path.abspath(os.path.join(
                self.hf_manager.get_local_gradient_directory(), "validator_gradients.pt"
            ))
        
        # version to push to averager
        model_gradients_averager_path = os.path.abspath(os.path.join(
                self.hf_manager.get_local_gradient_directory(), "averaged_model.pt"
            ))
        
        loss_path = os.path.abspath(os.path.join(
                self.hf_manager.get_local_gradient_directory(), "loss.pt"
            ))
            
        lora_weights = {k: v for k, v in self.model.state_dict().items() if 'lora' in k}
        torch.save(lora_weights, model_gradients_path) #FIXME validator should only send Dora weights
        torch.save(lora_weights, model_gradients_averager_path) #FIXME validator should only send Dora weights
        torch.save(final_averaged_loss, loss_path)

        self.hf_manager.push_gradients(["validator_gradients.pt", "loss.pt"])
        print("Successfully pushed the updated model to Hugging Face.")
        self.hf_manager.push_averaged_model(model_gradients_averager_path)
        print("Successfully pushed the averaged model to Hugging Face.")


        
        if self.commune_network.should_set_weights():
            self.commune_network.set_weights(self.scores)
                
    def start_periodic_validation(self):

        wandb.init(project="distributed-training-v4-1-1-1",entity="alizawahry1", name=f"validator-{str(time.time())}")
        self.hf_manager.clean_repo('gradient')
        while True:

            start_time = time.time()            
            self.validate_and_score()
            self.hf_manager.clear_hf_cache()
            
            self.data_loader._fetch_data_to_buffer(1) 
            
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
                        print(f"Error loading gradients: {e}")
