from bittensor import logging
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
import bitsandbytes
import time 
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW 
from datasets import load_dataset
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.config import Configurator
from hivetrain.dataset import SubsetFineWebEdu2Loader
from hivetrain.hf_manager import HFManager
import os
import math
import wandb
from copy import deepcopy

import random
import numpy as np



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility

apply_lora = True

logging.enable_debug()
print("Starting !")

# Initialize Weights & Biases
# uncomment here and add your profile on wandb
wandb.init(project="distributed-training-v2-10-2-1", entity="alizawahry1", name=f"miner-{str(time.time())}")

def flatten_list(nested_list):
    """Flatten a nested list."""
    if nested_list and isinstance(nested_list[0], list):
        # Assumes only one level of nesting
        return [item for sublist in nested_list for item in sublist]
    return nested_list


args = Configurator.combine_configs()

## Chain communication setup
BittensorNetwork.initialize(args)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)
set_seed(my_uid)

address_store = ChainMultiAddressStore(
    BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
)

#TODO add a while loop here to raise errors if failed to set or retry till success
current_address_in_store = address_store.retrieve_hf_repo(my_hotkey)
print(f"Current value in store: {current_address_in_store}")
if current_address_in_store != args.storage.gradient_repo:
    print(f"Storing new value: {args.storage.gradient_repo}")
    address_store.store_hf_repo(args.storage.gradient_repo)
    # FIXME return status and raise error/issue if failed

# Loading model =========================================================================

batch_size = args.miner.batch_size
num_epochs = args.miner.epochs  # Set to a more reasonable value
learning_rate = args.miner.learning_rate
send_interval = 30*60#args.storage.send_interval  # Every 60 seconds


quantization_config = None
model_name = "Qwen/Qwen2-1.5B"
model_cache_dir = './model_cache'  # Specify a local cache directory
os.makedirs(model_cache_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, cache_dir=model_cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

if apply_lora:
    config = LoraConfig(
        use_dora=True,
        r=32,
        lora_alpha=8,
        target_modules="all-linear",
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)

loader = SubsetFineWebEdu2Loader(
    batch_size=batch_size,
    sequence_length=args.model.sequence_length,
    num_pages=100,
    tokenizer=tokenizer,
)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def size_gradients(model):
    grad_sizes = {name: param.grad.numel() * param.grad.element_size() / (1024 * 1024) for name, param in model.named_parameters() if param.requires_grad and param.grad is not None}
    total_size = sum(grad_sizes.values())
    print(f"Total gradient size: {total_size:.2f} MB")
    return grad_sizes

## Training loop ==============================================================================================================

last_pull_time = 0
last_check_time = 0
last_send_time = time.time()

hf_manager = HFManager( gradient_repo_id=args.storage.gradient_repo, averaged_model_repo_id=args.storage.averaged_model_repo_id,
gradient_repo_local=args.storage.gradient_repo_local ,averaged_model_repo_local=args.storage.averaged_model_repo_local  )

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    total_loss = 0
    total_examples = 0
    
    for batch_idx, batch in tqdm(enumerate(loader)):

        ## FIXME separate send_interval from last_pull_time? 
        if time.time() - last_check_time >= send_interval:
            if hf_manager.check_for_new_submissions(hf_manager.averaged_model_repo_id) or last_check_time == 0:
                print("Averaged model updated on Hugging Face. Pulling latest model...")
                print("********Averaged model updated on Hugging Face. Pulling latest model...")
                hf_manager.pull_latest_model()
                time.sleep(10) #just to give enough time for pull
                model = hf_manager.update_model(model)
                del optimizer
                optimizer = AdamW(model.parameters(), lr=learning_rate)
                ##FIXME Save and load model state
            last_check_time = time.time()
        
        if batch_idx == 0:
            print("Beginning Training")
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Ensure inputs and labels are not empty
        if inputs.size(0) == 0 or inputs.size(0) == 0:
            continue

        optimizer.zero_grad()
        try:
            outputs = model(inputs, labels=labels)
        except Exception as e:
            logging.warning(f"Forward pass failed: {e}")
            continue
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total_loss += loss.item() * inputs.size(0)
        total_examples += inputs.size(0)

        if time.time() - last_send_time >= send_interval:
            average_loss = total_loss / total_examples
            perplexity = math.exp(average_loss)
            print(
                f"Epoch: {epoch}, Examples: {total_examples}, Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}"
            )
            wandb.log({
                    "step":batch_idx*(epoch+1),
                    "batch":batch_idx,
                    "epoch": epoch,
                    "training_loss": average_loss,
               })
            try:
                print(f"Attempting to send gradients")
                # Periodically save gradients
                delta_weights = {}
                for key in model.state_dict().keys():
                    if 'lora' in key:
                        delta_weights[key] = model.state_dict()[key].to("cpu")# - base_model.state_dict()[key].to("cpu")
                model_gradients_path = os.path.join(
                    hf_manager.get_local_gradient_directory(), "gradients.pt"
                )
                torch.save(delta_weights, model_gradients_path)
                hf_manager.push_gradients("gradients.pt")
            except Exception as e:
                logging.warning(f"Sending gradients failed: {e}")
                continue
            last_send_time = time.time()

    train_loss /= total_examples
    print(f"Epoch {epoch+1} completed. Loss: {train_loss:.2f}")
    loader._fetch_data_to_buffer(100)

