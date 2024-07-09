import os
import torch

from bittensor import logging
import logging

from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator
from hivetrain.dataset import SubsetFineWebEdu2Loader
from hivetrain.validation_logic import ModelValidator
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.hf_manager import HFManager

from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import wandb  # Import W&B

# Initialize Weights & Biases

args = Configurator.combine_configs()
BittensorNetwork.initialize(args, ignore_regs=True)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address

address_store = ChainMultiAddressStore(
    BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
)

current_address_in_store = address_store.retrieve_hf_repo(my_hotkey)
logging.info(f"Current value in store:{current_address_in_store}")
if current_address_in_store != args.storage.gradient_repo:
    logging.info(f"Storing new value: {args.storage.gradient_repo}")
    address_store.store_hf_repo(args.storage.gradient_repo)

batch_size = args.miner.batch_size
num_epochs = args.miner.epochs  # Set to a more reasonable value
learning_rate = args.miner.learning_rate
send_interval = args.storage.send_interval  # Every 60 seconds
sequence_length = args.model.sequence_length


quantization_config = None

model_name = "Qwen/Qwen2-1.5B"
model_cache_dir = './model_cache'  # Specify a local cache directory
os.makedirs(model_cache_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, cache_dir=model_cache_dir,
                                             torch_dtype=torch.float16,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
optimizer = AdamW(model.parameters(), lr=learning_rate)

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
    num_pages=18,
    tokenizer=tokenizer,
)


hf_manager = HFManager(gradient_repo_id=args.storage.gradient_repo, averaged_model_repo_id=args.storage.averaged_model_repo_id,
gradient_repo_local=args.storage.gradient_repo_local ,averaged_model_repo_local=args.storage.averaged_model_repo_local,
averaged_miner_assignment_repo_local = args.storage.averaged_miner_assignment_repo_local, averaged_miner_assignment_repo_id = args.storage.averaged_miner_assignment_repo_id)

validator_instance = ModelValidator(
        device="cuda",
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        check_update_interval=60*60,
        bittensor_network=BittensorNetwork,
        chain_manager=address_store,
        hf_manager=hf_manager,
        interval=60*60,
        data_loader=loader
    )

validator_instance.start_periodic_validation()

