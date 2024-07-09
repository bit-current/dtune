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
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.config import Configurator
from hivetrain.hf_manager import HFManager, model_hash
import os
import math
import wandb
from copy import deepcopy

import random
import numpy as np
from huggingface_hub import HfApi

args = Configurator.combine_configs()

## Chain communication setup
BittensorNetwork.initialize(args,ignore_regs=True )

def check_valis(chain_manager, hf_api, validator_uids, uid_to_hotkey):
    false_valis = []
    for validator_uid in validator_uids:
        validator_hotkey = uid_to_hotkey[validator_uid]

        hf_repo = chain_manager.retrieve_hf_repo(validator_hotkey)
        print(f"UID: {validator_uid} REPO: {hf_repo} Hotkey: {validator_hotkey}")
        if hf_repo is None:
            false_valis.append(validator_uid)
            continue 

        repo_files = hf_api.list_repo_files(hf_repo)

        if "gradients.pt" not in repo_files:
            false_valis.append(validator_uid)

    for false_vali in false_valis:
        validator_uids.remove(false_vali)
    return validator_uids

address_store = ChainMultiAddressStore(
    BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
)

print("Pre-check valis (By Stake Only)")
validator_uids = BittensorNetwork.get_validator_uids()
uid_to_hotkey = {uid:BittensorNetwork.metagraph.hotkeys[uid] for uid in range(len(BittensorNetwork.metagraph.hotkeys))}
print(validator_uids)
validator_uids = check_valis(address_store,HfApi(),validator_uids, uid_to_hotkey )
print("Post-check valis (Full repo)")
print(validator_uids)