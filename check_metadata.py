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

args = Configurator.combine_configs()

## Chain communication setup
BittensorNetwork.initialize(args,ignore_regs=True )

address_store = ChainMultiAddressStore(
    BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
)

for uid, hotkey in enumerate(BittensorNetwork.metagraph.hotkeys):
    current_address_in_store = address_store.retrieve_hf_repo(hotkey)
    print(f"UID: {uid} repo:{current_address_in_store} hotkey: {hotkey}")
