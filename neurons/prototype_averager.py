import os
import torch
from hivetrain.averaging_logic import Averager
from hivetrain.dataset import SubsetFineWebEdu2Loader
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig, AdamW, AutoModelForCausalLM, AutoTokenizer
from hivetrain.hf_manager import HFManager
from hivetrain.btt_connector import BittensorNetwork
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from hivetrain.config import Configurator
from hivetrain.chain_manager import ChainMultiAddressStore
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


args = Configurator.combine_configs()

batch_size = args.miner.batch_size

## Chain communication setup
BittensorNetwork.initialize(args,ignore_regs=True)

quantization_config = None

learning_rate = args.miner.learning_rate

model_name = "Qwen/Qwen2-1.5B"
model_cache_dir = '../trash_dir'  # Specify a local cache directory
os.makedirs(model_cache_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, cache_dir=model_cache_dir, 
                                             torch_dtype=torch.float16,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

config = LoraConfig(
    use_dora=True,
    r=32,
    lora_alpha=8,
    target_modules="all-linear",
    lora_dropout=0.1,
)
model = get_peft_model(model, config)

address_store = ChainMultiAddressStore(
    BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
)

hf_manager = HFManager( averaged_model_repo_id=args.storage.averaged_model_repo_id, averaged_model_repo_local=args.storage.averaged_model_repo_local, \
averaged_miner_assignment_repo_local = args.storage.averaged_miner_assignment_repo_local, averaged_miner_assignment_repo_id = args.storage.averaged_miner_assignment_repo_id)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

data_loader = SubsetFineWebEdu2Loader(
            batch_size=batch_size,
            sequence_length=args.model.sequence_length,
            num_pages=18,
            tokenizer=tokenizer
        )


#Averager Loading
averager = Averager(
        model,
        tokenizer,
        hf_manager,
        address_store,
        BittensorNetwork,
        hf_token=os.environ.get("HF_TOKEN"),
        device="cuda",
        data_loader=data_loader
    )

averager.run_periodic_averaging(60*60)