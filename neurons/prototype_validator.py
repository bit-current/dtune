import os
import torch
import time
import wandb  
#from bittensor import logging
from hivetrain.comm_connector import CommuneNetwork
from hivetrain.config import Configurator
from hivetrain.dataset import SubsetFineWebEdu2Loader
from hivetrain.validation_logic import ModelValidator
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.hf_manager import HFManager
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "openai-community/gpt2"

class Validation:
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.setup_bittensor()
        self.setup_model_and_tokenizer()
        self.setup_data_loader()
        self.setup_hf_manager()
        self.setup_validator()

    def setup_logging(self):
        #logging.basicConfig(level=logging.INFO)
        print("Setting up logging...")
    
    def setup_bittensor(self):
        """
        Initializes the Bittensor network with provided arguments, sets up the hotkey and unique identifier (UID)
        for this instance, and configures the seed based on the UID. It also initializes an address store to manage
        multiple addresses on the chain.
        
        The function also calls `update_address_store` to refresh the address store with the latest data from the network.
        """
        CommuneNetwork.initialize(self.args)
        self.my_hotkey = CommuneNetwork.my_hotkey
        self.my_uid = CommuneNetwork.my_uid
        self.address_store = ChainMultiAddressStore(
            CommuneNetwork.client, CommuneNetwork.netuid, CommuneNetwork.keypair, self.args.module_name
        )

    def setup_model_and_tokenizer(self):
        print("Setting up model and tokenizer...")
        model_name = MODEL_NAME
        model_cache_dir = './model_cache'
        os.makedirs(model_cache_dir, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=None,
            cache_dir=model_cache_dir,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.miner.learning_rate)

        config = LoraConfig(
            use_dora=True,
            r=32,
            lora_alpha=8,
            target_modules="all-linear",
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, config)
    
    def setup_data_loader(self):
        print("Setting up data loader...")
        self.loader = SubsetFineWebEdu2Loader(
            batch_size=self.args.miner.batch_size,
            sequence_length=self.args.model.sequence_length,
            num_pages=1,
            tokenizer=self.tokenizer,
        )
    
    def setup_hf_manager(self):
        print("Setting up HuggingFace manager...")
        self.hf_manager = HFManager(
            gradient_repo_id=self.args.storage.gradient_repo,
            averaged_model_repo_id=self.args.storage.averaged_model_repo_id,
            # gradient_repo_local=self.args.storage.gradient_repo_local,
            # averaged_model_repo_local=self.args.storage.averaged_model_repo_local,
            # averaged_miner_assignment_repo_local=self.args.storage.averaged_miner_assignment_repo_local,
            averaged_miner_assignment_repo_id=self.args.storage.averaged_miner_assignment_repo_id
        )

    def setup_validator(self):
        print("Setting up model validator...")
        self.validator = ModelValidator(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            check_update_interval=120*60,
            commune_network=CommuneNetwork,
            chain_manager=self.address_store,
            hf_manager=self.hf_manager,
            interval=20*60,
            data_loader=self.loader
        )

    def start_validation(self):
        print("Starting periodic validation...")
        self.validator.start_periodic_validation()

def main():
    args = Configurator.combine_configs()
    validator = Validation(args)
    validator.start_validation()

if __name__ == "__main__":
    main()



