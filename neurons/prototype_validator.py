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
        try:
            success = self.update_address_store()
            if success:
                print("Address store updated successfully")
        except RuntimeError as e:
            print(f"Failed to update address store: {str(e)}")

    def update_address_store(self):
        """
        Update the address store with the current gradient repository address.

        This method attempts to retrieve the current address from the store and update it
        if it differs from the current gradient repository address. It includes retry logic
        to handle transient failures in chain communication.
        """
        max_retries = 5
        retry_delay = 5  # seconds
        attempt = 0
        while attempt < max_retries:
            try:
                current_address = self.address_store.retrieve_hf_repo(self.my_uid)
                if current_address != self.args.storage.gradient_repo:
                    print(f"Storing new value: {self.args.storage.gradient_repo}")
                    success = self.address_store.store_hf_repo(
                        self.args.storage.gradient_repo
                    )
                    if not success:
                        raise ConnectionError(
                            "Failed to store new address in the chain"
                        )
                return True  # Successfully updated or no update needed
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {str(e)}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to update address store after {max_retries} attempts: {str(e)}"
                    )
        return False

    def setup_model_and_tokenizer(self):
        print("Setting up model and tokenizer...")
        model_name = MODEL_NAME
        model_cache_dir = './model_cache'
        os.makedirs(model_cache_dir, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=None,
            cache_dir=model_cache_dir,
            torch_dtype=torch.float16,
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


# args = Configurator.combine_configs()
# BittensorNetwork.initialize(args, ignore_regs=True)
# my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address

# address_store = ChainMultiAddressStore(
#     BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
# )

# current_address_in_store = address_store.retrieve_hf_repo(my_hotkey)
# logging.info(f"Current value in store:{current_address_in_store}")
# if current_address_in_store != args.storage.gradient_repo:
#     logging.info(f"Storing new value: {args.storage.gradient_repo}")
#     address_store.store_hf_repo(args.storage.gradient_repo)

# batch_size = args.miner.batch_size
# num_epochs = args.miner.epochs  # Set to a more reasonable value
# learning_rate = args.miner.learning_rate
# send_interval = args.storage.send_interval  # Every 60 seconds
# sequence_length = args.model.sequence_length


# quantization_config = None

# model_name = "Qwen/Qwen2-1.5B"
# model_cache_dir = './model_cache'  # Specify a local cache directory
# os.makedirs(model_cache_dir, exist_ok=True)

# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, cache_dir=model_cache_dir,
#                                              torch_dtype=torch.float16,device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# optimizer = AdamW(model.parameters(), lr=learning_rate)

# config = LoraConfig(
#     use_dora=True,
#     r=32,
#     lora_alpha=8,
#     target_modules="all-linear",
#     lora_dropout=0.1,
# )
# model = get_peft_model(model, config)

# loader = SubsetFineWebEdu2Loader(
#     batch_size=batch_size,
#     sequence_length=args.model.sequence_length,
#     num_pages=18,
#     tokenizer=tokenizer,
# )


# hf_manager = HFManager(gradient_repo_id=args.storage.gradient_repo, averaged_model_repo_id=args.storage.averaged_model_repo_id,
# gradient_repo_local=args.storage.gradient_repo_local ,averaged_model_repo_local=args.storage.averaged_model_repo_local,
# averaged_miner_assignment_repo_local = args.storage.averaged_miner_assignment_repo_local, averaged_miner_assignment_repo_id = args.storage.averaged_miner_assignment_repo_id)

# validator_instance = ModelValidator(
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         model=model,
#         tokenizer=tokenizer,
#         optimizer=optimizer,
#         check_update_interval=60*60,
#         bittensor_network=BittensorNetwork,
#         chain_manager=address_store,
#         hf_manager=hf_manager,
#         interval=60*60,
#         data_loader=loader
#     )

# validator_instance.start_periodic_validation()

