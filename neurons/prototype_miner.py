import os
import math
import wandb
import torch
import random
import numpy as np
import time
import bitsandbytes
from copy import deepcopy
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from hivetrain.comm_connector import CommuneNetwork
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.config import Configurator
from hivetrain.dataset import SubsetFineWebEdu2Loader
from hivetrain.hf_manager import HFManager


MODEL_NAME = "openai-community/gpt2"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Miner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_wandb()
        print("WANDB set")
        self.setup_bittensor()
        print("WANDB bittensor")
        self.setup_model_and_tokenizer()
        print("Model and optimizer setup")
        self.setup_data_loader()
        print("Dataloader setup")
        self.setup_optimizer()
        print("Optimizer Setup")
        self.hf_manager = HFManager(
            gradient_repo_id=args.storage.gradient_repo,
            averaged_model_repo_id=args.storage.averaged_model_repo_id,
            # gradient_repo_local=args.storage.gradient_repo_local,
            # averaged_model_repo_local=args.storage.averaged_model_repo_local
        )
        self.last_pull_time = 0
        self.last_check_time = 0
        self.last_send_time = time.time()

    def setup_wandb(self):
        """
        Initializes and configures a Weights & Biases (wandb) logging session for tracking and visualizing
        the model training process.
        """
        wandb.init(
            project="distributed-training-v4-1-1-1",
            entity="alizawahry1",
            name=f"miner-{str(time.time())}",
        )

    def setup_bittensor(self):
        """
        Initializes the Bittensor network with provided arguments, sets up the hotkey and unique identifier (UID)
        for this instance, and configures the seed based on the UID. It also initializes an address store to manage
        multiple addresses on the chain.
        
        The function also calls `update_address_store` to refresh the address store with the latest data from the network.
        """
        CommuneNetwork.initialize(self.args)
        set_seed(CommuneNetwork.my_uid)
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
        #FIXME am I a bug? 
        while attempt < max_retries:
            try:
                current_address = self.address_store.retrieve_hf_repo(CommuneNetwork.my_uid)
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
        """
        Sets up the model and tokenizer for training. Loads the specified transformer model and tokenizer
        from the given path, with options for LoRA modifications if specified in the args.
        The model and tokenizer are configured and moved to the appropriate device (e.g., GPU).
        """

        model_name = MODEL_NAME
        model_cache_dir = "./model_cache"
        os.makedirs(model_cache_dir, exist_ok=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=model_cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if True:  # apply_lora
            config = LoraConfig(
                use_dora=True,
                r=32,
                lora_alpha=8,
                target_modules="all-linear",
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, config)
        self.model.to(self.device)

    def setup_data_loader(self):
        self.loader = SubsetFineWebEdu2Loader(
            batch_size=self.args.miner.batch_size,
            sequence_length=self.args.model.sequence_length,
            num_pages=10,
            tokenizer=self.tokenizer,
        )

    def setup_optimizer(self):
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.args.miner.learning_rate
        )

    def train(self):
        print("Training Beginning")
        for epoch in range(self.args.miner.epochs):
            self.train_epoch(epoch)
            self.loader._fetch_data_to_buffer(100)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_examples = 0

        for batch_idx, batch in tqdm(enumerate(self.loader), desc=f"Epoch {epoch+1}"):
            try:
                self.check_for_model_updates()
            except:
                print("Failed to check for model updates")
            loss = self.train_step(batch)
            if loss is not None:
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_examples += batch["input_ids"].size(0)

            self.send_gradients_if_needed(batch_idx, epoch, total_loss, total_examples)

        avg_loss = total_loss / total_examples
        print(f"Epoch {epoch+1} completed. Loss: {avg_loss:.4f}")

    def check_for_model_updates(self):
        if time.time() - self.last_check_time >= self.args.storage.receive_interval:
            if (
                self.hf_manager.check_for_new_submissions(
                    self.hf_manager.averaged_model_repo_id
                )
                or self.last_check_time == 0
            ):
                print("Averaged model updated on Hugging Face. Pulling latest model...")
                self.hf_manager.pull_latest_model()
                time.sleep(10)
                new_model = self.hf_manager.update_model(self.model)
                if new_model is not None:
                    self.model = new_model
                self.setup_optimizer()
            self.last_check_time = time.time()

    def train_step(self, batch):
        inputs = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        if inputs.size(0) == 0 or labels.size(0) == 0:
            return None

        self.optimizer.zero_grad()
        try:
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            return loss
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return None

    def send_gradients_if_needed(self, batch_idx, epoch, total_loss, total_examples):
        if time.time() - self.last_send_time >= self.args.storage.send_interval:
            average_loss = total_loss / total_examples
            perplexity = math.exp(average_loss)
            print(
                f"Epoch: {epoch}, Examples: {total_examples}, Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}"
            )

            wandb.log(
                {
                    "step": batch_idx * (epoch + 1),
                    "batch": batch_idx,
                    "epoch": epoch,
                    "training_loss": average_loss,
                }
            )

            try:
                self.send_gradients()
            except Exception as e:
                print(f"Sending gradients failed: {e}")

            self.last_send_time = time.time()

    def send_gradients(self):
        print("Attempting to send gradients")
        delta_weights = {
            key: value.to("cpu")
            for key, value in self.model.state_dict().items()
            if "lora" in key
        }
        model_gradients_path = os.path.join(
            self.hf_manager.get_local_gradient_directory(), "gradients.pt"
        )
        torch.save(delta_weights, model_gradients_path)
        self.hf_manager.push_gradients("gradients.pt")


def main():
    print("Starting!")
    args = Configurator.combine_configs()
    trainer = Miner(args)
    trainer.train()


if __name__ == "__main__":
    main()
