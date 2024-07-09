import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from peft import get_peft_model, LoraConfig
from huggingface_hub import HfApi, HfFolder


parser = argparse.ArgumentParser(description="Push model to Hugging Face.")
parser.add_argument('--repo_id', type=str, default="mekaneeky/gpt-neo-averager", help='Hugging Face repository ID')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-1.5B", help='Name of the model to use')
args = parser.parse_args()

# Model initialization code
model_name = args.model_name
model_cache_dir = '../trash_dir'  # Specify a local cache directory
os.makedirs(model_cache_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

learning_rate = 5e-5  # You can adjust the learning rate if needed
optimizer = AdamW(model.parameters(), lr=learning_rate)

config = LoraConfig(
    use_dora=True,
    r=32,
    lora_alpha=8,
    target_modules="all-linear",
    lora_dropout=0.1,
)
model = get_peft_model(model, config)

averaged_model_path = os.path.abspath(os.path.join(
    "../trash_dir", "averaged_model.pt"
))

# Save only Lora weights
lora_weights = {k: v for k, v in model.state_dict().items() if 'lora' in k}
torch.save(lora_weights, averaged_model_path)

# Push model to Hugging Face
hf_api = HfApi()
hf_token = HfFolder.get_token()  # Ensure you have your Hugging Face token set up
repo_id = args.repo_id


hf_api.upload_file(
    path_or_fileobj=averaged_model_path,
    path_in_repo="averaged_model.pt",
    repo_id=repo_id,
    #repo_type="model",
    token=hf_token
)

validator_repos = [f"mekaneeky/testing-repo-{str(i)}" for i in range(1,101)] #FIXME add all 
validator_repos = validator_repos + ["mekaneeky/validate_me", "mekaneeky/help_me"]
for validator_repo in validator_repos:
    try:
        hf_api.delete_file(
            path_in_repo="gradients.pt",
            repo_id=validator_repo,
            #repo_type="model",
            token=hf_token
        )
    except Exception as e:
        print(f"Failed to delete for repo: {validator_repo}\nError:{e}")

os.remove(averaged_model_path)
print("Averaging + Validator model refreshed")