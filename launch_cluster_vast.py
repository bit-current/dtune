import csv
import subprocess
from io import StringIO
import pandas as pd
import re
import argparse
import json
import time 


# Function to run the vastai search command and get the output in memory
def run_search_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    return result.stdout

# Read and sort the TSV data by price using pandas
def read_and_sort_tsv(tsv_data):
    # Replace continuous whitespaces with a single space while preserving newlines
    tsv_data = re.sub(r'[^\S\n]+', ' ', tsv_data)
    tsv_data = tsv_data.replace(" \n", "\n").replace(" ", ",")
    df = pd.read_csv(StringIO(tsv_data), delimiter=',')
    sorted_df = df.sort_values(by='$/hr')
    return sorted_df

# Create an instance using vast.ai CLI
def create_instance(instance_id, netuid, mnemonic_cold, mnemonic_hot, averager_repo, weight_repo, batch_size, role, hf_token, wandb_token):
    print(f'Attempting to create: {instance_id}')
    command = f"""vastai create instance {instance_id} --entrypoint /RDbara/run_additional_commands.sh --image mekaneeky/test_bt_cluster --disk 200 --env '-e NETUID={netuid} -e MNEMONIC_COLD="{mnemonic_cold}" -e MNEMONIC_HOT="{mnemonic_hot}" -e WALLET_NAME=test_wallet -e WALLET_HOTKEY=test_hot -e AVERAGED_REPO={averager_repo} -e AVERAGED_REPO_DIR=averager_repo -e WEIGHT_REPO="mekaneeky/{weight_repo}" -e WEIGHT_REPO_DIR=weight_repo -e BATCH_SIZE={batch_size} -e HF_TOKEN={hf_token} -e ROLE={role} -e WANDB_TOKEN={wandb_token}'"""
    subprocess.run(command, shell=True)

# Load wallet info from JSON file
def load_wallet_info(file_path):
    with open(file_path, 'r') as file:
        wallet_data = json.load(file)
    return wallet_data

# Main function
def main():
    parser = argparse.ArgumentParser(description='Script to create instances using vast.ai CLI.')
    parser.add_argument('--netuid', required=True, help='Netuid for the instances')
    parser.add_argument('--wallet_info_file', required=True, help='Path to the wallet info JSON file')
    parser.add_argument('--averager_repo', required=True, help='Averager repository')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for the instances')
    parser.add_argument('--hf_token', required=True, help='Hugging Face token')
    parser.add_argument('--wandb_token', required=True, help='Weights & Biases token')
    
    parser.add_argument('--n_miners', type=int, default=5, help='Number of miners to create')
    parser.add_argument('--n_validators', type=int, default=1, help='Number of validators to create')
    parser.add_argument('--averager', action='store_true', help='Flag to use averager')
    parser.add_argument('--averager_delay', type=int, default=300, help='Averager delay in seconds')

    args = parser.parse_args()

    search_command = "vastai search offers 'reliability > 0.99 num_gpus==1 dph<=0.99 gpu_ram>=40 duration>=2 inet_up_cost<0.005 inet_down_cost<0.005 inet_down>=1000' -o 'dph-'"

    # Load wallet info from file
    wallet_info_list = load_wallet_info(args.wallet_info_file)
    
    # Run the search command and get the TSV data
    tsv_data = run_search_command(search_command)
    
    # Read and sort the TSV data
    sorted_offers = read_and_sort_tsv(tsv_data)

    # Get the top instance ID
    
    if args.n_miners > 0:
        # Use the first wallet info for this example (modify as needed)
        miner_wallet_info = wallet_info_list[:args.n_miners]
        miner_instance_ids = sorted_offers['ID'].iloc[0:args.n_miners]
        for instance_id, miner_wallet in zip(miner_instance_ids, miner_wallet_info): 
            create_instance(instance_id=instance_id, netuid=args.netuid, mnemonic_cold=miner_wallet["coldkey_mnemonic"], mnemonic_hot=miner_wallet["hotkey_mnemonic"], averager_repo=args.averager_repo, weight_repo=miner_wallet["hf_repo"], batch_size=args.batch_size, role="miner", hf_token=args.hf_token, wandb_token=args.wandb_token)
    
    if args.n_validators > 0:
        validator_wallet_info = wallet_info_list[89:89+args.n_validators]
        validator_instance_ids = sorted_offers['ID'].iloc[args.n_miners:args.n_miners+args.n_validators]
        for instance_id, validator_wallet in zip(validator_instance_ids, validator_wallet_info):
            create_instance(instance_id=instance_id, netuid=args.netuid, mnemonic_cold=validator_wallet["coldkey_mnemonic"], mnemonic_hot=validator_wallet["hotkey_mnemonic"], averager_repo=args.averager_repo, weight_repo=validator_wallet["hf_repo"], batch_size=args.batch_size, role="validator", hf_token=args.hf_token, wandb_token=args.wandb_token)

    if args.averager:
        # Create an instance
        time.sleep(args.averager_delay)
        previous_used_ids = args.n_miners+args.n_validators
        averager_instance_ids = sorted_offers['ID'].iloc[previous_used_ids:previous_used_ids+1]
        for averager_instance_id in averager_instance_ids:
            create_instance(instance_id=averager_instance_id, netuid=args.netuid, mnemonic_cold=None, mnemonic_hot=None, averager_repo=args.averager_repo, weight_repo=None, batch_size=args.batch_size, role="averager", hf_token=args.hf_token, wandb_token=args.wandb_token)

if __name__ == '__main__':
    main()
