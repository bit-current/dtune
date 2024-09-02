# DTune Miner + Validator Tutorial: Running Miner, Averager, and Validator

This tutorial will guide you through the process of setting up and running a node on the dtune subnet. We'll cover two methods of execution: using Docker and running the scripts manually.

## Prerequisites

Before you begin, make sure you have the following:

1. Docker installed (for the Docker method)
2. Python environment set up (for the manual method)
3. A Hugging Face account and API token
4. A Weights & Biases (wandb) account and API token

## Method 1: Using Docker

### Step 1: Create an HF weight repo + API

Go to Huggingface and create a new repo and a read + write token

### Step 2: Set up environment variables

First, set up the required environment variables. Replace the placeholders with your actual values:

```bash
export NETUID=16
export KEY_MNEMONIC="your mnemonic phrase here"
export AVERAGED_REPO="mekaneeky/gpt-neo-averager"
export WEIGHT_REPO="your_huggingface_username/your_weight_repo"
export BATCH_SIZE=1  # Adjust based on your GPU memory
export HF_TOKEN="your_huggingface_token"
export WANDB_TOKEN="your_wandb_token"
export AVERAGED_MINER_ASSIGNMENT_REPO="mekaneeky/averager-miner-assign"
```

### Step 3: Run the Docker container

Use the following command to run the Docker container:

```bash
docker run -d \
  --entrypoint /RDbara/run_additional_commands.sh \
  --image mekaneeky/test_comm_cluster \
  --disk 200 \
  -e NETUID=$NETUID \
  -e KEY_MNEMONIC="$KEY_MNEMONIC" \
  -e AVERAGED_REPO=$AVERAGED_REPO \
  -e AVERAGED_REPO_DIR=averager_repo \
  -e WEIGHT_REPO="$WEIGHT_REPO" \
  -e WEIGHT_REPO_DIR=weight_repo \
  -e BATCH_SIZE=$BATCH_SIZE \
  -e HF_TOKEN=$HF_TOKEN \
  -e ROLE=miner \
  -e WANDB_TOKEN=$WANDB_TOKEN \
  -e AVERAGED_MINER_ASSIGNMENT_REPO=$AVERAGED_MINER_ASSIGNMENT_REPO \
  -e AVERAGED_MINER_ASSIGNMENT_DIR=averaged_assignment
```

Note: Adjust the `ROLE` environment variable to `averager` or `validator` depending on which component you want to run.

## Method 2: Running Manually

### Step 1: Create an HF weight repo + API

Go to Huggingface and create a new repo and a read + write token

### Step 2: Set up environment variables

Set up the same environment variables as in the Docker method:

```bash
export NETUID=16
export KEY_MNEMONIC="your mnemonic phrase here"
export AVERAGED_REPO="mekaneeky/gpt-neo-averager"
export WEIGHT_REPO="your_huggingface_username/your_weight_repo"
export WEIGHT_REPO_DIR="./weight_repo"
export AVERAGED_REPO_DIR="./averaged_repo"
export BATCH_SIZE=32  # Adjust based on your GPU memory
export HF_TOKEN="your_huggingface_token"
export WANDB_TOKEN="your_wandb_token"
export AVERAGED_MINER_ASSIGNMENT_REPO="mekaneeky/averager-miner-assign"
export AVERAGED_MINER_ASSIGNMENT_DIR="averager_assign"
export MODULE_NAME="your_comm_module_name"
```

### Step 3: Clone repo

```
git clone https://github.com/bit-current/dtune
cd dtune
```

### Step 4: Install repo

```
pip install -e . 
```

### Step 5: Regen registered key

```
comx key regen runtime_key "$KEY_MNEMONIC"
```

### Step 5.1: Update registered module to have the hugginface repo as an address

```
comx module update runtime_key 16 --name "$WEIGHT_REPO"
```

### Step 6: Set up wandb

```
wandb login $WANDB_TOKEN
```
 
### Step 7: Set up huggingface

```
huggingface-cli login --token $HF_TOKEN
```

### Step 8: Install git lfs

```
sudo apt install git-lfs
git lfs install
```

### Step 9: Run the components

#### To run the miner:

```bash
python neurons/prototype_miner.py \
  --netuid $NETUID \
  --storage.gradient_repo $WEIGHT_REPO \
  --storage.averaged_model_repo_id $AVERAGED_REPO \
  --storage.gradient_repo_local "gradient_repo" \
  --storage.averaged_model_repo_local "averaged_repo" \
  --storage.averaged_miner_assignment_repo_id $AVERAGED_MINER_ASSIGNMENT_REPO \
  --storage.averaged_miner_assignment_repo_local "averager_assignment" \
  --miner.batch_size $BATCH_SIZE \
  --key_name runtime_key \
  > miner.log 2> miner.err
```

#### To run the averager:

```bash
python neurons/prototype_averager.py \
  --subtensor.network test \
  --netuid $NETUID \
  --storage.averaged_model_repo_id $AVERAGED_REPO \
  --storage.averaged_model_repo_local "averaged_repo" \
  --storage.averaged_miner_assignment_repo_id $AVERAGED_MINER_ASSIGNMENT_REPO \
  --storage.averaged_miner_assignment_repo_local "averager_assignment" \
  --miner.batch_size $BATCH_SIZE \
  > averager.log 2> averager.err
```

#### To run the validator:

```bash
python neurons/prototype_validator.py \
  --netuid $NETUID \
  --storage.gradient_repo $WEIGHT_REPO \
  --storage.averaged_model_repo_id $AVERAGED_REPO \
  --storage.gradient_repo_local "gradient_repo" \
  --storage.averaged_model_repo_local "averaged_repo" \
  --storage.averaged_miner_assignment_repo_id $AVERAGED_MINER_ASSIGNMENT_REPO \
  --storage.averaged_miner_assignment_repo_local "averager_assignment" \
  --miner.batch_size $BATCH_SIZE \
  --key_name runtime_key \
  > validator.log 2> validator.err
```

## Important Notes

1. For miners and validators, create a new Hugging Face repository to store weights and point the code to this repository using the `WEIGHT_REPO` variable.
2. The `AVERAGED_REPO` is a pre-existing Hugging Face repository provided in the tutorial for miners and validators to download the latest consensus/averaging model. 
3. The `AVERAGED_MINER_ASSIGNMENT_REPO` is used for storing miner assignments and should be set to the provided default value unless otherwise specified. 
4. Adjust the `BATCH_SIZE` to the largest value that doesn't cause out-of-memory (OOM) errors on your GPU.
5. Make sure to keep your mnemonic phrase and API tokens secure and never share them publicly.

By following this tutorial, you should be able to set up and run the dtune distributed training system using either Docker or manual execution. Remember to monitor the log files for any errors or important information during the execution.
