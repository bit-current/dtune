# Install required packages
#RUN pip install -r requirements.txt
env >> /etc/environment;
pip install expecttest
pip install sentencepiece
pip install -e .

# Set up Git credentials
git config --global credential.helper store

# Install your package in editable mode
#RUN pip install -e .
# Prompt user for Hugging Face token
apt install git-lfs -y
git lfs install 

# Login to Hugging Face CLI
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN

# Create HF repos (manually)
# Note: It's recommended to create the repos beforehand and update the script with the repo names

# Generate keys using btcli
if [ "$ROLE" = "averager" ]; then
    echo "RUNNING AVERAGER"
    python neurons/prototype_averager.py  --subtensor.network test --netuid $NETUID \
    --storage.averaged_model_repo_id $AVERAGED_REPO \
    --storage.averaged_model_repo_local $AVERAGED_REPO_DIR \
    --miner.batch_size $BATCH_SIZE \
    > averager.log 2> averager.err

else
    comx key regen runtime_key "$KEY_MNEMONIC"
    
    # Register hotkey if not registered (manual step)
    # Replace wallet names and netuid as needed
    # NOTE: For ease of use and lack of headache provide an already registered key mnemonic
    # btcli s register --netuid $NETUID --wallet.name $WALLET_NAME --wallet.hotkey $WALLET_HOTKEY --subtensor.network test --no_prompt

    if [ "$ROLE" = "miner" ]; then
        # Run the prototype miner
        echo "RUNNING MINER"
        python neurons/prototype_miner.py \
        --netuid $NETUID \
        --storage.gradient_repo $WEIGHT_REPO --storage.averaged_model_repo_id $AVERAGED_REPO \
        --storage.gradient_repo_local $WEIGHT_REPO_DIR --storage.averaged_model_repo_local $AVERAGED_REPO_DIR \
        --miner.batch_size $BATCH_SIZE \
        --key_name runtime_key \
        > miner.log 2> miner.err


    elif [ "$ROLE" = "validator" ]; then
        
        echo "RUNNING VALIDATOR"
        python neurons/prototype_validator.py \
        --netuid $NETUID \
        --storage.gradient_repo $WEIGHT_REPO --storage.averaged_model_repo_id $AVERAGED_REPO \
        --storage.gradient_repo_local $WEIGHT_REPO_DIR --storage.averaged_model_repo_local $AVERAGED_REPO_DIR \
        --miner.batch_size $BATCH_SIZE \
        --key_name runtime_key \
        > validator.log 2> validator.err
    
    fi
fi