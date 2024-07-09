import json
from substrateinterface import Keypair
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator

# Load the JSON file
with open('wallet_mnemonics.json') as file:
    wallets = json.load(file)

args = Configurator.combine_configs()
BittensorNetwork.initialize(args, ignore_regs=True)
subtensor = BittensorNetwork.subtensor
wallet = BittensorNetwork.wallet

validators_only = True

for idx, wallet_data in enumerate(wallets):
    # Get the mnemonic for the cold key
    print(f"sending to test-miner-{idx+1}")
    coldkey_mnemonic = wallet_data['coldkey_mnemonic']
    
    # Create a Keypair using the cold key mnemonic
    coldkey_keypair = Keypair.create_from_mnemonic(coldkey_mnemonic)
    
    # Get the SS58 address for the cold key
    receive_ss58_address = coldkey_keypair.ss58_address

    if not validators_only:
        subtensor.transfer(
            wallet=wallet,
            dest=receive_ss58_address,
            amount=0.1,
            wait_for_inclusion=True,
            prompt=False,
        )
    else:
        if idx != 89:
            continue
        
        subtensor.transfer(
            wallet=wallet,
            dest=receive_ss58_address,
            amount=10.0,
            wait_for_inclusion=True,
            prompt=False,
        )   
