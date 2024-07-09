import json
from substrateinterface import Keypair
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator
import time

# Load the JSON file
with open('wallet_mnemonics.json') as file:
    wallets = json.load(file)

args = Configurator.combine_configs()
BittensorNetwork.initialize(args, ignore_regs=True)
subtensor = BittensorNetwork.subtensor
wallet = BittensorNetwork.wallet
metagraph = BittensorNetwork.metagraph

register = False
stake_validator = True

for idx, wallet_data in enumerate(wallets):
    # Get the mnemonic for the cold key
    
    coldkey_mnemonic = wallet_data['coldkey_mnemonic']
    
    # Create a Keypair using the cold key mnemonic
    coldkey_keypair = Keypair.create_from_mnemonic(coldkey_mnemonic)
    
    # Get the SS58 address for the cold key
    receive_ss58_address = coldkey_keypair.ss58_address
    wallet.set_coldkey(coldkey_keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(coldkey_keypair, overwrite=True)
    
    hotkey_mnemonic = wallet_data['hotkey_mnemonic']
    hotkey_keypair = Keypair.create_from_mnemonic(hotkey_mnemonic)
    hotkey_address = hotkey_keypair.ss58_address
    wallet.set_hotkey(hotkey_keypair, encrypt=False, overwrite=True)

    if register:        
        if idx < 89:
           continue
        print(f"Registering test-wallet-{idx+1} on Netuid {args.netuid}")
        if hotkey_address in metagraph.hotkeys:
            print("Already Registered")
            continue
        subtensor.burned_register(
            wallet=wallet, netuid=args.netuid, prompt=False
        )
        time.sleep(10)

    if stake_validator:
        if idx < 89 or idx > 94:
            continue
        # result = subtensor.nominate(wallet)
        # if not result:
        #     print("Failed to nominate delegate")
        #     continue
        print(f"Staking to test-wallet-{idx+1}")
        subtensor.add_stake(
            wallet=wallet,
            hotkey_ss58=hotkey_address,
            amount=10.0,
            wait_for_inclusion=True,
            prompt=False,
        )

