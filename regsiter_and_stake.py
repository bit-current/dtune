import json
from substrateinterface import Keypair
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator
import time
import bittensor as bt

# Load the JSON file
with open('wallet_mnemonics.json') as file:
    wallets = json.load(file)
    breakpoint()

# Get the TAO in 5HgUf8QGQikfR2pQLXgYJNgAhvp1yy5KkS345KwHZQqMjH2r 

args = Configurator.combine_configs()
BittensorNetwork.initialize(args, ignore_regs=True)
subtensor = BittensorNetwork.subtensor
wallet = BittensorNetwork.wallet
temp_wallet = bt.wallet()
metagraph = BittensorNetwork.metagraph

register = False
stake_validator = True
show_balance = True

for idx, wallet_data in enumerate(wallets):
    # Get the mnemonic for the cold key
    
    coldkey_mnemonic = wallet_data['coldkey_mnemonic']
    
    # Create a Keypair using the cold key mnemonic
    coldkey_keypair = Keypair.create_from_mnemonic(coldkey_mnemonic)
    
    # Get the SS58 address for the cold key
    receive_ss58_address = coldkey_keypair.ss58_address
    temp_wallet.set_coldkey(coldkey_keypair, encrypt=False, overwrite=True)
    temp_wallet.set_coldkeypub(coldkey_keypair, overwrite=True)
    
    hotkey_mnemonic = wallet_data['hotkey_mnemonic']
    hotkey_keypair = Keypair.create_from_mnemonic(hotkey_mnemonic)
    hotkey_address = hotkey_keypair.ss58_address
    temp_wallet.set_hotkey(hotkey_keypair, encrypt=False, overwrite=True)

    if show_balance:
        if idx < 90 or idx > 90:
            continue

        cold_balance = subtensor.get_balance(receive_ss58_address)
        hot_balance = subtensor.get_total_stake_for_coldkey(hotkey_address)

        print(f"Cold Balance for test-wallet-{idx+1}: {cold_balance}")
        print(f"Hot Balance for test-wallet-{idx+1}: {cold_balance}")

    if register:        
        if idx < 89:
           continue
        print(f"Registering test-wallet-{idx+1} on Netuid {args.netuid}")
        if hotkey_address in metagraph.hotkeys:
            print("Already Registered")
            continue
        subtensor.burned_register(
            wallet=temp_wallet, netuid=args.netuid, prompt=False
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
        success = subtensor.add_stake(
            wallet=temp_wallet,
            hotkey_ss58=hotkey_address,
            amount=10.0,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            prompt=False,
        )
        if success:
            print(f"Success")
        else:
            print("Failed")

