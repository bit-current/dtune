import json
import time
import multiprocessing
import bittensor as bt
from substrateinterface import Keypair
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator

def registration_process(subtensor, wallet):
    success = False
    while not success:
        success = subtensor.register_subnetwork(
            wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True
        )
        if success:
            print(f"Registration successful for process {multiprocessing.current_process().name}")
        else:
            print(f"Trying again for process {multiprocessing.current_process().name}")
            #time.sleep(5)

def main():
    print("Initializing")
    args = Configurator.combine_configs()
    BittensorNetwork.initialize(args, ignore_regs=True)
    subtensor = BittensorNetwork.subtensor
    wallet = BittensorNetwork.wallet
    temp_wallet = bt.wallet()
    metagraph = BittensorNetwork.metagraph

    print("Initialized")

    processes = []
    num_processes = 7  # You can adjust this number

    for _ in range(num_processes):
        process = multiprocessing.Process(target=registration_process, args=(subtensor, wallet))
        processes.append(process)
        process.start()
        time.sleep(5)

    for process in processes:
        process.join()

    print("All registrations completed")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Necessary for Windows support
    main()