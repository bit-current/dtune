import json
import time
import threading
import bittensor as bt
from substrateinterface import Keypair
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator

class RegistrationThread(threading.Thread):
    def init(self, subtensor, wallet):
        threading.Thread.init(self)
        self.subtensor = subtensor
        self.wallet = wallet
        self.success = False
    def run(self):
        while not self.success:
            success = self.subtensor.register_subnetwork(
                self.wallet,
                wait_for_inclusion=True,
                wait_for_finalization=True
            )
            if success:
                self.success = True
                print(f"Registration successful for thread {self.ident}")
            else:
                print(f"Trying again for thread {self.ident}")
                time.sleep(1)

def main():
    args = Configurator.combine_configs()
    BittensorNetwork.initialize(args, ignore_regs=True)
    subtensor = BittensorNetwork.subtensor
    wallet = BittensorNetwork.wallet
    temp_wallet = bt.wallet()
    metagraph = BittensorNetwork.metagraph
    threads = []
    num_threads = 1  # You can adjust this number

    for _ in range(num_threads):
        thread = RegistrationThread(subtensor, wallet)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print("All registrations completed")
if __name__ == "__main__":
    main()