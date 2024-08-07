from communex._common import get_node_url

from communex.client import CommuneClient
from communex.compat.key import classic_load_key
import numpy as np
import torch
import time
from typing import List, Tuple
import threading
from . import __spec_version__
from communex.misc import get_map_modules
import functools
import multiprocessing
from typing import Any
import sys
import random 
from time import sleep


def retry(max_retries: int | None, retry_exceptions: list[type]):
    assert max_retries is None or max_retries > 0

    def decorator(func):
        def wrapper(*args, **kwargs):
            max_retries__ = max_retries or sys.maxsize  # TODO: fix this ugly thing
            for tries in range(max_retries__ + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if any(isinstance(e, exception_t) for exception_t in retry_exceptions):
                        func_name = func.__name__
                        print(f"An exception occurred in '{func_name} on try {tries}': {e}, but we'll retry.")
                        if tries < max_retries__:
                            delay = (1.4 ** tries) + random.uniform(0, 1)
                            sleep(delay)
                            continue
                    raise e
            raise Exception("Unreachable")
        return wrapper
    return decorator

@retry(5, [Exception])
def _make_client(node_url: str):
    return CommuneClient(url=node_url, num_connections=1, wait_for_finalization=False)


def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    try:
        result = func()
        queue.put(result)
    except (Exception, BaseException) as e:
        # Catch exceptions here to add them to the queue.
        queue.put(e)

def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result

class CommuneNetwork:
    _instance = None
    _lock = threading.Lock()  # Singleton lock
    _weights_lock = threading.Lock()  # Lock for set_weights
    _anomaly_lock = threading.Lock()  # Lock for detect_metric_anomaly
    _config_lock = threading.Lock()  # Lock for modifying config
    _rate_limit_lock = threading.Lock()
    metrics_data = {}
    model_checksums = {}
    request_counts = {}  # Track request counts
    blacklisted_addresses = {}  # Track blacklisted addresses
    last_sync_time = 0
    last_set_block = 0
    sync_interval = 600


    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CommuneNetwork, cls).__new__(cls)
                cls.config = None
        return cls._instance
    
    # Check that hotkey is registered on SN
    # modules_keys = self.client.query_map_key(syntia_netuid)
    # val_ss58 = self.key.ss58_address
    # if val_ss58 not in modules_keys.values():
    #     raise RuntimeError(f"validator key {val_ss58} is not registered in subnet")

    @classmethod
    def check_registered(cls):
        assert cls.my_hotkey in cls.hotkeys


    @classmethod
    @property
    def client(cls):
        node = get_node_url()
        current_client = _make_client(node)
        return current_client


    @classmethod
    def initialize(cls, config, ignore_regs=False):
        with cls._lock:
            
            #cls.client = CommuneClient(url="wss://commune-api-node-6.communeai.net")
            cls.keypair = classic_load_key(config.key_name) #TODO env var me TODO add assertion that key loaded well?
            cls.netuid = config.netuid
            
            cls.config = config
            cls.ignore_regs = ignore_regs

            temp_hotkeys = cls.client.query_map_key(cls.netuid)
            cls.hotkeys = [] #for backward compat and quick release
            for uid in range(len(temp_hotkeys.values())):
                cls.hotkeys.append(temp_hotkeys[uid])
                
            cls.my_hotkey = cls.keypair.ss58_address
            cls.my_uid = cls.hotkeys.index(
                cls.my_hotkey
            )

            if not cls.ignore_regs:
                try:
                    cls.check_registered()
                except AssertionError:
                    print(f"Key: {cls.my_hotkey} is not registered on netuid {config.netuid}. Please register the hotkey before trying again")
                    exit()
            
            cls.device="cpu"
            cls.base_scores = {}
        # Additional initialization logic here

    @classmethod
    @property
    def last_block(cls):
        return cls.client.get_block()['header']['number']

    @classmethod
    def set_weights(cls, scores):
        '''
            scores: dict of {uid:score}
            #comm client is from the comm connector class
            #netuid is from comm connector class,
            #keypair substrate pub/prv keypair is from comm connector class
        '''
        
        try:
            print(f"Scores: {scores}")
            #chain_weights = torch.zeros(cls.subtensor.subnetwork_n(netuid=cls.metagraph.netuid))
            min_new_score = min(scores.values())
            max_new_score = max(scores.values())
            normalized_new_scores = {k: (v - min_new_score) / (max_new_score - min_new_score) for k, v in scores.items()}
            print(f"Normalized: {normalized_new_scores}")

            uids = []
            for uid, public_address in enumerate(cls.hotkeys):
                try:
                    alpha = 0.333333 # T=5 (2/(5+1))
                    normalized_new_score = normalized_new_scores.get(public_address, 0.0)
                    try:
                        cls.base_scores[uid] = alpha * normalized_new_score + (1 - alpha) * cls.base_scores[uid]
                    except KeyError:
                        cls.base_scores[uid] = alpha * normalized_new_score
                except KeyError:
                    continue

            uids = torch.tensor(uids)
            print(f"raw_weights {cls.base_scores}")
            print(f"raw_weight_uids {uids}")
            # Process the raw weights to final_weights via subtensor limitations.
            #cls.base_scores
            cls.base_scores = {k: v for k, v in cls.base_scores.items() if v != 0.0}

            uids = list(cls.base_scores.keys())
            weights = list(cls.base_scores.values())

            # Set weights on comm
            cls.client.vote(key=cls.keypair, uids=uids, weights=weights, netuid=cls.netuid)
            cls.last_set_block = cls.last_block

                
        except Exception as e:
            print(f"Error setting weights: {e}")

    @classmethod
    def should_set_weights(cls) -> bool:
        with cls._lock:  # Assuming last_update modification is protected elsewhere with the same lock
            return (cls.last_block - cls.last_set_block) > cls.config.neuron.epoch_length

    @classmethod
    def resync_metagraph(cls,lite=True):
        
        # Fetch the latest state of the metagraph from the Bittensor network
        print("Resynchronizing metagraph...")
        # Update the metagraph with the latest information from the network
        temp_hotkeys = cls.client.query_map_key(cls.netuid)
        cls.hotkeys = [] #for backward compat and quick release
        for uid in range(len(temp_hotkeys.values())):
            cls.hotkeys.append(temp_hotkeys[uid])

        cls.get_stakes()

        if not cls.ignore_regs:
            try:
                cls.check_registered()
            except AssertionError:
                print(f"Key: {cls.my_hotkey} is no longer registered on netuid {cls.netuid}. Please register the hotkey before trying again")
                exit()
        
        print("Metagraph resynchronization complete.")

    @staticmethod
    def should_sync_metagraph(last_sync_time,sync_interval):
        current_time = time.time()
        return (current_time - last_sync_time) > sync_interval

    @classmethod
    def sync(cls, lite=True):
        if cls.should_sync_metagraph(cls.last_sync_time,cls.sync_interval):
            # Assuming resync_metagraph is a method to update the metagraph with the latest state from the network.
            # This method would need to be defined or adapted from the BaseNeuron implementation.
            try:
                cls.resync_metagraph(lite)
                cls.last_sync_time = time.time()
            except Exception as e:
                print(f"Failed to resync metagraph: {e}")
        else:
            print("Metagraph Sync Interval not yet passed")

    @classmethod
    def get_validator_uids(
        cls, vpermit_comm_limit: int = 1000
    ):
        """
        Check availability of all UIDs in a given subnet, returning their IP, port numbers, and hotkeys
        if they are serving and have at least vpermit_tao_limit stake, along with a list of strings
        formatted as 'ip:port' for each validator.

        Args:
            metagraph (bt.metagraph.Metagraph): Metagraph object.
            vpermit_tao_limit (int): Validator permit tao limit.

        Returns:
            Tuple[List[dict], List[str]]: A tuple where the first element is a list of dicts with details
                                            of available UIDs, including their IP, port, and hotkeys, and the
                                            second element is a list of strings formatted as 'ip:port'.
        """
        vpermit_comm_limit = vpermit_comm_limit* 1000000000
        validator_uids = []  # List to hold 'ip:port' strings
        for uid in range(len(cls.hotkeys)):
            if cls.stakes[uid] >= vpermit_comm_limit:
                validator_uids.append(uid)
        return validator_uids

    @staticmethod
    def sort_and_extract_attribute(modules, attribute):
        # Convert the dictionary values to a list of tuples: (uid, attribute_value)
        module_list = [(item['uid'], item[attribute]) for item in modules.values()]
        
        # Sort the list based on the uid
        sorted_modules = sorted(module_list, key=lambda x: x[0])

        #TODO add assertion that uids are continous with no missing values
        
        # Extract only the attribute values, maintaining the sorted order
        return [item[1] for item in sorted_modules]
    
    @classmethod
    def get_stakes(cls):
        """Retrieves and decompresses multiaddress on this subnet for specific hotkey"""
        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            get_map_modules, cls.client, netuid=cls.netuid, include_balances=False
        )
        try:
            modules = run_in_subprocess(partial, 120)
        except:
            modules = None
            print(f"Failed to retreive stakes")
            return

        cls.stakes = cls.sort_and_extract_attribute(modules, "stake")
