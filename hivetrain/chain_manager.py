#Thanks SN9
import multiprocessing
import functools
import os
import lzma
import base64
import multiprocessing
from typing import Optional, Any, Dict
from communex.misc import get_map_modules
import time 


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


class ChainMultiAddressStore:
    """Chain based implementation for storing and retrieving multiaddresses."""

    # To write
    # Steps before calling. 
    # 1) Keypair
    # 2) Module name (metagraph like obj?)
    # 3) Module Address ()
    # response = client.update_module(
    #         key=resolved_key,
    #         name=module["name"],
    #         address=module["address"],
    #         delegation_fee=20,#module["delegation_fee"],
    #         netuid=netuid,
    #         metadata=module["metadata"],
    #     )
    #
    #
    # To read
    # 
    # 

    def __init__(
        self,
        client,
        netuid,
        keypair,
        name
        
    ):
        self.client = client
        self.netuid = netuid
        self.keypair = keypair
        self.name = name
        self._cached_modules = None
        self._last_cache_time = 0
        self.CACHE_EXPIRATION_TIME = 600  # 10 minutes in seconds
        self.MAX_RETRIES = 5

    def store_hf_repo(self, hf_repo: str):
        """Stores compressed multiaddress on this subnet for a specific wallet."""
        if self.keypair is None:
            raise ValueError("No key available to write to the chain.")

        # Compress the multiaddress

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial( 
            self.client.update_module,
            key=self.keypair,
            name=self.name,
            address=hf_repo,
            metadata=None,
            netuid=self.netuid
        )
        
        run_in_subprocess(partial, 60)

    def _get_cached_modules(self) -> Optional[Dict[str, Any]]:
        """Fetches and caches the modules data, retrying every 10 seconds if modules are None."""
        current_time = time.time()
        if current_time - self._last_cache_time > self.CACHE_EXPIRATION_TIME:
            self.clear_cache()
        
        if self._cached_modules is None:
            partial = functools.partial(
                get_map_modules, self.client, netuid=self.netuid, include_balances=False
            )
            
            for attempt in range(self.MAX_RETRIES):
                try:
                    self._cached_modules = run_in_subprocess(partial, 120)
                    if self._cached_modules is not None:
                        self._last_cache_time = current_time
                        return self._cached_modules
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed to retrieve modules for netuid: {self.netuid}. Error: {str(e)}")
                
                if attempt < self.MAX_RETRIES - 1:
                    print(f"Retrying in 10 seconds...")
                    time.sleep(10)
            
            print(f"Failed to retrieve modules for netuid: {self.netuid} after {self.MAX_RETRIES} attempts")
            return None

        return self._cached_modules


    def retrieve_hf_repo(self, uid: int) -> Optional[str]: #TODO address whole subnet queries with a single request
        """Retrieves and decompresses multiaddress on this subnet for specific hotkey"""
        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        modules = self._get_cached_modules()
        if modules is None:
            print("Module empty -- WARNING")
            return None

        modules_to_list = [value for _, value in modules.items()]
        try:
            hf_repo = next((item for item in modules_to_list if item["uid"] == uid), None)['address']
        except Exception as e:
            print(f"Retreival failed: {e}")
            return None

        return hf_repo
    
    def clear_cache(self):
        """Clears the cached modules data."""
        self._cached_modules = None
        self._last_cache_time = 0
        
# Synchronous test cases for ChainMultiAddressStore
