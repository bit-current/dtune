## Thanks SN9 !!

import torch
import typing
import random
import time
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint

class SubsetFineWebEdu2Loader(IterableDataset):

    name: str = "HuggingFaceFW/fineweb-edu"    
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries
    
    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer=None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.num_rows_per_page = 100
        self.tokenizer = tokenizer

        self.buffer = []

        # Get the dataset configs and their row sizes
        self.configs_data = self.fetch_dataset_configs()

        # We first need to fetch the data and fill the loader buffer.
        # Since some sample files are broken, we first try to find `num_pages`
        # responsive samples, then we add them to the found pages `self.pages`
        if self.num_pages:
            self._fetch_data_to_buffer(self.num_pages)

        self.buffer_position = 0  # Add this line to track position in buffer

            
    def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        this method sets the `pages` property
        """
        
        self.pages = []
        attempts = 0
        
        while len(self.pages) < num_pages:

            # randomly sample one page
            config_name, page, split = self.get_random_pages(num_pages = 1)[0]
            
            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page,
                          limit=self.num_rows_per_page
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Add the page since the request was successful
                self.pages.append((config_name, page, split))
                
                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    self.buffer += self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += [self.tokenizer.eos_token_id]

            except requests.exceptions.RequestException as e:
                attempts += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def fetch_data_for_pages(self, pages):
        """
        Set the pages to be used to fill the buffer. Then fetch the page data
        to the buffer.
        """

        self.pages = pages
        
        # Empty the buffer if it is not.
        self.buffer = []

        for page in self.pages:
            self._fetch_data_for_page(page)

    def _fetch_data_for_page(self, page):

        retry_limit = 10
        
        attempt = 0
        while attempt < retry_limit:
            config_name, page, split = page

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page,
                          limit=self.num_rows_per_page
            )
            
            try:

                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    self.buffer += self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += [self.tokenizer.eos_token_id]
                    
                break  # If the request was successful, break out of the retry loop
            
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise
                
    def fetch_data_to_rows(self, num_pages):

        rows = []
        attempts = 0
        num_downloaded_pages = 0
        
        while num_downloaded_pages < num_pages:

            # randomly sample one page
            config_name, page, split = self.get_random_pages(num_pages = 1)[0]
            
            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page,
                          limit=self.num_rows_per_page
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                num_downloaded_pages += 1
                
                for row in response.json()["rows"]:
                    rows.append(row["row"]["text"])

            except requests.exceptions.RequestException as e:
                attempts += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

                
        return rows
    
    def get_random_pages(self, num_pages):
        """
        Randomly sample one page.
        A page is a row number of a given split of a given dataset dump.
        """
        pages = []
        
        for _ in range(num_pages):
            
            # Choose a random config
            config_name = random.choice(list(self.configs_data.keys()))

            # Choose a random page (row)
            page = random.randint(0,
                                  self.configs_data[config_name]['num_rows'] - 1 - self.num_rows_per_page)

            split = self.configs_data[config_name]['split']

            pages.append((config_name, page, split))

        return pages

    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple
        """

        page_names = []
        
        if hasattr(self, 'pages'):
            page_names = [f'{cfg_name}_{num_rows}_{split}' for
                           cfg_name, num_rows, split in self.pages]
            
        return page_names
        
    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(
            dataset = self.name
            )
        
        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.size_base_url, params=params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Extract the configs dict
                configs_dict = response.json()['size']['splits']

                # Now create a dict with config names (except 'default') as
                # keys, and the number of rows as values
                configs_data = {entry['config']: {'num_rows': entry['num_rows'] ,
                                                  'split': entry['split']}
                                for entry in configs_dict
                                if entry['config'] != 'default'
                                }                

                return configs_data
                    
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise
                
    def __iter__(self):
        self.buffer_position = 0  # Reset position at the start of iteration
        while self.buffer_position + self.sequence_length * self.batch_size <= len(self.buffer):
            batch_input_ids = []
            batch_attention_masks = []
            batch_labels = []
            for _ in range(self.batch_size):
                end_pos = self.buffer_position + self.sequence_length
                input_ids = self.buffer[self.buffer_position:end_pos]
                attention_mask = [1] * len(input_ids)
                
                # Pad sequences if necessary
                if len(input_ids) < self.sequence_length:
                    padding_length = self.sequence_length - len(input_ids)
                    input_ids += [self.tokenizer.pad_token_id] * padding_length
                    attention_mask += [0] * padding_length
                
                batch_input_ids.append(torch.tensor(input_ids))
                batch_attention_masks.append(torch.tensor(attention_mask))
                batch_labels.append(torch.tensor(input_ids))  # For causal language modeling, labels are the same as input_ids
                
                self.buffer_position = end_pos
            
            yield {
                "input_ids": torch.stack(batch_input_ids),
                "attention_mask": torch.stack(batch_attention_masks),
                "labels": torch.stack(batch_labels)
            }

    def __next__(self):
        if self.buffer_position + self.sequence_length * self.batch_size > len(self.buffer):
            self.buffer_position = 0  # Reset position for next iteration
            raise StopIteration
        
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        for _ in range(self.batch_size):
            end_pos = self.buffer_position + self.sequence_length
            input_ids = self.buffer[self.buffer_position:end_pos]
            attention_mask = [1] * len(input_ids)
            
            # Pad sequences if necessary
            if len(input_ids) < self.sequence_length:
                padding_length = self.sequence_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            
            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))
            batch_labels.append(torch.tensor(input_ids))  # For causal language modeling, labels are the same as input_ids
            
            self.buffer_position = end_pos
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_masks),
            "labels": torch.stack(batch_labels)
        }
    
    def refresh_data(self):
        """
        Refreshes the data by fetching new pages and updating the buffer.
        """
        bt.logging.info("Refreshing data with new pages...")
        
        # Clear the existing buffer
        self.buffer = []
        self.buffer_position = 0  # Reset buffer position
        
        # Fetch new pages
        if self.num_pages:
            self._fetch_data_to_buffer(self.num_pages)
        else:
            bt.logging.warning("No number of pages specified. Unable to refresh data.")
        
        bt.logging.info(f"Data refreshed. New buffer size: {len(self.buffer)} tokens")