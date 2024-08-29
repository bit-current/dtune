import os
import torch
import argparse
#from loguru import logger
from argparse import ArgumentParser
from .hivetrain_config import add_meta_miner_args, add_orchestrator_args, add_torch_miner_args #s, add_validator_args
from .base_subnet_config import add_neuron_args, add_validator_args, add_miner_args
import argparse
from typing import Any, Dict
from collections import defaultdict

# def check_config(cls, config: "bt.Config"):
#     r"""Checks/validates the config namespace object."""
#     bt.logging.check_config(config)

#     full_path = os.path.expanduser(
#         "{}/{}/{}/netuid{}/{}".format(
#             config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
#             config.wallet.name,
#             config.wallet.hotkey,
#             config.netuid,
#             config.neuron.name,
#         )
#     )
#     print("full path:", full_path)
#     config.neuron.full_path = os.path.expanduser(full_path)
#     if not os.path.exists(config.neuron.full_path):
#         os.makedirs(config.neuron.full_path, exist_ok=True)

#     if not config.neuron.dont_save_events:
#         # Add custom event logger for the events.
#         logger.level("EVENTS", no=38, icon="📝")
#         logger.add(
#             os.path.join(config.neuron.full_path, "events.log"),
#             rotation=config.neuron.events_retention_size,
#             serialize=True,
#             enqueue=True,
#             backtrace=False,
#             diagnose=False,
#             level="EVENTS",
#             format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
#         )

class NestedNamespace:
    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, NestedNamespace(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        return None

def nested_dict():
    return defaultdict(nested_dict)

def dict_to_nested_namespace(d):
    if not isinstance(d, dict):
        return d
    return NestedNamespace({k: dict_to_nested_namespace(v) for k, v in d.items()})

class Config:
    @staticmethod
    def create_config(parser):
        args = parser.parse_args()
        config_dict = nested_dict()

        for arg, value in vars(args).items():
            if value is not None:
                parts = arg.split('.')
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value

        return dict_to_nested_namespace(config_dict)

class Configurator:
    @staticmethod
    def combine_configs():
        parser = ArgumentParser(description="Unified Configuration for Bittensor")

        add_torch_miner_args(parser)
        add_meta_miner_args(parser)
        add_orchestrator_args(parser)
        add_neuron_args(parser)
        add_miner_args(parser)
        add_validator_args(parser)
        args = parser.parse_args()
        return Config.create_config(parser)