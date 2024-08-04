import os
import torch
import argparse
import bittensor as bt
import time


def add_meta_miner_args(parser):

    parser.add_argument("--key_name", type=str, help="Commune key name")
    parser.add_argument("--key_password", type=str, default=1024, help="Commune key decryption password")

    parser.add_argument("--model.sequence_length", type=int, default=1024, help="Sequence length of model in question")
    ## Standard Pytorch
    parser.add_argument("--miner.batch_size", type=int, default=64, help="Batch size per forward/backward pass") 
    parser.add_argument("--miner.epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--miner.learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference/training")

    ## Validator + Averager communication
    parser.add_argument("--storage.receive_interval", type=int, default=5*60, help="Interval to receive from averager in seconds")
    parser.add_argument("--storage.send_interval", type=int, default=10*60, help="Interval to send to validator in seconds")
    parser.add_argument('--storage.gradient_repo', type=str, help='Local path to gradients/weight deltas')
    parser.add_argument('--storage.averaged_model_repo_id', type=str, help='Huggingface repo for storing final model')
    parser.add_argument('--storage.averaged_model_repo_local', type=str, help='Local clone repo of averaged model')
    parser.add_argument('--storage.gradient_repo_local', type=str, help='Local clone repo of gradient repo')
    parser.add_argument('--storage.averaged_miner_assignment_repo_id', type=str, default="mekaneeky/averager-miner-assign"  ,help='Huggingface repo for storing final model')
    parser.add_argument('--storage.averaged_miner_assignment_repo_local', type=str, default="averager_assign" ,help='Huggingface repo for storing final model')

def add_torch_miner_args(parser):
    parser.add_argument('--rank', type=int, help='Rank of process/node in training run')
    parser.add_argument('--world-size', type=int, help='Number of processes/nodes in training run')
    parser.add_argument('--store-address', type=str,default="127.0.0.1", help='IP/URL of the TCPStore')#FIXME add the main from btt
    parser.add_argument('--store-port', type=int,default=4999, help='Port of the test TCPStore')#FIXME add the main from btt
    parser.add_argument(
    "--initial_peers",
    action="append",
    help="Add a peer. Can be used multiple times to pass multiple peers.",
    nargs="*",
    default=[],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="The largest batch size able to fit on your GPU.",
        default=1,
        const=1,
        nargs="?",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        help="Save the model every X global steps.",
        default=0,
        const=0,
        nargs="?",
    )



def add_orchestrator_args(parser):
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host-address', type=str, default="127.0.0.1")

# def add_validator_args(parser):
#     parser.add_argument('--port', type=int, default=5000, help="Port for the validator")
#     parser.add_argument('--host-address', type=str, default="127.0.0.1", help="Host address for the validator")
