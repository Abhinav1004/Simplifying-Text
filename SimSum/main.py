'''
Main Program: 
> python main.py
'''
# -- fix path --

import torch
# torch.multiprocessing.set_start_method('forkserver', force=True)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

from preprocessor import WIKI_DOC, D_WIKI, EXP_DIR
import time
import json

#from contextlib import contextmanager
import argparse

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

#from T5_2 import SumSim, train
#from Bart2 import SumSim, train
from Bart_baseline_finetuned import BartBaseLineFineTuned, train
#from T5_baseline_finetuned import T5BaseLineFineTuned, train


def parse_arguments():
    #p = SumSim.add_model_specific_args(p)
    p = ArgumentParser()

    p.add_argument('--seed', type=int, default=42, help='randomization seed')

    # Add your model-specific arguments
    p = BartBaseLineFineTuned.add_model_specific_args(p)
    # p = T5BaseLineFineTuned.add_model_specific_args(p)

    # Manually add the PyTorch Lightning Trainer arguments
    p.add_argument('--max_epochs', type=int, default=10, help='Max number of training epochs')
    p.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    p.add_argument('--precision', type=int, default=32, help='Precision for training')
    p.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clipping value')
    p.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate gradients over N batches')
    p.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    p.add_argument('--accelerator', type=str, default='cpu', help='Type of accelerator (cpu, gpu, tpu, etc.)')

    args, _ = p.parse_known_args()
    return args

# class MetricsCallback(pl.Callback):
#   def __init__(self):
#     super().__init__()
#     self.metrics = []
  
#   def on_validation_end(self, trainer, pl_module):
#       self.metrics.append(trainer.callback_metrics)

# Create experiment directory
def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir == True: path.mkdir(parents=True, exist_ok=True)
    return path

def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = dict()
    for key in kwargs:
        kwargs_str[key] = str(kwargs[key])
    json.dump(kwargs_str, filepath.open('w'), indent=4)



def run_training(args, dataset):

    args.output_dir = get_experiment_dir(create_dir=True)
    # logging the args
    log_params(args.output_dir / "params.json", vars(args))

    args.dataset = dataset
    print("Dataset: ",args.dataset)
    train(args)



dataset = WIKI_DOC
args = parse_arguments()
run_training(args, dataset)

