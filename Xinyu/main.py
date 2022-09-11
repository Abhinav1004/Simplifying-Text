'''
Main Program: 
> python main.py
'''
# -- fix path --

import torch
#torch.multiprocessing.set_start_method('forkserver', force=True)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

from preprocessor import D_WIKI,D_WIKI_SMALL,D_WIKI_CLEAN, D_WIKI_MATCH, WIKI_DOC_MID, TURKCORPUS_DATASET, EXP_DIR, WIKI_DOC, WIKI_PARAGH_SMALL, WIKI_DOC_Small, WIKI_PARA_DATASET, Preprocessor,EPFL_NEWS, WIKILARGE_DATASET,WIKILARGE_FILTER_DATASET,WIKI_PARAGH_FILTER_DATASET, WIKI_DOC_CLEAN
import time
import json

#from contextlib import contextmanager
import argparse
#from Ts_T5 import T5FineTuner, train
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
#from new_model import SumSim, train
#from Ts_BART import BartFineTuner, train
from T5_2 import SumSim, train
#from Bart2 import SumSim, train
#from Bart_baseline_finetuned import BartBaseLineFineTuned, train
#from T5_baseline_finetuned import T5BaseLineFineTuned, train


def parse_arguments():
    p = ArgumentParser()
                  
    p.add_argument('-t', '--trials', type=int, default=5,
                  help='number of trials for hyperparameter search')
    p.add_argument('--seed', type=int, default=42, help='randomization seed')
#     p.add_argument('--features_kwargs', default= {
#     # 'WordRatioFeature': {'target_ratio': 0.8},
#     'CharRatioFeature': {'target_ratio': 0.8},
#     'LevenshteinRatioFeature': {'target_ratio': 0.8},
#     'WordRankRatioFeature': {'target_ratio': 0.8},
#     'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
# })
    #p = T5FineTuner.add_model_specific_args(p)
    p = SumSim.add_model_specific_args(p)
    #p = BartFineTuner.add_model_specific_args(p)
    #p = BartBaseLineFineTuned.add_model_specific_args(p)
    #p = T5BaseLineFineTuned.add_model_specific_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args,_ = p.parse_known_args()
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

# @contextmanager
# def log_stdout(filepath, mute_stdout=False):
#     '''Context manager to write both to stdout and to a file'''

#     class MultipleStreamsWriter:
#         def __init__(self, streams):
#             self.streams = streams

#         def write(self, message):
#             for stream in self.streams:
#                 stream.write(message)

#         def flush(self):
#             for stream in self.streams:
#                 stream.flush()

#     save_stdout = sys.stdout
#     log_file = open(filepath, 'w')
#     if mute_stdout:
#         sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
#     else:
#         sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
#     try:
#         yield
#     finally:
#         sys.stdout = save_stdout
#         log_file.close()




def run_training(args, dataset):

    args.output_dir = get_experiment_dir(create_dir=True)
    # logging the args
    log_params(args.output_dir / "params.json", vars(args))

    # if without tokens, delete it
    # preprocessor = Preprocessor(args.features_kwargs)
    # preprocessor.preprocess_dataset(dataset)
    # dataset 
    args.dataset = dataset
    print("Dataset: ",args.dataset)
    train(args)

    #### add logging process if you want
    # with log_stdout(args.output_dir / "logs.txt"):
    #     #train_summary(args)
    #     train(args)



     ## MLO98 (tmux 1): T5_2 D_wiki 20Sim+1Sum-10CosSim(ReLU) 384Hidden_size (on epoch3)
     ## MLO95 (tmux 0): T5_2 D_Wiki (whole) 20Sim+1Sum+15KL 512_SeqDim (on epoch3)


    ## T5_2 D_wiki_small 1Sim+0.5Sum+0.5KL 1024_SeqDim (epoch 2 best) 0.54186
    ## T5_2 D_wiki_small original loss 0.54288
    ## T5_2 D_wiki_small 1Sim+0.5Sum+1KL 1024_SeqDim (epoch  best)0.54309
    ## T5_2 D_wiki_small 1Sim+0.5Sum-0.5CosSim(ReLU) 768Hidden 0.55825
## MLO95 (tmux 0): T5_2 D_wiki_match 1Sim+0.001Sum-0.1CosSim(ReLU) 384Hidden
## MLO94 (tmux 1): T5_2 D_wiki_match 1Sim+0.001Sum+0.1KL 1024_SeqDim
## MLO97 (tmux 0): T5_2 D_wiki_match original loss 
## MLO99 (tmux 0): T5_2 D-wiki_match 1Sim+0.001Sum+0.5KL 1024_SeqDim
## MLO96 (tmux 0): T5_2 D-wiki-match 1Sim+0.001Sum-0.05CosSim(ReLU) 384Hidden
## MLO98 (tmux 1): T5_2 D-wiki-match 1Sim+0.01Sum-0.1CosSim(ReLU) 384Hidden
dataset = D_WIKI_MATCH

args = parse_arguments()
run_training(args, dataset)

##########################
### fine-tuning
# dataset = EPFL_DATASET
# args.output_dir = get_experiment_dir(create_dir=True)
# log_params(args.output_dir / "params.json", vars(args))

# preprocessor = Preprocessor(args.features_kwargs)
# preprocessor.preprocess_dataset(dataset)
# args.dataset = dataset
# model = T5FineTuner.load_from_checkpoint('/home/xinyzhou/text_sum_sim/experiments/exp_1647193991142787/checkpoint-epoch=1.ckpt')

# def optuna_obj(trial, args, model):

#     args.learning_rate = trial.suggest_categorical('lr',[1e-6,5e-6,1e-5])
#     args.weight_decay = trial.suggest_categorical('weight_decay',[0.005,0.01,0.05])
#     metrics_callback = MetricsCallback()
#     cp_callback = pl.callbacks.ModelCheckpoint(
#       dirpath=args.output_dir,
#         filename="checkpoint-{epoch}",
#         monitor="val_loss",
#         verbose=True,
#         mode="min",
#         save_top_k=1
#     )
#     pr_callback = PyTorchLightningPruningCallback(trial,monitor = 'val_loss')
#     model = model

#     # train_params = dict(
#     #       accumulate_grad_batches=args.gradient_accumulation_steps,
#     #       gpus=args.n_gpu,
#     #       max_epochs=args.num_train_epochs,
#     #       # early_stop_callback=False,
#     #       # gradient_clip_val=args.max_grad_norm,
#     #       # checkpoint_callback=checkpoint_callback,
#     #       callbacks=[pr_callback, cp_callback,
#     #                  metrics_callback],
#     #       # logger=TensorBoardLogger(f'{args.output_dir}/logs'),
#     #       num_sanity_val_steps=args.nb_sanity_val_steps,  # skip sanity check to save time for debugging purpose
#     #       # plugins='ddp_sharded',
#     #       #progress_bar_refresh_rate=1,

#     # )
#     early_stopping = EarlyStopping(
#       monitor='val_loss',
#       patience=6
#     )
#     trainer = pl.Trainer.from_argparse_args(
#       args,
#       callbacks=[pr_callback, cp_callback,
#                  metrics_callback, early_stopping],
#       num_sanity_val_steps=args.nb_sanity_val_steps
#     )

#     trainer.fit(model)

#   #trainer.test(model)
#     return min([x['val_loss'].item() for x in metrics_callback.metrics])

# pruner = optuna.pruners.MedianPruner()
# study = optuna.create_study(direction='maximize',pruner=pruner)
# study.optimize(lambda x: optuna_obj(x, args, model),n_trials=args.trials)

