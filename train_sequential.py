import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config
from src import data as data_src
from src.neuralblox import config_training
from src.checkpoints import CheckpointIO
from src import layers
import pickle
import yaml
import src.neuralblox.helpers.sequential_trainer_utils as st_utils
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model simutaneously.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

#copy cfg to out_dir
os.makedirs(cfg['training']['out_dir'], exist_ok=True)

# Save the full cfg to a file in the output directory
config_path = os.path.join(cfg['training']['out_dir'], 'config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

log_comet = cfg['training']['log_comet']

if log_comet:
    from comet_ml import Experiment
    from comet_ml.integration.pytorch import log_model
    
    experiment = Experiment(
        api_key="PhozpUD8pYftjTWYPEI2hbrnw",
        project_name="train-simultaneously",
        workspace="robinoberson"
    )

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# device = torch.device("cpu")
# Set t0
t0 = time.time()

# Shorthands
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
learning_rate = cfg['training']['lr']
print_every = cfg['training']['print_every']
checkpoint_interval = 60 * cfg['training']['checkpoint_interval_minutes']


model_selection_metric = cfg['training']['model_selection_metric']

if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(cfg['training']['out_dir']):
    os.makedirs(cfg['training']['out_dir'])
    
# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

#set seed 
torch.manual_seed(42)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, 
    num_workers=cfg['training']['n_workers'], 
    shuffle=True,
    collate_fn=data_src.collate_remove_none,
    worker_init_fn=data_src.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, 
    num_workers=cfg['training']['n_workers'], 
    shuffle=False,
    collate_fn=data_src.collate_remove_none,
    worker_init_fn=data_src.worker_init_fn)

print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)
model_merging = layers.Conv3D_one_input().to(device)

checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=model)
checkpoint_io_merging = CheckpointIO(cfg['training']['out_dir'], model=model_merging)

try:
    checkpoint_io.load(cfg['training']['starting_model_backbone_file'])
    checkpoint_io_merging.load(cfg['training']['starting_model_merging_file'])
    
except FileExistsError as e:
    print(f'No checkpoint file found! {e}')
    
load_dict = dict()
    
optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=learning_rate)
trainer = config_training.get_trainer_sequential_shuffled(model, model_merging, optimizer, cfg, device=device)

if log_comet:
    trainer.set_experiment(experiment)
epoch_it = 0
it = 0

print('Loading model from epoch %d, iteration %d' % (epoch_it, it))

# Print model
nparameters = sum(p.numel() for p in model.parameters())
nparameters_merging = sum(p.numel() for p in model_merging.parameters())
print('Total number of parameters: %d' % nparameters)
print('Total number of parameters in merging model: %d' % nparameters_merging)

print('output path: ', cfg['training']['out_dir'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=cfg['training']['lr_patience'])

prev_lr = -1
last_checkpoint_time = time.time()

while True:
    torch.cuda.empty_cache()
    epoch_it += 1
    print(epoch_it)
    
    for batch in train_loader:
        #squeeze batch
        for key in batch:
            batch[key] = batch[key].squeeze(0)
            
        it += 1
    
        loss = trainer.train_sequence(batch)
        
        scheduler.step(loss)
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            if current_lr != prev_lr:
                print("Learning rate changed to:", current_lr)
                prev_lr = current_lr
                if log_comet:
                    experiment.log_metric("learning_rate", current_lr, step=it)   

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io_merging.save('model_merging_%d.pt' % it, epoch_it=epoch_it, it=it)
            checkpoint_io.save('model_backbone_%d.pt' % it, epoch_it=epoch_it, it=it)
        
        # Save checkpoint
        current_time = time.time()
        if (current_time - last_checkpoint_time) >= checkpoint_interval:
            last_checkpoint_time = current_time

            print('Saving checkpoint')
             
            checkpoint_io_merging.save(cfg['training']['model_merging'], epoch_it=epoch_it, it=it)
            checkpoint_io.save(cfg['training']['model_backbone'], epoch_it=epoch_it, it=it)
        print(f'epoch: {epoch_it}, it: {it}, loss: {loss}')
        
    # for batch in val_loader:
    #     with torch.no_grad():
    #         loss = trainer.validate_sequence(batch)
    
