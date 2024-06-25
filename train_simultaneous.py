import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
import shutil
from src import layers
import pickle

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

from tensorboardX import SummaryWriter

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
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
    train_dataset, batch_size=batch_size, 
    num_workers=cfg['training']['n_workers'], 
    shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, 
    num_workers=cfg['training']['n_workers'], 
    shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Model for merging
# num_blocks = 5
# num_channels = [256, 256, 192, 192, 192, 128, 128, 128]
# model_merging = layers.Conv3D_one_input(num_blocks=num_blocks, num_channels=num_channels).to(device)
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
trainer = config.get_trainer_sequential(model, model_merging, optimizer, cfg, device=device)

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
    for idx_batch, batch in enumerate(train_loader):
        it += 1
        
        loss = trainer.train_sequence_window(batch)
        
        scheduler.step(loss)
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            if current_lr != prev_lr:
                print("Learning rate changed to:", current_lr)
                prev_lr = current_lr
                if log_comet:
                    experiment.log_metric("learning_rate", current_lr, step=it)
                    
        if log_comet: 
            experiment.log_metric('train_loss', loss, step=it)
        
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
            
            if log_comet: 
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    experiment.log_metric("learning_rate", lr, step=it)

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

        path = os.path.join(cfg['training']['out_dir'], 'data_viz')
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.cuda.empty_cache()
        checkpoint_io_merging.save(cfg['training']['model_merging'], epoch_it=epoch_it, it=it)
        checkpoint_io.save(cfg['training']['model_backbone'], epoch_it=epoch_it, it=it)
        if cfg['training']['save_data_viz']: 

            with torch.no_grad():
                trainer.model.eval()
                trainer.model_merge.eval()
                
                print(f'Saving {os.path.join(path, f"data_viz_batch_idx_{epoch_it}.pkl")}')

                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx > 10 :
                        break
                    tup = trainer.validate_sequence(batch)
                
                    #dump all files 
                    with open(os.path.join(path, f"data_viz_train_{batch_idx}_{epoch_it}.pkl"), 'wb') as f:
                        pickle.dump(tup, f)
                    # print(f'Saved {os.path.join(path, f"data_viz_{batch_idx}_{epoch_it}.pkl")}')

                tup = []
                val_loss = 0
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx > 10:
                        continue 
                    
                    tup = trainer.validate_sequence(batch)                    #dump all files 
                    val_loss += tup[-1]
                    
                    with open(os.path.join(path, f"data_viz_val_{batch_idx}_{epoch_it}.pkl"), 'wb') as f:
                        pickle.dump(tup, f)
                    # print(f'Saved {os.path.join(path, f"data_viz_{batch_idx}_{epoch_it}.pkl")}')
            
                    
                val_loss = val_loss / len(val_loader)
                print(f'val_loss = {val_loss}, {len(val_loader)}')

                if log_comet: 
                    experiment.log_metric('val_loss', val_loss, step=it)
    else:
        print('Not time to save checkpoint')