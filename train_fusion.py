
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
    description='Train a 3D reconstruction model.'
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
    project_name="training-fusion",
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
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
learning_rate = cfg['training']['learning_rate']
limited_gpu = cfg['training']['limited_gpu']
exit_after = args.exit_after

gt_query = cfg['training']['gt_query']

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = config.get_dataset('train', cfg)

batch_sampler = data.CategoryBatchSampler(train_dataset, batch_size=batch_size, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, num_workers=cfg['training']['n_workers'],
    collate_fn=data.collate_remove_none,
    batch_sampler=batch_sampler,
    worker_init_fn=data.worker_init_fn)
# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Model for merging
model_merging = layers.Conv3D_one_input().to(device)

if model_merging is not None:
    # Freeze ConvONet parameters
    print(f'Freezing {model.__class__.__name__} parameters...')
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=learning_rate)
    trainer = config.get_trainer_sequence(model, model_merging, optimizer, cfg, device=device)
else:
    raise ValueError('model_merging is None')

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io_merging = CheckpointIO(out_dir, model=model_merging, optimizer = optimizer)

try:
    checkpoint_io.load(os.path.join(os.getcwd(), cfg['training']['backbone_file']))
    load_dict = checkpoint_io_merging.load(cfg['training']['starting_model'])
    # load_dict = dict()

except FileExistsError as e:
    print(f'No checkpoint file found! {e}')
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', -1)

print('Loading model from epoch %d, iteration %d' % (epoch_it, it))

metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
nparameters_merging = sum(p.numel() for p in model_merging.parameters())
print('Total number of parameters: %d' % nparameters)
print('Total number of parameters in merging model: %d' % nparameters_merging)

print('output path: ', cfg['training']['out_dir'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=150)

# pcd = o3d.geometry.PointCloud()
# base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
prev_lr = -1

while True:
    epoch_it += 1
    print(epoch_it)
    for batch in train_loader:
        it += 1

        p_in = batch.get('inputs').to(device)        
        batch_size, T, D = p_in.size() 
        if batch_size < cfg['training']['batch_size']:
            print('Batch size too small, skipping batch')
            continue

        categories = batch.get('category')
        unique_cat = np.unique(categories)
        if len(unique_cat) > 1:
            print('Multiple categories found in batch, skipping batch')
            continue
        category = unique_cat[0]
        path_gt_points = os.path.join(cfg['data']['path_gt'], category, '000000', cfg['data']['gt_file_name'])
        
        points_gt_npz = np.load(path_gt_points)
        points_gt_3D = points_gt_npz['points']
        points_gt_3D = torch.from_numpy(points_gt_3D).to(device).float()

        points_gt_occ = np.unpackbits(points_gt_npz['occupancies'])
        points_gt_occ = torch.from_numpy(points_gt_occ).to(device).float().unsqueeze(-1)
        
        random_indices = np.random.choice(points_gt_3D.shape[0], size=T, replace=False)
        points_gt_3D = points_gt_3D[random_indices]
        points_gt_occ = points_gt_occ[random_indices]
        
        points_gt = torch.cat((points_gt_3D, points_gt_occ), dim=-1)
        
        loss, losses = trainer.train_sequence_window(batch, points_gt)
        
        scheduler.step(loss)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            if current_lr != prev_lr:
                print("Learning rate changed to:", current_lr)
                prev_lr = current_lr
        
        logger.add_scalar('train/loss', loss, it)
        if log_comet: 
            experiment.log_metric('train_loss', loss, step=it)
            for idx_elem, elem in enumerate(losses):
                experiment.log_metric(f'train_loss_{idx_elem}', elem, step=it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
            
            if log_comet: 
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    experiment.log_metric("learning_rate", lr, step=it)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io_merging.save('model_merging.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            if cfg['training']['save_data_viz']: 
                print('Saving data for visualizing in %s' % out_dir)
                latent_map_gt, latent_map_sampled_merged, logits_gt, logits_sampled, p_stacked, p_n_stacked, inputs_distributed = trainer.save_data_visualization(batch, points_gt)
            
                #dump all files 
                path = os.path.join(out_dir, 'data_viz')
                if not os.path.exists(path):
                    os.makedirs(path)
                
                with open(os.path.join(path, f'data_viz_{it}.pkl'), 'wb') as f:
                    pickle.dump([latent_map_gt, latent_map_sampled_merged, logits_gt, logits_sampled, p_stacked, p_n_stacked, inputs_distributed], f)
                del latent_map_gt, latent_map_sampled_merged, logits_gt, logits_sampled, p_stacked, p_n_stacked, inputs_distributed
                torch.cuda.empty_cache()
        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io_merging.save('model_merging_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io_merging.save('model_merging.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
