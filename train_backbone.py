from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
from tqdm import trange
import pickle
import cProfile
import pstats
import pynvml


experiment = Experiment(
  api_key="PhozpUD8pYftjTWYPEI2hbrnw",
  project_name="backbone-training",
  workspace="robinoberson"
)
# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after

reset_training = cfg['training']['reset_training']
starting_model = cfg['training']['starting_model']

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
val_dataset = config.get_dataset('val', cfg, return_idx=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

model_counter = defaultdict(int)
data_vis_list = []

# Build a data dictionary for visualization
print("Build a data dictionary for visualization")
iterator = iter(vis_loader)
for i in trange(int(len(vis_loader)/1)):
    data_vis = next(iterator)
    idx = data_vis['idx'].item()
    model_dict = val_dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    category_name = category_name.split(',')[0]
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)


try:
    if reset_training:
        checkpoint_io = CheckpointIO(out_dir, model=model)
        _ = checkpoint_io.load(starting_model)
        load_dict = dict()
        print(f'Loaded {starting_model}')
    else:
        checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
        load_dict = checkpoint_io.load(starting_model)
    
        print(f'Loaded model.pt from {load_dict["epoch_it"]}')
    
except FileExistsError:
    print(f'Training beginning from scratch')
    load_dict = dict()
    
epoch_it = load_dict.get('epoch_it', 1)
it = load_dict.get('it', -1)
it0 = load_dict.get('it', -1)

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
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])

# profiler = cProfile.Profile()
# profiler.enable()
monitor_gpu_usage = cfg['training']['monitor_gpu_usage']
if monitor_gpu_usage:
    pynvml.nvmlInit()

    try:
        # Get the number of available GPUs
        device_count = pynvml.nvmlDeviceGetCount()
        
        # Initialize handles for each GPU
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
    except Exception as e:
        experiment.log_text(f"Error: {e}")
        print(f"Error: {e}")
        monitor_gpu_usage = False        

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
                
        loss = trainer.train_step(batch)
        experiment.log_metric('train_loss', loss, step=it)
        
        if monitor_gpu_usage:
            total_memory_used = 0
                
            for i, handle in enumerate(handles):
                # Get current memory usage for each GPU
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = info.used
                
                # Print free and used memory for each GPU
                # print(f"GPU {i} Memory - Used: {memory_used / (1024**2):.2f} MB, Free: {info.free / (1024**2):.2f} MB")
                experiment.log_text(f"GPU {i} Memory - Used: {memory_used / (1024**2):.2f} MB, Free: {info.free / (1024**2):.2f} MB") 
                experiment.log_metric(f"GPU_{i}_memory_used", memory_used / (1024**2), step=it)
                # Calculate total memory usage
                total_memory_used += memory_used
            
            # Print total memory usage across all GPUs
            # print(f"Total Memory Used Across all GPUs: {total_memory_used / (1024**2):.2f} MB")
            experiment.log_text(f"Total Memory Used Across all GPUs: {total_memory_used / (1024**2):.2f} MB")
                
        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
            experiment.log_text(f'[Epoch {epoch_it}] it={it}, loss={loss}')

        # if it - it0 == 1100:
        #     profiler.disable()
        #     print(f'Dumping profiler stats')
        #     profiler.dump_stats('/home/roberson/MasterThesis/master_thesis/Playground/Training/debug/train.prof')
            
        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            for data_vis in data_vis_list:
                if cfg['generation']['sliding_window']:
                    out = generator.generate_mesh_sliding(data_vis['data'])    
                else:
                    out = generator.generate_mesh(data_vis['data'])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                    
                export_name_mesh = os.path.join(out_dir, 'vis', '{}_{}_{}.obj'.format(it, data_vis['category'], data_vis['it']))
                export_name_data = os.path.join(out_dir, 'vis', '{}_{}_{}.pkl'.format(it, data_vis['category'], data_vis['it']))
                mesh.export(export_name_mesh)
                
                with open(export_name_data, 'wb') as f:
                    pickle.dump(data_vis['data'], f)
                

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            # experiment.log_asset("model.pt")

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('aug_model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            # experiment.log_asset("aug_model_%d.pt" % it)
        # Run validation
        if (validate_every > 0 and (it % validate_every) == 0) or (it0 + 1 == it):
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)
                experiment.log_metric(f'val_{k}', v, step=it)


            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                
                # experiment.log_asset("model_best.pt")

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            # experiment.log_asset("model.pt")

            exit(3)
