
import os
import sys
from scipy.spatial.transform import Rotation as R
import gc

current_directory = os.getcwd()
master_thesis_path = os.path.join(os.path.sep, *current_directory.split(os.path.sep))

# Print the master thesis path
print(master_thesis_path)

sys.path.append(os.path.join(master_thesis_path, 'neuralblox'))
sys.path.append(os.path.join(master_thesis_path, 'neuralblox', 'configs'))

#cd to neuralblox folder
os.chdir(os.path.join(master_thesis_path, 'neuralblox'))

import torch
import numpy as np

import open3d as o3d
from src.utils.debug_utils import *
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
print(device)
import src.neuralblox.helpers.visualization_utils as vis_utils
import src.neuralblox.helpers.sequential_trainer_utils as st_utils
import src.neuralblox.helpers.metrics_utils as metrics_utils

if 'robin' in os.getcwd():
    bool_location = 1
    cfg_path = 'configs/evaluation_cfg.yaml'
    print(f'On home')
elif 'cluster' in os.getcwd():
    bool_location = 2
    print(f'On euler')
else:
    bool_location = 0
    cfg_path = 'configs/evaluation_cfg_local.yaml'
    print(f'On local')

cfg_default_path = 'configs/default.yaml'

cfg = config.load_config(cfg_path, cfg_default_path)

from src.neuralblox import config_generators
generator_robot = config_generators.get_generator_sequential(cfg, device=device)

from src import config, data as data_src

test_dataset = config.get_dataset('test', cfg)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, 
    num_workers=cfg['training']['n_workers'], 
    shuffle=False,
    collate_fn=data_src.collate_remove_none,
    worker_init_fn=data_src.worker_init_fn)

iter = 0
if not os.path.exists(cfg['evaluation']['processed_data_dir']):
    os.makedirs(cfg['evaluation']['processed_data_dir'])

for n_max_inputs in cfg['evaluation']['n_max_inputs']:
    iter = 0
    for batch in test_loader:
        torch.cuda.empty_cache()
        gc.collect()
        print(f'preprocessing batch {iter} for n_max_inputs = {n_max_inputs}')
        with torch.no_grad():
            batch_subsampled_reduced = metrics_utils.process_batch(batch, cfg, n_max_inputs)
            
            logits_sampled, query_points = metrics_utils.generate_logits(generator_robot, batch_subsampled_reduced, cfg, 20)
            generator_robot = config_generators.get_generator_sequential(cfg, device=device)

            data_saving = metrics_utils.prepare_data_saving(batch_subsampled_reduced, query_points, logits_sampled, cfg)
            out_path = os.path.join(cfg['evaluation']['processed_data_dir'], f'data_saving_{n_max_inputs}_{iter}.pth')
            torch.save(data_saving, out_path)
            iter += 1
            print(f'Saved to {out_path}')
        
print(f"scp -r {cfg['evaluation']['processed_data_dir']} roberson@129.132.39.165:/scratch/roberson/data/simultaneous/sequential_training")
