
import os
import sys
from scipy.spatial.transform import Rotation as R

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

from src import data
from src.neuralblox import config_training
from src.checkpoints import CheckpointIO
from src import layers
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
    print(f'On home')

elif 'cluster' in os.getcwd():
    bool_location = 2
    print(f'On euler')
else:
    bool_location = 0
    print(f'On local')

cfg_default_path = 'configs/default.yaml'
cfg_path = 'configs/evaluation_cfg.yaml'

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

accuracy = 0
completeness = 0
recall = 0

for batch in test_loader:
    batch_subsampled_reduced = metrics_utils.process_batch(batch, cfg)
    
    logits_sampled, query_points = metrics_utils.generate_logits(generator_robot, batch_subsampled_reduced, cfg, 20)
    
    model_infos = batch['model_infos']
    terrain = model_infos[0]['category'][0]
    
    gt_mesh_o3d, gt_mesh_points = metrics_utils.load_ground_truth(cfg, terrain)
    gt_mesh_o3d, gt_mesh_points = metrics_utils.apply_transformations(batch_subsampled_reduced, gt_mesh_o3d, gt_mesh_points)
    
    if cfg['evaluation']['discard_ground']:
        gt_mesh_points, query_points, logits_sampled = metrics_utils.filter_ground_points(gt_mesh_points, query_points, logits_sampled, batch_subsampled_reduced)
        
    if cfg['evaluation']['visualize']:
        input_points = batch_subsampled_reduced['inputs'].squeeze(0).cpu().numpy().reshape(-1, 3)
        input_occ = batch_subsampled_reduced['inputs.occ'].squeeze(0).cpu().numpy().reshape(-1)
        
        pcd_inputs = o3d.geometry.PointCloud()
        pcd_inputs.points = o3d.utility.Vector3dVector(input_points[input_occ == 1])
        pcd_inputs.paint_uniform_color([1, 0, 1])
        
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_mesh_points)
        pcd_gt.paint_uniform_color([0, 1, 0])
        
        pcd_query = o3d.geometry.PointCloud()
        occ_sampled = 1 / (1 + np.exp(-logits_sampled))
        occ_sampled = (occ_sampled > 0.1)
        
        query_pts_occ = query_points[occ_sampled]
        pcd_query.points = o3d.utility.Vector3dVector(query_pts_occ)
        pcd_query.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([pcd_inputs, pcd_gt, pcd_query, gt_mesh_o3d])

    model_infos = batch['model_infos']
    terrain = model_infos[0]['category'][0]

    acc, comp, rec = metrics_utils.evaluate_points(query_points, logits_sampled, cfg, gt_mesh_o3d, gt_mesh_points)

    accuracy += acc
    completeness += comp
    recall += rec

print(f'Accuracy: {accuracy/len(test_loader)}')
print(f'Completeness: {completeness/len(test_loader)}')
print(f'Recall: {recall/len(test_loader)}')