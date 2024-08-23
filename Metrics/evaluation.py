
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

# for n_max_inputs in cfg['evaluation']['n_max_inputs']:
#     for iter in range(5):

iter = 0
if cfg['evaluation']['is_neuralblox']:
    processed_data_dir = cfg['evaluation']['processed_data_dir_neuralblox']
else:
    processed_data_dir = cfg['evaluation']['processed_data_dir']

full_metrics = {}
for n_max_inputs in cfg['evaluation']['n_max_inputs']:
    # get the number of files that begin with f'data_saving_{n_max_inputs}'
    # in the processed_data_dir
    metrics_n_max_inputs = {}
    
    files = sorted([f for f in sorted(os.listdir(processed_data_dir)) if f.startswith(f'data_saving_{n_max_inputs}')])
    n_files = len(files)
    print(f'Number of files for n_max_inputs = {n_max_inputs}: {n_files}')
    for file in files:
        batch_path = os.path.join(processed_data_dir, file)
        print(f'Computing metrics for {batch_path}, is neuralblox = {cfg["evaluation"]["is_neuralblox"]}')
        
        batch = torch.load(batch_path)
            
        accuracy = 0
        completeness = 0
        recall = 0

        model_infos = batch['model_infos']
        terrain = model_infos[0]['category'][0]
        
        transform = batch['transform']
    
        
        query_points = batch['query_points']
        logits_sampled = batch['logits_sampled']
        
        if torch.is_tensor(query_points):
            query_points = query_points.cpu().numpy()
            logits_sampled = logits_sampled.cpu().numpy() # Convert to float32 for NumPy arrays

        query_points = query_points.astype(np.float32)
            
        gt_mesh_o3d, gt_mesh_points = metrics_utils.load_ground_truth(batch['cfg'], terrain)
        gt_mesh_o3d, gt_mesh_points = metrics_utils.apply_transformations(transform, gt_mesh_o3d, gt_mesh_points)
        
        if cfg['evaluation']['discard_ground']:
            input_points = batch['inputs'].squeeze(0).cpu().numpy().reshape(-1, 3)
            gt_mesh_points, query_points, logits_sampled = metrics_utils.filter_ground_points(gt_mesh_points, query_points, logits_sampled, input_points)
            
        if cfg['evaluation']['visualize']:
            input_points = batch['inputs'].squeeze(0).cpu().numpy().reshape(-1, 3)
            input_occ = batch['inputs.occ'].squeeze(0).cpu().numpy().reshape(-1)
            
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
            base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

            o3d.visualization.draw_geometries([pcd_inputs, pcd_gt, pcd_query, gt_mesh_o3d, base_axis])

        model_infos = batch['model_infos']
        terrain = model_infos[0]['category'][0]
        
        if cfg['evaluation']['is_neuralblox']:
            thresholds = [0.01]
        else:
            thresholds = [0.1, 0.2, 0.3, 0.4]

        best_metrics, metrics = metrics_utils.evaluate_points(query_points, logits_sampled, cfg, gt_mesh_o3d, gt_mesh_points, thresholds = thresholds)

        metrics_n_max_inputs[file] = {'metrics': metrics, 'best_metrics': best_metrics}

    full_metrics[n_max_inputs] = metrics_n_max_inputs

torch.save(full_metrics, os.path.join(processed_data_dir, 'full_metrics.pth'))