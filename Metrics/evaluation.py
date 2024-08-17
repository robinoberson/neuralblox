
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
    bool_location = 0
    cfg_path_simultaneous_training = 'configs/simultaneous/train_simultaneous_home.yaml'
    output_dir_prefix = '/home/robin/Dev/MasterThesis/data/simultaneous'
    scene_folder = '/home/robin/Dev/MasterThesis/data/parkour_line_scenes/'
    preprocessed_scene_paths = '/home/robin/Dev/MasterThesis/data/flat_anymal_c/exported'
    print(f'On home')

elif 'cluster' in os.getcwd():
    bool_location = 1
    cfg_path_simultaneous_training = 'configs/simultaneous/train_simultaneous_euler.yaml'
    output_dir_prefix = '/cluster/scratch/roberson/data/simultaneous'
    scene_folder = '/cluster/scratch/roberson/data/parkour_line_scenes'
    print(f'On euler')
    
else:
    bool_location = 2
    cfg_path_simultaneous_training = 'configs/simultaneous/train_simultaneous_local.yaml'
    output_dir_prefix = '/scratch/roberson/data/simultaneous'
    scene_folder = '/media/roberson/T7/terrains'
    preprocessed_scene_paths = '/scratch/roberson/flat_anymal_c/exported/27_06_2024'
    print(f'On local')

cfg_default_path = 'configs/default.yaml'
    
cfg = config.load_config(cfg_path_simultaneous_training, cfg_default_path)
    
config_metrics = metrics_utils.load_config('configs/metrics_cfg.yaml')

from src.neuralblox import config_generators
generator_robot = config_generators.get_generator_sequential(cfg, device=device)

from src import config, data as data_src

train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

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

batch = next(iter(train_loader))

batch_subsampled_reduced = metrics_utils.process_batch(batch, config_metrics)

with torch.no_grad():
    generator_robot.trainer.n_max_points_inputs = config_metrics['generator_robot']['n_max_points_inputs']
    generator_robot.points_threshold = config_metrics['generator_robot']['points_threshold']
    
    logits_tup = generator_robot.generate_logits_at_index(batch_subsampled_reduced, 20, 0, config_metrics['generation']['n_logits'])
    
logits_sampled, query_points, inputs_frame, centers_frame = logits_tup[-1]

threshold = config_metrics['generation']['threshold_prob']

print(f'query_points.shape = {query_points.shape}')
occ_sampled = torch.sigmoid(logits_sampled)
occ_sampled = (occ_sampled > threshold)

pcd_query = o3d.geometry.PointCloud()
query_pts_occ = query_points.reshape(-1, 3)[occ_sampled.reshape(-1)]

print(f'query_pts_occ.shape = {query_pts_occ.shape}')

pcd_query.points = o3d.utility.Vector3dVector(query_pts_occ.cpu().numpy() + np.array([0, 0.02, 0]))


terrain = os.listdir(cfg['data']['path'])[0]

# Path to your ground truth mesh file (replace with your actual file path)
ground_truth_mesh_path = os.path.join(scene_folder, terrain, 'terrain.obj') # 'terrain.obj'
ground_truth_mesh_points_path = os.path.join(scene_folder, terrain, 'surface_points.npy')

# Load the ground truth mesh
ground_truth_mesh_o3d = o3d.io.read_triangle_mesh(ground_truth_mesh_path)
ground_truth_mesh_points = np.load(ground_truth_mesh_points_path)
ground_truth_mesh_points = metrics_utils.filter_points(ground_truth_mesh_points, 2,3)

pcd_ground_truth = o3d.geometry.PointCloud()
pcd_ground_truth.points = o3d.utility.Vector3dVector(ground_truth_mesh_points)

ground_truth_mesh_o3d.compute_vertex_normals()
R_gravity = ground_truth_mesh_o3d.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
ground_truth_mesh_o3d.rotate(R_gravity, center=(0, 0, 0))
pcd_ground_truth.rotate(R_gravity, center=(0, 0, 0))

angle_x = cfg['data']['transform']['angle_x']
angle_y = cfg['data']['transform']['angle_y']
angle_z = cfg['data']['transform']['angle_z']

# Create rotation object
rand_trans = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)

# Convert to rotation matrix
R_additional = rand_trans.as_matrix()

# Apply the additional rotation
ground_truth_mesh_o3d.rotate(R_additional, center=(0, 0, 0))
pcd_ground_truth.rotate(R_additional, center=(0, 0, 0))

pcd_ground_truth.paint_uniform_color([0.5, 0.5, 0.])
# Convert to Open3D's tensor geometry
o3d.visualization.draw_geometries([pcd_query, pcd_ground_truth, ground_truth_mesh_o3d])
ground_truth_mesh = o3d.t.geometry.TriangleMesh.from_legacy(ground_truth_mesh_o3d)

# Initialize the RaycastingScene
scene = o3d.t.geometry.RaycastingScene()

# Add the ground truth mesh to the scene
_ = scene.add_triangles(ground_truth_mesh)

predicted_points = query_pts_occ.cpu().numpy()
ground_truth_mesh_points = np.asarray(pcd_ground_truth.points)
accuracy = metrics_utils.compute_accuracy(scene, predicted_points)
completeness = metrics_utils.compute_completeness(ground_truth_mesh_points, predicted_points)
recall = metrics_utils.compute_recall(ground_truth_mesh_points, predicted_points, tau_r=0.5)

print(f"Accuracy: {accuracy}")
print(f"Completeness: {completeness}")
print(f"Recall: {recall}")