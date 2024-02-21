import gc
import pickle
import shutil
import sys
import torch
import os
import argparse
from src import config
from src.checkpoints import CheckpointIO
from src import layers
import numpy as np
import open3d as o3d
from tqdm import trange
from os.path import join
from src.common import define_align_matrix, get_shift_noise, get_yaw_noise_matrix
from pathlib import Path
from memory_profiler import profile
from src.utils.test_utils import generate_and_save_mesh, process_and_generate_latent

np.random.seed(42) # for reproducibility

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

max_frames = cfg['test']['max_frames']
interval = cfg['test']['frames_interval']
noise_shift_std = cfg['test']['scene']['noise']['shift']['shift_std']
shift_on_gravity = cfg['test']['scene']['noise']['shift'].get("on_gravity", False)
noise_yaw_std = cfg['test']['scene']['noise']['yaw_std']
export_pc = cfg['test']['export_pointcloud']
export_each_frame = cfg['test']['export_each_frame']
merge_cameras = cfg['test']['merge_cameras']

data_path = cfg['data']['path']
intrinsics = cfg['data']['intrinsics']

cameras = cfg['cameras_list']

num_files = min([len(os.listdir(data_path + f'/{cam}_pcld')) for cam in cameras])

if num_files%2 != 0:
    num_files = num_files - 1

if intrinsics == None:
    cam = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
else:
    intrinsic = np.loadtxt(join(data_path, "camera-intrinsics.txt")).tolist()
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = intrinsic

out_dir = cfg['test']['out_dir']
out_name = cfg['test']['out_name']

#clear out_dir 
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

# Model
model = config.get_model(cfg, device=device)
model_merging = layers.Conv3D_one_input().to(device)
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io_merging = CheckpointIO(out_dir, model=model_merging)
checkpoint_io.load(join(os.getcwd(), cfg['test']['model_file']))
checkpoint_io_merging.load(join(os.getcwd(),cfg['test']['merging_model_file']))

# Get aligning matrix
align_matrix = define_align_matrix(cfg['data']['align'])
align_matrix = torch.from_numpy(align_matrix).to(device).float()

print("Getting scene bounds from dataset sampled every {} frames".format(interval))
sample_points = torch.empty(1, 0, 3, device=device)

# print(f'Using {min_n_files} files')
for i in trange(1, max_frames * interval, interval):
    # print(f'Frame {i}')
    for cam in cameras:
        pcl_unaligned_merged = np.load(join(data_path, f'{cam}_pcld', "pcld-%06d.npy"%(i)))
        #transform to world
        pcl_unaligned_merged = pcl_unaligned_merged[::10]
        N = pcl_unaligned_merged.shape[0]
        one = np.ones((N, 1))
        pcl_unaligned_merged = np.hstack((pcl_unaligned_merged, one))
        
        pcl_unaligned_merged = torch.from_numpy(pcl_unaligned_merged).to(device).float()
        pcl_world = torch.matmul(align_matrix, pcl_unaligned_merged.T).T
        pcl_world = pcl_world[:,:3]

        sample_points = torch.cat([sample_points, pcl_world.unsqueeze(0)], dim=1)

# Generate
model.eval()
model_merging.eval()
generator = config.get_generator_fusion(model, model_merging, sample_points, cfg, device=device)

if export_pc==True:
    sampled_pcl = np.array([])

for i in trange(1, max_frames * interval, interval):
    if merge_cameras:
        pcl_unaligned_merged = np.array([])

        for cam in cameras:
            pcl_unaligned_cam = np.load(join(data_path, f'{cam}_pcld', "pcld-%06d.npy"%(i)))
            
            if len(pcl_unaligned_merged)==0:
                pcl_unaligned_merged = pcl_unaligned_cam
            else:
                pcl_unaligned_merged = np.vstack((pcl_unaligned_merged, pcl_unaligned_cam))
            
        latent = process_and_generate_latent(pcl_unaligned_merged, align_matrix, generator, device, export_pc, out_dir, f'pcl-merged-{i:06d}')
        
        if export_each_frame==True:
            generate_and_save_mesh(generator, latent.clone(), out_dir, f'merged-{i:06d}')
        
    else:
        for cam in cameras:
            pcl_unaligned_cam = np.load(join(data_path, f'{cam}_pcld', "pcld-%06d.npy"%(i)))
            
            latent = process_and_generate_latent(pcl_unaligned_cam, align_matrix, generator, device, export_pc, out_dir, f'pcl-{cam}-{i:06d}')
            
            if export_each_frame==True:
                generate_and_save_mesh(generator, latent.clone(), out_dir, f'{cam}-{i:06d}')              
    
if not export_each_frame:
    generate_and_save_mesh(generator, latent, out_dir, f'final_mesh')   

