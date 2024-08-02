import os
import torch
import math
import numpy as np
import time
import yaml
import torch.nn as nn
from torch.nn import functional as F
import random
import gc
import src.neuralblox.helpers.sequential_trainer_utils as st_utils
import src.neuralblox.helpers.visualization_utils as vis_utils
# import open3d as o3d

torch.manual_seed(42)

from src.common import (
    add_key, coord2index, normalize_coord
)
from src.training import BaseTrainer
from src.neuralblox.helpers.voxel_grid import VoxelGrid

class SequentialTrainerShuffled(BaseTrainer):
    ''' Trainer object for fusion network.

    Args:
        model (nn.Module): Convolutional Occupancy Network model
        model_merge (nn.Module): fusion network
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        query_n (int): number of query points per voxel
        hdim (int): hidden dimension
        depth (int): U-Net depth (3 -> hdim 32 to 128)

    '''


    def __init__(self, model, model_merge, optimizer, cfg, input_crop_size = 1.6, query_crop_size = 1.0, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, query_n = 8192, unet_hdim = 32, unet_depth = 2, grid_reso = 24, limited_gpu = False, n_voxels_max = 20, n_max_points = 2048, n_max_points_query = 8192, occ_per_query = 0.5, return_flat = True,
                 sigma = 0.8):
        self.model = model
        self.model_merge = model_merge
        self.optimizer = optimizer
        self.device = device
        self.input_crop_size = input_crop_size
        self.query_crop_size = query_crop_size
        self.vis_dir = vis_dir
        self.hdim = unet_hdim
        self.unet = None
        self.limited_gpu = limited_gpu
        self.reso = grid_reso
        self.iteration = 0
        self.n_voxels_max = n_voxels_max
        self.n_max_points_input = n_max_points
        self.n_max_points_query = n_max_points_query
        self.occ_per_query = occ_per_query
        self.voxel_grid = VoxelGrid()
        self.return_flat = return_flat
        self.sigma = sigma
        # self.voxel_grid.verbose = True
        self.empty_latent_code = self.get_empty_latent_representation()
        self.timing_counter = 0
        self.experiment = None
        self.log_experiment = False
        self.shifts = torch.tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)

        self.debug = False
                
        current_dir = os.getcwd()
        
        if 'robin' in current_dir:
            self.location = 'home'
        elif 'cluster' in current_dir:
            self.location = 'euler'
        else:
            self.location = 'local'

        print(f'Location: {self.location}')
        
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
    
    def set_experiment(self, experiment):
        self.experiment = experiment
        self.log_experiment = True
    
    def train_sequence(self, full_batch):
        with torch.no_grad():
            inputs_distributed_neighboured, query_distributed, centers_neighboured = self.precompute_sequence(full_batch)
        print(f'Finished precomputing')
        loss = self.train_batch(inputs_distributed_neighboured, query_distributed, centers_neighboured)
        return loss
    
    def validate_sequence(self, full_batch):
        inputs_distributed_neighboured, query_distributed, centers_neighboured = self.precompute_sequence(full_batch)
        loss = self.validate_batch(inputs_distributed_neighboured, query_distributed, centers_neighboured)
        return loss

    def precompute_sequence(self, full_batch):
        self.model.eval()
        self.model_merge.eval()
        
        with torch.no_grad():
            #batch is [12, 20, 40960, 4] for points (gt points)
            #batch is [12, 20, 4096, 4] for inputs 
            p_in_full, p_query_full = st_utils.get_inputs_from_scene(full_batch, self.device)

            inputs_scene_distributed_list = []
            query_scene_distributed_list = []
            centers_scene_list = []
            axis_n_crop_scene_list = []
            
            n_scenes = p_in_full.shape[0]
            n_frames = p_in_full.shape[1]
            # n_frames = 1
            
            n_voxels = torch.zeros(n_scenes, n_frames, dtype = torch.int32).to(device = self.device)
                        
            for scene_idx in range(n_scenes):
                inputs_frame_distributed_list = []
                query_frame_distributed_list = []
                centers_frame_list = []
                axis_n_crop_frame_list = []
                # for frame_idx in range(p_in_full.shape[1]):
                for frame_idx in range(n_frames):
                    inputs_frame = p_in_full[scene_idx, frame_idx]
                    inputs_frame_distributed, centers_frame, axis_n_crop = self.get_distributed_inputs(inputs_frame, self.n_max_points_input, 1.0, return_empty = True, isquery = False)
                    
                    query_frame = p_query_full[scene_idx, frame_idx]
                    query_frame_distributed, centers_frame_query, axis_n_crop = self.get_distributed_inputs(query_frame, self.n_max_points_query, self.occ_per_query, return_empty = True, isquery = True)

                    frame_mask = st_utils.compute_mask_occupied(centers_frame, centers_frame_query)
                    query_mask = st_utils.compute_mask_occupied(centers_frame_query, centers_frame)
                    
                    if frame_mask.sum() != frame_mask.shape[0]:
                        print(f'PROBLEM: Frame {frame_idx} has {frame_mask.sum()} occupied voxels instead of {frame_mask.shape[0]}')
                        
                    centers_frame = centers_frame[frame_mask]
                    centers_frame_query = centers_frame_query[query_mask]
                    
                    inputs_frame_distributed = inputs_frame_distributed[frame_mask]
                    query_frame_distributed = query_frame_distributed[query_mask]
                    
                    inputs_frame_distributed_list.append(inputs_frame_distributed)
                    query_frame_distributed_list.append(query_frame_distributed)
                    centers_frame_list.append(centers_frame)
                    axis_n_crop_frame_list.append(axis_n_crop)
                    
                    n_x, n_y, n_z = axis_n_crop
                    n_voxels[scene_idx, frame_idx] = (n_x+2)*(n_y+2)*(n_z+2)
                
                inputs_scene_distributed_list.append(inputs_frame_distributed_list)
                query_scene_distributed_list.append(query_frame_distributed_list)
                centers_scene_list.append(centers_frame_list)
                axis_n_crop_scene_list.append(axis_n_crop_frame_list)
                       
            n_voxels_total = int(n_voxels.sum().item())
            print(f'n_voxels_total = {n_voxels_total}')
            
            c, h, w, d = self.empty_latent_code.shape

            storage_centers_idx = torch.zeros(n_voxels_total, 3).to(torch.device('cpu'))
            storage_inputs = torch.zeros(n_voxels_total, self.n_max_points_input, 4).to(torch.device('cpu'))
            storage_frame_shapes = torch.zeros(n_voxels_total, 3).to(torch.device('cpu'))
            storage_indexes_lookup = torch.zeros(n_voxels_total, 2).to(torch.device('cpu'))
            
            storage_latents_padded = torch.zeros(n_voxels_total, c, h, w, d).to(torch.device('cpu'))
                        
            dtype_size = torch.tensor([], dtype=torch.float32).element_size()
            total_elements = n_voxels_total * c * h * w * d
            memory_size_bytes = total_elements * dtype_size
            memory_size_gb = memory_size_bytes / (1024 ** 3)
            print(f'memory_size_gb = {memory_size_gb}')

            if self.debug:
                self.merged_inputs = torch.zeros(n_voxels_total, self.n_max_points_input, 4).to(torch.device('cpu'))
            
            idx_start = 0
            for scene_idx in range(n_scenes):
                for frame_idx in range(n_frames):
                    idx_end = idx_start + n_voxels[scene_idx, frame_idx]
                    
                    n_voxels_frame = n_voxels[scene_idx, frame_idx]
                    
                    centers_frame = centers_scene_list[scene_idx][frame_idx]
                    inputs_frame = inputs_scene_distributed_list[scene_idx][frame_idx]
                    
                    indexes_lookup_frame = torch.tensor([idx_start, idx_end]).repeat(n_voxels_frame, 1).to(torch.device('cpu'))
                    
                    if frame_idx == 0:
                        prev_merged_latents_padded, prev_centers_frame, vol_bounds_padded_prev = self.process_frame_cold_start(centers_frame, inputs_frame)
                    else: 
                        return centers_frame, inputs_frame, prev_merged_latents_padded, prev_centers_frame, vol_bounds_padded_prev
                        prev_merged_latents_padded = self.process_frame(centers_frame, inputs_frame, prev_merged_latents_padded, prev_centers_frame, vol_bounds_padded_prev)
            
            torch.cuda.empty_cache()
            gc.collect()

            return inputs_distributed_neighboured, query_distributed, centers_neighboured
    def init_empty_latent_grid(self, vol_bounds_frame_padded):
        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop']
        
        expanded_shape = (vol_bounds_frame_padded['n_crop'],) + self.empty_latent_code.shape #padded 
        empty_latent_grid = self.empty_latent_code.unsqueeze(0)
        empty_latent_grid = empty_latent_grid.expand(expanded_shape)
        empty_latent_grid = empty_latent_grid.reshape(n_x, n_y, n_z, *self.empty_latent_code.shape).clone()
        
        return empty_latent_grid
    
    def encode_occupied_inputs(self, inputs_frame, centers_frame, vol_bounds_frame_padded):
        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop']
        occupied_frame_mask = inputs_frame[..., 3].sum(dim = -1) > 10
        
        latents_occupied_frame = self.encode_distributed(inputs_frame[occupied_frame_mask], centers_frame[occupied_frame_mask])

        grid_latents_frame_current = self.init_empty_latent_grid(vol_bounds_frame_padded)
        
        occupied_current_mask_padded = torch.nn.functional.pad(occupied_frame_mask.reshape(n_x-2, n_y-2, n_z-2), (1, 1, 1, 1, 1, 1), value=False)

        grid_latents_frame_current[occupied_current_mask_padded] = latents_occupied_frame.clone()
        grid_latents_frame_current = grid_latents_frame_current.reshape(-1, *self.empty_latent_code.shape)
        
        return grid_latents_frame_current
    
    def pad_inputs(self, inputs_frame, vol_bounds_frame_padded):
        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop']
        inputs_frame_padded = torch.zeros(n_x, n_y, n_z, self.n_max_points_input, 4).to(self.device)
        inputs_frame_padded[1:-1, 1:-1, 1:-1] = inputs_frame.reshape(n_x-2, n_y-2, n_z-2, self.n_max_points_input, 4)
        inputs_frame_padded = inputs_frame_padded.reshape(-1, self.n_max_points_input, 4)
        
        return inputs_frame_padded
    def process_frame_cold_start(self, centers_frame, inputs_frame):
        vol_bounds_frame_padded, centers_frame_padded = st_utils.compute_vol_bound(centers_frame, self.query_crop_size, self.input_crop_size, padding = True)

        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop'] #padded
        
        if len(centers_frame) != (n_x-2)*(n_y-2)*(n_z-2):
            print(f'PROBLEM: {len(centers_frame)} != {(n_x-2)*(n_y-2)*(n_z-2)}')
            
        grid_latents_frame_current = self.encode_occupied_inputs(inputs_frame, centers_frame, vol_bounds_frame_padded)
        
        centers_lookup = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame), 1).to(self.device)
        grid_shapes = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame), 1).to(self.device)
        centers_frame_idx = st_utils.centers_to_grid_indexes(centers_frame, vol_bounds_frame_padded['lb'], self.query_crop_size).int().reshape(-1, 3)
        
        distributed_latents = st_utils.get_distributed_voxel(centers_frame_idx, grid_latents_frame_current, grid_shapes, centers_lookup, self.shifts.to(self.device))

        inputs_frame_padded = self.pad_inputs(inputs_frame, vol_bounds_frame_padded)
        distributed_inputs = st_utils.get_distributed_voxel(centers_frame_idx, inputs_frame_padded, grid_shapes, centers_lookup, self.shifts.to(self.device))
        
        c, h, w, d = self.empty_latent_code.shape

        distributed_latents = distributed_latents.reshape(-1, c, 3*h, 3*w, 3*d)
        stacked_frame = torch.cat((distributed_latents, distributed_latents), dim = 1)
        
        merged_latents = self.merge_latent_map(stacked_frame)
        merged_latents_padded = self.init_empty_latent_grid(vol_bounds_frame_padded)

        merged_latents_padded[1:-1, 1:-1, 1:-1] = merged_latents.reshape(n_x-2, n_y-2, n_z-2, *self.empty_latent_code.shape)
        return merged_latents_padded, centers_frame, vol_bounds_frame_padded
    

    def process_frame(self, centers_frame, inputs_frame, prev_merged_latents_padded, prev_centers_frame, vol_bounds_padded_prev):
        vol_bounds_frame_padded, centers_frame_padded = st_utils.compute_vol_bound(centers_frame, self.query_crop_size, self.input_crop_size, padding = True)

        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop'] #padded
        
        if len(centers_frame) != (n_x-2)*(n_y-2)*(n_z-2):
            print(f'PROBLEM: {len(centers_frame)} != {(n_x-2)*(n_y-2)*(n_z-2)}')
            
        grid_latents_frame_current = self.encode_occupied_inputs(inputs_frame, centers_frame, vol_bounds_frame_padded)
        
        centers_lookup = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame), 1).to(self.device)
        grid_shapes = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame), 1).to(self.device)
        centers_frame_idx = st_utils.centers_to_grid_indexes(centers_frame, vol_bounds_frame_padded['lb'], self.query_crop_size).int().reshape(-1, 3)
        
        distributed_latents = st_utils.get_distributed_voxel(centers_frame_idx, grid_latents_frame_current, grid_shapes, centers_lookup, self.shifts.to(self.device))

        inputs_frame_padded = self.pad_inputs(inputs_frame, vol_bounds_frame_padded)
        distributed_inputs = st_utils.get_distributed_voxel(centers_frame_idx, inputs_frame_padded, grid_shapes, centers_lookup, self.shifts.to(self.device))
        
        c, h, w, d = self.empty_latent_code.shape

        distributed_latents = distributed_latents.reshape(-1, c, 3*h, 3*w, 3*d)
        
        #retrieve prev latent
        n_x_p, n_y_p, n_z_p = vol_bounds_padded_prev['axis_n_crop'] #padded
        centers_lookup_prev = torch.tensor([0, (n_x_p * n_y_p * n_z_p)]).repeat(len(prev_centers_frame), 1).to(self.device)
        grid_shapes_prev = torch.tensor([n_x_p, n_y_p, n_z_p]).repeat(len(prev_centers_frame), 1).to(self.device)
        centers_frame_idx_prev = st_utils.centers_to_grid_indexes(prev_centers_frame, vol_bounds_padded_prev['lb'], self.query_crop_size).int().reshape(-1, 3)
        
        distributed_latents_prev = st_utils.get_distributed_voxel(centers_frame_idx_prev, prev_merged_latents_padded, grid_shapes_prev, centers_lookup_prev, self.shifts.to(self.device))

        #centers from prev frame present in this frame
        mask_centers_prev_in_current = st_utils.compute_mask_occupied(prev_centers_frame, centers_frame)
        mask_centers_current_in_prev = st_utils.compute_mask_occupied(centers_frame, prev_centers_frame)
        
        distributed_latents_prev_temp = distributed_latents.clone()
        distributed_latents_prev_temp[mask_centers_current_in_prev] = distributed_latents_prev[mask_centers_prev_in_current]
        
        stacked_frame = torch.cat((distributed_latents, distributed_latents_prev_temp), dim = 1)
        
        merged_latents = self.merge_latent_map(stacked_frame)
        merged_latents_padded = self.init_empty_latent_grid(vol_bounds_frame_padded)

        merged_latents_padded[1:-1, 1:-1, 1:-1] = merged_latents.reshape(n_x-2, n_y-2, n_z-2, *self.empty_latent_code.shape)
        return merged_latents_padded
        
    def fuse(self, inputs_list, centers_list, scene_idx, frame_idx, start_idx_prev, end_idx_prev):
        centers_interior_current = centers_list[scene_idx][frame_idx]
        inputs_interior_current = inputs_list[scene_idx][frame_idx]
        occupied_current_mask = inputs_interior_current[..., 3].sum(dim = -1) > 10
        
        vol_bounds, centers_temp = st_utils.compute_vol_bound(centers_interior_current, self.query_crop_size, self.input_crop_size, padding = True)
        n_x, n_y, n_z = vol_bounds['axis_n_crop']

        occupied_inputs_interior_current = inputs_interior_current[occupied_current_mask]
        occupied_centers_interior_current = centers_interior_current[occupied_current_mask]
        
        latents_current = self.encode_distributed(occupied_inputs_interior_current, occupied_centers_interior_current)
        if self.debug:
            latents_existing, inputs_existing, mask_current_in_existing = self.retrieve_existing_latents(centers_list, inputs_list, scene_idx, frame_idx, start_idx_prev, end_idx_prev)
        else:
            latents_existing, mask_current_in_existing = self.retrieve_existing_latents(centers_list, inputs_list, scene_idx, frame_idx, start_idx_prev, end_idx_prev)
        
        expanded_shape = (vol_bounds['n_crop'],) + self.empty_latent_code.shape #padded 
        
        grid_latents_frame_current = self.empty_latent_code.unsqueeze(0)
        grid_latents_frame_current = grid_latents_frame_current.expand(expanded_shape)
        grid_latents_frame_current = grid_latents_frame_current.reshape(n_x, n_y, n_z, *self.empty_latent_code.shape).clone()
        
        grid_latents_frame_existing = grid_latents_frame_current.clone()        
        
        occupied_current_mask_padded = torch.zeros((n_x, n_y, n_z), dtype = torch.bool).to(self.device)
        occupied_existing_mask_padded = torch.zeros((n_x, n_y, n_z), dtype = torch.bool).to(self.device)
        
        occupied_current_mask_padded[1:-1, 1:-1, 1:-1] = occupied_current_mask.reshape(n_x-2, n_y-2, n_z-2)
        occupied_existing_mask_padded[1:-1, 1:-1, 1:-1] = mask_current_in_existing.reshape(n_x-2, n_y-2, n_z-2)
        
        grid_latents_frame_current[occupied_current_mask_padded] = latents_current.clone()
        grid_latents_frame_existing[occupied_existing_mask_padded] = latents_existing.clone().to(self.device)
            
        mask_complete_current = (~occupied_current_mask_padded) & occupied_existing_mask_padded # current is empty and existing is occupied
        mask_complete_existing = (~occupied_existing_mask_padded) & occupied_current_mask_padded # existing is empty and current is occupied
        
        grid_latents_frame_current[mask_complete_current] = grid_latents_frame_existing[mask_complete_current]
        grid_latents_frame_existing[mask_complete_existing] = grid_latents_frame_current[mask_complete_existing]
        
        # print(f'mask_complete_current = {mask_complete_current.sum()}, mask_complete_existing = {mask_complete_existing.sum()}')
        
        lb = vol_bounds['lb']
        
        shifts = torch.tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)
        centers_current_shifted = st_utils.centers_to_grid_indexes(centers_interior_current.clone(), lb, self.query_crop_size).int() #.reshape(*centers_temp_interior.shape)
        centers_current_shifted = centers_current_shifted.unsqueeze(1) + shifts.unsqueeze(0)
        
        # centers_temp_shifted = centers_temp_shifted.reshape(*centers_temp.shape).int()
        # return centers_temp_shifted, grid_latents_frame
        latent_current_temp = grid_latents_frame_current[centers_current_shifted[..., 0], centers_current_shifted[..., 1], centers_current_shifted[..., 2]]
        latent_existing_temp = grid_latents_frame_existing[centers_current_shifted[..., 0], centers_current_shifted[..., 1], centers_current_shifted[..., 2]]
        
        # inputs_temp = grid_inputs_frame[centers_temp_shifted[..., 0], centers_temp_shifted[..., 1], centers_temp_shifted[..., 2]]
        c, h, w, d = self.empty_latent_code.shape

        latent_current_temp_stacked = latent_current_temp.reshape(len(centers_interior_current), c, 3 * h, 3 * w, 3 * d)
        latent_existing_temp_stacked = latent_existing_temp.reshape(len(centers_interior_current), c, 3 * h, 3 * w, 3 * d)
        
        latent_map_stacked = torch.cat((latent_current_temp_stacked, latent_existing_temp_stacked), dim = 1)
        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)

        return latent_map_stacked_merged, latent_existing_temp_stacked
    
    def retrieve_existing_latents(self, centers_list, inputs_list, scene_idx, frame_idx, start_idx_prev, end_idx_prev):
        centers_interior_current = centers_list[scene_idx][frame_idx]
        centers_interior_existing = centers_list[scene_idx][frame_idx - 1]

        if self.debug: inputs_interior_existing = self.merged_inputs[start_idx_prev:end_idx_prev].to(self.device)
        latents_merged_existing = self.merged_latents[start_idx_prev:end_idx_prev].to(self.device)
        
        mask_existing_in_current = st_utils.compute_mask_occupied(centers_interior_existing, centers_interior_current)
        mask_current_in_existing = st_utils.compute_mask_occupied(centers_interior_current, centers_interior_existing)
        
        if sum(mask_existing_in_current) == 0:
            raise Exception('No overlap found')
        
        latents_existing = latents_merged_existing[mask_existing_in_current]
        if self.debug: 
            inputs_existing = inputs_interior_existing[mask_existing_in_current]
            return latents_existing, inputs_existing, mask_current_in_existing    
            
        return latents_existing, mask_current_in_existing
    
    def fuse_cold_start(self, inputs_temp_interior, centers_temp_interior):
        # centers_temp_interior = centers_temp.reshape(-1, 3,3,3,3)[:,1,1,1,:]
        # inputs_temp_interior = inputs_temp.reshape(-1, 3, 3, 3, self.n_max_points_input, 4)[:, 1,1,1, :, :]
        # print(f'inputs_temp_centers.shape = {inputs_temp_interior.shape}')        
        occupied_mask = inputs_temp_interior[..., 3].sum(dim = -1) > 10
        
        vol_bounds, centers_temp = st_utils.compute_vol_bound(centers_temp_interior, self.query_crop_size, self.input_crop_size, padding = True)
        n_x, n_y, n_z = vol_bounds['axis_n_crop']

        occupied_inputs_interior = inputs_temp_interior[occupied_mask]
        occupied_centers_interior = centers_temp_interior[occupied_mask]
        
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # points = occupied_inputs_interior[..., :3].detach().cpu().numpy()
        # occ = occupied_inputs_interior[..., 3].detach().cpu().numpy()
        # pcd.points = o3d.utility.Vector3dVector(points[occ == 1].reshape(-1, 3))

        # o3d.visualization.draw_geometries([pcd])
        
        latents = self.encode_distributed(occupied_inputs_interior, occupied_centers_interior)
        
        expanded_shape = (vol_bounds['n_crop'],) + self.empty_latent_code.shape #padded 
        
        grid_latents_frame = self.empty_latent_code.unsqueeze(0)
        grid_latents_frame = grid_latents_frame.expand(expanded_shape)
        grid_latents_frame = grid_latents_frame.reshape(n_x, n_y, n_z, *self.empty_latent_code.shape).clone()
        
        occupied_mask_padded = torch.zeros((n_x, n_y, n_z), dtype = torch.bool).to(self.device)
        occupied_mask_padded[1:-1, 1:-1, 1:-1] = occupied_mask.reshape(n_x-2, n_y-2, n_z-2)
        # occupied_mask_padded = occupied_mask_padded.reshape(-1)
        grid_latents_frame[occupied_mask_padded] = latents.clone()
        
        # grid_inputs_frame = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(self.device)
        # grid_inputs_frame[occupied_mask_padded] = occupied_inputs_interior
        
        lb = vol_bounds['lb']
        
        shifts = torch.tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)
        centers_temp_shifted = st_utils.centers_to_grid_indexes(centers_temp_interior.clone(), lb, self.query_crop_size).int() #.reshape(*centers_temp_interior.shape)
        centers_temp_shifted = centers_temp_shifted.unsqueeze(1) + shifts.unsqueeze(0)
        
        # centers_temp_shifted = centers_temp_shifted.reshape(*centers_temp.shape).int()
        # return centers_temp_shifted, grid_latents_frame
        latent_temp = grid_latents_frame[centers_temp_shifted[..., 0], centers_temp_shifted[..., 1], centers_temp_shifted[..., 2]]
        # inputs_temp = grid_inputs_frame[centers_temp_shifted[..., 0], centers_temp_shifted[..., 1], centers_temp_shifted[..., 2]]
        c, h, w, d = self.empty_latent_code.shape

        latent_map_stacked_temp = latent_temp.reshape(len(centers_temp_interior), c, 3 * h, 3 * w, 3 * d)
        latent_map_stacked = torch.cat((latent_map_stacked_temp, latent_map_stacked_temp), dim = 1)
        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)

        # inputs_map_stacked = inputs_temp.reshape(len(centers_temp_interior), 27*self.n_max_points_input, 4)
        
        # return latent_map_stacked, inputs_map_stacked
        return latent_map_stacked_merged, latent_map_stacked_temp

    def fill_inputs_centers(self, inputs_distributed_neighboured, centers_frame):
        shifts = torch.tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)

        
        vol_bounds_p, centers_grid_padded = st_utils.compute_vol_bound(centers_frame, self.query_crop_size, self.input_crop_size, padding = True)
        n_x_p, n_y_p, n_z_p = vol_bounds_p['axis_n_crop']
        n_x, n_y, n_z = n_x_p - 2, n_y_p - 2, n_z_p - 2
        zeroed_padded_inputs = torch.zeros(n_x_p, n_y_p, n_z_p, self.n_max_points_input, 4).to(self.device)
        zeroed_padded_inputs[1:-1, 1:-1, 1:-1] = inputs_distributed_neighboured.reshape(n_x, n_y, n_z, self.n_max_points_input, 4)
                
        centers_grid_padded = centers_grid_padded.reshape(n_x_p, n_y_p, n_z_p, 3)

        lb = vol_bounds_p['lb'] #+ torch.Tensor([self.query_crop_size, self.query_crop_size, self.query_crop_size]).to(self.device)
        centers_grid_shifted = st_utils.centers_to_grid_indexes(centers_frame, lb, self.query_crop_size)
        centers_grid_shifted = centers_grid_shifted.unsqueeze(1) + shifts.unsqueeze(0)
        centers_grid_shifted = centers_grid_shifted.int()
        
        return zeroed_padded_inputs, centers_grid_padded, centers_grid_shifted
        
    def cube_distribution(self, source_tensor, centers_grid_shifted):
        return source_tensor[centers_grid_shifted[..., 0], centers_grid_shifted[..., 1], centers_grid_shifted[..., 2]]
    
    def get_logits(self, p_stacked, latents, centers):
    
        n_crops_total = p_stacked.shape[0]
                
        vol_bound = st_utils.get_grid_from_centers(centers, self.input_crop_size)
                
        # import open3d as o3d
        # for i in range(p_stacked.shape[0]):
        #     pcd = o3d.geometry.PointCloud()
        #     points = p_stacked[i]
        #     pcd.points = o3d.utility.Vector3dVector(points[points[:, 3] == 1, :3].detach().cpu().numpy())
        #     pcd.paint_uniform_color([0, 1, 0])
            
        #     vol_bound_i = vol_bound[i]
        #     bb_min = vol_bound_i[0]
        #     bb_max = vol_bound_i[1]
        #     bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bb_min.detach().cpu().numpy(), max_bound=bb_max.detach().cpu().numpy())
            
        #     o3d.visualization.draw_geometries([pcd, bbox])
        p_n_stacked = normalize_coord(p_stacked, vol_bound)

        n_batch = int(np.ceil(n_crops_total / self.n_voxels_max))

        logits_stacked = None  # Initialize logits directly

        torch.cuda.empty_cache()
        for i in range(n_batch):
            start = i * self.n_voxels_max
            end = min((i + 1) * self.n_voxels_max, n_crops_total)

            p_stacked_batch = p_stacked[start:end]
            p_n_stacked_batch = p_n_stacked[start:end]
            latent_map_full_batch = latents[start:end]
            
            kwargs = {}
            pi_in = p_stacked_batch[..., :3].clone()
            pi_in = {'p': pi_in}
            p_n = {}
            p_n['grid'] = p_n_stacked_batch[..., :3].clone()
            pi_in['p_n'] = p_n
            c = {}
            latent_map_full_batch_decoded = self.unet.decode(latent_map_full_batch, self.features_shapes)
            
            c['grid'] = latent_map_full_batch_decoded.clone()
            logits_decoded = self.model.decode(pi_in, c, **kwargs).logits
            
            if logits_stacked is None:
                logits_stacked = logits_decoded.clone()  # Initialize logits directly with the first batch
            else:
                logits_stacked = torch.cat((logits_stacked, logits_decoded), dim=0)  # Concatenate logits
        
        return logits_stacked
    
    
        
    def merge_latent_map(self, latent_map):
        n_samples = latent_map.shape[0]
        n_batches = int(math.ceil(n_samples / self.n_voxels_max))
        
        merged_latent_map = None
        
        for i in range(n_batches):
            # Determine the start and end indices for the current batch
            start_idx = i * self.n_voxels_max
            end_idx = min((i + 1) * self.n_voxels_max, n_samples)
            
            # Extract the current batch from the latent_map
            latent_map_batch = latent_map[start_idx:end_idx]
            
            # Create a dictionary as required by the model_merge function
            fea_dict = {'latent': latent_map_batch}
            
            # Merge the current batch using the model_merge function
            merged_batch = self.model_merge(fea_dict)
            
            # Concatenate the merged batch to the final merged_latent_map
            if merged_latent_map is None:
                merged_latent_map = merged_batch  # Initialize with the first batch
            else:
                merged_latent_map = torch.cat((merged_latent_map, merged_batch), dim=0)
    
        return merged_latent_map
    
    def get_empty_latent_representation(self):
        center = torch.tensor([0,0,0]).unsqueeze(0).to(self.device)
        
        empty_inputs = st_utils.get_empty_inputs(center, self.query_crop_size, n_max_points = 2048)
        occ = torch.zeros(*empty_inputs.shape[0:2], 1).to(self.device)
        empty_inputs = torch.cat((empty_inputs, occ), axis = -1)
        empty_latent_code = self.encode_distributed(empty_inputs, center)

        return empty_latent_code.squeeze(0)
    
    def encode_distributed(self, inputs, centers):
        
        vol_bound = st_utils.get_grid_from_centers(centers, self.input_crop_size)

        n_crop = inputs.shape[0]
        
        n_batch_voxels = int(math.ceil(n_crop / self.n_voxels_max))

        latent_map = None

        for i in range(n_batch_voxels): 
            
            if self.limited_gpu: torch.cuda.empty_cache()
            
            start = i * self.n_voxels_max
            end = min((i + 1) * self.n_voxels_max, inputs.shape[0])
            inputs_distributed_batch = inputs[start:end]
            inputs_distributed_batch_3D = inputs_distributed_batch[..., :3]
            vol_bound_batch = vol_bound[start:end]
            
            kwargs = {}
            fea_type = 'grid'
            index = {}
            ind = coord2index(inputs_distributed_batch, vol_bound_batch, reso=self.reso, plane=fea_type)
            index[fea_type] = ind
            input_cur_batch = add_key(inputs_distributed_batch_3D, index, 'points', 'index', device=self.device)
            
            self.t0 = time.time()
            
            if self.unet is None:
                fea, self.unet = self.model.encode_inputs(input_cur_batch)
            else:
                fea, _ = self.model.encode_inputs(input_cur_batch)
                            
            _, latent_map_batch, self.features_shapes = self.unet(fea, return_feature_maps=True, decode = False, limited_gpu = False)


            if latent_map is None:
                latent_map = latent_map_batch  # Initialize latent_map with the first batch
            else:
                latent_map = torch.cat((latent_map, latent_map_batch), dim=0)  # Concatenate latent maps
                
        latent_map_shape = latent_map.shape
        return latent_map.reshape(n_crop, *latent_map_shape[1:])

    def get_distributed_inputs(self, distributed_inputs_raw, n_max = 2048, occ_perc = 1.0, return_empty = False, isquery = False):
        # Clone the input tensor
        distributed_inputs = distributed_inputs_raw.clone()
        
        vol_bound, centers = st_utils.compute_vol_bound(distributed_inputs[:, :3].reshape(-1, 3), self.query_crop_size, self.input_crop_size)
        
        n_crops = vol_bound['n_crop']
        
        distributed_inputs = distributed_inputs.repeat(n_crops, 1, 1)

        # Convert vol_bound['input_vol'] to a torch tensor
        
        # Create masks for each condition
        mask_1 = distributed_inputs[:, :, 0] < vol_bound['input_vol'][:, 0, 0].unsqueeze(1)
        mask_2 = distributed_inputs[:, :, 0] > vol_bound['input_vol'][:, 1, 0].unsqueeze(1)
        mask_3 = distributed_inputs[:, :, 1] < vol_bound['input_vol'][:, 0, 1].unsqueeze(1)
        mask_4 = distributed_inputs[:, :, 1] > vol_bound['input_vol'][:, 1, 1].unsqueeze(1)
        mask_5 = distributed_inputs[:, :, 2] < vol_bound['input_vol'][:, 0, 2].unsqueeze(1)
        mask_6 = distributed_inputs[:, :, 2] > vol_bound['input_vol'][:, 1, 2].unsqueeze(1)

        # Combine masks
        final_mask = mask_1 | mask_2 | mask_3 | mask_4 | mask_5 | mask_6

        # Set values to 0 where conditions are met
        distributed_inputs[:, :, 3][final_mask] = 0
        
        # Create a mask for selecting points with label 1
        indexes_keep = distributed_inputs[..., 3] == 1

        random_points = st_utils.get_empty_inputs(centers, self.input_crop_size, n_max_points = n_max)
        
        distributed_inputs_short = torch.zeros(n_crops, n_max, 4, device=self.device)
        distributed_inputs_short[:, :, :3] = random_points.reshape(n_crops, n_max, 3)
                
        # Select n_max points
        new_line_template = torch.zeros(indexes_keep.shape[1], dtype=torch.bool)
        for i in range(distributed_inputs.shape[0]):
            indexes_line = indexes_keep[i, :]
            if indexes_line.sum() > int(n_max*occ_perc): # We still want unooccupied points 
                indexes_true = torch.where(indexes_line)[0]
                random_indexes_keep = torch.randperm(indexes_true.shape[0])[:int(n_max*occ_perc)]
                
                # Create a new line with the randomly selected indexes set to True in a single step
                new_line = new_line_template.clone()
                new_line[indexes_true[random_indexes_keep]] = True
                
                # Update the indexes_keep tensor with the new line for the current sample
                indexes_keep[i] = new_line
            if indexes_line.sum() < int(n_max*0.02): # TODO move this to config, // if the number of occupied points is less than 2%, set as empty
                indexes_keep[i] = torch.zeros_like(indexes_keep[i])
        # Select n_max points
        n_points = indexes_keep.sum(axis=1)
        mask = torch.arange(n_max).expand(n_crops, n_max).to(device=self.device) < n_points.unsqueeze(-1)
                
        distributed_inputs_short[mask] = distributed_inputs[indexes_keep]
        
        voxels_occupied = distributed_inputs_short[..., 3].sum(dim=1).int() > 25 #TODO move this to config

        if isquery:
            for i in range(distributed_inputs.shape[0]):
                if not voxels_occupied[i]: continue
                centers_remove = centers[i:i+1]
                crop_size = self.input_crop_size
                random_points = distributed_inputs_short[i]
                occupied_inputs = distributed_inputs_short[i, distributed_inputs_short[i, :, 3] == 1]
                unoccupied_inputs = distributed_inputs_short[i, distributed_inputs_short[i, :, 3] == 0]
                n_sample = unoccupied_inputs.shape[0]
                thresh = 0.03 #distance to occupied points
                
                new_samples = st_utils.maintain_n_sample_points(centers_remove, crop_size, random_points, occupied_inputs, n_sample, thresh).clone()
                distributed_inputs_short[i] = torch.cat((occupied_inputs, new_samples), dim=0)
                
                
            inputs_frame_occupied = distributed_inputs_short
            centers_frame_occupied = centers

        else:
            if return_empty:
                inputs_frame_occupied = distributed_inputs_short
                centers_frame_occupied = centers
                
            else:
                inputs_frame_occupied = distributed_inputs_short[voxels_occupied]
                centers_frame_occupied = centers[voxels_occupied]
            
        return inputs_frame_occupied, centers_frame_occupied, vol_bound['axis_n_crop']