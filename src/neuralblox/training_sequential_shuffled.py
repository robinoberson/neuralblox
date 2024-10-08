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
from src.neuralblox.helpers.gpu_monitor import GPUMonitor

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


    def __init__(self, model, model_merge, optimizer_backbone, optimizer_merge, cfg, input_crop_size = 1.6, query_crop_size = 1.0, device=None, input_type='pointcloud',
                 threshold=0.5, query_n = 8192, unet_hdim = 32, unet_depth = 2, grid_reso = 24, limited_gpu = False, n_voxels_max = 20, n_batch = 40, n_max_points = 2048, n_max_points_query = 8192, occ_per_query = 0.5, return_flat = True,
                 sigma = 0.8):
        self.model = model
        self.model_merge = model_merge
        self.optimizer_backbone = optimizer_backbone
        self.optimizer_merge = optimizer_merge
        self.device = device
        self.input_crop_size = input_crop_size
        self.query_crop_size = query_crop_size
        self.hdim = unet_hdim
        self.unet = None
        self.limited_gpu = limited_gpu
        self.reso = grid_reso
        self.iteration = 0
        self.n_voxels_max = n_voxels_max
        self.n_batch = n_batch
        self.n_max_points_input = n_max_points
        self.n_max_points_query = n_max_points_query
        self.occ_per_query = occ_per_query
        self.voxel_grid = VoxelGrid()
        self.return_flat = return_flat
        self.sigma = sigma
        self.empty_latent_code = self.get_empty_latent_representation()
        self.timing_counter = 0
        self.experiment = None
        self.log_experiment = False
        self.shifts = torch.tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)
        self.cfg = cfg
        self.points_threshold = 20 #TODO move to cfg
        self.keep_empty_per = 0.7 #TODO move to cfg

        self.debug = False
                
        current_dir = os.getcwd()
        
        if 'robin' in current_dir:
            self.location = 'home'
        elif 'cluster' in current_dir:
            self.location = 'euler'
        else:
            self.location = 'local'

        if self.location != 'euler':
            self.GPU_monitor = GPUMonitor()
            self.GPU_monitor.update_memory_usage()
            
        print(f'Location: {self.location}')
    
    def set_experiment(self, experiment):
        self.experiment = experiment
        self.log_experiment = True
    
    def train_sequence(self, full_batch):
        with torch.no_grad():
            centers_idx_full_current, centers_idx_full_prev, query_points_full_, grid_shapes_full_current, grid_shapes_full_prev, centers_lookup_full_current, centers_lookup_full_prev, mask_centers_full, latents_full, inputs_full, centers_coord_full = self.precompute_sequence(full_batch)
        print(f'Finished precomputing')
        torch.cuda.empty_cache()
        gc.collect()
        
        loss = self.train_batch(centers_idx_full_current, centers_idx_full_prev, query_points_full_, grid_shapes_full_current, grid_shapes_full_prev, centers_lookup_full_current, centers_lookup_full_prev, mask_centers_full, latents_full, inputs_full, centers_coord_full)
        # print(f'Finished training batch, loss = {loss}')
        return loss
    
    def validate_sequence(self, full_batch):
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            centers_idx_full_current, centers_idx_full_prev, query_points_full_, grid_shapes_full_current, grid_shapes_full_prev, centers_lookup_full_current, centers_lookup_full_prev, mask_centers_full, latents_full, inputs_full, centers_coord_full = self.precompute_sequence(full_batch)
            print(f'Finished precomputing validation')
            loss = self.train_batch(centers_idx_full_current, centers_idx_full_prev, query_points_full_, grid_shapes_full_current, grid_shapes_full_prev, centers_lookup_full_current, centers_lookup_full_prev, mask_centers_full, latents_full, inputs_full, centers_coord_full, training=False)

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
            prob = self.cfg['training']['prob']
            save_mask = torch.rand(n_scenes, n_frames, device=self.device) < prob         
            if self.iteration == 0: print(f'first pass through data')
            for scene_idx in range(n_scenes):
                inputs_frame_distributed_padded_list = []
                query_frame_distributed_padded_list = []
                centers_frame_list = []
                axis_n_crop_frame_list = []
                # for frame_idx in range(p_in_full.shape[1]):
                for frame_idx in range(n_frames):
                    inputs_frame = p_in_full[scene_idx, frame_idx]
                    inputs_frame_distributed_padded, centers_frame, vol_bound_frame = self.get_distributed_inputs(inputs_frame, self.n_max_points_input, 1.0, return_empty = True, isquery = False, padding = True)
                    axis_n_crop = vol_bound_frame['axis_n_crop'] #padded

                    query_frame = p_query_full[scene_idx, frame_idx]
                    query_frame_distributed_padded, centers_frame_query, vol_bound_frame_query = self.get_distributed_inputs(query_frame, self.n_max_points_query, self.occ_per_query, return_empty = True, isquery = True, padding = True)

                    frame_mask = st_utils.compute_mask_occupied(centers_frame, centers_frame_query)
                    query_mask = st_utils.compute_mask_occupied(centers_frame_query, centers_frame)
                    
                    frame_mask_sum = frame_mask.sum()
                    query_mask_sum = query_mask.sum()
                    
                    # if frame_mask.sum() != frame_mask.shape[0]:
                    #     # print(f'PROBLEM: Frame {frame_idx} has {frame_mask.sum()} occupied voxels instead of {frame_mask.shape[0]}')
                    #     out_dir = os.path.join(self.cfg['training']['out_dir'], 'debug')
                    #     if not os.path.exists(out_dir):
                    #         os.mkdir(out_dir)
                            
                    #     torch.save(inputs_frame, os.path.join(out_dir, f'inputs_frame_{self.iteration}_{frame_idx}.pt'))
                    #     torch.save(query_frame, os.path.join(out_dir, f'query_frame_{self.iteration}_{frame_idx}.pt'))
                    #     torch.save(axis_n_crop, os.path.join(out_dir, f'axis_n_crop_{self.iteration}_{frame_idx}.pt'))
                        
                    centers_frame = centers_frame[frame_mask].reshape(-1, 3)
                    centers_frame_query = centers_frame_query[query_mask].reshape(-1, 3)
                    
                    inputs_frame_distributed_padded = inputs_frame_distributed_padded[frame_mask].reshape(-1, self.n_max_points_input, 4)
                    query_frame_distributed_padded = query_frame_distributed_padded[query_mask].reshape(-1, self.n_max_points_query, 4)
                    
                    inputs_frame_distributed_padded_list.append(inputs_frame_distributed_padded)
                    query_frame_distributed_padded_list.append(query_frame_distributed_padded)
                    centers_frame_list.append(centers_frame)
                    axis_n_crop_frame_list.append(axis_n_crop)
                    
                    n_x, n_y, n_z = axis_n_crop
                    n_voxels[scene_idx, frame_idx] = (n_x)*(n_y)*(n_z) #padded
                    
                    # return inputs_frame_distributed_padded, query_frame_distributed_padded, centers_frame
                
                inputs_scene_distributed_list.append(inputs_frame_distributed_padded_list)
                query_scene_distributed_list.append(query_frame_distributed_padded_list)
                centers_scene_list.append(centers_frame_list)
                axis_n_crop_scene_list.append(axis_n_crop_frame_list)
            
            # print(f'Second pass through data')
            n_voxels_total = torch.sum(n_voxels[save_mask])
            
            # print(f'n_voxels_total = {n_voxels_total}')
            
            c, h, w, d = self.empty_latent_code.shape

            # storage_centers_idx = torch.zeros(n_voxels_total, 3).to(torch.device('cpu'))
            # storage_inputs = torch.zeros(n_voxels_total, self.n_max_points_input, 4).to(torch.device('cpu'))
            # storage_frame_shapes = torch.zeros(n_voxels_total, 3).to(torch.device('cpu'))
            # storage_indexes_lookup = torch.zeros(n_voxels_total, 2).to(torch.device('cpu'))
            
            # storage_latents_padded = torch.zeros(n_voxels_total, c, h, w, d).to(torch.device('cpu'))
            
            centers_idx_full_current = torch.zeros(n_voxels_total, 3, dtype = torch.int32).to(torch.device('cpu')) # all centers contained in the frame with grid indices
            centers_idx_full_prev = torch.zeros(n_voxels_total, 3, dtype = torch.int32).to(torch.device('cpu')) # all centers contained in the frame with grid indices

            query_points_full = torch.zeros(n_voxels_total, self.n_max_points_query, 4).to(torch.device('cpu'))

            grid_shapes_full_current = torch.zeros(n_voxels_total, 3, dtype = torch.int32).to(torch.device('cpu')) # grid shapes contained in the frame
            grid_shapes_full_prev = torch.zeros(n_voxels_total, 3, dtype = torch.int32).to(torch.device('cpu')) # grid shapes contained in the frame
            
            centers_lookup_full_current = torch.zeros(n_voxels_total, 2, dtype = torch.int32).to(torch.device('cpu')) # centers lookup contained in the frame, allows to retrieve the correct centers 
            centers_lookup_full_prev = torch.zeros(n_voxels_total, 2, dtype = torch.int32).to(torch.device('cpu')) # centers lookup contained in the frame, allows to retrieve the correct centers
            
            mask_centers_full = torch.zeros(n_voxels_total, dtype = torch.bool).to(torch.device('cpu'))
            
            latents_full = torch.zeros(n_voxels_total, *self.empty_latent_code.shape).to(torch.device('cpu'))
            inputs_full = torch.zeros(n_voxels_total, self.n_max_points_input, 4).to(torch.device('cpu'))
            centers_coord_full = torch.zeros(n_voxels_total, 3).to(torch.device('cpu'))

            mask_force_save_empty = torch.zeros(n_voxels_total, dtype = torch.bool).to(torch.device('cpu'))

            # dtype_size = torch.tensor([], dtype=torch.float32).element_size()
            # total_elements = n_voxels_total * c * h * w * d
            # memory_size_bytes = total_elements * dtype_size
            # memory_size_gb = memory_size_bytes / (1024 ** 3)
            # print(f'memory_size_gb = {memory_size_gb}')
            
            idx_start = 0
            
            for scene_idx in range(n_scenes):
                torch.cuda.empty_cache()
                gc.collect()
                for frame_idx in range(n_frames):
                    centers_frame_padded = centers_scene_list[scene_idx][frame_idx]
                    inputs_frame_padded = inputs_scene_distributed_list[scene_idx][frame_idx]
                    query_frame_padded = query_scene_distributed_list[scene_idx][frame_idx].reshape(-1, self.n_max_points_query, 4)
                    
                    if frame_idx == 0:
                        # return centers_frame_padded, inputs_frame_padded, query_frame_padded
                        tup = self.process_frame_cold_start(centers_frame_padded, inputs_frame_padded)

                    else: 
                        # if torch.equal(centers_frame_padded, prev_centers_frame_padded):
                        #     print('WARNING: centers_frame_padded == prev_centers_frame_padded')
                        # else:
                            # print('centers_frame_padded != prev_centers_frame_padded')
                        # return centers_frame_padded, inputs_frame_padded, merged_latents_padded, prev_centers_frame_padded, prev_vol_bounds_padded, prev_inputs_frame_padded
                        tup = self.process_frame(centers_frame_padded, inputs_frame_padded, merged_latents_padded, prev_centers_frame_padded, prev_vol_bounds_padded, prev_inputs_frame_padded)
                        if self.debug: return tup
                    
                    centers_idx_current, centers_idx_prev, grid_shapes_current, grid_shapes_prev, centers_lookup_frame_current, centers_lookup_frame_prev, mask_centers, merged_latents_padded, vol_bounds_frame_padded = tup

                    prev_centers_frame_padded = centers_frame_padded.clone()
                    prev_centers_idx_prev = centers_idx_current.clone()
                    prev_inputs_frame_padded = inputs_frame_padded.clone()
                    prev_vol_bounds_padded = vol_bounds_frame_padded.copy()
                    
                    n_voxels_frame = centers_idx_current.shape[0]
                    
                    idx_end = idx_start + n_voxels_frame #n_voxels = n_x * n_y * n_z

                    if save_mask[scene_idx, frame_idx]: #Decide if we drop the frame to get more diverse data

                        centers_idx_full_current[idx_start:idx_end] = centers_idx_current.clone()
                        centers_idx_full_prev[idx_start:idx_end] = centers_idx_prev.clone()
                        
                        query_points_full[idx_start:idx_end] = query_frame_padded.clone()
                        
                        grid_shapes_full_current[idx_start:idx_end] = grid_shapes_current.clone() # grid shapes contained in the frame
                        grid_shapes_full_prev[idx_start:idx_end] = grid_shapes_prev.clone()

                        centers_lookup_full_current[idx_start:idx_end] = (centers_lookup_frame_current + idx_start).clone() # centers lookup contained in the frame, allows to retrieve the correct centers 
                        centers_lookup_full_prev[idx_start:idx_end] = (centers_lookup_frame_prev + idx_start).clone()
                        mask_centers_full[idx_start:idx_end] = mask_centers.clone()

                        latents_full[idx_start:idx_end] = merged_latents_padded.clone()
                        inputs_full[idx_start:idx_end] = inputs_frame_padded.clone()
                        centers_coord_full[idx_start:idx_end] = centers_frame_padded.clone()

                        idx_start = idx_end
            
            # Create a mask for occupied inputs
            mask_occupied = inputs_full[..., 3].sum(dim=-1) > self.points_threshold
            # sum_inputs = inputs[:, :, :, 3].sum(dim=2)
            # mask1 = sum_inputs > self.points_threshold #n points 
            # mask_occupied = mask1.sum(dim=1) > 6 #n neighbouring voxels occupied

            # Apply the mask to filter the tensors
            filtered_centers_idx_full_current = centers_idx_full_current[mask_occupied]
            filtered_centers_idx_full_prev = centers_idx_full_prev[mask_occupied]
            
            filtered_query_points_full = query_points_full[mask_occupied]
            
            filtered_grid_shapes_full_current = grid_shapes_full_current[mask_occupied]
            filtered_grid_shapes_full_prev = grid_shapes_full_prev[mask_occupied]
            
            filtered_centers_lookup_full_current = centers_lookup_full_current[mask_occupied]  
            filtered_centers_lookup_full_prev = centers_lookup_full_prev[mask_occupied]
            filtered_mask_centers_full = mask_centers_full[mask_occupied]
                      
            # #add empty voxels as well to learn the empty latent code
            # centers_idx_current, query_frame_occupied, grid_shapes_padded, centers_lookup_padded, merged_latents_padded, inputs_frame_occupied, centers_frame_occupied = self.generate_empty_inputs(inputs_full.shape[0])
            tup_empty = self.generate_empty_inputs(inputs_full.shape[0])
            centers_idx_empty, query_frame_padded_empty, grid_shapes_empty, centers_lookup_frame_empty, mask_centers_empty, merged_latents_padded_empty, inputs_frame_occupied_empty, centers_frame_occupied_empty = tup_empty
            
            filtered_centers_idx_full_current = torch.cat([filtered_centers_idx_full_current, centers_idx_empty.to(torch.device('cpu'))], dim = 0)
            filtered_centers_idx_full_prev = torch.cat([filtered_centers_idx_full_prev, centers_idx_empty.to(torch.device('cpu'))], dim = 0)
            
            filtered_query_points_full = torch.cat([filtered_query_points_full, query_frame_padded_empty.to(torch.device('cpu'))], dim = 0)
            
            filtered_grid_shapes_full_current = torch.cat([filtered_grid_shapes_full_current, grid_shapes_empty.to(torch.device('cpu'))], dim = 0)
            filtered_grid_shapes_full_prev = torch.cat([filtered_grid_shapes_full_prev, grid_shapes_empty.to(torch.device('cpu'))], dim = 0)

            filtered_centers_lookup_full_current = torch.cat([filtered_centers_lookup_full_current, centers_lookup_frame_empty.to(torch.device('cpu'))], dim = 0)
            filtered_centers_lookup_full_prev = torch.cat([filtered_centers_lookup_full_prev, centers_lookup_frame_empty.to(torch.device('cpu'))], dim = 0)
            filtered_mask_centers_full = torch.cat([filtered_mask_centers_full, mask_centers_empty.to(torch.device('cpu'))], dim = 0)
            
            latents_full = torch.cat([latents_full, merged_latents_padded_empty.to(torch.device('cpu'))], dim = 0)
            inputs_full = torch.cat([inputs_full, inputs_frame_occupied_empty.to(torch.device('cpu'))], dim = 0)
            centers_coord_full = torch.cat([centers_coord_full, centers_frame_occupied_empty.to(torch.device('cpu'))], dim = 0)
            # filtered_centers_idx_full = torch.cat([filtered_centers_idx_full, centers_idx_current.to(torch.device('cpu'))], dim = 0)
            # filtered_query_points_full = torch.cat([filtered_query_points_full, query_frame_occupied.to(torch.device('cpu'))], dim = 0)
            # filtered_grid_shapes_full = torch.cat([filtered_grid_shapes_full, grid_shapes_padded.to(torch.device('cpu'))], dim = 0)
            # filtered_centers_lookup_full = torch.cat([filtered_centers_lookup_full, centers_lookup_padded.to(torch.device('cpu'))], dim = 0)
            
            # latents_full = torch.cat([latents_full, merged_latents_padded.to(torch.device('cpu'))], dim = 0)
            # inputs_full = torch.cat([inputs_full, inputs_frame_occupied.to(torch.device('cpu'))], dim = 0)
            # centers_coord_full = torch.cat([centers_coord_full, centers_frame_occupied.to(torch.device('cpu'))], dim = 0)
            
            # Generate a random permutation
            permutation_mask = torch.randperm(filtered_centers_idx_full_current.shape[0])
            
            # Apply the permutation mask
            centers_idx_full_current = filtered_centers_idx_full_current[permutation_mask]
            centers_idx_full_prev = filtered_centers_idx_full_prev[permutation_mask]
            
            query_points_full = filtered_query_points_full[permutation_mask]
            
            grid_shapes_full_current = filtered_grid_shapes_full_current[permutation_mask]
            grid_shapes_full_prev = filtered_grid_shapes_full_prev[permutation_mask]
            
            centers_lookup_full_current = filtered_centers_lookup_full_current[permutation_mask]
            centers_lookup_full_prev = filtered_centers_lookup_full_prev[permutation_mask]
            mask_centers_full = filtered_mask_centers_full[permutation_mask]
            
            # centers_idx_full_current = filtered_centers_idx_full_current
            # centers_idx_full_prev = filtered_centers_idx_full_prev
            
            # query_points_full = filtered_query_points_full
            
            # grid_shapes_full_current = filtered_grid_shapes_full_current
            # grid_shapes_full_prev = filtered_grid_shapes_full_prev
            
            # centers_lookup_full_current = filtered_centers_lookup_full_current
            # centers_lookup_full_prev = filtered_centers_lookup_full_prev
            
            # mask_centers_full = filtered_mask_centers_full
            
            print(f'Batch contains {centers_idx_full_current.shape[0]} voxels')
            
            # if self.debug:
            #     centers_coord_full_debug = filtered_centers_coord_full[permutation_mask]
            #     inputs_full_debug = filtered_inputs_coord_full[permutation_mask]
            #     return centers_idx_full, query_points_full, grid_shapes_full, centers_lookup_full, latents_full, inputs_full, centers_coord_full, centers_coord_full_debug, inputs_full_debug

            # print(f'finished precomputation batch')
            return centers_idx_full_current, centers_idx_full_prev, query_points_full, grid_shapes_full_current, grid_shapes_full_prev, centers_lookup_full_current, centers_lookup_full_prev, mask_centers_full, latents_full, inputs_full, centers_coord_full
        
    def init_empty_latent_grid(self, vol_bounds_frame_padded):
        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop']
        
        expanded_shape = (vol_bounds_frame_padded['n_crop'],) + self.empty_latent_code.shape #padded 
        empty_latent_grid = self.empty_latent_code.unsqueeze(0)
        empty_latent_grid = empty_latent_grid.expand(expanded_shape)
        empty_latent_grid = empty_latent_grid.reshape(n_x, n_y, n_z, *self.empty_latent_code.shape).clone()
        
        return empty_latent_grid
    
    def encode_occupied_inputs_frame(self, inputs_frame, centers_frame, vol_bounds_frame_padded, threshold = None):
        if threshold is None:
            threshold = self.points_threshold
            
        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop']
        occupied_frame_mask = inputs_frame[..., 3].sum(dim = -1) > threshold
        grid_latents_frame_current = self.init_empty_latent_grid(vol_bounds_frame_padded)

        if occupied_frame_mask.sum() == 0:
            return grid_latents_frame_current.reshape(-1, *self.empty_latent_code.shape)
        
        latents_occupied_frame = self.encode_distributed(inputs_frame[occupied_frame_mask], centers_frame[occupied_frame_mask])

        occupied_current_mask_padded = torch.nn.functional.pad(occupied_frame_mask.reshape(n_x-2, n_y-2, n_z-2), (1, 1, 1, 1, 1, 1), value=False)

        grid_latents_frame_current[occupied_current_mask_padded] = latents_occupied_frame.clone()
        grid_latents_frame_current = grid_latents_frame_current.reshape(-1, *self.empty_latent_code.shape)
        
        return grid_latents_frame_current
    
    def encode_occupied_inputs_flat(self, inputs_frame, centers_frame):
        assert inputs_frame.shape[0] == centers_frame.shape[0]
        if len(inputs_frame.shape) != 3:
            inputs_frame = inputs_frame.reshape(-1, self.n_max_points_input, 4)
            centers_frame = centers_frame.reshape(-1, 3)
                    
        occupied_frame_mask = inputs_frame[..., 3].sum(dim = -1) > self.points_threshold
        
        latents_occupied_frame = self.encode_distributed(inputs_frame[occupied_frame_mask], centers_frame[occupied_frame_mask])
       
        grid_latents_frame_current = self.empty_latent_code.repeat(inputs_frame.shape[0], 1, 1, 1, 1).clone()
        grid_latents_frame_current[occupied_frame_mask] = latents_occupied_frame
        
        return grid_latents_frame_current
    
    
    def pad_points(self, inputs_frame, vol_bounds_frame_padded, n_points):
        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop']
        inputs_frame_padded = torch.zeros(n_x, n_y, n_z, n_points, 4).to(self.device)
        inputs_frame_padded[1:-1, 1:-1, 1:-1] = inputs_frame.reshape(n_x-2, n_y-2, n_z-2, n_points, 4)
        inputs_frame_padded = inputs_frame_padded.reshape(-1, n_points, 4)
        
        return inputs_frame_padded
    def process_frame_cold_start(self, centers_frame_padded, inputs_frame_padded):
        vol_bounds_frame_padded, _ = st_utils.compute_vol_bound(centers_frame_padded, self.query_crop_size, self.input_crop_size)

        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop'] #padded
        centers_frame = centers_frame_padded.reshape(n_x, n_y, n_z, 3)[1:-1, 1:-1, 1:-1, :].reshape(-1, 3)
        inputs_frame = inputs_frame_padded.reshape(n_x, n_y, n_z, self.n_max_points_input, 4)[1:-1, 1:-1, 1:-1, :, :].reshape(-1, self.n_max_points_input, 4)

        if len(centers_frame) != (n_x-2)*(n_y-2)*(n_z-2):
            print(f'PROBLEM: {len(centers_frame)} != {(n_x-2)*(n_y-2)*(n_z-2)}')
            
        grid_latents_frame_current = self.encode_occupied_inputs_frame(inputs_frame, centers_frame, vol_bounds_frame_padded) #contains padded empty latents
        
        centers_lookup = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame), 1).to(self.device)
        centers_lookup_padded = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame_padded), 1).to(self.device)

        grid_shapes = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame), 1).to(self.device)
        grid_shapes_padded = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame_padded), 1).to(self.device)
        
        centers_idx_current = st_utils.centers_to_grid_indexes(centers_frame, vol_bounds_frame_padded['lb'], self.query_crop_size).int().reshape(-1, 3)
        centers_idx_current_padded = st_utils.centers_to_grid_indexes(centers_frame_padded, vol_bounds_frame_padded['lb'], self.query_crop_size).int().reshape(-1, 3)

        distributed_latents = st_utils.get_distributed_voxel(centers_idx_current, grid_latents_frame_current, grid_shapes, centers_lookup, self.shifts)
        inputs_frame_padded = self.pad_points(inputs_frame, vol_bounds_frame_padded, self.n_max_points_input)
        distributed_inputs = st_utils.get_distributed_voxel(centers_idx_current, inputs_frame_padded, grid_shapes, centers_lookup, self.shifts)
        
        c, h, w, d = self.empty_latent_code.shape

        distributed_latents = distributed_latents.reshape(-1, c, 3*h, 3*w, 3*d)
        stacked_frame = torch.cat((distributed_latents, distributed_latents), dim = 1)
        
        merged_latents = self.merge_latent_map(stacked_frame)
        merged_latents_padded = self.init_empty_latent_grid(vol_bounds_frame_padded)

        merged_latents_padded[1:-1, 1:-1, 1:-1] = merged_latents.reshape(n_x-2, n_y-2, n_z-2, *self.empty_latent_code.shape)
        merged_latents_padded = merged_latents_padded.reshape(-1, *self.empty_latent_code.shape)
        
        centers_lookup_frame_current = centers_lookup_padded
        centers_lookup_frame_prev = centers_lookup_padded 
        
        mask_centers = torch.ones(len(centers_frame_padded)).to(self.device)
         
        return_tup = [centers_idx_current_padded, centers_idx_current_padded, grid_shapes_padded, grid_shapes_padded, centers_lookup_padded, centers_lookup_padded, mask_centers, merged_latents_padded, vol_bounds_frame_padded]
        # if self.debug:
        #     return_tup.append(distributed_inputs)
            #  [merged_latents_padded, centers_frame, vol_bounds_frame_padded, centers_idx_current, distributed_inputs, inputs_frame_padded]

        return return_tup
    

    def process_frame(self, centers_frame_padded, inputs_frame_padded, prev_merged_latents_padded, prev_centers_frame_padded, vol_bounds_padded_prev, prev_undist_inputs_padded = None):
        vol_bounds_frame_padded, _ = st_utils.compute_vol_bound(centers_frame_padded, self.query_crop_size, self.input_crop_size)

        n_x, n_y, n_z = vol_bounds_frame_padded['axis_n_crop'] #padded
        centers_frame = centers_frame_padded.reshape(n_x, n_y, n_z, 3)[1:-1, 1:-1, 1:-1, :].reshape(-1, 3)
        inputs_frame = inputs_frame_padded.reshape(n_x, n_y, n_z, self.n_max_points_input, 4)[1:-1, 1:-1, 1:-1, :, :].reshape(-1, self.n_max_points_input, 4)

        if len(centers_frame) != (n_x-2)*(n_y-2)*(n_z-2):
            print(f'PROBLEM: {len(centers_frame)} != {(n_x-2)*(n_y-2)*(n_z-2)}')
            
        grid_latents_frame_current = self.encode_occupied_inputs_frame(inputs_frame, centers_frame, vol_bounds_frame_padded)
        
        centers_lookup = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame), 1).to(self.device)
        centers_lookup_padded = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame_padded), 1).to(self.device)
        grid_shapes = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame), 1).to(self.device)
        grid_shapes_padded = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame_padded), 1).to(self.device)
        centers_idx_current = st_utils.centers_to_grid_indexes(centers_frame, vol_bounds_frame_padded['lb'], self.query_crop_size).int().reshape(-1, 3)
        centers_idx_current_padded = st_utils.centers_to_grid_indexes(centers_frame_padded, vol_bounds_frame_padded['lb'], self.query_crop_size).int().reshape(-1, 3)
        
        distributed_latents = st_utils.get_distributed_voxel(centers_idx_current, grid_latents_frame_current, grid_shapes, centers_lookup, self.shifts)

        if self.debug:
            distributed_inputs = st_utils.get_distributed_voxel(centers_idx_current, inputs_frame_padded, grid_shapes, centers_lookup, self.shifts)
            distributed_centers = st_utils.get_distributed_voxel(centers_idx_current, centers_frame_padded, grid_shapes, centers_lookup, self.shifts)
        c, h, w, d = self.empty_latent_code.shape

        distributed_latents = distributed_latents.reshape(-1, c, 3*h, 3*w, 3*d)
        
        #retrieve prev latent
        n_x_p, n_y_p, n_z_p = vol_bounds_padded_prev['axis_n_crop'] #padded
        prev_centers_frame = prev_centers_frame_padded.reshape(n_x_p, n_y_p, n_z_p, 3)[1:-1, 1:-1, 1:-1, :].reshape(-1, 3)
        centers_lookup_prev = torch.tensor([0, (n_x_p * n_y_p * n_z_p)]).repeat(len(prev_centers_frame), 1).to(self.device)
        centers_lookup_prev_padded = torch.tensor([0, (n_x_p * n_y_p * n_z_p)]).repeat(len(prev_centers_frame_padded), 1).to(self.device) - (n_x_p * n_y_p * n_z_p) #will be positive after we add start_idx, allows to retrieve data in the previous frame
        grid_shapes_prev = torch.tensor([n_x_p, n_y_p, n_z_p]).repeat(len(prev_centers_frame), 1).to(self.device)
        grid_shapes_prev_padded = torch.tensor([n_x_p, n_y_p, n_z_p]).repeat(len(prev_centers_frame_padded), 1).to(self.device)
        
        centers_idx_prev = st_utils.centers_to_grid_indexes(prev_centers_frame, vol_bounds_padded_prev['lb'], self.query_crop_size).int().reshape(-1, 3)
        centers_idx_prev_padded = st_utils.centers_to_grid_indexes(prev_centers_frame_padded, vol_bounds_padded_prev['lb'], self.query_crop_size).int().reshape(-1, 3)
        
        distributed_latents_prev = st_utils.get_distributed_voxel(centers_idx_prev, prev_merged_latents_padded, grid_shapes_prev, centers_lookup_prev, self.shifts)
        distributed_latents_prev = distributed_latents_prev.reshape(-1, c, 3*h, 3*w, 3*d)

        if self.debug:
            distributed_inputs_prev = st_utils.get_distributed_voxel(centers_idx_prev, prev_undist_inputs_padded, grid_shapes_prev, centers_lookup_prev, self.shifts)
            distributed_centers_prev = st_utils.get_distributed_voxel(centers_idx_prev, prev_centers_frame_padded, grid_shapes_prev, centers_lookup_prev, self.shifts)
            return distributed_inputs, distributed_inputs_prev, distributed_centers, distributed_centers_prev, inputs_frame, prev_undist_inputs_padded

        #centers from prev frame present in this frame
        mask_centers_prev_in_current = st_utils.compute_mask_occupied(prev_centers_frame, centers_frame)
        mask_centers_current_in_prev = st_utils.compute_mask_occupied(centers_frame, prev_centers_frame)
        
        # mask_centers_current_in_prev_padded = st_utils.compute_mask_occupied(centers_frame_padded, prev_centers_frame_padded)
        # mask_centers_prev_in_current_padded = st_utils.compute_mask_occupied(prev_centers_frame_padded, centers_frame_padded)
        mask_centers_prev_in_current_reshaped = mask_centers_prev_in_current.reshape(n_x_p-2, n_y_p-2, n_z_p-2)
        mask_centers_current_in_prev_reshaped = mask_centers_current_in_prev.reshape(n_x-2, n_y-2, n_z-2)
        
        mask_centers_current_in_prev_padded = torch.nn.functional.pad(mask_centers_current_in_prev_reshaped, (1, 1, 1, 1, 1, 1), value = False).reshape(-1)
        mask_centers_prev_in_current_padded = torch.nn.functional.pad(mask_centers_prev_in_current_reshaped, (1, 1, 1, 1, 1, 1), value = False).reshape(-1)
        
        distributed_latents_prev_temp = distributed_latents.clone() # allows to merge with itself where possible 
        # print(f'distributed_latents_prev_temp: {distributed_latents_prev_temp.shape}, mask_centers_current_in_prev: {mask_centers_current_in_prev.shape}')
        # print(f'mask_centers_prev_in_current shape: {mask_centers_prev_in_current.shape}, sum: {mask_centers_prev_in_current.sum()}')
        distributed_latents_prev_temp[mask_centers_current_in_prev] = distributed_latents_prev[mask_centers_prev_in_current]
        

        # print(f'Process frame, overlapping voxels : {mask_centers_current_in_prev.sum()}')
        stacked_frame = torch.cat((distributed_latents, distributed_latents_prev_temp), dim = 1)
        
        merged_latents = self.merge_latent_map(stacked_frame)
        merged_latents_padded = self.init_empty_latent_grid(vol_bounds_frame_padded)

        merged_latents_padded[1:-1, 1:-1, 1:-1] = merged_latents.reshape(n_x-2, n_y-2, n_z-2, *self.empty_latent_code.shape)
        merged_latents_padded = merged_latents_padded.reshape(-1, *self.empty_latent_code.shape)
        
        #Prepare data for return
        # Idea here is the following: we use the current data as default (would then merge with itself) 
        # Overwrite the indices where we have overlap with the previous frame data 
        # centers_idx_prev_padded allows to search in the correct frame (previous where overlap, current by default)
        
        centers_idx_current = centers_idx_current_padded
        centers_idx_prev_return = centers_idx_current.clone()
        centers_idx_prev_return[mask_centers_current_in_prev_padded] = centers_idx_prev_padded[mask_centers_prev_in_current_padded]
        
        grid_shapes_current = grid_shapes_padded
        grid_shapes_prev_return = grid_shapes_current.clone()
        grid_shapes_prev_return[mask_centers_current_in_prev_padded] = grid_shapes_prev_padded[mask_centers_prev_in_current_padded]
        
        centers_lookup_frame_current = centers_lookup_padded
        centers_lookup_frame_prev = centers_lookup_frame_current.clone()
        centers_lookup_frame_prev[mask_centers_current_in_prev_padded] = centers_lookup_prev_padded[mask_centers_prev_in_current_padded]
        
        mask_centers = mask_centers_current_in_prev_padded #this mask indicates where overlap between current and previous happens

        return_tup = [centers_idx_current, centers_idx_prev_return, grid_shapes_current, grid_shapes_prev_return, centers_lookup_frame_current, centers_lookup_frame_prev, mask_centers, merged_latents_padded, vol_bounds_frame_padded]
        
        if self.debug:
            return_tup.append(distributed_inputs)
            return_tup.append(distributed_inputs_prev)
            return_tup.append(mask_centers_prev_in_current)
            return_tup.append(mask_centers_current_in_prev)

        return return_tup
    
    def generate_empty_inputs(self, start_idx):
        n_points = 10000
        points = torch.stack([torch.rand(n_points) * 19.99 + 0.01, torch.rand(n_points) * 2 - 1, torch.rand(n_points) * 19.99 + 0.01], dim=1)
        occ = torch.zeros(n_points, 1)

        inputs = torch.cat([points, occ], dim = -1).to(self.device)

        inputs_frame_occupied, centers_frame_occupied, vol_bound = self.get_distributed_inputs(inputs, n_max = self.n_max_points_input, return_empty = True)
        query_frame_occupied, centers_query_occupied, vol_bound_query = self.get_distributed_inputs(inputs, n_max = self.n_max_points_query, return_empty = True)

        tup = self.process_frame_cold_start(centers_frame_occupied, inputs_frame_occupied)
        
        centers_idx_current_padded, centers_idx_current_padded, grid_shapes_padded, grid_shapes_padded, centers_lookup_padded, centers_lookup_padded, mask_centers, merged_latents_padded, vol_bounds_frame_padded = tup

        n_x, n_y, n_z = vol_bound['axis_n_crop']
        mask_keep_int = torch.ones(n_x-2, n_y-2, n_z-2, dtype = torch.bool).to(self.device)
        mask_keep = torch.nn.functional.pad(mask_keep_int, (1, 1, 1, 1, 1, 1), value=False)
        mask_keep = mask_keep.reshape(-1)
        
        mask_centers_empty = torch.ones_like(mask_keep)
        return_tup = (
            centers_idx_current_padded[mask_keep], 
            query_frame_occupied[mask_keep], 
            grid_shapes_padded[mask_keep], 
            centers_lookup_padded[mask_keep] + start_idx, 
            mask_centers_empty, 
            merged_latents_padded, 
            inputs_frame_occupied, 
            centers_frame_occupied)

        # print(f'Generated empty inputs, {torch.sum(mask_keep)} voxels')
        return return_tup
    
    def train_batch(self, centers_idx_full_current, centers_idx_full_prev, query_points_full, grid_shapes_full_current, grid_shapes_full_prev, centers_lookup_full_current, centers_lookup_full_prev, mask_centers_full, latents_full, inputs_full, centers_coord_full, training = True):
        if training:
            self.model.train()
            self.model_merge.train()
        else:
            self.model.eval()
            self.model_merge.eval()
                
        n_batch_div = len(centers_idx_full_current) // self.n_batch
        idx_start = 0
        iter_batch = 0
        loss_full = 0
        
        c, h, w, d = self.empty_latent_code.shape

        for i in range(n_batch_div):
            torch.cuda.empty_cache()
           
            self.optimizer_backbone.zero_grad()
            self.optimizer_merge.zero_grad()

            self.empty_latent_code = self.get_empty_latent_representation()

            idx_end = idx_start + self.n_batch
    
            # print(f'Process batch {i} out of {n_batch_div}, starting from {idx_start} to {idx_end}, tot length: {len(centers_idx_full)}')           
            centers_idx_batch_current = centers_idx_full_current[idx_start:idx_end].to(self.device) # centers indices corresponding to the voxel in its frame
            grid_shapes_batch_current = grid_shapes_full_current[idx_start:idx_end].to(self.device) # grid shapes corresponding to the voxel in its frame
            centers_lookup_batch_current = centers_lookup_full_current[idx_start:idx_end].to(self.device) # centers lookup corresponding to the voxel in its frame
            query_points_batch = query_points_full[idx_start:idx_end].to(self.device) # query points corresponding to the voxel          
            mask = mask_centers_full[idx_start:idx_end].to(self.device) # mask indicates if there is overlap between the current and previous frame, if True: use merged latent and encoded latent, if False: use only encoded latent (double stack)
            
            centers_idx_batch_prev = centers_idx_full_prev[idx_start:idx_end].to(self.device) # centers indices corresponding to the voxel to concat with current: will be itself of of the previous frame
            grid_shapes_batch_prev = grid_shapes_full_prev[idx_start:idx_end].to(self.device) # grid shapes corresponding to the voxel to concat with current: will be itself of of the previous frame
            centers_lookup_batch_prev = centers_lookup_full_prev[idx_start:idx_end].to(self.device) # centers lookup corresponding to the voxel to concat with current: will be itself of of the previous frame

            if len(centers_idx_batch_current) == 0:
                continue
            try:
                latents_prev = st_utils.get_distributed_voxel(centers_idx_batch_prev[mask], latents_full, grid_shapes_batch_prev[mask], centers_lookup_batch_prev[mask], self.shifts).to(self.device)
            except ValueError as e:
                print('*' * 50)
                print(e)
                print(f'mask sum {torch.sum(mask)}')
                print('*' * 50)
                continue
            
            latents_prev = latents_prev.reshape(-1, c, 3*h, 3*w, 3*d)
            centers_distributed_batch_prev = st_utils.get_distributed_voxel(centers_idx_batch_prev[mask], centers_coord_full, grid_shapes_batch_prev[mask], centers_lookup_batch_prev[mask], self.shifts).to(self.device)
            centers_coord_batch_prev = centers_distributed_batch_prev.reshape(-1, 3, 3, 3, 3)[:, 1, 1, 1, :].reshape(-1, 3)
            inputs_batch_prev = st_utils.get_distributed_voxel(centers_idx_batch_prev[mask], inputs_full, grid_shapes_batch_prev[mask], centers_lookup_batch_prev[mask], self.shifts).to(self.device)
            
            # compute the latents of the current frame
            inputs_batch_current = st_utils.get_distributed_voxel(centers_idx_batch_current, inputs_full, grid_shapes_batch_current, centers_lookup_batch_current, self.shifts).to(self.device)
            centers_distributed_batch_current = st_utils.get_distributed_voxel(centers_idx_batch_current, centers_coord_full, grid_shapes_batch_current, centers_lookup_batch_current, self.shifts).to(self.device)

            centers_coord_batch_current = centers_distributed_batch_current.reshape(-1, 3, 3, 3, 3)[:, 1, 1, 1, :].reshape(-1, 3)
        
            latents_current = self.encode_occupied_inputs_flat(inputs_batch_current, centers_distributed_batch_current)
            latents_current = latents_current.reshape(-1, c, 3*h, 3*w, 3*d)
            
            # stack the latents 
            # mask_centers_current_in_prev = st_utils.compute_mask_occupied(centers_coord_batch_current, centers_coord_batch_prev)
            # mask_centers_prev_in_current = st_utils.compute_mask_occupied(centers_coord_batch_prev, centers_coord_batch_current)
            
            # sum_centers_current_in_prev = torch.sum(mask_centers_current_in_prev)
            # sum_centers_prev_in_current = torch.sum(mask_centers_prev_in_current)
            
            # if sum_centers_current_in_prev != sum_centers_prev_in_current:
            #     print(f'WARNING: sum_centers_current_in_prev != sum_centers_prev_in_current: {sum_centers_current_in_prev} vs {sum_centers_prev_in_current}')
            
            latents_prev_temp = latents_current.clone() #Use current latents as default
            latents_prev_temp[mask] = latents_prev #overwrite current latents with precomputed merged latents where there is overlap
            
            inputs_batch_prev_temp = inputs_batch_current.clone()
            inputs_batch_prev_temp[mask] = inputs_batch_prev

            stacked_latents = torch.cat((latents_current, latents_prev_temp), dim = 1)
        
            merged_latents = self.merge_latent_map(stacked_latents)
            
            # compute logits : latents
            p_stacked = query_points_batch.reshape(-1, self.n_max_points_query, 4)
            centers = centers_coord_batch_current.reshape(-1, 3)
            occ = query_points_batch[..., 3]

            # return p_stacked, occ, merged_latents, centers
            logits_sampled = self.get_logits(p_stacked, merged_latents, centers)
            
            # compare logits with query 
            inputs_batch_current_int = inputs_batch_current.reshape(self.n_batch, 3, 3, 3, self.n_max_points_input, 4)[:, 1, 1, 1, ...].reshape(-1, self.n_max_points_input, 4)
            # weights = st_utils.compute_gaussian_weights(p_stacked, inputs_batch_current_int, sigma = self.sigma)
            inputs_batch_prev_int = inputs_batch_prev_temp.reshape(self.n_batch, 3, 3, 3, self.n_max_points_input, 4)[:, 1, 1, 1, ...].reshape(-1, self.n_max_points_input, 4)
            # max_weights = torch.max(weights)
            # min_weights = torch.min(weights)
            # pos_weight_value = st_utils.compute_pos_weight(occ)
            # pos_weight = torch.full_like(occ, pos_weight_value)
            criterion = nn.BCEWithLogitsLoss(reduction='none')

            # loss_batch_unweighted = F.binary_cross_entropy_with_logits(logits_sampled, occ, reduction='none')
            loss_batch_unweighted = criterion(logits_sampled, occ)
            # dynamic_weights = loss_batch_unweighted**2
            loss_batch = loss_batch_unweighted #* dynamic_weights

            # return logits_sampled, p_stacked, centers_coord_batch_current, centers_coord_batch_prev, centers_distributed_batch_current, centers_distributed_batch_prev, loss_batch, inputs_batch_current, inputs_batch_prev_temp
            # vis_utils.visualize_logits_voxel(logits_sampled, p_stacked, centers_coord_batch_current, centers_coord_batch_prev, centers_distributed_batch_current, centers_distributed_batch_prev, loss_batch, inputs_batch_current, inputs_batch_prev_temp, location = self.location, threshold = 0.02)
            # vis_utils.visualize_logits(logits_sampled, p_stacked, centers, loss_batch, self.location, weights = inputs_batch_current, inputs_distributed = inputs_batch_current, force_viz = False)
            # vis_utils.visualize_logits(logits_sampled, p_stacked, centers, loss_batch, self.location, weights = inputs_batch_current_int, inputs_distributed = inputs_batch_current_int, force_viz = False)
            
            loss_batch = loss_batch.mean()

            # print(f'loss_batch: {loss_batch}')
            if training:    
                loss_batch.backward()
            
                st_utils.print_gradient_norms(self.iteration, self.model_merge, print_every = 100)  # Print gradient norms
                st_utils.print_gradient_norms(self.iteration, self.model, print_every = 100)  # Print gradient norms
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.model_merge.parameters(), max_norm=2.0)
                
                self.optimizer_backbone.step()
                self.optimizer_merge.step()
                
                self.iteration += 1
                
                if self.log_experiment: self.experiment.log_metric('loss', loss_batch.item(), step = self.iteration)
            # print(f'iteration: {self.iteration}, loss: {loss_batch:.2f}')
            iter_batch += 1

            loss_full += loss_batch
            
            idx_start = idx_end
            
            centers_idx_batch_current = centers_idx_batch_current.to(torch.device('cpu')) # centers indices corresponding to the voxel in its frame
            grid_shapes_batch_current = grid_shapes_batch_current.to(torch.device('cpu')) # grid shapes corresponding to the voxel in its frame
            centers_lookup_batch_current = centers_lookup_batch_current.to(torch.device('cpu')) # centers lookup corresponding to the voxel in its frame
            query_points_batch = query_points_batch.to(torch.device('cpu')) # query points corresponding to the voxel
            mask = mask.to(torch.device('cpu'))
            
            centers_idx_batch_prev = centers_idx_batch_prev.to(torch.device('cpu')) # centers indices corresponding to the voxel in its frame
            grid_shapes_batch_prev = grid_shapes_batch_prev.to(torch.device('cpu')) # grid shapes corresponding to the voxel in its frame
            centers_lookup_batch_prev = centers_lookup_batch_prev.to(torch.device('cpu')) # centers lookup corresponding to the voxel in its frame

            torch.cuda.empty_cache()
        
        if iter_batch == 0: return loss_full
        else: return loss_full / iter_batch
            
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
        n_max = self.n_voxels_max

        n_batch = int(np.ceil(n_crops_total / n_max))

        logits_stacked = None  # Initialize logits directly

        torch.cuda.empty_cache()
        
        for i in range(n_batch):
            start = i * n_max
            end = min((i + 1) * n_max, n_crops_total)

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
    
    def encode_distributed(self, inputs_raw, centers_raw):
        
        if len(inputs_raw.shape) != 3 or len(centers_raw.shape) != 2: #n_batch, n_points, 4
            inputs = inputs_raw.reshape(-1, self.n_max_points_input, 4)
            centers = centers_raw.reshape(-1, 3)
        else:
            inputs = inputs_raw
            centers = centers_raw
        
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
        latent_map = latent_map.reshape(n_crop, *latent_map_shape[1:])
            
        return latent_map

    def get_distributed_inputs(self, distributed_inputs_raw, n_max = 2048, occ_perc = 1.0, return_empty = False, isquery = False, padding = False):
        # Clone the input tensor
        distributed_inputs = distributed_inputs_raw.clone()
        
        vol_bound, centers = st_utils.compute_vol_bound(distributed_inputs[:, :3].reshape(-1, 3), self.query_crop_size, self.input_crop_size, padding = padding)

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
        
        voxels_occupied = distributed_inputs_short[..., 3].sum(dim=-1).int() > self.points_threshold #TODO move this to config

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
    
            
        return inputs_frame_occupied, centers_frame_occupied, vol_bound