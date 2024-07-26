import os
import torch
import math
import numpy as np
import time
import yaml
import torch.nn as nn
from torch.nn import functional as F
import random
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
        
    def precompute_sequence(self, full_batch):
        self.model.eval()
        self.model_merge.eval()
                
        with torch.no_grad():
            #batch is [12, 20, 40960, 4] for points (gt points)
            #batch is [12, 20, 4096, 4] for inputs 
            p_in_full, p_query_full = st_utils.get_inputs_from_scene(full_batch, self.device)

            inputs_scene_distributed_list = []
            centers_scene_list = []
            input_vol_scene_list = []
            n_voxels = torch.zeros(p_in_full.shape[0], p_in_full.shape[1], dtype = torch.int32).to(device = self.device)
                        
            for scene_idx in range(p_in_full.shape[0]):
                inputs_frame_distributed_list = []
                centers_frame_list = []
                input_vol_frame_list = []
                for frame_idx in range(p_in_full.shape[1]):
                    inputs_frame = p_in_full[scene_idx, frame_idx]
                    inputs_frame_distributed, centers_frame, vol_bounds_frame = self.get_distributed_inputs(inputs_frame, self.n_max_points_input, self.occ_per_query, return_empty = True, isquery = False)
                    
                    inputs_frame_distributed_list.append(inputs_frame_distributed)
                    centers_frame_list.append(centers_frame)
                    input_vol_frame_list.append(vol_bounds_frame['input_vol'])
                    n_voxels[scene_idx, frame_idx] = inputs_frame_distributed.shape[0]
                
                inputs_scene_distributed_list.append(inputs_frame_distributed_list)
                centers_scene_list.append(centers_frame_list)
                input_vol_scene_list.append(input_vol_frame_list)
            
            
            centers_flattened = [centers for sublist in centers_scene_list for centers in sublist]
            centers_full = torch.cat(centers_flattened, dim = 0)
            centers_unique, indices_centers_unique = torch.unique(centers_full, dim = 0, return_inverse = True)
            
            c, h, w, d = self.empty_latent_code.shape
            self.precomputed_stacked_latents = torch.empty(centers_unique.shape[0], c, 3 * h, 3 * w, 3 * d).to(torch.device('cpu'))
            
            n_voxels_total = int(n_voxels.sum().item())
            inputs_distributed_neighboured = torch.zeros(n_voxels_total, 27, self.n_max_points_input, 4).to(self.device)
            centers_neighboured = torch.zeros(n_voxels_total, 27, 3).to(self.device)
            input_vol_neighboured = torch.zeros(n_voxels_total, 27, 2, 3).to(self.device)
            
            idx_start = 0
            for scene_idx in range(n_voxels.shape[0]):
                for frame_idx in range(n_voxels.shape[1]):
                    centers_frame = centers_scene_list[scene_idx][frame_idx]
                    inputs_frame = inputs_scene_distributed_list[scene_idx][frame_idx]
                    input_vol_frame = input_vol_scene_list[scene_idx][frame_idx]

                    inputs_temp, centers_temp, input_vol_temp = self.fill_inputs_centers(inputs_frame, centers_frame, input_vol_frame)

                    idx_end = idx_start + n_voxels[scene_idx, frame_idx]
                                        
                    inputs_distributed_neighboured[idx_start:idx_end] = inputs_temp
                    centers_neighboured[idx_start:idx_end] = centers_temp
                    input_vol_neighboured[idx_start:idx_end] = input_vol_temp
                    
                    return inputs_temp, centers_temp, input_vol_temp, frame_idx, centers_unique
                    self.precompute_latents_frame(inputs_temp, centers_temp, input_vol_temp, frame_idx)                    
                    
                    idx_start = idx_end
                    
    def precompute_latents_frame(self, inputs_temp, centers_temp, input_vol_temp, frame_idx):
        if frame_idx == 0:
            self.fuse_cold_start(inputs_temp, centers_temp, input_vol_temp)
        else:
            self.fuse(inputs_temp, centers_temp, input_vol_temp)
            
    def fuse_cold_start(self, inputs_temp, centers_temp, input_vol_temp):
        inputs_temp_reshaped = inputs_temp.reshape(-1, 3, 3, 3, self.n_max_points_input, 4)
        inputs_temp_centers = inputs_temp_reshaped[:, 1,1,1, :, 3]
        print(f'inputs_temp_centers.shape = {inputs_temp_centers.shape}')        
        occupied_mask = inputs_temp_centers.sum(dim = -1) > 10
        
        # print(f'occupied_mask = {occupied_mask}, sum = {occupied_mask.sum(), occupied_mask.shape}')

        occupied_centers = centers_temp[occupied_mask]
        occupied_inputs = inputs_temp[occupied_mask]
        occupied_input_vol = input_vol_temp[occupied_mask]
        print(f'occupied_inputs.shape = {occupied_inputs.shape}')
        print(f'occupied_centers.shape = {occupied_centers.shape}')
        
        occupied_inputs_interior = occupied_inputs.reshape(-1, 3, 3, 3, self.n_max_points_input, 4)[:,1,1,1,:,:]
        occupied_centers_interior = occupied_centers.reshape(-1, 3,3,3,3)[:,1,1,1,:]
        
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # points = occupied_inputs_interior[..., :3].detach().cpu().numpy()
        # occ = occupied_inputs_interior[..., 3].detach().cpu().numpy()
        # pcd.points = o3d.utility.Vector3dVector(points[occ == 1].reshape(-1, 3))

        # o3d.visualization.draw_geometries([pcd])
        
        latents = self.encode_distributed(occupied_inputs_interior, occupied_centers_interior)
        latent_map = torch.zeros(centers_temp.shape[0], *self.empty_latent_code.shape).to(self.device)
        latent_map[occupied_mask] = latents
        
        expanded_shape = (len(occupied_mask),) + self.empty_latent_code.shape
        grid_latents_frame = self.empty_latent_code.unsqueeze(0)
        grid_latents_frame = grid_latents_frame.expand(expanded_shape)
        
        grid_latents_frame[occupied_mask] = latents
        
        #shift the centers_occupied_shifted
        lb = torch.min(centers_temp, dim = 0)[0] - self.query_crop_size/2
        centers_occupied_shifted = st_utils.centers_to_grid_indexes(occupied_centers.reshape(-1, 3).clone(), lb, self.query_crop_size).reshape(*occupied_centers.shape)

        latent_temp = grid_latents[centers_occupied_shifted[..., 0], centers_occupied_shifted[..., 1], centers_occupied_shifted[..., 2]]
        c, h, w, d = self.voxel_grid.latent_shape

        latent_map_stacked = latent_temp.reshape(len(centers_occupied_shifted), c, 3 * h, 3 * w, 3 * d)

        
    def fill_inputs_centers(self, inputs_distributed_neighboured, centers_frame, input_vol_frame):
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
        
        input_vol_padded = torch.zeros(n_x_p, n_y_p, n_z_p, input_vol_frame.shape[1], input_vol_frame.shape[2]).to(self.device)
        input_vol_padded[1:-1, 1:-1, 1:-1] = input_vol_frame.reshape(n_x, n_y, n_z, input_vol_frame.shape[1], input_vol_frame.shape[2])
        
        inputs_temp = zeroed_padded_inputs[centers_grid_shifted[..., 0], centers_grid_shifted[..., 1], centers_grid_shifted[..., 2]]
        centers_temp = centers_grid_padded[centers_grid_shifted[..., 0], centers_grid_shifted[..., 1], centers_grid_shifted[..., 2]]
        input_vol_temp = input_vol_padded[centers_grid_shifted[..., 0], centers_grid_shifted[..., 1], centers_grid_shifted[..., 2]]

        return inputs_temp, centers_temp, input_vol_temp
            
    def process_sequence(self, full_batch, is_training, return_flat = True):
        
        n_scenes = len(full_batch)
        n_sequences = full_batch[0]['points'].shape[1]
        
        keys = [f'{idx_scene}_{idx_sequence}' for idx_scene in range(n_scenes) for idx_sequence in range(n_sequences)]
        #shuffle the keys
        random.shuffle(keys)
        
        total_loss = 0
        results = []
           
        for key in keys:
            scene = int(key.split('_')[0])
            idx_sequence = int(key.split('_')[1])
            self.precomputed_storage.scene_idx = scene
            self.precomputed_storage.sequence_idx = idx_sequence 
            
            p_in, p_query = st_utils.get_inputs_from_scene(full_batch[scene], self.device)

            inputs_frame = p_in[idx_sequence]

            if idx_sequence == 0:
                self.voxel_grid.reset()
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.fuse_cold_start(inputs_frame, encode_empty = is_training)
            else:
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.fuse_inputs(inputs_frame, encode_empty = is_training)
            
            p_query_distributed, centers_query = self.get_distributed_inputs(p_query[idx_sequence], self.n_max_points_query, self.occ_per_query, isquery = True)
            p_stacked, latents, centers, occ, mask_frame = self.prepare_data_logits(latent_map_stacked_merged, centers_frame_occupied, p_query_distributed, centers_query)

            logits_sampled = self.get_logits(p_stacked, latents, centers)
            loss_unweighted = F.binary_cross_entropy_with_logits(logits_sampled, occ, reduction='none')
            
            weights = st_utils.compute_gaussian_weights(p_stacked, inputs_frame_distributed[mask_frame], sigma = self.sigma)

            loss_weighted = loss_unweighted 
            
            loss = loss_weighted.sum(dim=-1).mean()
            
            if self.log_experiment: self.experiment.log_metric('loss', loss.item(), step = self.iteration)
            
            if is_training:         
                vis_utils.visualize_logits(logits_sampled, p_stacked, self.location, weights = loss_weighted, inputs_distributed = inputs_frame_distributed[mask_frame], force_viz = False)
                loss.backward()
                
                st_utils.print_gradient_norms(self.iteration, self.model_merge, print_every = 100)  # Print gradient norms
                st_utils.print_gradient_norms(self.iteration, self.model, print_every = 100)  # Print gradient norms
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(self.model_merge.parameters(), max_norm=2.0)
               
                self.voxel_grid.detach_latents()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.iteration += 1
            else:
                results.append([p_stacked.cpu(), latents.cpu(), inputs_frame.cpu(), logits_sampled.cpu(), loss.item()])
            total_loss += loss.item()

        if is_training:
            if self.log_experiment: self.experiment.log_metric('total_loss', total_loss / n_sequences / n_scenes, step = self.iteration)
            return total_loss / n_sequences
        else:
            results.append(total_loss / n_sequences / n_scenes)
            return results
        
    def train_sequence_window(self, full_data_batch):
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.train()
        self.model_merge.train()
        self.optimizer.zero_grad()
               
        return self.process_sequence(full_data_batch, is_training=True, return_flat = self.return_flat)
    
    def validate_sequence(self, full_data_batch):
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.eval()
        self.model_merge.eval()
        
        with torch.no_grad():
            return self.process_sequence(full_data_batch, is_training=False, return_flat = True)

    # def fuse_cold_start(self, inputs_frame, encode_empty = True):
    #     latents_frame_occupied, inputs_frame_distributed_occupied, centers_frame_occupied = self.encode_inputs_frame(inputs_frame)

    #     for idx, (center, encoded_latent) in enumerate(zip(centers_frame_occupied, latents_frame_occupied)):
    #         if encoded_latent.shape != self.empty_latent_code.shape:
    #             print(f'encoded_latent.shape: {encoded_latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
    #         self.voxel_grid.add_voxel(center, encoded_latent, overwrite = False)

    #     # for center, input in zip(centers_frame_occupied, inputs_frame_distributed_occupied): #TODO Remove, was used for debugging
    #     #     self.voxel_grid.add_pcd(center, input)
        
    #     latent_map_stacked = self.stack_latents_cold_start(centers_frame_occupied, encode_empty = encode_empty)

    #     latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)

    #     for center, encoded_latent in zip(centers_frame_occupied, latent_map_stacked_merged):
    #         if encoded_latent.shape != self.empty_latent_code.shape:
    #             print(f'encoded_latent.shape: {encoded_latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
    #         self.voxel_grid.add_voxel(center, encoded_latent, overwrite = True)
            
    #     return latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed_occupied
    
    def fuse_inputs(self, inputs_frame, encode_empty = True, is_precomputing = False):
        latents_frame_occupied, inputs_frame_distributed_occupied, centers_frame_occupied = self.encode_inputs_frame(inputs_frame)
                
        voxel_grid_temp = VoxelGrid()

        for idx, (center, encoded_latent) in enumerate(zip(centers_frame_occupied, latents_frame_occupied)):
            if encoded_latent.shape != self.empty_latent_code.shape:
                print(f'encoded_latent.shape: {encoded_latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
            voxel_grid_temp.add_voxel(center, encoded_latent)
        # for center, input in zip(centers_frame_occupied, inputs_frame_distributed_occupied): #TODO Remove, was used for debugging
        #     voxel_grid_temp.add_pcd(center, input)
            
        #stack the latents
        latent_map_stacked = self.stack_latents(centers_frame_occupied, voxel_grid_temp, encode_empty = encode_empty, is_precomputing = is_precomputing)

        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)

        for i in range(len(centers_frame_occupied)):
            self.voxel_grid.add_voxel(centers_frame_occupied[i], latent_map_stacked_merged[i], overwrite = True)

        # for i in range(len(centers_frame_occupied)): #TODO Remove, was used for debugging
        #     self.voxel_grid.add_pcd(centers_frame_occupied[i], inputs_frame_distributed_occupied[i])
            
        return latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed_occupied

    def prepare_data_logits(self, latent_map, centers_frame_occupied, p_query_distributed, centers_query):
        p_query_dict, mask_query, mask_frame = self.select_query_points(p_query_distributed, centers_query, centers_frame_occupied)
        latents = latent_map[mask_frame]
        centers = centers_frame_occupied[mask_frame]
        
        p_stacked = torch.zeros(latents.shape[0], self.n_max_points_query, 4).to(self.device)
        
        for idx, center in enumerate(centers):
            h = self.voxel_grid.compute_hash(center)
            if h in p_query_dict:
                p_stacked[idx] = p_query_dict[h]
        
        occ = p_stacked[..., 3]
        
        return p_stacked, latents, centers, occ, mask_frame
    
    def get_logits(self, p_stacked, latents, centers):
    
        n_crops_total = p_stacked.shape[0]
                
        vol_bound = st_utils.get_grid_from_centers(centers, self.input_crop_size)
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
    
    def select_query_points(self, p_query_distributed: torch.Tensor, centers_query: torch.Tensor, centers_frame_occupied: torch.Tensor):        
        error = 0.1

        mask_query = (torch.norm(centers_query.unsqueeze(1) - centers_frame_occupied, dim=2) <= error).any(dim=1)
        mask_frame = (torch.norm(centers_frame_occupied.unsqueeze(1) - centers_query, dim=2) <= error).any(dim=1)
        
        p_query_dict = {}
        
        for p_query, center in zip(p_query_distributed[mask_query], centers_query[mask_query]):
            hash = self.voxel_grid.compute_hash(center)
            p_query_dict[hash] = p_query
        
        return p_query_dict, mask_query, mask_frame
        
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
    
    def encode_inputs_frame(self, inputs_frame, encode_empty = True):
        
        inputs_frame_distributed_occupied, centers_frame_occupied = self.get_distributed_inputs(inputs_frame, self.n_max_points_input)

        latents_frame_occupied = self.encode_distributed(inputs_frame_distributed_occupied, centers_frame_occupied)
                    
        return latents_frame_occupied, inputs_frame_distributed_occupied, centers_frame_occupied

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
            return inputs_frame_occupied, centers_frame_occupied
        else:
            if return_empty:
                inputs_frame_occupied = distributed_inputs_short
                centers_frame_occupied = centers
                
                return inputs_frame_occupied, centers_frame_occupied, vol_bound
            else:
                inputs_frame_occupied = distributed_inputs_short[voxels_occupied]
                centers_frame_occupied = centers[voxels_occupied]
                
                return inputs_frame_occupied, centers_frame_occupied
    
    def populate_grid_latent_cold_start(self, centers_frame_occupied, voxel_grid_occ, encode_empty = True): 
        vol_bounds, centers_grid = st_utils.compute_vol_bound(centers_frame_occupied, self.query_crop_size, self.input_crop_size, padding = True)
        n_x, n_y, n_z = vol_bounds['axis_n_crop']

        centers_occupied_full = torch.stack(list(voxel_grid_occ.centers_table.values()))

        mask_centers_grid_occupied = st_utils.compute_mask_occupied(centers_grid, centers_occupied_full)
        
        if encode_empty == False:
            empty_latent = self.empty_latent_code
        else:
            empty_latent = self.get_empty_latent_representation()
            
        expanded_shape = (len(centers_grid),) + self.empty_latent_code.shape
        grid_latents = empty_latent.unsqueeze(0)
        grid_latents = grid_latents.expand(expanded_shape)
        grid_latents = grid_latents.reshape(n_x, n_y, n_z, *self.empty_latent_code.shape).to(self.device).clone()
        
        #shift back the centers 
        
        lb = vol_bounds['lb']
        centers_grid_shifted = st_utils.centers_to_grid_indexes(centers_grid, lb, self.query_crop_size)
                   
        for i in range(centers_grid.shape[0]):
            occupancy = mask_centers_grid_occupied[i]
            
            if occupancy:
                index_x = centers_grid_shifted[i, 0].int().item()
                index_y = centers_grid_shifted[i, 1].int().item()
                index_z = centers_grid_shifted[i, 2].int().item()

                latent = voxel_grid_occ.get_latent(centers_grid[i])
                grid_latents[index_x, index_y, index_z] = latent
                
        return grid_latents, vol_bounds
    
    def populate_grid_latent(self, centers_frame_occupied, voxel_grid_occ_frame, voxel_grid_occ_existing, encode_empty = True, is_precomputing = False): 
        # centers_frame_occupied + centers_frame_unocc = centers_frame_grid
        vol_bounds, centers_grid = st_utils.compute_vol_bound(centers_frame_occupied, self.query_crop_size, self.input_crop_size, padding = True)
        
        n_x, n_y, n_z = vol_bounds['axis_n_crop']

        if encode_empty == False:
            empty_latent = self.empty_latent_code
        else:
            empty_latent = self.get_empty_latent_representation()

        expanded_shape = (len(centers_grid),) + self.empty_latent_code.shape
        grid_latents_frame = empty_latent.unsqueeze(0)
        grid_latents_frame = grid_latents_frame.expand(expanded_shape)
        grid_latents_frame = grid_latents_frame.reshape(n_x, n_y, n_z, *self.empty_latent_code.shape).to(self.device).clone()
        
        grid_latents_existing = grid_latents_frame.clone()
        
        lb = vol_bounds['lb']
        centers_grid_shifted = st_utils.centers_to_grid_indexes(centers_grid, lb, self.query_crop_size)
        
        if is_precomputing:
            centers_occupied_existing_full = torch.stack(list(voxel_grid_occ_existing.centers_table.values()))
        else:
            latents_precomputed, indexes_precomputed, centers_occupied_existing_full  = self.precomputed_storage.load_idx(self.precomputed_storage.scene_idx, self.precomputed_storage.sequence_idx)
            grid_latents_existing[indexes_precomputed[:, 0], indexes_precomputed[:, 1], indexes_precomputed[:, 2], :] = latents_precomputed
            
            # self.precomputed_storage.visualize_pcd() #TODO Remove used for debug
            
        centers_occupied_frame_full = torch.stack(list(voxel_grid_occ_frame.centers_table.values()))

        mask_centers_occupied_frame_occupied = st_utils.compute_mask_occupied(centers_grid, centers_occupied_frame_full)
        mask_centers_occupied_existing_occupied = st_utils.compute_mask_occupied(centers_grid, centers_occupied_existing_full)
        
        if is_precomputing:
            precomputed_latents = torch.empty(sum(mask_centers_occupied_existing_occupied), *self.empty_latent_code.shape).to(self.device)
            precomputed_indices = torch.empty(sum(mask_centers_occupied_existing_occupied), 3, dtype = torch.int).to(self.device)
            precomputed_centers = torch.empty(sum(mask_centers_occupied_existing_occupied), 3).to(self.device) #TODO Remove, used for debug
            
        idx_indexes = 0
        for i in range(centers_grid_shifted.shape[0]):
            occupancy_frame = mask_centers_occupied_frame_occupied[i]
            occupancy_existing = mask_centers_occupied_existing_occupied[i]
            
            if occupancy_frame or (occupancy_existing and is_precomputing):
                index_x = centers_grid_shifted[i, 0].int().item()
                index_y = centers_grid_shifted[i, 1].int().item()
                index_z = centers_grid_shifted[i, 2].int().item()
                
                if occupancy_frame:
                    latent = voxel_grid_occ_frame.get_latent(centers_grid[i])
                    grid_latents_frame[index_x, index_y, index_z, :] = latent
                    
                if (occupancy_existing and is_precomputing):
                    latent = voxel_grid_occ_existing.get_latent(centers_grid[i])
                    grid_latents_existing[index_x, index_y, index_z, :] = latent
                    precomputed_indices[idx_indexes] = torch.tensor([index_x, index_y, index_z])
                    precomputed_centers[idx_indexes] = centers_grid[i] #TODO Remove, used for debug
                    pcd = voxel_grid_occ_existing.get_pcd(centers_grid[i])
                    idx_indexes += 1
                        
        if is_precomputing:           
            precomputed_latents = grid_latents_existing[precomputed_indices[:, 0], precomputed_indices[:, 1], precomputed_indices[:, 2], ...].clone()
            self.precomputed_storage.add_sequence_idx(precomputed_latents, precomputed_indices, centers_occupied_existing_full, torch.tensor([n_x, n_y, n_z]))
            # self.precomputed_storage.create_pcd_full(precomputed_centers, voxel_grid_occ_existing, color = np.array([1.0, 0.5, 1.0])) #TODO Remove, used for debug
            
        #complete the empty voxels by crossing 
        mask_complete_frame = (~mask_centers_occupied_frame_occupied) & mask_centers_occupied_existing_occupied #returns True where frame is empty and existing is occupied
        mask_complete_existing = (~mask_centers_occupied_existing_occupied) & mask_centers_occupied_frame_occupied #returns True where existing is empty and frame is occupied
        
        grid_latents_frame.view(-1, *self.empty_latent_code.shape)[mask_complete_frame] = grid_latents_existing.view(-1, *self.empty_latent_code.shape)[mask_complete_frame]
        grid_latents_existing.view(-1, *self.empty_latent_code.shape)[mask_complete_existing] = grid_latents_frame.view(-1, *self.empty_latent_code.shape)[mask_complete_existing]
        
        return grid_latents_frame, grid_latents_existing, vol_bounds
    
    def create_stacked_map(self, grid_latents, centers_occupied, lb):

        centers_occupied_shifted = st_utils.centers_to_grid_indexes(centers_occupied, lb, self.query_crop_size)
        
        shifts = torch.tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)
        c, h, w, d = self.voxel_grid.latent_shape
        latent_map_stacked = torch.zeros(len(centers_occupied_shifted), c, 3 * h, 3 * w, 3 * d).to(self.device)
                
        # Calculate all possible shifts for the centers
        centers_occupied_shifted = centers_occupied_shifted.unsqueeze(1) + shifts.unsqueeze(0)  # Shape: (num_centers, 27, 3)
        centers_occupied_shifted = centers_occupied_shifted.int()
        
        latent_temp = grid_latents[centers_occupied_shifted[..., 0], centers_occupied_shifted[..., 1], centers_occupied_shifted[..., 2]]

        # Reshape the gathered latents and assign to latent_map_stacked
        latent_map_stacked = latent_temp.reshape(len(centers_occupied_shifted), c, 3 * h, 3 * w, 3 * d)
        return latent_map_stacked

    def stack_latents_cold_start(self, centers_frame_occupied, encode_empty = True):
        grid_latents, vol_bounds = self.populate_grid_latent_cold_start(centers_frame_occupied, self.voxel_grid, encode_empty = encode_empty)
        
        latent_map_stacked = self.create_stacked_map(grid_latents, centers_frame_occupied, vol_bounds)
        latent_map_stacked_reshaped = latent_map_stacked.reshape(latent_map_stacked.shape[0], -1)
        
        latent_map_stacked_sumed = latent_map_stacked_reshaped.sum(dim = 1)

        return torch.cat((latent_map_stacked, latent_map_stacked), dim = 1)
    
    def stack_latents(self, centers_frame_occupied, voxel_grid_temp, encode_empty = True, is_precomputing = False):
        grid_latents_frame, grid_latents_existing, vol_bounds = self.populate_grid_latent(centers_frame_occupied, voxel_grid_temp, self.voxel_grid, encode_empty = encode_empty, is_precomputing = is_precomputing)
        latent_map_stacked_frame = self.create_stacked_map(grid_latents_frame, centers_frame_occupied, vol_bounds)
        latent_map_stacked_existing = self.create_stacked_map(grid_latents_existing, centers_frame_occupied, vol_bounds)

        return torch.cat((latent_map_stacked_frame, latent_map_stacked_existing), dim = 1)