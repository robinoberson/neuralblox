import os
import torch
import math
from torch.profiler import profile, record_function, ProfilerActivity

from src.common import (
    add_key, coord2index, normalize_coord
)
from src.training import BaseTrainer
import numpy as np
import pickle
import time
import yaml
from torch.nn import functional as F
from torch.autograd import gradcheck
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import copy

torch.manual_seed(42)

class VoxelGrid:
    def __init__(self):
        self.points_table = {}
        self.centers_table = {}
        self.latents_table = {}
        self.pcd_table = {}
        self.latent_shape = None
        self.verbose = False
    def compute_hash(self, center):
        # Assuming center is a tuple or a tensor with (x, y, z) coordinates
        return f"{center[0]}_{center[1]}_{center[2]}"
    
    def add_voxel(self, center, points, latent, pcd = None, overwrite=False):
        h = self.compute_hash(center)
        if overwrite or h not in self.points_table:
            if self.verbose:
            #print if overwriting has been requested
                if overwrite and h in list(self.centers_table.keys()):
                    print('Overwriting', h)
                
            self.points_table[h] = points
            self.centers_table[h] = center
            self.latents_table[h] = latent
            
            # if pcd is not None:
            #     self.pcd_table[h] = pcd
        
        if self.latent_shape is None:
            self.latent_shape = latent.shape    
    
    def add_voxels_batch(self, centers, points_list, latents, pcds=None, overwrite=False):
        for idx, center in enumerate(centers):
            points = points_list[idx]
            latent = latents[idx]
            pcd = pcds[idx] if pcds is not None else None
            h = self.compute_hash(center)
            if overwrite or h not in self.points_table:
                if self.verbose and overwrite and h in self.points_table:
                    print('Overwriting', h)
                
                self.points_table[h] = points
                self.centers_table[h] = center
                self.latents_table[h] = latent
                
                if pcd is not None:
                    self.pcd_table[h] = pcd

        if self.latent_shape is None and latents:
            self.latent_shape = latents[0].shape
        
    def get_voxel(self, center):
        h = self.compute_hash(center)
        points = self.points_table.get(h, None)
        center = self.centers_table.get(h, None)
        latent = self.latents_table.get(h, None)
        return points, center, latent
    
    def get_latent(self, center):
        h = self.compute_hash(center)
        latent = self.latents_table.get(h, None)
        return latent
    
    def get_points(self, center):
        h = self.compute_hash(center)
        points = self.points_table.get(h, None)
        return points
    
    def get_pcd(self, center):
        h = self.compute_hash(center)
        pcd = self.pcd_table.get(h, None)
        return pcd
    
    def detach_latents(self):
        self.latents_table = {k: v.detach() for k, v in self.latents_table.items()}
        
    def reset(self):
        self.points_table = {}
        self.centers_table = {}
        self.latents_table = {}
        self.pcd_table = {}
        self.latent_shape = None
        
    def copy(self):
        return copy.deepcopy(self)
    
    def is_empty(self):
        return len(self.latents_table) == 0
        
class SequentialTrainer(BaseTrainer):
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
                 vis_dir=None, threshold=0.5, query_n = 8192, unet_hdim = 32, unet_depth = 2, grid_reso = 24, limited_gpu = False, n_voxels_max = 20, n_max_points = 2048, n_max_points_query = 8192, occ_per_query = 0.3):
        self.model = model
        self.model_merge = model_merge
        self.optimizer = optimizer
        self.device = device
        self.device_og = device
        self.input_type = input_type
        self.input_crop_size = input_crop_size
        self.query_crop_size = query_crop_size
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.max_crop_with_change = None
        self.query_n = query_n
        self.hdim = unet_hdim
        self.factor = 2**unet_depth
        self.unet = None
        self.limited_gpu = limited_gpu
        self.reso = grid_reso
        self.iteration = 0
        self.cfg = cfg
        self.n_voxels_max = n_voxels_max
        self.n_max_points = n_max_points
        self.n_max_points_query = n_max_points_query
        self.occ_per_query = occ_per_query
        self.voxel_grid = VoxelGrid()
        # self.voxel_grid.verbose = True
        self.voxel_grid_empty = VoxelGrid()
        self.empty_latent_code = self.get_empty_latent_representation()
        current_dir = os.getcwd()
        
        if 'robin' in current_dir:
            self.location = 'home'
        elif 'roberson' in current_dir:
            self.location = 'local'
        else:
            self.location = 'euler'

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
           
    def print_timing(self, operation):
        t1 = time.time()
        print(f'Time elapsed, {self.timing_counter}: {t1 - self.t0:.3f}, {operation}')

        self.t0 = time.time()
        self.timing_counter += 1
        
    def process_sequence(self, p_in, p_query, is_training):
        n_sequence = p_in.shape[0]
        total_loss = 0
        results = []

        with torch.no_grad():
            
            for idx_sequence in range(n_sequence):
                # idx_sequence = 0
                inputs_frame = p_in[idx_sequence]
                p_query_distributed, centers_query = self.get_distributed_inputs(p_query[idx_sequence], self.n_max_points_query, self.occ_per_query)

                if idx_sequence == 0:
                    self.voxel_grid.reset()
                    latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.fuse_cold_start(inputs_frame)
                else:
                    latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.fuse_inputs(inputs_frame)

                p_stacked, latents, centers, occ = self.prepare_data_logits(latent_map_stacked_merged, centers_frame_occupied, p_query_distributed, centers_query)
                logits_sampled = self.get_logits(p_stacked, latents, centers)
                loss = F.binary_cross_entropy_with_logits(logits_sampled, occ, reduction='none').sum(-1).mean()
                
                if is_training:                        
                    self.visualize_logits(logits_sampled, p_stacked, inputs_frame_distributed)
                    # loss.backward()
                    self.voxel_grid.detach_latents()
                    self.optimizer.step()
                    self.iteration += 1
                else:
                    results.append([p_stacked, latents, inputs_frame, logits_sampled, loss.item()])

            total_loss += loss.item()
        
        if is_training:
            return total_loss
        else:
            results.append(total_loss)
            return results
        
    def train_sequence_window(self, data_batch):
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.train()
        self.model_merge.train()
        self.optimizer.zero_grad()
        
        p_in, p_query = self.get_inputs_from_batch(data_batch)
        
        return self.process_sequence(p_in, p_query, is_training=True)
    
    def validate_sequence(self, data_batch):
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.eval()
        self.model_merge.eval()
        
        with torch.no_grad():
            p_in, p_query = self.get_inputs_from_batch(data_batch)
            return self.process_sequence(p_in, p_query, is_training=False)

    def fuse_cold_start(self, inputs_frame, encode_empty = True):
        if encode_empty:
            latents_frame, inputs_frame_distributed, inputs_frame_distributed_occupied, centers_frame, centers_frame_occupied  = self.encode_distributed_inputs(inputs_frame, encode_empty = True)
        else:
            latents_frame, inputs_frame_distributed, centers_frame, centers_frame_occupied = self.encode_distributed_inputs(inputs_frame, encode_empty = False)
            inputs_frame_distributed_occupied = inputs_frame_distributed

        for idx, (center, points, encoded_latent) in enumerate(zip(centers_frame, inputs_frame_distributed, latents_frame)):
            if idx < len(centers_frame_occupied):
                self.voxel_grid.add_voxel(center, points, encoded_latent, overwrite = False)
            else:
                if encode_empty:
                    self.voxel_grid_empty.add_voxel(center, points, encoded_latent, overwrite = True)
                else:
                    break
                
        latent_map_stacked = self.stack_latents_cold_start(centers_frame_occupied, encode_empty = encode_empty)
        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)

        for center, points, encoded_latent in zip(centers_frame_occupied, inputs_frame_distributed_occupied, latent_map_stacked_merged):
            self.voxel_grid.add_voxel(center, points, encoded_latent, overwrite = True)
            
        return latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed_occupied
    
    def fuse_inputs(self, inputs_frame, encode_empty = True):
        self.t0 = time.time()
        self.timing_counter = 0
        
        if encode_empty:
            latents_frame, inputs_frame_distributed, inputs_frame_distributed_occupied, centers_frame, centers_frame_occupied  = self.encode_distributed_inputs(inputs_frame, encode_empty = True)
        else:
            latents_frame, inputs_frame_distributed, centers_frame, centers_frame_occupied = self.encode_distributed_inputs(inputs_frame, encode_empty = False)
            inputs_frame_distributed_occupied = inputs_frame_distributed
        
        self.print_timing('encode time')
        
        voxel_grid_temp = VoxelGrid()
        voxel_grid_empty_temp = None
        
        if encode_empty:
            voxel_grid_empty_temp = VoxelGrid()

        for idx, (center, points, encoded_latent) in enumerate(zip(centers_frame, inputs_frame_distributed, latents_frame)):
            if idx < len(centers_frame_occupied):
                voxel_grid_temp.add_voxel(center, points, encoded_latent)
                # self.voxel_grid.add_voxel(center, points, encoded_latent, overwrite = False)
            else:
                if encode_empty:
                    voxel_grid_empty_temp.add_voxel(center, points, encoded_latent, overwrite = True)                    
                # self.voxel_grid_empty.add_voxel(center, points, encoded_latent, overwrite = True)
        self.print_timing('add voxel')
        #stack the latents
        latent_map_stacked = self.stack_latents(centers_frame_occupied, voxel_grid_temp, voxel_grid_empty_temp, encode_empty = encode_empty)
        self.print_timing('stack latent')
        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)
        self.print_timing('merge latent')
        centers_frame_occupied_list = [centers_frame_occupied[i] for i in range(len(centers_frame_occupied))]
        inputs_frame_distributed_occupied_list = [inputs_frame_distributed_occupied[i].cpu().numpy() for i in range(len(inputs_frame_distributed_occupied))]
        latent_map_stacked_merged_list = [latent_map_stacked_merged[i] for i in range(len(latent_map_stacked_merged))]
        self.print_timing('listify latent')
        for i in range(len(centers_frame_occupied)):
            self.voxel_grid.add_voxel(centers_frame_occupied_list[i], inputs_frame_distributed_occupied_list[i], latent_map_stacked_merged_list[i], overwrite = True)
        
        # for center, points, encoded_latent in zip(centers_frame_occupied, inputs_frame_distributed_occupied, latent_map_stacked_merged):
        #     self.voxel_grid.add_voxel(center, points, encoded_latent, overwrite = True)
        self.print_timing('add voxel')
        return latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed_occupied
        # return stacked_points, centers_frame_occupied
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
        
        return p_stacked, latents, centers, occ
    
    def get_logits(self, p_stacked, latents, centers):
    
        n_crops_total = p_stacked.shape[0]
                
        vol_bound = self.get_grid_from_centers(centers, self.input_crop_size)
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
        mask_query = (centers_query.unsqueeze(1) == centers_frame_occupied).all(dim=2).any(dim=1)
        mask_frame = (centers_frame_occupied.unsqueeze(1) == centers_query).all(dim=2).any(dim=1)
        # print(f'{mask_query.sum()}, {mask_frame.sum()}')
        
        p_query_dict = {}
        
        for p_query, center in zip(p_query_distributed[mask_query], centers_query[mask_query]):
            hash = self.voxel_grid.compute_hash(center)
            p_query_dict[hash] = p_query
        
        return p_query_dict, mask_query, mask_frame
        
    def merge_latent_map(self, latent_map):
        fea_dict = {}
        fea_dict['latent'] = latent_map
        
        latent_map = self.model_merge(fea_dict)
        return latent_map
    
    def get_empty_latent_representation(self):
        center = torch.tensor([0,0,0]).unsqueeze(0).to(self.device)
        
        empty_inputs = self.get_empty_inputs(center, self.query_crop_size, n_max_points = 2048)
        occ = torch.zeros(*empty_inputs.shape[0:2], 1).to(self.device)
        empty_inputs = torch.cat((empty_inputs, occ), axis = -1)
        empty_latent_code = self.encode_distributed(empty_inputs, center)

        return empty_latent_code.squeeze(0)
    
    def encode_distributed_inputs(self, inputs_frame, encode_empty = True):
        self.t0 = time.time()
        
        inputs_frame_distributed_occupied, centers_frame_occupied = self.get_distributed_inputs(inputs_frame, self.n_max_points)
                
        centers_neighbours = self.compute_neighbours_and_bounds(centers_frame_occupied)
        centers_frame_empty = self.get_empty_neighbours_centers(centers_neighbours, centers_frame_occupied)
        
        centers_frame = torch.cat((centers_frame_occupied, centers_frame_empty), axis = 0)
        
        if encode_empty:
            inputs_frame_distributed_empty, centers_frame_empty = self.get_empty_neighbours(centers_frame_empty)
            inputs_frame_distributed = torch.cat((inputs_frame_distributed_occupied, inputs_frame_distributed_empty), axis = 0)

            latents_frame = self.encode_distributed(inputs_frame_distributed, centers_frame)
            
            return latents_frame, inputs_frame_distributed, inputs_frame_distributed_occupied, centers_frame, centers_frame_occupied

        else:           
            latents_frame_occupied = self.encode_distributed(inputs_frame_distributed_occupied, centers_frame_occupied)
                        
            expanded_shape = (len(centers_neighbours),) + self.empty_latent_code.shape
            latents_frame_empty = self.empty_latent_code.expand(expanded_shape)
            
            latents_frame = torch.cat((latents_frame_occupied, latents_frame_empty), axis = 0)
            return latents_frame, inputs_frame_distributed_occupied, centers_frame, centers_frame_occupied

            
        
    def stack_latents_cold_start(self, centers_frame_occupied, encode_empty = True):
        c, h, w, d = self.voxel_grid.latent_shape 
        
        latent_map_stacked = torch.zeros(len(centers_frame_occupied), 2 * c, 3*h, 3*w, 3*d).to(device=self.device)
        
        offsets = torch.tensor([(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)], device = self.device)
        latent_cache = {}

        for idx_center, center in enumerate(centers_frame_occupied):
            
            stacked_voxel_frame = None
            
            #Retrieve latent 
            for idx_offset, offset in enumerate(offsets):
                center_with_offset = center + offset

                # check if in cache, if not check in voxel grid if not in empty
                if center_with_offset not in latent_cache:
                    latent_cache[center_with_offset] = self.voxel_grid.get_latent(center_with_offset)
                    
                    if latent_cache[center_with_offset] is None:
                        if encode_empty:
                            latent_cache[center_with_offset] = self.voxel_grid_empty.get_latent(center_with_offset)
                        else:
                            latent_cache[center_with_offset] = self.empty_latent_code
                        
                latent = latent_cache[center_with_offset]
                
                if latent is None:
                    print('latent_existing is None, problem in cold start')
                    
                if stacked_voxel_frame is None:
                    stacked_voxel_frame = latent
                    
                else:
                    stacked_voxel_frame = torch.cat((stacked_voxel_frame, latent), axis = 0)
            
            stacked_voxel_frame = stacked_voxel_frame.reshape(c, 3*h, 3*w, 3*d)
            
            latent_map_stacked[idx_center] = torch.cat((stacked_voxel_frame, stacked_voxel_frame), axis = 0)

        return latent_map_stacked
    
    def stack_points_cold_start(self, centers_frame_occupied):
        
        latent_map_stacked = torch.zeros(len(centers_frame_occupied), 2, 27 * self.n_max_points, 4).to(device=self.device)      
          
        offsets = torch.tensor([(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)], device = self.device)
        latent_cache = {}

        for idx_center, center in enumerate(centers_frame_occupied):
            
            stacked_voxel_frame = None
            
            #Retrieve latent 
            for idx_offset, offset in enumerate(offsets):
                center_with_offset = center + offset

                # check if in cache, if not check in voxel grid if not in empty
                if center_with_offset not in latent_cache:
                    latent_cache[center_with_offset] = self.voxel_grid.get_points(center_with_offset)
                    
                    if latent_cache[center_with_offset] is None:
                        latent_cache[center_with_offset] = self.voxel_grid_empty.get_points(center_with_offset)
                        
                latent = latent_cache[center_with_offset]
                
                if latent is None:
                    print('latent_existing is None, problem in cold start')
                    
                if stacked_voxel_frame is None:
                    stacked_voxel_frame = latent
                    
                else:
                    stacked_voxel_frame = torch.cat((stacked_voxel_frame, latent), axis = 0)
            
            stacked_voxel_frame = stacked_voxel_frame.reshape(self.n_max_points*27, 4).unsqueeze(0)

            inter = torch.cat((stacked_voxel_frame, stacked_voxel_frame), axis = 0)

            latent_map_stacked[idx_center] = inter

        return latent_map_stacked
    
    def stack_latents(self, centers_frame_occupied, voxel_grid_temp, voxel_grid_temp_empty, encode_empty = True):
        #voxel_grid_temp is the precomputed voxel grid of the new frame
        
        c, h, w, d = voxel_grid_temp.latent_shape  # Use voxel_grid_temp to get latent_shape
        
        latent_map_stacked = torch.zeros(len(centers_frame_occupied), 2 * c, 3*h, 3*w, 3*d, device=self.device)
        
        offsets = torch.tensor([(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)], device=self.device)
        
        # Cache latents for efficiency
        latent_cache = {}
        latent_cache_empty = {}

        for idx_center, center in enumerate(centers_frame_occupied):
            centers_with_offsets = center + offsets
            
            # Prepare latents in one batch
            stacked_voxel_frame = []
            stacked_voxel_existing = []
            
            for center_with_offset in centers_with_offsets:
                
                is_empty = False

                # Retrieve latents frame 
                if center_with_offset not in latent_cache:
                    latent_cache[center_with_offset] = voxel_grid_temp.get_latent(center_with_offset)

                    if latent_cache[center_with_offset] is None:
                        if encode_empty:
                            latent_cache_empty[center_with_offset] = voxel_grid_temp_empty.get_latent(center_with_offset)
                        else:
                            latent_cache_empty[center_with_offset] = self.empty_latent_code
                            
                        is_empty = True
                        
                if is_empty:
                    latent_frame = self.voxel_grid.get_latent(center_with_offset)
                    if latent_frame is None:
                        latent_frame = latent_cache_empty[center_with_offset]
                else:
                    latent_frame = latent_cache[center_with_offset]
                
                # Retrieve latents for fusion
                latent_existing = self.voxel_grid.get_latent(center_with_offset)
               
                if latent_existing is None:
                    latent_existing = latent_frame # If the position has no latent, we use the latent from the frame to double stack (fuse with itself)

                # Append latents to batch
                stacked_voxel_frame.append(latent_frame.unsqueeze(0))
                stacked_voxel_existing.append(latent_existing.unsqueeze(0))
            
            stacked_voxel_frame = torch.cat(stacked_voxel_frame, dim=0)
            stacked_voxel_existing = torch.cat(stacked_voxel_existing, dim=0)
            
            # Reshape latents for neighborhood             
            stacked_voxel_frame = stacked_voxel_frame.reshape(c, 3*h, 3*w, 3*d)
            stacked_voxel_existing = stacked_voxel_existing.reshape(c, 3*h, 3*w, 3*d)
            
            # Stack latents frame and existing
            latent_map_stacked[idx_center] = torch.cat((stacked_voxel_frame, stacked_voxel_existing), dim=0)

        return latent_map_stacked
    
    def stack_points(self, centers_frame_occupied, voxel_grid_temp):
        #voxel_grid_temp is the precomputed voxel grid of the new frame
                
        latent_map_stacked = torch.zeros(len(centers_frame_occupied), 2, 27 * self.n_max_points, 4).to(device=self.device)      
        
        offsets = torch.tensor([(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)], device=self.device)
        
        # Cache latents for efficiency
        latent_cache = {}
        

        for idx_center, center in enumerate(centers_frame_occupied):
            centers_with_offsets = center + offsets
            
            # Prepare latents in one batch
            stacked_voxel_frame = []
            stacked_voxel_existing = []
            n_fusion_frame = 0
            n_fusion_existing = 0
            for center_with_offset in centers_with_offsets:
                
                # Retrieve latents frame 
                if center_with_offset not in latent_cache:
                    latent_cache[center_with_offset] = voxel_grid_temp.get_points(center_with_offset)
                
                latent_frame = latent_cache[center_with_offset]
                
                # Retrieve latents for fusion
                if self.voxel_grid.get_points(center_with_offset) is not None:
                    latent_existing = self.voxel_grid.get_points(center_with_offset)
                    n_fusion_existing += 1
                else: 
                    # latent_existing = self.voxel_grid_empty.get_latent(center_with_offset) # If the position has no latent, we use the latent from the empty voxel grid
                    latent_existing = latent_frame # If the position has no latent, we use the latent from the frame to double stack (fuse with itself)
                    n_fusion_frame += 1
                # Append latents to batch
                stacked_voxel_frame.append(latent_frame.unsqueeze(0))
                stacked_voxel_existing.append(latent_existing.unsqueeze(0))
            
            print(f'n_fusion_frame: {n_fusion_frame}, n_fusion_existing: {n_fusion_existing}, should add to {len(centers_with_offsets)}, is {n_fusion_frame + n_fusion_existing}')
            stacked_voxel_frame = torch.cat(stacked_voxel_frame, dim=0)
            stacked_voxel_existing = torch.cat(stacked_voxel_existing, dim=0)
            
            # Reshape latents for neighborhood             
            stacked_voxel_frame = stacked_voxel_frame.reshape(self.n_max_points*27, 4).unsqueeze(0)
            stacked_voxel_existing = stacked_voxel_existing.reshape(self.n_max_points*27, 4).unsqueeze(0)

            inter = torch.cat((stacked_voxel_frame, stacked_voxel_existing), axis = 0)

            latent_map_stacked[idx_center] = inter
            
        return latent_map_stacked
    def generate_points(self, n, lb = [0.0, 0.0, 0.0], ub = [1.0, 1.0, 1.0]):
        """
        Generate n points within the bounds lb and ub.

        Args:
        - n (int): Number of points to generate.
        - lb (list of float): Lower bound for each dimension.
        - ub (list of float): Upper bound for each dimension.

        Returns:
        - torch.Tensor: Tensor of shape (n, 3) containing the generated points.
        """
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        
        # Generate n points with each dimension in the range [0, 1)
        points = torch.rand(n, 3)
        
        # Scale points to the range [lb, ub)
        points = lb + points * (ub - lb)
        
        return points
    
    def get_empty_inputs(self, centers, crop_size, n_max_points = 2048):
        lb_input = centers - crop_size / 2
        ub_input = centers + crop_size / 2
        
        vol_bound_inputs = torch.stack([lb_input, ub_input], axis=1)
        
        n_crops = vol_bound_inputs.shape[0]
        
        bb_min = vol_bound_inputs[:, 0, :].unsqueeze(1) # Shape: (n_crops, 3, 1)
        bb_max = vol_bound_inputs[:, 1, :].unsqueeze(1)  # Shape: (n_crops, 3, 1)
        bb_size = bb_max - bb_min  # Shape: (n_crops, 3, 1)

        random_points = self.generate_points(n_max_points)
        random_points = random_points.repeat(n_crops, 1, 1).to(device=self.device)
        random_points *= bb_size  # Scale points to fit inside each bounding box
        random_points += bb_min  # Translate points to be within each bounding box

        return random_points
    def get_empty_neighbours_centers(self, centers_neighbours, centers_frame):
        mask = (centers_neighbours.unsqueeze(1) == centers_frame).all(dim=2).any(dim=1)

        centers_neighbours_empty = centers_neighbours[~mask]

        return centers_neighbours_empty
        
    def get_empty_neighbours(self, centers_neighbours_empty):
        
        random_points = self.get_empty_inputs(centers_neighbours_empty, self.input_crop_size, self.n_max_points)
        occupancies = torch.zeros(*random_points.shape[:-1], 1, device=self.device)
        
        random_points = torch.cat((random_points, occupancies), axis = -1)
        
        return random_points, centers_neighbours_empty
        
        
    def compute_neighbours_and_bounds(self, centers):
        offsets = torch.tensor(
            [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2) if not (i == 0 and j == 0 and k == 0)],
            device=self.device
        )
        
        # Expand the centers tensor to match the offsets tensor
        centers_expanded = centers.unsqueeze(1).expand(-1, offsets.size(0), -1)
        
        # Calculate the neighbouring centers
        neighbouring_centers_full = centers_expanded + offsets

        # Reshape the tensor to merge the first two dimensions
        neighbouring_centers_full = neighbouring_centers_full.reshape(-1, 3)

        # Remove duplicates by converting to a set of tuples and back to a tensor
        unique_neighbouring_centers = torch.unique(neighbouring_centers_full, dim=0)

        return unique_neighbouring_centers

    def encode_distributed(self, inputs, centers):
        
        vol_bound = self.get_grid_from_centers(centers, self.input_crop_size)

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
            
            # self.print_timing('encode_inputs')
                
            _, latent_map_batch, self.features_shapes = self.unet(fea, return_feature_maps=True, decode = False, limited_gpu = False)

            # self.print_timing('unet')
            # if self.limited_gpu: latent_map_batch = latent_map_batch.to('cpu')

            if latent_map is None:
                latent_map = latent_map_batch  # Initialize latent_map with the first batch
                # latent_map_decoded = latent_map_batch_decoded
            else:
                latent_map = torch.cat((latent_map, latent_map_batch), dim=0)  # Concatenate latent maps
                
        latent_map_shape = latent_map.shape
        return latent_map.reshape(n_crop, *latent_map_shape[1:])

    def get_inputs_from_batch(self, batch):
        p_in_3D = batch.get('inputs').to(self.device).squeeze(0)
        p_in_occ = batch.get('inputs.occ').to(self.device).squeeze(0).unsqueeze(-1)
                
        p_query_3D = batch.get('points').to(self.device).squeeze(0)
        p_query_occ = batch.get('points.occ').to(self.device).squeeze(0).unsqueeze(-1)
        
        # print(f'p_in_3D: {p_in_3D.shape}, p_in_occ: {p_in_occ.shape}, p_query_3D: {p_query_3D.shape}, p_query_occ: {p_query_occ.shape}')
        p_in = torch.cat((p_in_3D, p_in_occ), dim=-1)
        p_query = torch.cat((p_query_3D, p_query_occ), dim=-1)
        
        return p_in, p_query
    
    def get_grid_from_centers(self, centers, crop_size):
        lb = centers - crop_size / 2
        ub = centers + crop_size / 2
        vol_bounds = torch.stack([lb, ub], dim=1)
        
        return vol_bounds

    def compute_vol_bound(self, inputs):
        # inputs must have shape (n_points, 3)
        assert inputs.shape[1] == 3 and inputs.shape[0] > 0 and len(inputs.shape) == 2

        vol_bound = {}

        lb_p = torch.min(inputs, dim=0).values - torch.tensor([0.1, 0.1, 0.1], device=inputs.device)
        ub_p = torch.max(inputs, dim=0).values
        
        # print(lb_p, ub_p)

        lb = torch.round((lb_p - lb_p % self.query_crop_size) * 1e6) / 1e6
        ub = torch.round((((ub_p - ub_p % self.query_crop_size).int() / self.query_crop_size) + 1) * self.query_crop_size * 1e6) / 1e6

        lb_query = torch.stack(torch.meshgrid(
            torch.arange(lb[0], ub[0], self.query_crop_size, device=inputs.device),
            torch.arange(lb[1], ub[1], self.query_crop_size, device=inputs.device),
            torch.arange(lb[2], ub[2], self.query_crop_size, device=inputs.device),
        ), dim=-1).reshape(-1, 3)

        ub_query = lb_query + self.query_crop_size
        centers = (lb_query + ub_query) / 2

        # Number of crops alongside x, y, z axis
        vol_bound['axis_n_crop'] = torch.ceil((ub - lb) / self.query_crop_size).int()

        # Total number of crops
        num_crop = torch.prod(vol_bound['axis_n_crop']).item()
        vol_bound['n_crop'] = num_crop
        vol_bound['input_vol'] = self.get_grid_from_centers(centers, self.input_crop_size)
        vol_bound['query_vol'] = self.get_grid_from_centers(centers, self.query_crop_size)

        return vol_bound, centers

    
    def get_distributed_inputs(self, distributed_inputs_raw, n_max = 2048, occ_perc = 1.0):
        # Clone the input tensor
        distributed_inputs = distributed_inputs_raw.clone()
        
        vol_bound, centers = self.compute_vol_bound(distributed_inputs[:, :3].reshape(-1, 3))
        
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

        random_points = self.get_empty_inputs(centers, self.input_crop_size, n_max_points = n_max)
        
        distributed_inputs_short = torch.zeros(n_crops, n_max, 4, device=self.device)
        distributed_inputs_short[:, :, :3] = random_points.reshape(n_crops, n_max, 3)
                
        # Select n_max points
        new_line_template = torch.zeros(indexes_keep.shape[1], dtype=torch.bool)
        for i in range(distributed_inputs.shape[0]):
            indexes_line = indexes_keep[i, :]
            if indexes_line.sum() > int(n_max*occ_perc):
                indexes_true = torch.where(indexes_line)[0]
                random_indexes_keep = torch.randperm(indexes_true.shape[0])[:int(n_max*occ_perc)]
                
                # Create a new line with the randomly selected indexes set to True in a single step
                new_line = new_line_template.clone()
                new_line[indexes_true[random_indexes_keep]] = True
                
                # Update the indexes_keep tensor with the new line for the current sample
                indexes_keep[i] = new_line

        # Select n_max points
        n_points = indexes_keep.sum(axis=1)
        mask = torch.arange(n_max).expand(n_crops, n_max).to(device=self.device) < n_points.unsqueeze(-1)
                
        distributed_inputs_short[mask] = distributed_inputs[indexes_keep]
        
        voxels_occupied = distributed_inputs_short[..., 3].sum(dim=1).int() > 2
        
        inputs_frame_occupied = distributed_inputs_short[voxels_occupied]
        centers_frame_occupied = centers[voxels_occupied]

        return inputs_frame_occupied, centers_frame_occupied
    
    def visualize_logits(self, logits_sampled, p_query, inputs_distributed=None, force_viz = False):
        geos = []
        
        current_dir = os.getcwd()
            
        # file_path = f'/home/roberson/MasterThesis/master_thesis/neuralblox/configs/simultaneous/train_simultaneous_{self.location}.yaml'
        file_path = '/home/robin/Dev/MasterThesis/GithubRepos/master_thesis/neuralblox/configs/fusion/train_fusion_home.yaml'

        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        except:
            return
            
        if not(force_viz or config['visualization']):
            return

        import open3d as o3d
        p_stacked = p_query[..., :3]
        
        p_full = p_stacked.detach().cpu().numpy().reshape(-1, 3)

        occ_sampled = logits_sampled.detach().cpu().numpy()

        values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
        
        values_sampled = values_sampled.reshape(-1)

        threshold = 0.5

        values_sampled[values_sampled < threshold] = 0
        values_sampled[values_sampled >= threshold] = 1
        
        values_gt = p_query[..., -1].reshape(-1).detach().cpu().numpy()

        both_occ = np.logical_and(values_gt, values_sampled)
        
        pcd = o3d.geometry.PointCloud()
        colors = np.zeros((values_gt.shape[0], 3))
        colors[values_gt == 1] = [1, 0, 0] # red
        colors[values_sampled == 1] = [0, 0, 1] # blue
        colors[both_occ == 1] = [0, 1, 0] # green
        
        mask = np.any(colors != [0, 0, 0], axis=1)
        # print(mask.shape, values_gt.shape, values_sampled.shape, colors.shape)
        if inputs_distributed is not None:
            pcd_inputs = o3d.geometry.PointCloud()
            inputs_reshaped = inputs_distributed.reshape(-1, 4).detach().cpu().numpy()
            pcd_inputs.points = o3d.utility.Vector3dVector(inputs_reshaped[inputs_reshaped[..., -1] == 1, :3])
            pcd_inputs.paint_uniform_color([1., 0.5, 1.0]) # blue
            geos += [pcd_inputs]
            
        colors = colors[mask]
        pcd.points = o3d.utility.Vector3dVector(p_full[mask])
        bb_min_points = np.min(p_full[mask], axis=0)
        bb_max_points = np.max(p_full[mask], axis=0)
        # print(bb_min_points, bb_max_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        geos += [pcd, base_axis]
        o3d.visualization.draw_geometries(geos)