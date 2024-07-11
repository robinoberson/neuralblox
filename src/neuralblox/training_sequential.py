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
        self.centers_table = {}
        self.latents_table = {}
        self.latent_shape = None
        self.verbose = False
    def compute_hash(self, center):
        # Assuming center is a tuple or a tensor with (x, y, z) coordinates
        return f"{torch.round(center[0]*1000)/1000:.3f}_{torch.round(center[1]*1000)/1000:.3f}_{torch.round(center[2]*1000)/1000:.3f}"
    
    def add_voxel(self, center, latent, overwrite=False):
        h = self.compute_hash(center)
        list_keys = list(self.centers_table.keys())
        if overwrite or h not in list(self.centers_table.keys()):
            if self.verbose:
            #print if overwriting has been requested
                if overwrite and h in list(self.centers_table.keys()):
                    print('Overwriting', h)
                
            self.centers_table[h] = center
            self.latents_table[h] = latent
            
        if self.latent_shape is None:
            self.latent_shape = latent.shape    
        
    def get_voxel(self, center):
        h = self.compute_hash(center)
        center = self.centers_table.get(h, None)
        latent = self.latents_table.get(h, None)
        return center, latent
    
    def get_latent(self, center):
        h = self.compute_hash(center)
        latent = self.latents_table.get(h, None)
        return latent
    
    def detach_latents(self):
        self.latents_table = {k: v.detach() for k, v in self.latents_table.items()}
        
    def reset(self):
        self.centers_table = {}
        self.latents_table = {}
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
                 vis_dir=None, threshold=0.5, query_n = 8192, unet_hdim = 32, unet_depth = 2, grid_reso = 24, limited_gpu = False, n_voxels_max = 20, n_max_points = 2048, n_max_points_query = 8192, occ_per_query = 0.3, return_flat = True,
                 sigma = 0.8):
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
        self.return_flat = return_flat
        self.sigma = sigma
        # self.voxel_grid.verbose = True
        self.empty_latent_code = self.get_empty_latent_representation()
        self.timing_counter = 0
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
           
    def print_timing(self, operation):
        torch.cuda.synchronize()
        t1 = time.time()
        print(f'Time elapsed, {self.timing_counter}: {t1 - self.t0:.3f}, {operation}')

        self.t0 = time.time()
        self.timing_counter += 1
        
    def process_sequence(self, p_in, p_query, is_training, return_flat = True):
        # self.print_timing('process_sequence')
        n_sequence = p_in.shape[0]
        total_loss = 0
        results = []

        # with torch.no_grad():
            
        for idx_sequence in range(n_sequence):
            # idx_sequence = 0
            # self.print_timing(f'process_sequence start')
            inputs_frame = p_in[idx_sequence]
            p_query_distributed, centers_query = self.get_distributed_inputs(p_query[idx_sequence], self.n_max_points_query, self.occ_per_query)

            # self.print_timing(f'get_distributed_inputs')
            if idx_sequence == 0:
                self.voxel_grid.reset()
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.fuse_cold_start(inputs_frame, encode_empty = is_training)
            else:
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.fuse_inputs(inputs_frame, encode_empty = is_training)

            # self.print_timing(f'fuse_inputs')
            
            # if not return_flat:
            #     mask_elevation = self.get_elevation_mask(inputs_frame_distributed)
            #     latent_map_stacked_merged = latent_map_stacked_merged[mask_elevation]
            #     centers_frame_occupied = centers_frame_occupied[mask_elevation]
            #     inputs_frame_distributed = inputs_frame_distributed[mask_elevation]
            
            
            p_stacked, latents, centers, occ, mask_frame = self.prepare_data_logits(latent_map_stacked_merged, centers_frame_occupied, p_query_distributed, centers_query)

            # self.visualize_cost(p_stacked, inputs_frame_distributed, mask_frame)
            logits_sampled = self.get_logits(p_stacked, latents, centers)
            loss_unweighted = F.binary_cross_entropy_with_logits(logits_sampled, occ, reduction='none')
            
            weights = self.compute_gaussian_weights(p_stacked, inputs_frame_distributed[mask_frame], sigma = self.sigma)

            loss = (loss_unweighted * weights).sum(dim=-1).mean()
            # self.print_timing('loss done')
            if is_training:         
                self.visualize_logits(logits_sampled, p_stacked, weights = weights, inputs_distributed = inputs_frame_distributed[mask_frame], force_viz = False)
                loss.backward()
                # self.print_timing('backward done')
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
        # self.print_timing('loop done')
        
    def visualize_cost(self, p_stacked, inputs_frame_distributed, mask_frame):
        import open3d as o3d
        
        pcd_stacked = o3d.geometry.PointCloud()
        pcd_stacked_points = p_stacked.detach().cpu().numpy().reshape(-1, 4)
        pcd_stacked.points = o3d.utility.Vector3dVector(pcd_stacked_points[pcd_stacked_points[:, 3] == 1][:, :3])
        pcd_stacked.paint_uniform_color([1.0, 0.0, 0.0])
        
        pcd_distributed = o3d.geometry.PointCloud()
        pcd_distributed_points = inputs_frame_distributed.detach().cpu().numpy().reshape(-1, 4)
        pcd_distributed.points = o3d.utility.Vector3dVector(pcd_distributed_points[pcd_distributed_points[:, 3] == 1][:, :3])
        pcd_distributed.paint_uniform_color([0.0, 0.0, 1.0])
        
        pcd_distributed_masked = o3d.geometry.PointCloud()
        pcd_distributed_masked_points = inputs_frame_distributed[mask_frame].detach().cpu().numpy().reshape(-1, 4)
        pcd_distributed_masked.points = o3d.utility.Vector3dVector(pcd_distributed_masked_points[pcd_distributed_masked_points[:, 3] == 1][:, :3])
        pcd_distributed_masked.paint_uniform_color([0.0, 1.0, 0.0])
        
        o3d.visualization.draw_geometries([pcd_stacked, pcd_distributed, pcd_distributed_masked])
        
    def train_sequence_window(self, data_batch):
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.train()
        self.model_merge.train()
        self.optimizer.zero_grad()
        
        p_in, p_query = self.get_inputs_from_batch(data_batch)
        
        return self.process_sequence(p_in, p_query, is_training=True, return_flat = self.return_flat)
    
    def validate_sequence(self, data_batch):
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.eval()
        self.model_merge.eval()
        
        with torch.no_grad():
            p_in, p_query = self.get_inputs_from_batch(data_batch)
            return self.process_sequence(p_in, p_query, is_training=False, return_flat = True)
        
    def compute_gaussian_weights(self, gt_points_batch, input_points_batch, sigma=1.0):
        
        batch_size = gt_points_batch.shape[0]
        n_gt = gt_points_batch.shape[1]
        
        weights_batch = torch.zeros(batch_size, n_gt, device=self.device)

        for b in range(batch_size):
            gt_points = gt_points_batch[b]
            input_points = input_points_batch[b]
            
            # Flatten to handle each batch element independently
            gt_points_flat = gt_points[..., :3].reshape(-1, 3)
            inputs_flat = input_points[input_points[..., 3] == 1, :3].reshape(-1, 3)
            
            # Compute pairwise distances
            distances = torch.cdist(gt_points_flat, inputs_flat)
            
            # Find the minimum distance for each gt point
            min_distances, _ = distances.min(dim=1)
            
            # Compute Gaussian weights
            weights = torch.exp(-min_distances ** 2 / (2 * sigma ** 2))
            weights_batch[b] = weights
            
        return weights_batch

    def fuse_cold_start(self, inputs_frame, encode_empty = True):
        latents_frame_occupied, inputs_frame_distributed_occupied, centers_frame_occupied = self.encode_inputs_frame(inputs_frame)

        for idx, (center, encoded_latent) in enumerate(zip(centers_frame_occupied, latents_frame_occupied)):
            if encoded_latent.shape != self.empty_latent_code.shape:
                print(f'encoded_latent.shape: {encoded_latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
            self.voxel_grid.add_voxel(center, encoded_latent, overwrite = False)
        
        # voxel_grid_latents = torch.stack(list(self.voxel_grid.latents_table.values()))
        # voxel_grid_latents_reshaped = voxel_grid_latents.reshape(voxel_grid_latents.shape[0], -1)
        # voxel_grid_latents_sumed = voxel_grid_latents_reshaped.sum(dim = 1)
        # unique_lines = torch.unique(voxel_grid_latents_reshaped, dim = 0)
        
        # print(f'fuse_cold_start voxel_grid_latents, Unique lines: {unique_lines.shape[0]}, Total lines: {voxel_grid_latents_reshaped.shape[0]}')
        
        latent_map_stacked = self.stack_latents_cold_start(centers_frame_occupied, encode_empty = encode_empty)
        
        # latent_map_stacked_reshaped = latent_map_stacked.reshape(latent_map_stacked.shape[0], -1)
        # latent_map_stacked_reshaped_sumed = latent_map_stacked_reshaped.sum(dim = 1)
        # unique_lines = torch.unique(latent_map_stacked_reshaped, dim = 0)
        
        # print(f'fuse_cold_start latent_map_stacked, Unique lines: {unique_lines.shape[0]}, Total lines: {latent_map_stacked_reshaped.shape[0]}')
        
        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)

        for center, encoded_latent in zip(centers_frame_occupied, latent_map_stacked_merged):
            if encoded_latent.shape != self.empty_latent_code.shape:
                print(f'encoded_latent.shape: {encoded_latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
            self.voxel_grid.add_voxel(center, encoded_latent, overwrite = True)
            
        return latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed_occupied
    
    def fuse_inputs(self, inputs_frame, encode_empty = True):
        self.t0 = time.time()
        # self.print_timing('start')
        latents_frame_occupied, inputs_frame_distributed_occupied, centers_frame_occupied = self.encode_inputs_frame(inputs_frame)
        
        # self.print_timing('encode')
        
        voxel_grid_temp = VoxelGrid()

        for idx, (center, encoded_latent) in enumerate(zip(centers_frame_occupied, latents_frame_occupied)):
            if encoded_latent.shape != self.empty_latent_code.shape:
                print(f'encoded_latent.shape: {encoded_latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
            voxel_grid_temp.add_voxel(center, encoded_latent)
        
             
        # self.print_timing('add voxel')
        #stack the latents
        latent_map_stacked = self.stack_latents(centers_frame_occupied, voxel_grid_temp, encode_empty = encode_empty)
        # self.print_timing('stack latent')
        # latent_map_stacked_reshaped = latent_map_stacked.reshape(latent_map_stacked.shape[0], -1)
        # latent_map_stacked_reshaped_sumed = latent_map_stacked_reshaped.sum(dim = 1)
        # unique_lines = torch.unique(latent_map_stacked_reshaped, dim = 0)
        
        # print(f'fuse_inputs latent_map_stacked, Unique lines: {unique_lines.shape[0]}, Total lines: {latent_map_stacked_reshaped.shape[0]}')
        
        latent_map_stacked_merged = self.merge_latent_map(latent_map_stacked)
        # self.print_timing('merge latent')
        # centers_frame_occupied_list = [centers_frame_occupied[i] for i in range(len(centers_frame_occupied))]
        # latent_map_stacked_merged_list = [latent_map_stacked_merged[i] for i in range(len(latent_map_stacked_merged))]
        
        for i in range(len(centers_frame_occupied)):
            self.voxel_grid.add_voxel(centers_frame_occupied[i], latent_map_stacked_merged[i], overwrite = True)
        # self.print_timing('add voxel')
        # for center, points, encoded_latent in zip(centers_frame_occupied, inputs_frame_distributed_occupied, latent_map_stacked_merged):
        #     self.voxel_grid.add_voxel(center, points, encoded_latent, overwrite = True)
        # self.print_timing('add voxel')
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
        
        return p_stacked, latents, centers, occ, mask_frame
    
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
        error = 0.1

        mask_query = (torch.norm(centers_query.unsqueeze(1) - centers_frame_occupied, dim=2) <= error).any(dim=1)
        mask_frame = (torch.norm(centers_frame_occupied.unsqueeze(1) - centers_query, dim=2) <= error).any(dim=1)
        # print(f'{mask_query.sum()}, {mask_frame.sum()}')
        
        p_query_dict = {}
        
        for p_query, center in zip(p_query_distributed[mask_query], centers_query[mask_query]):
            hash = self.voxel_grid.compute_hash(center)
            p_query_dict[hash] = p_query
        
        return p_query_dict, mask_query, mask_frame
        
    def merge_latent_map(self, latent_map):
        n_samples = latent_map.shape[0]
        batch_size = 5
        n_batches = int(math.ceil(n_samples / batch_size))
        
        merged_latent_map = None
        
        for i in range(n_batches):
            # Determine the start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
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
        
        empty_inputs = self.get_empty_inputs(center, self.query_crop_size, n_max_points = 2048)
        occ = torch.zeros(*empty_inputs.shape[0:2], 1).to(self.device)
        empty_inputs = torch.cat((empty_inputs, occ), axis = -1)
        empty_latent_code = self.encode_distributed(empty_inputs, center)

        return empty_latent_code.squeeze(0)
    
    def encode_inputs_frame(self, inputs_frame, encode_empty = True):
        
        inputs_frame_distributed_occupied, centers_frame_occupied = self.get_distributed_inputs(inputs_frame, self.n_max_points)
                
        # centers_neighbours = self.compute_neighbours_and_bounds(centers_frame_occupied)
        # centers_frame_empty = self.get_empty_neighbours_centers(centers_neighbours, centers_frame_occupied) #does not contains occupied centers
        
        # centers_frame = torch.cat((centers_frame_occupied, centers_frame_empty), axis = 0) #contains occupied + empty neighbours
        
        # if encode_empty:
        #     self.empty_latent_code = self.get_empty_latent_representation()

        latents_frame_occupied = self.encode_distributed(inputs_frame_distributed_occupied, centers_frame_occupied)
                    
        # expanded_shape = (len(centers_neighbours),) + self.empty_latent_code.shape
        # latents_frame_empty = self.empty_latent_code.expand(expanded_shape)
        
        # latents_frame = torch.cat((latents_frame_occupied, latents_frame_empty), axis = 0)
        return latents_frame_occupied, inputs_frame_distributed_occupied, centers_frame_occupied
    
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

    def compute_vol_bound(self, inputs, padding = False):
        # inputs must have shape (n_points, 3)
        assert inputs.shape[1] == 3 and inputs.shape[0] > 0 and len(inputs.shape) == 2

        vol_bound = {}

        lb_p = torch.min(inputs, dim=0).values - torch.tensor([0.01, 0.01, 0.01], device=inputs.device)
        ub_p = torch.max(inputs, dim=0).values
        
        # print(lb_p, ub_p)

        lb = torch.round((lb_p - lb_p % self.query_crop_size) * 1e6) / 1e6
        ub = torch.round((((ub_p - ub_p % self.query_crop_size) / self.query_crop_size) + 1) * self.query_crop_size * 1e6) / 1e6

        if padding:
            lb -= self.query_crop_size
            ub += self.query_crop_size
        
        lb_query = torch.stack(torch.meshgrid(
            torch.arange(lb[0], ub[0] - 0.01, self.query_crop_size, device='cuda:0'),
            torch.arange(lb[1], ub[1] - 0.01, self.query_crop_size, device='cuda:0'),
            torch.arange(lb[2], ub[2] - 0.01, self.query_crop_size, device='cuda:0'),
        ), dim=-1).reshape(-1, 3)

        ub_query = lb_query + self.query_crop_size
        centers = (lb_query + ub_query) / 2

        # Number of crops alongside x, y, z axis
        vol_bound['axis_n_crop'] = torch.ceil((ub - lb - 0.01) / self.query_crop_size).int()

        # Total number of crops
        num_crop = torch.prod(vol_bound['axis_n_crop']).item()
        vol_bound['n_crop'] = num_crop
        vol_bound['input_vol'] = self.get_grid_from_centers(centers, self.input_crop_size)
        vol_bound['query_vol'] = self.get_grid_from_centers(centers, self.query_crop_size)
        vol_bound['lb'] = lb
        vol_bound['ub'] = ub

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
    
    def get_elevation_mask(self, inputs_frame_occupied):
        elevation_mask = torch.rand(len(inputs_frame_occupied), device=self.device) < 0.5
        # elevation_mask = torch.zeros(len(inputs_frame_occupied)).bool()
        # Mask for occupied points
        occupied_mask = inputs_frame_occupied[:, :, -1] == 1

        # Compute bounding boxes for all frames
        bb_min = torch.min(torch.where(occupied_mask.unsqueeze(-1), inputs_frame_occupied[:, :, :3], float('inf')), dim=1)[0]
        bb_max = torch.max(torch.where(occupied_mask.unsqueeze(-1), inputs_frame_occupied[:, :, :3], float('-inf')), dim=1)[0]
        
        # Compute y_diff
        y_diff = bb_max[:, 1] - bb_min[:, 1]
        
        # print(y_diff)
        # Update elevation_mask based on y_diff
        elevation_mask = elevation_mask | (y_diff > 0.1) #max elevation is self.query_crop_size[1]
        
        # Count the number of flat and elevated frames
        n_flat = (~elevation_mask).sum()
        n_elevated = elevation_mask.sum()
        
        
        return elevation_mask
    def visualize_weights(self, weights, p_query, inputs_distributed):
        """
        Visualize point cloud `p_query` with colors based on `weights`.

        Parameters:
        weights (torch.Tensor): Weights for each point in `p_query`, shape (n_points,)
        p_query (torch.Tensor): Point cloud data, shape (n_points, 3)

        Returns:
        None
        """
        import matplotlib.pyplot as plt
        import open3d as o3d
        # Convert torch tensors to numpy arrays
        inputs_distributed_np = inputs_distributed.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        p_query_np = p_query[..., :3].detach().cpu().numpy()
        
        n_skip = 20
        geos = []
        for i in range(p_query_np.shape[0]):

            # Create Open3D PointCloud
            pcd_query = o3d.geometry.PointCloud()
            pcd_query.points = o3d.utility.Vector3dVector(p_query_np[i][::n_skip])
            
            pcd_in = o3d.geometry.PointCloud()
            inputs_points = inputs_distributed_np[i]
            pcd_in.points = o3d.utility.Vector3dVector(inputs_points[inputs_points[:, 3] == 1][:, :3])
            pcd_in.paint_uniform_color([.0, 1., 0.])

            # Normalize weights to [0, 1] for colormap
            colors = plt.cm.jet(weights_np[i])[::n_skip]

            # Assign colors to PointCloud
            pcd_query.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Use RGB values from colormap
            # geos.append(pcd_query)
            geos.append(pcd_in)
            
            # o3d.visualization.draw_geometries([pcd_query, pcd_in])
            # Visualize using Open3D
        o3d.visualization.draw_geometries(geos)
        
    def visualize_logits(self, logits_sampled, p_query, weights = None, inputs_distributed=None, force_viz = False):

        geos = []
        
        current_dir = os.getcwd()
            
        file_path = f'configs/simultaneous/train_simultaneous_{self.location}.yaml'
        # file_path = '/home/robin/Dev/MasterThesis/GithubRepos/master_thesis/neuralblox/configs/fusion/train_fusion_home.yaml'

        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        except:
            return
            
        if not(force_viz or config['visualization']):
            return

        import open3d as o3d
        
        if weights is not None:
            self.visualize_weights(weights, p_query, inputs_distributed)
        
        p_stacked = p_query[..., :3]
        
        p_full = p_stacked.detach().cpu().numpy().reshape(-1, 3)

        occ_sampled = logits_sampled.detach().cpu().numpy()

        values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
        
        values_sampled = values_sampled.reshape(-1)

        threshold = 0.1

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
    
    def compute_mask_occupied(self, centers_grid, centers_occupied):
        centers_grid_expanded = centers_grid.unsqueeze(1)  
        centers_occupied_full_expanded = centers_occupied.unsqueeze(0)  
        
        error = 0.1

        matches = (torch.norm(centers_grid_expanded - centers_occupied_full_expanded, dim=2) <= error)

        mask_centers_grid_occupied = matches.any(dim=1)
        
        return mask_centers_grid_occupied
    
    def centers_to_grid_indexes(self, centers, lb):
        centers_shifted = torch.round((centers - (lb + self.query_crop_size / 2)) / self.query_crop_size * 10e4) / 10e4

        return centers_shifted
    def populate_grid_latent_cold_start(self, centers_frame_occupied, voxel_grid_occ, encode_empty = True): 
        vol_bounds, centers_grid = self.compute_vol_bound(centers_frame_occupied, padding = True)
        n_x, n_y, n_z = vol_bounds['axis_n_crop']

        centers_occupied_full = torch.stack(list(voxel_grid_occ.centers_table.values()))

        mask_centers_grid_occupied = self.compute_mask_occupied(centers_grid, centers_occupied_full)
        
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
        centers_grid_shifted = self.centers_to_grid_indexes(centers_grid, lb)
        
        # path = '/home/roberson/MasterThesis/master_thesis/Playground/Training/Sequential_training/test_data'
        # torch.save(centers_grid_shifted, path + '/centers_grid_shifted.pt')
        # indexes_used = []
        for i in range(centers_grid.shape[0]):
            occupancy = mask_centers_grid_occupied[i]
            
            if occupancy:
                index_x = centers_grid_shifted[i, 0].int().item()
                index_y = centers_grid_shifted[i, 1].int().item()
                index_z = centers_grid_shifted[i, 2].int().item()
                # indexes_used.append((index_x, index_y, index_z))
                latent = voxel_grid_occ.get_latent(centers_grid[i])
                grid_latents[index_x, index_y, index_z] = latent
                # grid_latents[centers_grid_shifted[i, 0], centers_grid_shifted[i, 1], centers_grid_shifted[i, 2], :] = voxel_grid_occ.get_latent(centers_grid[i])
        # grid_latents_reshaped = grid_latents.reshape(centers_grid.shape[0], -1)
        # unique_lines = torch.unique(grid_latents_reshaped, dim=0)
        # print(f'populate_grid_latent_cold_start Unique lines: {unique_lines.shape[0]}, n_centers_occ {centers_frame_occupied.shape[0]}')
        return grid_latents, vol_bounds
    
    def populate_grid_latent(self, centers_frame_occupied, voxel_grid_occ_frame, voxel_grid_occ_existing, encode_empty = True): 
        # centers_frame_occupied + centers_frame_unocc = centers_frame_grid
        vol_bounds, centers_grid = self.compute_vol_bound(centers_frame_occupied, padding = True)
        
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
        centers_grid_shifted = self.centers_to_grid_indexes(centers_grid, lb)
        
        centers_occupied_frame_full = torch.stack(list(voxel_grid_occ_frame.centers_table.values()))
        centers_occupied_existing_full = torch.stack(list(voxel_grid_occ_existing.centers_table.values()))

        mask_centers_occupied_frame_occupied = self.compute_mask_occupied(centers_grid, centers_occupied_frame_full)
        mask_centers_occupied_existing_occupied = self.compute_mask_occupied(centers_grid, centers_occupied_existing_full)

        indexes_used = []
        for i in range(centers_grid_shifted.shape[0]):
            occupancy_frame = mask_centers_occupied_frame_occupied[i]
            occupancy_existing = mask_centers_occupied_existing_occupied[i]
            
            if occupancy_frame or occupancy_existing:
                index_x = centers_grid_shifted[i, 0].int().item()
                index_y = centers_grid_shifted[i, 1].int().item()
                index_z = centers_grid_shifted[i, 2].int().item()
                indexes_used.append((index_x, index_y, index_z))
                if occupancy_frame:
                    latent = voxel_grid_occ_frame.get_latent(centers_grid[i])
                    # if latent.shape != self.empty_latent_code.shape:
                    #     print(f'latent.shape: {latent.shape}, empty_latent_code.shape: {self.empty_latent_code.shape}')
                    grid_latents_frame[index_x, index_y, index_z, :] = latent
                    
                if occupancy_existing:
                    latent = voxel_grid_occ_existing.get_latent(centers_grid[i])
                    grid_latents_existing[index_x, index_y, index_z, :] = latent

        #complete the empty voxels by crossing 
        mask_complete_frame = (~mask_centers_occupied_frame_occupied) & mask_centers_occupied_existing_occupied #returns True where frame is empty and existing is occupied
        mask_complete_existing = (~mask_centers_occupied_existing_occupied) & mask_centers_occupied_frame_occupied #returns True where existing is empty and frame is occupied
        
        grid_latents_frame.view(-1, *self.empty_latent_code.shape)[mask_complete_frame] = grid_latents_existing.view(-1, *self.empty_latent_code.shape)[mask_complete_frame]
        grid_latents_existing.view(-1, *self.empty_latent_code.shape)[mask_complete_existing] = grid_latents_frame.view(-1, *self.empty_latent_code.shape)[mask_complete_existing]
        
        return grid_latents_frame, grid_latents_existing, vol_bounds
    
    def create_stacked_map(self, grid_latents, centers_occupied, vol_bounds):
        # self.print_timing('create stacked map start')
        # print('')
        lb = vol_bounds['lb']
        centers_occupied_shifted = self.centers_to_grid_indexes(centers_occupied, lb)
        
        shifts = torch.Tensor([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]).to(self.device)
        # self.print_timing('create shifts')
        c, h, w, d = self.voxel_grid.latent_shape
        # self.print_timing('create indices')
        latent_map_stacked = torch.zeros(len(centers_occupied_shifted), c, 3 * h, 3 * w, 3 * d).to(self.device)
        
        # self.print_timing('allocate latent map')
        
        # Calculate all possible shifts for the centers
        centers_occupied_shifted = centers_occupied_shifted.unsqueeze(1) + shifts.unsqueeze(0)  # Shape: (num_centers, 27, 3)
        centers_occupied_shifted = centers_occupied_shifted.int()
        # Clip the indices to ensure they are within the bounds of the grid_latents
        # min_bound = torch.tensor([0, 0, 0], device=self.device)
        # max_bound = torch.tensor(grid_latents.shape[:3], device=self.device) - 1
        # centers_occupied_shifted = torch.clamp(centers_occupied_shifted, min_bound, max_bound.unsqueeze(0).unsqueeze(1))

        # Gather the latents for all the shifted centers    
        #save centers_occupied_shifted
        # path = '/home/roberson/MasterThesis/master_thesis/Playground/Training/Sequential_training/test_data'
        # torch.save(centers_occupied_shifted, path + '/centers_occupied_shifted.pt')
        # torch.save(centers_occupied, path + '/centers_occupied.pt')
        # centers_occupied_shifted = (torch.round(centers_occupied_shifted*1e4)/1e4)
        
        latent_temp = grid_latents[centers_occupied_shifted[..., 0], centers_occupied_shifted[..., 1], centers_occupied_shifted[..., 2]]
        # self.print_timing('gather latents')
        # latent_temp_reshaped = latent_temp.reshape(latent_temp.shape[0], -1)
        # latent_temp_sumed = latent_temp_reshaped.sum(dim = 1)
        # unique_lines = torch.unique(latent_temp_reshaped, dim = 0)

        # print(f'create_stacked_map, Unique lines: {unique_lines.shape[0]}, centers_occupied_shifted: {len(centers_occupied_shifted)}')
        
        # Reshape the gathered latents and assign to latent_map_stacked
        latent_map_stacked = latent_temp.reshape(len(centers_occupied_shifted), c, 3 * h, 3 * w, 3 * d)
        # self.print_timing('reshape stacked map')
        return latent_map_stacked

    def stack_latents_cold_start(self, centers_frame_occupied, encode_empty = True):
        grid_latents, vol_bounds = self.populate_grid_latent_cold_start(centers_frame_occupied, self.voxel_grid, encode_empty = encode_empty)
        
        # self.print_timing('populate grid latent')
        latent_map_stacked = self.create_stacked_map(grid_latents, centers_frame_occupied, vol_bounds)
        latent_map_stacked_reshaped = latent_map_stacked.reshape(latent_map_stacked.shape[0], -1)
        
        latent_map_stacked_sumed = latent_map_stacked_reshaped.sum(dim = 1)
        # unique_lines = torch.unique(latent_map_stacked_reshaped, dim = 0)
        
        # print(f'stack_latents_cold_start Unique lines: {unique_lines.shape[0]}, centers_frame_occupied: {len(centers_frame_occupied)}')
        # self.print_timing('create stacked map')
        return torch.cat((latent_map_stacked, latent_map_stacked), dim = 1)
    def stack_latents(self, centers_frame_occupied, voxel_grid_temp, encode_empty = True):
        # self.print_timing('stack latents start')
        grid_latents_frame, grid_latents_existing, vol_bounds = self.populate_grid_latent(centers_frame_occupied, self.voxel_grid, voxel_grid_temp, encode_empty = encode_empty)
        # self.print_timing('populate grid latent')
        latent_map_stacked_frame = self.create_stacked_map(grid_latents_frame, centers_frame_occupied, vol_bounds)
        # self.print_timing('create stacked map frame')
        latent_map_stacked_existing = self.create_stacked_map(grid_latents_existing, centers_frame_occupied, vol_bounds)
        # self.print_timing('create stacked map existing')
        # self.print_timing('create stacked map')
        return torch.cat((latent_map_stacked_frame, latent_map_stacked_existing), dim = 1)