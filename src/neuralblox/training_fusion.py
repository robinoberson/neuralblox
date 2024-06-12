import os
import torch
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


torch.manual_seed(42)

class Trainer(BaseTrainer):
    ''' Trainer object for fusion network.

    Args:
        model (nn.Module): Convolutional Occupancy Network model
        model_merge (nn.Module): fusion network
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
        query_n (int): number of query points per voxel
        hdim (int): hidden dimension
        depth (int): U-Net depth (3 -> hdim 32 to 128)

    '''


    def __init__(self, model, model_merge, optimizer, cfg, input_crop_size = 1.6, query_crop_size = 1.0, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False, query_n = 8192, unet_hdim = 32, unet_depth = 2, grid_reso = 24, limited_gpu = False):
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
        self.eval_sample = eval_sample
        self.max_crop_with_change = None
        self.query_n = query_n
        self.hdim = unet_hdim
        self.factor = 2**unet_depth
        self.reso = None
        self.unet = None
        self.limited_gpu = limited_gpu
        self.reso = grid_reso
        self.iteration = 0
        self.cfg = cfg
        self.rand_points_inputs = (torch.load('pretrained_models/fusion/empty_inputs.pt').to(self.device) + 0.56) / 1.12
        # self.rand_points_inputs = torch.rand(20000, 3) 
        
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
        pass
        # t1 = time.time()
        # print(f'Time elapsed, {self.timing_counter}: {t1 - self.t0:.3f}, {operation}')

        # self.t0 = time.time()
        # self.timing_counter += 1
        
    def train_sequence_window(self, data_batch):
        self.t0 = time.time()
        self.timing_counter = 0
                
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.train()
        self.model_merge.train()
        self.optimizer.zero_grad()
        
        self.print_timing('Initialization')
        
        p_in, p_query = self.get_inputs_from_batch(data_batch)
        #Step 1 get the bounding box of the scene
        self.vol_bound_all = self.get_crop_bound(p_in.view(-1, 4), self.input_crop_size, self.query_crop_size)

        self.print_timing('Get crop bounds')
        
        inputs_distributed = self.distribute_inputs(p_in.unsqueeze(1), self.vol_bound_all)

        self.print_timing('Distribute inputs')
        # Encode latents 
        latent_map_sampled, _ = self.encode_latent_map(inputs_distributed, torch.tensor(self.vol_bound_all['input_vol'], device = self.device))
        
        self.print_timing('Encode latents')
          
        # Stack latents 
        latent_map_sampled_stacked = self.stack_latents_safe(latent_map_sampled, self.vol_bound_all)
        
        self.print_timing('Stack latents')
        # Merge latents
        latent_map_sampled_merged = self.merge_latent_map(latent_map_sampled_stacked) 
        # n_crops = latent_map_sampled_stacked.shape[0] * latent_map_sampled_stacked.shape[1] * latent_map_sampled_stacked.shape[2]
        # latent_map_sampled_merged = latent_map_sampled_stacked.reshape(n_crops, *latent_map_sampled_stacked.shape[-4:])[:,:128, 6:12, 6:12, 6:12]
        self.print_timing('Merge latents')
        
        if self.limited_gpu: torch.cuda.empty_cache()
        # Compute gt latent
        
        # return latent_map_sampled_merged, p_query

        p_query_distributed = self.distribute_query(p_query, self.vol_bound_all)
        
        self.print_timing('Distribute query')
        
        logits_sampled = self.get_logits(latent_map_sampled_merged, p_query_distributed, torch.tensor(self.vol_bound_all['input_vol'], device = self.device), remove_padding=True)
        occ = p_query_distributed[..., -1].squeeze(0)
        
        self.print_timing('Get logits')
        
        loss = F.binary_cross_entropy_with_logits(logits_sampled, occ, reduction='none').sum(-1).mean()
        # compute cost
        # print(f'Model')
        # self.check_model_device(self.model)
        # print(f'Model merge')
        # self.check_model_device(self.model_merge)

        loss.backward()
        self.optimizer.step()

        self.print_timing('Backward')
        self.visualize_logits(logits_sampled, p_query_distributed, inputs_distributed)

        self.iteration += 1
        self.print_timing('Finish')
        return loss
    
    def validate_sequence_window(self, data_batch):
        self.t0 = time.time()
        self.timing_counter = 0
        
        n_inputs = 2
        
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.eval()
        self.model_merge.eval()
                
        p_in, p_query = self.get_inputs_from_batch(data_batch)
        p_full = torch.cat((p_in.view(-1, 4), p_query.view(-1, 4)), dim=0)
        #Step 1 get the bounding box of the scene
        self.vol_bound_all = self.get_crop_bound(p_in.view(-1, 4), self.input_crop_size, self.query_crop_size)

        
        inputs_distributed = self.distribute_inputs(p_in.unsqueeze(1), self.vol_bound_all)
        # Encode latents 
        latent_map_sampled, _ = self.encode_latent_map(inputs_distributed, torch.tensor(self.vol_bound_all['input_vol'], device = self.device))
                  
        # Stack latents 
        latent_map_sampled_stacked = self.stack_latents_safe(latent_map_sampled, self.vol_bound_all)
        # Merge latents
        latent_map_sampled_merged = self.merge_latent_map(latent_map_sampled_stacked) 
        # n_crops = latent_map_sampled_stacked.shape[0] * latent_map_sampled_stacked.shape[1] * latent_map_sampled_stacked.shape[2]
        # latent_map_sampled_merged = latent_map_sampled_stacked.reshape(n_crops, *latent_map_sampled_stacked.shape[-4:])[:,:128, 6:12, 6:12, 6:12]
        
        if self.limited_gpu: torch.cuda.empty_cache()        

        p_query_distributed = self.distribute_query(p_query, self.vol_bound_all)
        
        
        logits_sampled = self.get_logits(latent_map_sampled_merged, p_query_distributed, torch.tensor(self.vol_bound_all['input_vol'], device = self.device), remove_padding=True)
        occ = p_query_distributed[..., -1].squeeze(0)
                
        loss = F.binary_cross_entropy_with_logits(logits_sampled, occ, reduction='none').sum(-1).mean()
        return loss

    def get_elevated_voxels(self, inputs):
        inputs_reshaped = inputs.reshape(-1, *inputs.shape[2:]).detach().cpu().numpy()
        elevation_voxels = torch.zeros(inputs_reshaped.shape[0], dtype=torch.bool)

        n_crit = 0
        for i in range(inputs_reshaped.shape[0]):
            points = inputs_reshaped[i][inputs_reshaped[i, ..., -1] == 1]
            if points.shape[0] > 1:
                bb_min = np.min(points, axis=0)
                bb_max = np.max(points, axis=0)
                y_diff = bb_max[1] - bb_min[1]
                if y_diff > 0.1:
                    n_crit += 1 
                    elevation_voxels[i] = True 

        return elevation_voxels
    
    def save_data_visualization(self, data_batch):
        n_inputs = 2
        
        if self.limited_gpu: torch.cuda.empty_cache()
        
        self.model.eval()
        self.model_merge.eval()
        
        with torch.no_grad():
            p_in, p_query = self.get_inputs_from_batch(data_batch)
            self.vol_bound_all = self.get_crop_bound(p_in.view(-1, 4), self.input_crop_size, self.query_crop_size)
            inputs_distributed = self.distribute_inputs(p_in.unsqueeze(1), self.vol_bound_all)
            latent_map_sampled, _ = self.encode_latent_map(inputs_distributed, torch.tensor(self.vol_bound_all['input_vol'], device = self.device))
            latent_map_sampled_stacked = self.stack_latents_safe(latent_map_sampled, self.vol_bound_all)
            latent_map_sampled_merged = self.merge_latent_map(latent_map_sampled_stacked) 
            p_query_distributed = self.distribute_query(p_query, self.vol_bound_all)
            logits_sampled = self.get_logits(latent_map_sampled_merged, p_query_distributed, torch.tensor(self.vol_bound_all['input_vol'], device = self.device), remove_padding=True)
            
            n_crops = latent_map_sampled_stacked.shape[0] * latent_map_sampled_stacked.shape[1] * latent_map_sampled_stacked.shape[2]
            latent_map_sampled_int = latent_map_sampled_stacked.reshape(n_crops, *latent_map_sampled_stacked.shape[-4:])[:,:128, 6:12, 6:12, 6:12]
            
        return logits_sampled, p_query_distributed, inputs_distributed, latent_map_sampled_int
    
    def visualize_logits(self, logits_sampled, p_query, inputs_distributed=None, force_viz = False):
        geos = []
        
        current_dir = os.getcwd()

            
        file_path = f'/home/roberson/MasterThesis/master_thesis/neuralblox/configs/simultaneous/train_simultaneous_{self.location}.yaml'
        # file_path = '/home/robin/Dev/MasterThesis/GithubRepos/master_thesis/neuralblox/configs/fusion/train_fusion_home.yaml'

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
        # colors[values_sampled == 0] = [0, 0, 0.5]
        # colors[both_occ == 1] = [0, 0, 1] # green
        
        mask = np.any(colors != [0, 0, 0], axis=1)
        # print(mask.shape, values_gt.shape, values_sampled.shape, colors.shape)
        if inputs_distributed is not None:
            points_second = inputs_distributed
            pcd_inputs = o3d.geometry.PointCloud()
            inputs_reshaped = inputs_distributed.reshape(-1, 4).detach().cpu().numpy()
            pcd_inputs.points = o3d.utility.Vector3dVector(inputs_reshaped[inputs_reshaped[..., -1] == 1, :3])
            pcd_inputs.paint_uniform_color([1., 0.5, 0]) # blue
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

        # o3d.io.write_point_cloud("/media/roberson/T7/visualization/test.ply", pcd)
        # o3d.io.write_point_cloud("/media/roberson/T7/visualization/test_inputs.ply", pcd_inputs)
        
    def get_inputs_from_batch(self, batch):
        p_in_3D = batch.get('inputs').to(self.device).squeeze(0)
        p_in_occ = batch.get('inputs.occ').to(self.device).squeeze(0).unsqueeze(-1)
                
        p_query_3D = batch.get('points').to(self.device)
        p_query_occ = batch.get('points.occ').to(self.device).unsqueeze(-1)
        
        p_in = torch.cat((p_in_3D, p_in_occ), dim=-1)
        p_query = torch.cat((p_query_3D, p_query_occ), dim=-1)
        
        return p_in, p_query
    
    def remove_padding_single_dim(self, tensor):
        other_dims = tensor.shape[1:]
        H, W, D = self.vol_bound_all['axis_n_crop'][0], self.vol_bound_all['axis_n_crop'][1], self.vol_bound_all['axis_n_crop'][2]
        tensor = tensor.reshape(H, W, D, *other_dims)
        return tensor[1:-1, 1:-1, 1:-1]
    
    def remove_padding_single_dim_batch(self, tensor):
        other_dims = tensor.shape[2:]
        n_batch = tensor.shape[0]
        H, W, D = self.vol_bound_all['axis_n_crop'][0], self.vol_bound_all['axis_n_crop'][1], self.vol_bound_all['axis_n_crop'][2]
        tensor = tensor.reshape(n_batch, H, W, D, *other_dims)
        return tensor[:, 1:-1, 1:-1, 1:-1]
    
    def remove_padding(self, tensor):
        is_latent = False
        H, W, D = self.vol_bound_all['axis_n_crop'][0], self.vol_bound_all['axis_n_crop'][1], self.vol_bound_all['axis_n_crop'][2]

        if len(tensor.shape) == 7:
            H, W, D, c, h, w, d = tensor.shape
            is_latent = True

        elif len(tensor.shape) == 5:
            _, c, h, w, d = tensor.shape
            is_latent = True

        elif len(tensor.shape) == 4:
            n_in, H, W, D = tensor.shape

        elif len(tensor.shape) == 2:
            n_in = tensor.shape[0]

        else:
            raise ValueError
        
        if is_latent:
            tensor = tensor.reshape(H, W, D, c, h, w, d)
            return tensor[1:-1, 1:-1, 1:-1]
        else:
            return tensor.reshape(n_in, H, W, D)
                
    def get_logits(self, latent_map_full, p_stacked, vol_bound, remove_padding):
        n_inputs, n_crop, n_points, _ = p_stacked.shape
        
        n_crops_total = latent_map_full.shape[0]
        
        if n_crops_total != n_inputs * n_crop:
            raise ValueError
    
        if remove_padding:
            vol_bound = self.remove_padding_single_dim(vol_bound).reshape(-1, *vol_bound.shape[-2:])
        vol_bound = vol_bound.unsqueeze(0).repeat(n_inputs, 1, 1, 1)
        
        p_stacked = p_stacked.reshape(n_inputs * n_crop, n_points, 4)
        vol_bound = vol_bound.reshape(n_inputs * n_crop, 2, 3)
        
        p_n_stacked = normalize_coord(p_stacked, vol_bound)
        
        fea_type = 'grid'
        if self.limited_gpu:
            n_batch_max = 10
        else:
            n_batch_max = 1000

        n_batch = int(np.ceil(n_crops_total / n_batch_max))

        logits_stacked = None  # Initialize logits directly

        for i in range(n_batch):
            start = i * n_batch_max
            end = min((i + 1) * n_batch_max, n_crops_total)

            p_stacked_batch = p_stacked[start:end]
            p_n_stacked_batch = p_n_stacked[start:end]
            latent_map_full_batch = latent_map_full[start:end]

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
    
    def get_query_points(self, input_size, query_size_voxel = 1.0, unpadding = True):
        # Pay attention to wording #TODO change this 
        # query_crop_size is used as the voxel size for distributing the inputs 
        # query_size_voxel is used to determine the size of the query on a voxel latent 
        random_points = torch.rand(1, self.query_n, 3).to(self.device) - 0.5

        if unpadding: n_crops = (self.vol_bound_all['axis_n_crop'][0] - 2) * (self.vol_bound_all['axis_n_crop'][1] - 2) * (self.vol_bound_all['axis_n_crop'][2] - 2)
        else: n_crops = (self.vol_bound_all['axis_n_crop'][0]) * (self.vol_bound_all['axis_n_crop'][1]) * (self.vol_bound_all['axis_n_crop'][2])
        
        random_points_stacked = random_points.repeat(n_crops, 1, 1)

        centers = torch.from_numpy((self.vol_bound_all['input_vol'][:,1] - self.vol_bound_all['input_vol'][:,0])/2.0 + self.vol_bound_all['input_vol'][:,0]).to(self.device) #shape [n_crops, 3]
        if unpadding: centers = self.remove_padding_single_dim(centers).reshape(-1,3)
        centers = centers.unsqueeze(1)
        
        p = ((random_points_stacked.clone()) * input_size + centers)
        # p_n = ((random_points_stacked.clone() - 0.5) * query_size_voxel / input_size + 0.5)[occupied_voxels]
        p_n = random_points_stacked + 0.5

        return p, p_n
    
    def get_latent_gt(self, points_gt):
        inputs_distributed_gt = self.distribute_inputs(points_gt.unsqueeze(1), self.vol_bound_all, remove_padding = True)
        
        latent_map_gt, _ = self.encode_latent_map(inputs_distributed_gt, torch.tensor(self.vol_bound_all['input_vol'], device = self.device), remove_padding = True)
        # print(latent_map_gt.shape)
        return latent_map_gt.squeeze(0), inputs_distributed_gt
    def merge_latent_map(self, latent_map):
        H, W, D, c, h, w, d = latent_map.size()
        latent_map = latent_map.reshape(-1, c, h, w, d)
        
        fea_dict = {}
        fea_dict['latent'] = latent_map
        
        latent_map = self.model_merge(fea_dict)
        return latent_map


    def stack_latents(self, latent_map, vol_bound):
        n_in, n_crops, c, h, w, d = latent_map.shape
        H, W, D = vol_bound['axis_n_crop'][0], vol_bound['axis_n_crop'][1], vol_bound['axis_n_crop'][2]

        latent_map = torch.reshape(latent_map, (n_in, H, W, D, c, h, w, d))
        
        latent_map_neighbored = torch.zeros(H-2, W-2, D-2, c*n_in, h*3, w*3, d*3).to(self.device) #take padding off

        #TODO: Rework with unfolding
        # Stack neighbors and take padding in account
        for idx_n_in in range(n_in):
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    for k in range(1, D - 1):
                        stacked_vector = latent_map[idx_n_in, i-1:i+2, j-1:j+2, k-1:k+2].reshape(c, h*3, w*3, d*3)
                        latent_map_neighbored[i-1, j-1, k-1, (idx_n_in)*c:(idx_n_in+1)*c] = stacked_vector
                        
        return latent_map_neighbored

    def stack_latents_safe(self, latent_map, vol_bound): #TODO: make faster
        n_in, n_crops, c, h, w, d = latent_map.shape
        H, W, D = vol_bound['axis_n_crop'][0], vol_bound['axis_n_crop'][1], vol_bound['axis_n_crop'][2]

        latent_map = torch.reshape(latent_map, (n_in, H, W, D, c, h, w, d))

        latent_map_neighbored = torch.zeros(H-2, W-2, D-2, c*n_in, h*3, w*3, d*3).to(self.device) #take padding off

        for idx_n_in in range(n_in):
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    for k in range(1, D - 1):
                        center = torch.tensor([i, j, k]).to(self.device)
                        for l in range(3):
                            for m in range(3):
                                for n in range(3):
                                    index = torch.tensor([l-1, m-1, n-1]).to(self.device) + center
                                    latent_map_neighbored[i-1, j-1, k-1, (idx_n_in)*c:(idx_n_in+1)*c, l*h:(l+1)*h, m*w:(m+1)*w, n*d:(n+1)*d] = latent_map[idx_n_in, index[0], index[1], index[2]]
        
        return latent_map_neighbored
    
    def encode_latent_map(self, inputs_distributed, vol_bound, remove_padding=False):
        n_inputs, n_crop, n_points, _ = inputs_distributed.shape
        d = self.reso // self.factor

        # Initialize latent_map to accumulate latent maps
        latent_map = None  

        if remove_padding:
            vol_bound = self.remove_padding_single_dim(vol_bound).reshape(-1, *vol_bound.shape[-2:])
        vol_bound = vol_bound.unsqueeze(0).repeat(n_inputs, 1, 1, 1)
        
        inputs_distributed = inputs_distributed.reshape(n_inputs * n_crop, n_points, 4)
        vol_bound = vol_bound.reshape(n_inputs * n_crop, 2, 3)

        if self.limited_gpu:
            n_voxels_max = 40
        else:
            n_voxels_max = 1000
        
        n_batch_voxels = int(np.ceil(inputs_distributed.shape[0] / n_voxels_max))
        
        for i in range(n_batch_voxels): 
            
            if self.limited_gpu: torch.cuda.empty_cache()
            start = i * n_voxels_max
            end = min((i + 1) * n_voxels_max, inputs_distributed.shape[0])
            inputs_distributed_batch = inputs_distributed[start:end]
            inputs_distributed_batch_3D = inputs_distributed_batch[..., :3]
            vol_bound_batch = vol_bound[start:end]

            kwargs = {}
            fea_type = 'grid'
            index = {}
            ind = coord2index(inputs_distributed_batch, vol_bound_batch, reso=self.reso, plane=fea_type)
            index[fea_type] = ind
            input_cur_batch = add_key(inputs_distributed_batch_3D, index, 'points', 'index', device=self.device)
            
            if self.unet is None:
                fea, self.unet = self.model.encode_inputs(input_cur_batch)
            else:
                fea, _ = self.model.encode_inputs(input_cur_batch)
                
            _, latent_map_batch, self.features_shapes = self.unet(fea, True)
            # if self.limited_gpu: latent_map_batch = latent_map_batch.to('cpu')
            if self.limited_gpu: 
                del fea
                torch.cuda.empty_cache()

            if latent_map is None:
                latent_map = latent_map_batch  # Initialize latent_map with the first batch
                # latent_map_decoded = latent_map_batch_decoded
            else:
                latent_map = torch.cat((latent_map, latent_map_batch), dim=0)  # Concatenate latent maps


        latent_map_shape = latent_map.shape
        
        # latent_map_decoded_shape = latent_map_decoded.shape
        # return latent_map.reshape(n_inputs, n_crop, *latent_map_shape[1:]), latent_map_decoded.reshape(n_inputs, n_crop, *latent_map_decoded_shape[1:])
        return latent_map.reshape(n_inputs, n_crop, *latent_map_shape[1:]), None
    def distribute_query(self, query, vol_bound, remove_padding = True):
        distributed_inputs = query.clone()
        n_inputs = distributed_inputs.shape[0]
        n_crops = vol_bound['n_crop']
        
        if remove_padding: 
            H, W, D = self.vol_bound_all['axis_n_crop'][0], self.vol_bound_all['axis_n_crop'][1], self.vol_bound_all['axis_n_crop'][2]
            n_crops = (H-2) * (W-2) * (D-2)
        
        distributed_inputs = distributed_inputs.repeat(1, n_crops, 1, 1)

        # Convert vol_bound['input_vol'] to a torch tensor
        vol_bound_tensor = torch.tensor(vol_bound['input_vol']).to(self.device)
        if remove_padding: vol_bound_tensor = self.remove_padding_single_dim(vol_bound_tensor).reshape(-1, *vol_bound_tensor.shape[-2:])
        
        # Permute dimensions
        distributed_inputs = distributed_inputs.permute(1, 0, 2, 3) #from n_inputs, n_crop, n_points, 4 to n_crop, n_inputs, n_points, 4
        distributed_inputs = distributed_inputs.reshape(distributed_inputs.shape[0], -1, 4)

        # Create masks for each condition
        mask_1 = distributed_inputs[:, :, 0] < vol_bound_tensor[:, 0, 0].unsqueeze(1)
        mask_2 = distributed_inputs[:, :, 0] > vol_bound_tensor[:, 1, 0].unsqueeze(1)
        mask_3 = distributed_inputs[:, :, 1] < vol_bound_tensor[:, 0, 1].unsqueeze(1)
        mask_4 = distributed_inputs[:, :, 1] > vol_bound_tensor[:, 1, 1].unsqueeze(1)
        mask_5 = distributed_inputs[:, :, 2] < vol_bound_tensor[:, 0, 2].unsqueeze(1)
        mask_6 = distributed_inputs[:, :, 2] > vol_bound_tensor[:, 1, 2].unsqueeze(1)

        # Combine masks
        final_mask = mask_1 | mask_2 | mask_3 | mask_4 | mask_5 | mask_6

        distributed_inputs = torch.cat([distributed_inputs, (~final_mask).unsqueeze(-1).float()], dim=-1)
        
        # Reshape back to original shape
        distributed_inputs = distributed_inputs.reshape(n_crops, n_inputs, -1, 5)
        distributed_inputs = distributed_inputs.permute(1, 0, 2, 3)
        
        n_max = int(distributed_inputs[..., 4].sum(dim=2).max().item())
        # print(f'Distributing inputs with n_max = {n_max}')
        # Create a mask for selecting points with label 1
        indexes_keep = distributed_inputs[..., 4] == 1

        # Calculate the number of points to keep for each input and crop
        n_points = distributed_inputs[..., 4].sum(dim=2).int()

        # Create a mask for broadcasting
        mask = torch.arange(n_max).expand(n_inputs, n_crops, n_max).to(device=self.device) < n_points.unsqueeze(-1)

        bb_min = vol_bound_tensor[:, 0, :].unsqueeze(1).repeat(n_inputs, 1, 1) # Shape: (n_crops, 3, 1)
        bb_max = vol_bound_tensor[:, 1, :].unsqueeze(1).repeat(n_inputs, 1, 1)  # Shape: (n_crops, 3, 1)
        bb_size = bb_max - bb_min  # Shape: (n_crops, 3, 1)
        
        if self.rand_points_inputs.shape[0] < n_max:
            n_repeat = np.ceil(n_max / self.rand_points_inputs.shape[0])
            self.rand_points_inputs = self.rand_points_inputs.repeat(int(n_repeat), 1)
        random_points_sel = self.rand_points_inputs[:n_max]
        random_points = random_points_sel.repeat(n_inputs*n_crops,1, 1).to(device=self.device)
        random_points *= bb_size  # Scale points to fit inside each bounding box
        random_points += bb_min  # Translate points to be within each bounding box
        
        distributed_inputs_short = torch.zeros(n_inputs, n_crops, n_max, 4, device=self.device)
        distributed_inputs_short[:, :, :, :3] = random_points.reshape(n_inputs, n_crops, n_max, 3)
        
        # Assign values to the shortened tensor using advanced indexing and broadcasting
        distributed_inputs_short[mask] = distributed_inputs[indexes_keep][..., :4]

        return distributed_inputs_short

    def distribute_inputs(self, distributed_inputs_raw, vol_bound, remove_padding = False):
        # Clone the input tensor
        distributed_inputs = distributed_inputs_raw.clone()
        n_inputs = distributed_inputs.shape[0]
        n_crops = vol_bound['n_crop']
        
        if remove_padding: 
            H, W, D = self.vol_bound_all['axis_n_crop'][0], self.vol_bound_all['axis_n_crop'][1], self.vol_bound_all['axis_n_crop'][2]
            n_crops = (H-2) * (W-2) * (D-2)
        
        distributed_inputs = distributed_inputs.repeat(1, n_crops, 1, 1)

        # Convert vol_bound['input_vol'] to a torch tensor
        vol_bound_tensor = torch.tensor(vol_bound['input_vol']).to(self.device)
        if remove_padding: vol_bound_tensor = self.remove_padding_single_dim(vol_bound_tensor).reshape(-1, *vol_bound_tensor.shape[-2:])
        
        # Permute dimensions
        distributed_inputs = distributed_inputs.permute(1, 0, 2, 3) #from n_inputs, n_crop, n_points, 4 to n_crop, n_inputs, n_points, 4
        distributed_inputs = distributed_inputs.reshape(distributed_inputs.shape[0], -1, 4)

        # Create masks for each condition
        mask_1 = distributed_inputs[:, :, 0] < vol_bound_tensor[:, 0, 0].unsqueeze(1)
        mask_2 = distributed_inputs[:, :, 0] > vol_bound_tensor[:, 1, 0].unsqueeze(1)
        mask_3 = distributed_inputs[:, :, 1] < vol_bound_tensor[:, 0, 1].unsqueeze(1)
        mask_4 = distributed_inputs[:, :, 1] > vol_bound_tensor[:, 1, 1].unsqueeze(1)
        mask_5 = distributed_inputs[:, :, 2] < vol_bound_tensor[:, 0, 2].unsqueeze(1)
        mask_6 = distributed_inputs[:, :, 2] > vol_bound_tensor[:, 1, 2].unsqueeze(1)

        # Combine masks
        final_mask = mask_1 | mask_2 | mask_3 | mask_4 | mask_5 | mask_6

        # Set values to 0 where conditions are met
        distributed_inputs[:, :, 3][final_mask] = 0

        # Reshape back to original shape
        distributed_inputs = distributed_inputs.reshape(n_crops, n_inputs, -1, 4)
        distributed_inputs = distributed_inputs.permute(1, 0, 2, 3)
        
        # Calculate n_max
        n_max = int(distributed_inputs[..., 3].sum(dim=2).max().item())
        # print(f'Distributing inputs with n_max = {n_max}')
        # Create a mask for selecting points with label 1
        indexes_keep = distributed_inputs[..., 3] == 1

        # Calculate the number of points to keep for each input and crop
        n_points = distributed_inputs[..., 3].sum(dim=2).int()

        # Create a mask for broadcasting
        mask = torch.arange(n_max).expand(n_inputs, n_crops, n_max).to(device=self.device) < n_points.unsqueeze(-1)

        bb_min = vol_bound_tensor[:, 0, :].unsqueeze(1).repeat(n_inputs, 1, 1) # Shape: (n_crops, 3, 1)
        bb_max = vol_bound_tensor[:, 1, :].unsqueeze(1).repeat(n_inputs, 1, 1)  # Shape: (n_crops, 3, 1)
        bb_size = bb_max - bb_min  # Shape: (n_crops, 3, 1)
        
        if self.rand_points_inputs.shape[0] < n_max:
            n_repeat = np.ceil(n_max / self.rand_points_inputs.shape[0])
            self.rand_points_inputs = self.rand_points_inputs.repeat(int(n_repeat), 1)
        random_points_sel = self.rand_points_inputs[:n_max]
        random_points = random_points_sel.repeat(n_inputs*n_crops,1, 1).to(device=self.device)
        random_points *= bb_size  # Scale points to fit inside each bounding box
        random_points += bb_min  # Translate points to be within each bounding box
        
        distributed_inputs_short = torch.zeros(n_inputs, n_crops, n_max, 4, device=self.device)
        distributed_inputs_short[:, :, :, :3] = random_points.reshape(n_inputs, n_crops, n_max, 3)
        
        # Assign values to the shortened tensor using advanced indexing and broadcasting
        distributed_inputs_short[mask] = distributed_inputs[indexes_keep]

        return distributed_inputs_short

    def encode_crop(self, inputs, inputs_normalized, fea_type = 'grid'):      
        
        index = {}
        vol_bound_norm = torch.tensor([[0., 0., 0.], [1., 1., 1.]]).to(self.device)
        ind = coord2index(inputs_normalized, vol_bound_norm, reso=self.reso, plane=fea_type, normalize_coords=False)
        # print(f'ind shape: {ind.shape}')
        index[fea_type] = ind
        input_cur = add_key(inputs.clone()[...,:3], index, 'points', 'index', device=self.device)
        fea, unet = self.model.encode_inputs(input_cur)
        return fea, unet
        
    def concat_points(self, p_in, p_gt, n_in):
        # random_indices = torch.randint(0, p_in.size(0), (n_in,))
        selected_p_in = p_in[:n_in]
        
        return selected_p_in, torch.cat([selected_p_in, p_gt], dim=0)

        
    def get_crop_bound(self, inputs_raw, input_crop_size, query_crop_size):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        '''
        if inputs_raw.shape[-1] == 4:
            inputs = inputs_raw[:, :3]
        else:
            inputs = inputs_raw
            
        vol_bound = {}

        lb = torch.min(inputs, dim=0).values.cpu().numpy() - query_crop_size - np.array([0.01, 0.01, 0.01]) # - np.array([0, 0.2, 0]) #Padded once
        ub = torch.max(inputs, dim=0).values.cpu().numpy() + query_crop_size #Padded once
        
        lb_query = np.mgrid[lb[0]:ub[0]:query_crop_size,
                            lb[1]:ub[1]:query_crop_size,
                            lb[2]:ub[2]:query_crop_size].reshape(3, -1).T
        
        ub_query = lb_query + query_crop_size
        center = (lb_query + ub_query) / 2
        lb_input = center - input_crop_size / 2
        ub_input = center + input_crop_size / 2
        # number of crops alongside x,y, z axis
        vol_bound['axis_n_crop'] = np.ceil((ub - lb) / query_crop_size).astype(int)
        # total number of crops
        num_crop = np.prod(vol_bound['axis_n_crop'])
        vol_bound['n_crop'] = num_crop
        vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)

        return vol_bound