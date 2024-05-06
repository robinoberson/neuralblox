import os
import torch
from src.common import (
    add_key, coord2index, normalize_coord
)
from src.training import BaseTrainer
import numpy as np
import pickle

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

    def __init__(self, model, model_merge, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False, query_n = 8192, unet_hdim = 32, unet_depth = 2):
        self.model = model
        self.model_merge = model_merge
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.max_crop_with_change = None
        self.query_n = query_n
        self.hdim = unet_hdim
        self.factor = 2**unet_depth
        self.reso = None
        self.unet = None
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_sequence_window(self, data_batch, points_gt, input_crop_size, query_crop_size, grid_reso):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        
        n_inputs = 2
        
        torch.cuda.empty_cache()
        
        self.model.train()
        self.model_merge.train()
        self.optimizer.zero_grad()

        self.reso = grid_reso
        d = self.reso//self.factor
        
        p_in, points_gt = self.get_inputs_from_batch(data_batch, points_gt)
    
        #Step 1 get the bounding box of the scene
        inputs, inputs_full = self.concat_points(p_in, points_gt, n_inputs) #Everything 4D
        self.vol_bound_all = self.get_crop_bound(inputs_full.view(-1, 4), input_crop_size, query_crop_size)
        
        inputs_distributed = self.distribute_inputs(inputs.unsqueeze(1), self.vol_bound_all)
        # return inputs_distributed
        # Encode latents 
        latent_map_sampled = self.encode_latent_map(inputs_distributed, self.vol_bound_all['input_vol'])
        # return latent_map_sampled
        # Stack latents 
        latent_map_sampled_stacked = self.stack_latents(latent_map_sampled, self.vol_bound_all)

        # Merge latents
        latent_map_sampled_merged = self.merge_latent_map(latent_map_sampled_stacked) 
        
        # del latent_map_sampled, latent_map_sampled_stacked
        # torch.cuda.empty_cache()
        # Compute gt latent
        latent_map_gt = self.get_latent_gt(points_gt)

        # get logits
        random_points = torch.rand(1, self.query_n, 3).to(self.device)
        
        # return latent_map_gt, latent_map_sampled_merged, random_points, input_crop_size, query_crop_size
    
        logits_sampled, query_points_sampled = self.get_logits(latent_map_sampled_merged, occupied_voxels = torch.ones(latent_map_sampled_merged.shape[0], device = self.device, dtype = torch.bool), random_points=random_points, query_size = query_crop_size, input_size = input_crop_size)
        # del latent_map_sampled_merged
        # torch.cuda.empty_cache()
        
        logits_gt, query_points_gt = self.get_logits(latent_map_gt, occupied_voxels = torch.ones(latent_map_sampled_merged.shape[0], device = self.device, dtype = torch.bool), random_points=random_points, query_size = query_crop_size, input_size = input_crop_size)
        
        if logits_gt.shape != logits_sampled.shape:
            print(logits_gt.shape, logits_sampled.shape)
            raise ValueError
        
        
        # compute cost
        loss_logits = torch.nn.L1Loss(reduction='mean')
        loss = 1 * loss_logits(logits_sampled, logits_gt)

        loss.backward()
        self.optimizer.step()
        
        # self.visualize_logits(logits_gt, logits_sampled, query_points_sampled)
        
        return loss
    
    # def visualize_logits(self, logits_gt, logits_sampled, query_points):
    #     import open3d as o3d
        
    #     points_gt_np = query_points['p'].detach().cpu().numpy().reshape(-1, 3)
    #     points_sampled_np = query_points['p'].detach().cpu().numpy().reshape(-1, 3)

    #     occ_gt = logits_gt.detach().cpu().numpy()
    #     occ_sampled = logits_sampled.detach().cpu().numpy()

    #     values_gt = np.exp(occ_gt) / (1 + np.exp(occ_gt))
    #     values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
        
    #     values_gt = values_gt.reshape(-1)
    #     values_sampled = values_sampled.reshape(-1)

    #     threshold = 0.5

    #     values_gt[values_gt < threshold] = 0
    #     values_gt[values_gt >= threshold] = 1

    #     values_sampled[values_sampled < threshold] = 0
    #     values_sampled[values_sampled >= threshold] = 1

    #     pcd_occ_gt = o3d.geometry.PointCloud()
    #     pcd_unoccc_gt = o3d.geometry.PointCloud()
    #     pcd_occ_sampled = o3d.geometry.PointCloud()
    #     pcd_unoccc_sampled = o3d.geometry.PointCloud()

    #     pcd_occ_gt.points = o3d.utility.Vector3dVector(points_gt_np[values_gt == 1])
    #     pcd_unoccc_gt.points = o3d.utility.Vector3dVector(points_gt_np[values_gt == 0])

    #     pcd_occ_sampled.points = o3d.utility.Vector3dVector(points_sampled_np[values_sampled == 1])
    #     pcd_unoccc_sampled.points = o3d.utility.Vector3dVector(points_sampled_np[values_sampled == 0])

    #     pcd_occ_gt.paint_uniform_color([0, 1, 0])
    #     pcd_unoccc_gt.paint_uniform_color([1, 0, 0])

    #     pcd_occ_sampled.paint_uniform_color([1, 0, 0])
    #     pcd_unoccc_sampled.paint_uniform_color([1, 0, 0])

    #     o3d.visualization.draw_geometries([pcd_occ_gt, pcd_occ_sampled])
    
    def get_inputs_from_batch(self, batch, points_gt):
        p_in_3D = batch.get('inputs').to(self.device)
        p_in_occ = batch.get('inputs.occ').to(self.device).unsqueeze(-1)
        
        p_in = torch.cat([p_in_3D, p_in_occ], dim = -1)
        points_gt = points_gt.unsqueeze(0)
        
        indices = torch.randperm(points_gt.shape[1])[:p_in_3D.shape[1]]
        points_gt = points_gt[:, indices]
        
        return p_in, points_gt
    
    def remove_padding_single_dim(self, tensor):
        other_dims = tensor.shape[1:]
        H, W, D = self.vol_bound_all['axis_n_crop'][0], self.vol_bound_all['axis_n_crop'][1], self.vol_bound_all['axis_n_crop'][2]
        tensor = tensor.reshape(H, W, D, *other_dims)
        return tensor[1:-1, 1:-1, 1:-1]
    
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
                
    def get_logits(self, latent_map_full, occupied_voxels, random_points, query_size, input_size):
        
        n_crops_total = latent_map_full.shape[0]
        n_crops_valid = occupied_voxels.sum().item()

        query_points_stacked = self.get_query_points(random_points, occupied_voxels, query_size=query_size, input_size=input_size)

        # Get latent codes of valid crops
        kwargs = {}
        fea = {}
        fea['unet3d'] = self.unet
        fea['latent'] = latent_map_full[occupied_voxels]

        p_r = self.model.decode(query_points_stacked, fea, **kwargs)
        logits_decoded = p_r.logits
        logits = torch.full((n_crops_total, random_points.shape[1]), -100.).to(self.device)
        logits[occupied_voxels] = logits_decoded
        
        return logits, query_points_stacked
    def get_query_points(self, random_points, occupied_voxels, query_size, input_size):
        n_crops_unpadded = (self.vol_bound_all['axis_n_crop'][0] - 2) * (self.vol_bound_all['axis_n_crop'][1] - 2) * (self.vol_bound_all['axis_n_crop'][2] - 2)
        random_points_stacked = random_points.repeat(n_crops_unpadded, 1, 1)

        centers = torch.from_numpy((self.vol_bound_all['input_vol'][:,1] - self.vol_bound_all['input_vol'][:,0])/2.0 + self.vol_bound_all['input_vol'][:,0]).to(self.device) #shape [n_crops, 3]
        centers = self.remove_padding_single_dim(centers).reshape(-1,3)
        centers = centers.unsqueeze(1)
        
        p = (random_points_stacked.clone() + centers)[occupied_voxels]
        p_n = ((random_points_stacked.clone() - 0.5) * query_size / input_size + 0.5)[occupied_voxels]

        p_n_dict = {'grid': p_n}
                        
        pi_in = {'p': p}
        pi_in['p_n'] = p_n_dict
        
        return pi_in
    
    def get_latent_gt(self, points_gt):
        inputs_distributed_gt = self.distribute_inputs(points_gt.unsqueeze(1), self.vol_bound_all)
        
        latent_map_gt = self.encode_latent_map(inputs_distributed_gt, self.vol_bound_all['input_vol']).squeeze(0)
        latent_map_gt = self.remove_padding(latent_map_gt)
        print(latent_map_gt.shape)
        return latent_map_gt.reshape(-1, *tuple(latent_map_gt.shape[3:]))
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
    
    # def encode_latent_map(self, inputs_distributed, occupied_voxels, vol_bound_valid):
    #     n_inputs = occupied_voxels.shape[0]
    #     n_crop = occupied_voxels.shape[1]
    #     d = self.reso//self.factor
    #     latent_map = torch.zeros(n_inputs, n_crop, self.hdim*self.factor, d, d, d).to(self.device)
    
    #     for idx_pin in range(n_inputs):
    #         n_valid_crops = torch.sum(occupied_voxels, dim=1)[idx_pin].item()
    #         fea = torch.zeros(n_valid_crops, self.hdim, self.reso, self.reso, self.reso).to(self.device)

    #         idx_input_valid = 0
    #         for bool_valid in occupied_voxels[idx_pin]:
    #             if bool_valid:
    #                 fea[idx_input_valid], self.unet = self.encode_crop_sequential(inputs_distributed[idx_pin][idx_input_valid].unsqueeze(0).float(), self.device, vol_bound = vol_bound_valid[idx_pin][idx_input_valid])
            
    #                 idx_input_valid += 1
            
    #         _, latent_map[idx_pin, occupied_voxels[idx_pin]] = self.unet(fea) #down and upsample
            

        
    #     return latent_map
    
    def encode_latent_map(self, inputs_distributed, vol_bound):
        
        n_inputs, n_crop, n_points, _ = inputs_distributed.shape
        
        d = self.reso//self.factor
        # print(inputs_distributed.shape, vol_bound.shape)
        fea, self.unet = self.encode_crop(inputs_distributed, vol_bound)
        # print(fea.shape)
        _, latent_map = self.unet(fea) #down and upsample
        
        latent_map_shape = latent_map.shape
        return latent_map.reshape(n_inputs, n_crop, *latent_map_shape[1:])

    def normalize_inputs(self, inputs, vol_bound):
        vol_bound = torch.from_numpy(vol_bound).to(self.device)
        n_inputs, n_crops, n_max, _ = inputs.shape
        inputs = inputs.reshape(n_inputs*n_crops, n_max, 4)
        
        coords = inputs[:, :, :3]
        occ = inputs[:, :, 3].unsqueeze(-1)
        
        bb_min = vol_bound[:, 0, :].unsqueeze(1).repeat(n_inputs, 1, 1) # Shape: (n_crops, 3, 1)
        bb_max = vol_bound[:, 1, :].unsqueeze(1).repeat(n_inputs, 1, 1)  # Shape: (n_crops, 3, 1)
        bb_size = bb_max - bb_min  # Shape: (n_crops, 3, 1)
        
        coords = (coords - bb_min) / bb_size
        # bb_min_inputs = torch.min(coords, dim=1).values
        # bb_max_inputs = torch.max(coords, dim=1).values
        # print("bb_min_inputs", bb_min_inputs)
        # print("bb_max_inputs", bb_max_inputs)
        
        inputs = torch.cat([coords, occ], dim=-1)
        inputs = inputs.reshape(n_inputs, n_crops, n_max, 4)
        
        return inputs
        
    def distribute_inputs(self, distributed_inputs_raw, vol_bound):
        # Clone the input tensor
        distributed_inputs = distributed_inputs_raw.clone()
        n_inputs = distributed_inputs.shape[0]
        n_crops = vol_bound['n_crop']
        
        distributed_inputs = distributed_inputs.repeat(1, n_crops, 1, 1)

        # Convert vol_bound['input_vol'] to a torch tensor
        vol_bound_tensor = torch.tensor(vol_bound['input_vol']).to(self.device)

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

        # Create a mask for selecting points with label 1
        indexes_keep = distributed_inputs[..., 3] == 1

        # Calculate the number of points to keep for each input and crop
        n_points = distributed_inputs[..., 3].sum(dim=2).int()

        # Create a mask for broadcasting
        mask = torch.arange(n_max).expand(n_inputs, n_crops, n_max).to(device=self.device) < n_points.unsqueeze(-1)

        # Create a tensor to hold the shortened inputs
        input_vol = torch.from_numpy(vol_bound['input_vol']).float() 

        bb_min = input_vol[:, 0, :].unsqueeze(1).repeat(n_inputs, 1, 1) # Shape: (n_crops, 3, 1)
        bb_max = input_vol[:, 1, :].unsqueeze(1).repeat(n_inputs, 1, 1)  # Shape: (n_crops, 3, 1)
        bb_size = bb_max - bb_min  # Shape: (n_crops, 3, 1)
        
        random_points = torch.rand(n_inputs * n_crops, n_max, 3, device=self.device)  # Shape: (n_inputs, n_crops, n_max, 3)
        random_points *= bb_size  # Scale points to fit inside each bounding box
        random_points += bb_min  # Translate points to be within each bounding box
        
        distributed_inputs_short = torch.zeros(n_inputs, n_crops, n_max, 4, device=self.device)
        distributed_inputs_short[:, :, :, :3] = random_points.reshape(n_inputs, n_crops, n_max, 3)
        
        # Assign values to the shortened tensor using advanced indexing and broadcasting
        distributed_inputs_short[mask] = distributed_inputs[indexes_keep]

        return distributed_inputs_short

    def encode_crop(self, inputs, vol_bound, fea = 'grid'):
        n_inputs, n_crop, n_points, _ = inputs.shape
        
        inputs_normalized = self.normalize_inputs(inputs, vol_bound)
        
        inputs = inputs.reshape(n_inputs*n_crop, n_points, 4)
        inputs_normalized = inputs_normalized.reshape(n_inputs*n_crop, n_points, 4)
        
        index = {}
        vol_bound_norm = torch.tensor([[0., 0., 0.], [1., 1., 1.]]).to(self.device)
        ind = coord2index(inputs_normalized, vol_bound_norm, reso=self.reso, plane=fea, normalize_coords=False)
        # print(f'ind shape: {ind.shape}')
        index[fea] = ind
        input_cur = add_key(inputs.clone()[...,:3], index, 'points', 'index', device=self.device)
        fea, unet = self.model.encode_inputs(input_cur)
        return fea, unet
        
        
    def encode_crop_sequential(self, inputs, device, vol_bound, fea = 'grid'):
        ''' Encode a crop to feature volumes

        Args:
            inputs (tensor): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''

        index = {}
        grid_reso = self.reso
        ind = coord2index(inputs.clone(), vol_bound, reso=grid_reso, plane=fea)
        index[fea] = ind
        input_cur = add_key(inputs.clone()[...,:3], index, 'points', 'index', device=device)

        fea, unet = self.model.encode_inputs(input_cur)

        return fea, unet
        
    def concat_points(self, p_in, p_gt, n_in):
        random_indices = torch.randint(0, p_in.size(0), (n_in,))
        selected_p_in = p_in[random_indices]
        
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

        lb = torch.min(inputs, dim=0).values.cpu().numpy() - 0.01 - query_crop_size #Padded once
        ub = torch.max(inputs, dim=0).values.cpu().numpy() + 0.01 + query_crop_size #Padded once
        
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