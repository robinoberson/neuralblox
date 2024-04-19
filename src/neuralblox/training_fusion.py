import os
import torch
from src.common import (
    add_key, coord2index
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

    def __init__(self, model, model_merge, optimizer, stack_latents = False, device=None, input_type='pointcloud',
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
        self.stack_latents = stack_latents
        self.unet = None

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_sequence_window(self, data, points_gt, input_crop_size, query_crop_size, grid_reso):
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

        device = self.device
        p_in = data.get('inputs').to(device)
        batch_size, T, D = p_in.size()  # seq_length, T, 3
        query_sample_size = self.query_n
        
        #Step 1 get the bounding box of the scene
        inputs, inputs_full = self.concat_points(p_in, points_gt, n_inputs)
        self.vol_bound_all = self.get_crop_bound(inputs_full.view(-1, 3), input_crop_size, query_crop_size)
        inputs_distributed, occupied_voxels, vol_bound_valid = self.distribute_inputs(inputs, self.vol_bound_all)

        # Encode latents 
        latent_map_sampled = self.encode_latent_map(inputs_distributed, occupied_voxels, vol_bound_valid)
        return latent_map_sampled
        # # Stack latents 
        latent_map_sampled_neighbored = self.stack_neighbors(latent_map_sampled, self.vol_bound_all)
        # return latent_map_sampled_neighbored
        # latent_map_sampled_stacked = self.stack_latents(latent_map_sampled)

        # # Merge latents
        # latent_map_sampled_merged = self.merge_latent_map(latent_map_sampled_stacked)        
        
        
        
        # n_in, H, W, D, c, h, w, d = n_inputs, vol_bound_all['axis_n_crop'][0], vol_bound_all['axis_n_crop'][1], vol_bound_all['axis_n_crop'][2], 256, 18, 18, 18
        # latent_map_sampled_stacked = torch.randn((H, W, D, c*n_in, h, w, d))
        # Merge latents 
        
        
        # return latent_map_sampled_merged
    
    def merge_latent_map(self, latent_map):
        H, W, D, c, h, w, d = latent_map.size()
        latent_map = latent_map.reshape(-1, c, h, w, d)
        
        fea_dict = {}
        fea_dict['latent'] = latent_map
        
        latent_map = self.model_merge(fea_dict)
        return latent_map

    def stack_neighbors(self, latent_map, vol_bound):
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
    
    def encode_latent_map(self, inputs_distributed, occupied_voxels, vol_bound_valid):
        n_inputs = occupied_voxels.shape[0]
        n_crop = occupied_voxels.shape[1]
        d = self.reso//self.factor
        latent_map = torch.zeros(n_inputs, n_crop, self.hdim*self.factor, d, d, d).to(self.device)
    
        for idx_pin in range(n_inputs):
            n_valid_crops = torch.sum(occupied_voxels, dim=1)[idx_pin].item()
            fea = torch.zeros(n_valid_crops, self.hdim, self.reso, self.reso, self.reso).to(self.device)

            idx_input_valid = 0
            for bool_valid in occupied_voxels[idx_pin]:
                if bool_valid:
                    fea[idx_input_valid], self.unet = self.encode_crop_sequential(inputs_distributed[idx_pin][idx_input_valid].unsqueeze(0).float(), self.device, vol_bound = vol_bound_valid[idx_pin][idx_input_valid])
            
                    idx_input_valid += 1
            
            _, latent_map[idx_pin, occupied_voxels[idx_pin]] = self.unet(fea) #down and upsample
            
        
            
            
            # fea = torch.zeros(n_crop, self.hdim, self.reso, self.reso, self.reso).to(self.device)

            # for i in range(n_crop):
                
            #     inputs_crop = self.get_input_crop(p_in[idx_pin].unsqueeze(0), vol_bound_all['input_vol'][i])
            #     if inputs_crop.shape[0] == 0:
            #         latent_map[idx_pin, i] = torch.zeros(self.hdim, self.reso, self.reso, self.reso).to(self.device)
            #         # inputs_crop = torch.zeros(100, 3).to(self.device) + torch.tensor(vol_bound_all['input_vol'][i][0]).to(self.device)
            #     else: 
            #         fea[i], unet = self.encode_crop_sequential(inputs_crop.unsqueeze(0).float(), self.device, vol_bound = vol_bound_all['input_vol'][i])
            #         _, latent_map[idx_pin] = unet(fea) #down and upsample
        
        return latent_map
    
    def distribute_inputs(self, p_input, vol_bound_all):
        n_inputs = p_input.shape[0]
        occpied_voxels = torch.zeros(n_inputs, vol_bound_all['n_crop'], dtype=torch.bool).to(self.device)
        inputs_croped_list = [[] for i in range(n_inputs)]
        vol_bound_valid = [[] for i in range(n_inputs)]
        
        for idx_pin in range(n_inputs):
            for i in range(vol_bound_all['n_crop']):
                inputs_crop = self.get_input_crop(p_input[idx_pin], vol_bound_all['input_vol'][i])
                
                if inputs_crop.shape[0] > 0:
                    inputs_croped_list[idx_pin].append(inputs_crop)
                    vol_bound_valid[idx_pin].append(vol_bound_all['input_vol'][i])
                    occpied_voxels[idx_pin, i] = True
                    
        return inputs_croped_list, occpied_voxels, vol_bound_valid

            
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
        input_cur = add_key(inputs.clone(), index, 'points', 'index', device=device)

        fea, unet = self.model.encode_inputs(input_cur)

        return fea, unet

    def get_input_crop(self, p_input, vol_bound):
        
        mask_x = (p_input[..., 0] >= vol_bound[0][0]) & \
                    (p_input[..., 0] < vol_bound[1][0])
        mask_y = (p_input[..., 1] >= vol_bound[0][1]) & \
                    (p_input[..., 1] < vol_bound[1][1])
        mask_z = (p_input[..., 2] >= vol_bound[0][2]) & \
                    (p_input[..., 2] < vol_bound[1][2])
        mask = mask_x & mask_y & mask_z
        
        return p_input[mask]
        
    def concat_points(self, p_in, p_gt, n_in):
        random_indices = torch.randint(0, p_in.size(0), (n_in,))
        selected_p_in = p_in[random_indices]
        
        return selected_p_in, torch.cat([selected_p_in, p_gt.unsqueeze(0)], dim=0)

        
    def get_crop_bound(self, inputs, input_crop_size, query_crop_size):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        '''

        vol_bound = {}

        lb = torch.min(inputs, dim=0).values.cpu().numpy() - 0.01 - query_crop_size #Padded once
        ub = torch.max(inputs, dim=0).values.cpu().numpy() + 0.01 + query_crop_size
        
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