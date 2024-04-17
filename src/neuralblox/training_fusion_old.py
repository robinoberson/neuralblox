import os
import torch
from src.common import (
    add_key, coord2index
)
from src.training import BaseTrainer
import numpy as np
import pickle

def get_crop_bound(inputs, input_crop_size, query_crop_size):
    ''' Divide a scene into crops, get boundary for each crop

    Args:
        inputs (dict): input point cloud
    '''

    vol_bound = {}

    lb = inputs.min(axis=1).values[0].cpu().numpy() - 0.01
    ub = inputs.max(axis=1).values[0].cpu().numpy() + 0.01
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

def get_crop_with_change(p_input, n_crop, vol_bound_all, crop_with_change_count=None):
    
    if crop_with_change_count is None:
            crop_with_change_count = [0] * n_crop
    
    for i in range(n_crop):

        # Get bound of the current crop
        vol_bound = {}
        vol_bound['input_vol'] = vol_bound_all['input_vol'][i]

        # Obtain mask
        mask_x = (p_input[:, :, 0] >= vol_bound['input_vol'][0][0]) & \
                    (p_input[:, :, 0] < vol_bound['input_vol'][1][0])
        mask_y = (p_input[:, :, 1] >= vol_bound['input_vol'][0][1]) & \
                    (p_input[:, :, 1] < vol_bound['input_vol'][1][1])
        mask_z = (p_input[:, :, 2] >= vol_bound['input_vol'][0][2]) & \
                    (p_input[:, :, 2] < vol_bound['input_vol'][1][2])
        mask = mask_x & mask_y & mask_z

        p_input_mask = p_input[mask]

        # If first scan is empty in the crop, then continue
        if p_input_mask.shape[0] == 0:  # no points in the current crop
            continue
        else:
            crop_with_change_count[i] += 1
    
    return crop_with_change_count

def combine_lists(list_sampled, list_gt):
    return_list = []

    for elem1, elem2 in zip(list_sampled, list_gt):
        if elem1 == 0 or elem2 == 0:
            return_list.append(0)
        else:
            return_list.append(elem1)

    return return_list

def get_query_points(random_points, crop_with_change):
    pi_in = {'p': random_points[crop_with_change]}
    p_n = {}
    p_n['grid'] = random_points[crop_with_change]
    pi_in['p_n'] = p_n
    query_points = pi_in
    
    return query_points

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

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_sequence_window(self, data, points_gt, input_crop_size, query_crop_size, grid_reso, gt_query, iter, init_it, window = 8):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
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

        # Get bounding box from sampled all scans in a batch
        sample_points = p_in[:, :, :]
        sample_points = sample_points.view(-1, 3).unsqueeze(0)

        # Shuffle p_in
        p_in =p_in[torch.randperm(p_in.size()[0])]
        
        #concat sample points and points_gt
        points_gt = points_gt.unsqueeze(0)
        
        # save the point s
        # with open('points.pkl', 'wb') as f:
        #     pickle.dump([points_gt.view(-1, 3).cpu().numpy(), sample_points.view(-1, 3).cpu().numpy()], f)

        combined_points = torch.cat([sample_points, points_gt], dim=1)

        vol_bound_all = get_crop_bound(combined_points, input_crop_size, query_crop_size)
        
        # dump vol_bound_all
        # with open('vol_bound_all.pkl', 'wb') as f:
        #     pickle.dump(vol_bound_all, f)
            
        n_crop = vol_bound_all['n_crop']
        n_crop_axis = vol_bound_all['axis_n_crop']

        ## Initialize latent map (prediction)
        latent_map_pred = torch.zeros(n_crop_axis[0], n_crop_axis[1], n_crop_axis[2],
                                      self.hdim*self.factor, d, d, d).to(device)

        loss_all = 0
        crop_with_change_count_sampled = None

        counter = 0

        crop_with_change_count_gt = get_crop_with_change(points_gt, n_crop, vol_bound_all)
        
        for n in range(batch_size):
            p_input_n = p_in[n].unsqueeze(0)

            # Get prediction
            latent_map_pred, unet, crop_with_change_count_sampled = self.update_latent_map_window(p_input_n, latent_map_pred,
                                                                                       n_crop,
                                                                                       vol_bound_all, crop_with_change_count_sampled)

            if (n+1)%window==0:
                # Get vol bounds of updated grids
                
                
                crop_with_change_gt = list(map(bool, crop_with_change_count_gt))
                crop_with_change_sampled = list(map(bool, crop_with_change_count_sampled))

                vol_bound_valid_gt = []

                for i in range(len(crop_with_change_gt)):
                    if crop_with_change_count_gt[i]==True:
                        # Get bound of the current crop
                        vol_bound = {}
                        vol_bound['input_vol'] = vol_bound_all['input_vol'][i]

                        vol_bound_valid_gt.append(vol_bound)
                        
                # with open('vol_bound_valid.pkl', 'wb') as f:
                #     pickle.dump(vol_bound_valid, f)
                bb_low = (1 - query_crop_size / input_crop_size) / 2
                bb_high = (1 + query_crop_size/ input_crop_size) / 2
                random_points = torch.rand(n_crop, query_sample_size, 3).to(self.device) * (bb_high - bb_low) + bb_low
                
                if init_it: print(f'min point over all points {torch.min(random_points)}, max {torch.max(random_points)}')
                
                query_points_sampled = get_query_points(random_points, crop_with_change_sampled)
                query_points_gt = get_query_points(random_points, crop_with_change_gt)                

                del random_points
                torch.cuda.empty_cache()
                
                # Merge
                if self.stack_latents:
                    latent_update_pred = self.merge_latent_codes_neighbours(latent_map_pred, crop_with_change_count_sampled)
                else:
                    latent_update_pred = self.merge_latent_codes(latent_map_pred, crop_with_change_count_sampled)

                # Get prediction logits
                logits_pred = self.get_logits_and_latent(latent_update_pred, unet, crop_with_change_sampled,
                                                                       query_sample_size, query_points_sampled)

                del query_points_sampled
                torch.cuda.empty_cache()
                
                # latent_update_gt, unet_gt, p_input_masked_list = self.update_latent_map_gt(points_accumulated, vol_bound_valid_gt)
                latent_update_gt, unet_gt = self.update_latent_map_gt(points_gt, vol_bound_valid_gt)
                
                # Get ground truth logits
                logits_gt = self.get_logits_and_latent(latent_update_gt, unet_gt, crop_with_change_gt,
                                                          query_sample_size, query_points_gt)
                
                
                # with open('latent_update_gt.pkl', 'wb') as f:
                #     pickle.dump(latent_update_gt, f)
                
                # Get ground truth
                # points_accumulated = p_in[(n+1-window):(n + 1), :, :]
                # points_accumulated = points_accumulated.view(-1, 3).unsqueeze(0)
                # with open('points_accumulated.pkl', 'wb') as f:
                #     pickle.dump(points_accumulated, f)
                
                # with open('gt_points.pkl', 'wb') as f:
                #     pickle.dump(points_gt, f)
                    
                # with open('crop_with_change_count.pkl', 'wb') as f:
                #     pickle.dump(crop_with_change_count, f)
                    
                # with open('vol_bound_all.pkl', 'wb') as f:
                #     pickle.dump(vol_bound_all, f)
                
                crop_with_change_count_inter = combine_lists(crop_with_change_count_sampled, crop_with_change_count_gt)
                crop_with_change_inter = list(map(bool, crop_with_change_count_inter))
                
                mask_sampled = []
                mask_gt = []
                
                torch.cuda.empty_cache()
                
                for elem1, elem2 in zip(crop_with_change_sampled, crop_with_change_gt):
                    if elem1:
                        if elem2:
                            mask_sampled.append(True)
                        else:
                            mask_sampled.append(False)
                    if elem2:
                        if elem1:
                            mask_gt.append(True)
                        else:
                            mask_gt.append(False)                        

                # print(logits_pred[mask_sampled].shape)
                # print(logits_gt[mask_gt].shape)
                # print(latent_update_pred[mask_sampled].shape)
                # print(latent_update_gt[mask_gt].shape)
                
                # Calculate loss
                prediction = {}
                gt = {}
                prediction['logits'] = logits_pred[mask_sampled]
                gt['logits'] = logits_gt[mask_gt]

                prediction['latent'] = latent_update_pred[mask_sampled]
                gt['latent'] = latent_update_gt[mask_gt] 

                loss = self.compute_sequential_loss(prediction, gt, latent_loss=True)
                loss_all += loss.item()
                counter += 1
                loss.backward()
                self.optimizer.step()

                # save_dict = {}
                # save_dict['prediction'] = prediction
                # save_dict['gt'] = gt
                # save_dict['crop_with_change_count'] = crop_with_change_count_sampled
                # save_dict['latent_map_pred'] = latent_map_pred
                # save_dict['iter'] = iter
                # save_dict['loss'] = loss
                
                # path = '/home/roberson/MasterThesis/master_thesis/Playground/Training/debug/loss_inputs_gt/' + str(iter) + '.pkl'
                # if not os.path.exists(os.path.dirname(path)):
                #     os.makedirs(os.path.dirname(path))
                
                # if iter%10 == 0:
                #     with open(path, 'wb') as f:
                #         pickle.dump(save_dict, f)
                
                # Re-initialize
                latent_map_pred = torch.zeros(n_crop_axis[0], n_crop_axis[1], n_crop_axis[2],
                                              self.hdim*self.factor, d, d, d).to(device)
                
                crop_with_change_count_sampled = None


        return loss_all / counter

    def update_latent_map_window(self, p_input, latent_map_pred, n_crop, vol_bound_all, crop_with_change_count):
        """
        Sum latent codes and keep track of counts of updated grids
        """
        p_input_mask_list = []
        vol_bound_valid = []
        mask_valid = []

        H, W, D, c, h, w, d = latent_map_pred.size()
        latent_map_pred = latent_map_pred.view(-1, c, h, w, d)

        updated_crop = [False]*n_crop

        for i in range(n_crop):

            # Get bound of the current crop
            vol_bound = {}
            vol_bound['input_vol'] = vol_bound_all['input_vol'][i]

            # Obtain mask
            mask_x = (p_input[:, :, 0] >= vol_bound['input_vol'][0][0]) & \
                       (p_input[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (p_input[:, :, 1] >= vol_bound['input_vol'][0][1]) & \
                       (p_input[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (p_input[:, :, 2] >= vol_bound['input_vol'][0][2]) & \
                       (p_input[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z

            p_input_mask = p_input[mask]

            # If first scan is empty in the crop, then continue
            if p_input_mask.shape[0] == 0:  # no points in the current crop
                continue
            else:
                if self.max_crop_with_change is not None:
                    crop_with_change = list(map(bool, crop_with_change_count))
                    if sum(crop_with_change) == self.max_crop_with_change:
                        break

                crop_with_change_count[i] += 1
                p_input_mask_list.append(p_input_mask)
                vol_bound_valid.append(vol_bound)
                mask_valid.append(mask)

                updated_crop[i] = True

        valid_crop_index = np.where(updated_crop)[0].tolist()
        n_crop_update = sum(updated_crop)

        fea = torch.zeros(n_crop_update, self.hdim, self.reso, self.reso, self.reso).to(self.device)

        _, unet = self.encode_crop_sequential(p_input_mask_list[0], self.device, vol_bound=vol_bound_valid[0])
        for i in range(n_crop_update):
            fea[i], _ = self.encode_crop_sequential(p_input_mask_list[i], self.device, vol_bound=vol_bound_valid[i])

        fea, latent_update = unet(fea)
        latent_map_pred[valid_crop_index] += latent_update

        latent_map_pred = latent_map_pred.view(H, W, D, c, h, w, d)

        return latent_map_pred, unet, crop_with_change_count

    def merge_latent_codes(self, latent_map_pred, crop_with_change_count):
        H, W, D, c, h, w, d = latent_map_pred.size()
        latent_map_pred = latent_map_pred.view(-1, c, h, w, d)

        crop_with_change = list(map(bool, crop_with_change_count))

        divisor = torch.FloatTensor(crop_with_change_count)[crop_with_change] #keep only the crops with change
        divisor = divisor.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(self.device)

        latent_update = latent_map_pred[crop_with_change] #keep only the crops with change
        latent_update = torch.div(latent_update, divisor)
        
        #dump latent_update
        # with open('latent_update.pkl', 'wb') as f:
        #     pickle.dump(latent_update, f)

        fea_dict = {}
        fea_dict['latent'] = latent_update
        latent_update = self.model_merge(fea_dict)

        return latent_update

    def merge_latent_codes_neighbours(self, latent_map_pred, crop_with_change_count):
        H, W, D, c, h, w, d = latent_map_pred.size()
        latent_map_pred = latent_map_pred.view(-1, c, h, w, d)

        crop_with_change = list(map(bool, crop_with_change_count))

        divisor = torch.clone(crop_with_change_count)
        divisor[~crop_with_change] = 1
        divisor = divisor.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(self.device)

        latent_update = torch.div(latent_map_pred, divisor)
        
        latent_update = latent_update.view(H, W, D, c, h, w, d)
        grid_occ = crop_with_change.view(H, W, D, 1)
        latent_update_stacked, crop_with_change_stacked = self.stack_neighbors(grid_values=latent_update, grid_occ=grid_occ)

        fea_dict = {}
        fea_dict['latent'] = latent_update_stacked
        latent_update_stacked = self.model_merge(fea_dict)
        
        return latent_update

    def update_latent_map_gt(self, p_input, vol_bound_valid):
        

        fea = torch.zeros(len(vol_bound_valid), self.hdim, self.reso, self.reso, self.reso).to(self.device)

        for i in range(len(vol_bound_valid)):
            mask_x = (p_input[:, :, 0] >= vol_bound_valid[i]['input_vol'][0][0]) & \
                     (p_input[:, :, 0] < vol_bound_valid[i]['input_vol'][1][0])
            mask_y = (p_input[:, :, 1] >= vol_bound_valid[i]['input_vol'][0][1]) & \
                     (p_input[:, :, 1] < vol_bound_valid[i]['input_vol'][1][1])
            mask_z = (p_input[:, :, 2] >= vol_bound_valid[i]['input_vol'][0][2]) & \
                     (p_input[:, :, 2] < vol_bound_valid[i]['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z

            fea[i], unet = self.encode_crop_sequential(p_input[mask], self.device,
                                                                        vol_bound=vol_bound_valid[i])

        fea, latent_update = unet(fea)


        return latent_update, unet

    def get_logits_and_latent(self, latent_update, unet, crop_with_change, query_sample_size, query_points=None):

        if query_points is None:
            #raise error
            raise ValueError('query_points is None, please provide query_points')
        # Initialize logits list
        num_valid_crops = sum(crop_with_change)

        # Get latent codes of valid crops
        kwargs = {}
        fea = {}
        fea['unet3d'] = unet
        fea['latent'] = latent_update

        p_r = self.model.decode(query_points, fea, **kwargs)
        logits = p_r.logits

        return logits

    def encode_crop_sequential(self, inputs, device, fea = 'grid', vol_bound=None):
        ''' Encode a crop to feature volumes

        Args:
            inputs (tensor): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''

        index = {}
        grid_reso = self.reso
        ind = coord2index(inputs.clone().unsqueeze(0), vol_bound['input_vol'], reso=grid_reso, plane=fea)
        index[fea] = ind
        input_cur = add_key(inputs.unsqueeze(0), index, 'points', 'index', device=device)

        fea, unet = self.model.encode_inputs(input_cur)

        return fea, unet

    def compute_sequential_loss(self, prediction, gt, latent_loss = False):

        loss_logits = torch.nn.L1Loss(reduction='mean')
        loss_i = 1 * loss_logits(prediction['logits'], gt['logits'])
        loss = loss_i

        if latent_loss == True:
            loss_latent = torch.nn.L1Loss(reduction='mean')
            loss_ii = 1 * loss_latent(prediction['latent'], gt['latent'])
            loss += loss_ii

        return loss

    def find_cells_with_all_neighbors(self, grid_occ):
        """
        Find grid cells that have neighbors on all sides, including diagonals.
        
        Parameters:
            grid_occ (torch.Tensor): 3D grid of boolean values.
            
        Returns:
            torch.Tensor: Boolean mask indicating cells with all neighbors.
        """
        # Create an empty mask of the same shape as the grid_occ
        mask = torch.zeros_like(grid_occ, dtype=torch.bool)
        
        # Pad the grid_occ with False values to avoid boundary checks
        padded_grid_occ = torch.nn.functional.pad(grid_occ, (1, 1, 1, 1, 1, 1), value=False)
        
        # Generate a sliding window view of the padded grid_occ to efficiently check neighbors
        sliding_window = padded_grid_occ.unfold(0, 3, 1).unfold(1, 3, 1).unfold(2, 3, 1)
        
        # Count the number of True values in each window
        neighbor_count = sliding_window.sum(dim=(3, 4, 5))
        
        # Update the mask where the neighbor count is 26
        mask = neighbor_count == 27
        
        return mask


    def stack_neighbors(self, grid_values, grid_occ):
        full_neighbors_mask = find_cells_with_all_neighbors(grid_occ)
        H, W, D, c, h, w, d = grid_values.shape
        stacked_grid = torch.zeros((H, W, D, c, h*3, w*3, d*3), dtype=torch.float32)
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                for k in range(1, D - 1):
                    if full_neighbors_mask[i, j, k]:
                        stacked_vector = grid_values[i-1:i+2, j-1:j+2, k-1:k+2].view(c, h*3, w*3, d*3)
                        stacked_grid[i, j, k] = stacked_vector
                        
        return stacked_grid, full_neighbors_mask.view(-1)