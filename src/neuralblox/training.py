import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
from math import sin,cos,radians,sqrt
import random
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
import numpy as np

def get_bounding_box(points):
    # Ensure points array has shape (n, 3)
    assert points.shape[1] == 3, "Points array must have shape (n, 3)"
    
    # Calculate min and max values along each axis
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    
    # Print bounding box in a single line
    print(f"Bounding Box: X: {min_x:.2f} - {max_x:.2f}, Y: {min_y:.2f} - {max_y:.2f}, Z: {min_z:.2f} - {max_z:.2f}")


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, vol_bound = None, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.vol_bound = vol_bound
        self.grid_reso = vol_bound['reso']

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def get_crop_bound(self, inputs):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (torch.Tensor): input point cloud
        '''
        query_crop_size = self.vol_bound['query_crop_size']
        input_crop_size = self.vol_bound['input_crop_size']

        # Get the minimum and maximum values along each dimension for each element in the batch
        
        min_vals, _ = torch.min(inputs, dim=2) 
        max_vals, _ = torch.max(inputs, dim=2) 
         
        lb = min_vals.unsqueeze(2).values[0].cpu().numpy() - 0.01
        ub = max_vals.unsqueeze(2).values[0].cpu().numpy() + 0.01
        
        # lb_query = 
        
        # Create grid coordinates for each element in the batch
        lb_query = np.stack(np.meshgrid(np.arange(lb[0], ub[0], query_crop_size),
                                        np.arange(lb[1], ub[1], query_crop_size),
                                        np.arange(lb[2], ub[2], query_crop_size),
                                        indexing='ij'), axis=-1).reshape(-1, 3)
        
        # Calculate upper bound query coordinates
        ub_query = lb_query + query_crop_size
        # Calculate center coordinates
        center = (lb_query + ub_query) / 2
        # Calculate input bounds
        lb_input = center - input_crop_size / 2
        ub_input = center + input_crop_size / 2
        
        # Number of crops alongside x, y, z axis
        self.vol_bound['axis_n_crop'] = np.ceil((ub - lb) / query_crop_size).astype(int)
        # Total number of crops
        num_crop = np.prod(self.vol_bound['axis_n_crop'])
        self.vol_bound['n_crop'] = num_crop
        self.vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)

        
    def train_step(self, data, DEGREES = 0):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
            DEGREES (integer): degree range in which object is going to be rotated
        '''
        self.model.train()
        self.optimizer.zero_grad()
        # loss = self.compute_loss(data, DEGREES = DEGREES)
        loss = self.compute_loss_voxels(data, DEGREES = DEGREES)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def encode_crop(self, inputs, device, vol_bound=None):
        ''' Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''

        if vol_bound == None:
            vol_bound = self.vol_bound

        index = {}
        for fea in self.vol_bound['fea_type']:
            # crop the input point cloud
            mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) & \
                     (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) & \
                     (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) & \
                     (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z

            p_input = inputs[mask]
            
            if p_input.shape[0] == 0:  # no points in the current crop
                p_input = inputs.squeeze()
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
                if fea == 'grid':
                    ind[~mask] = self.vol_bound['reso'] ** 3
                else:
                    ind[~mask] = self.vol_bound['reso'] ** 2
            else:
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
            index[fea] = ind.unsqueeze(0)
            input_cur = add_key(p_input.unsqueeze(0), index, 'points', 'index', device=device)

            c = self.model.encode_inputs(input_cur)
            
        return c, input_cur
        
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, 
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data, DEGREES = 0):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        if (DEGREES != 0):
            inputs, rotation = self.rotate_points(inputs, DEGREES=DEGREES)
            p = self.rotate_points(p, use_rotation_tensor=True)
        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        logits = self.model.decode(p, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss
    
    def compute_loss_voxels(self, data, DEGREES=0, input_crop_size=None, query_crop_size=None, grid_reso=None, resolution0=64):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
            DEGREES (int): degree of rotation
            input_crop_size (tuple): size of input crop
            query_crop_size (tuple): size of query crop
            grid_reso (int): grid resolution
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        # Apply rotation if DEGREES is not zero
        if DEGREES != 0:
            inputs, rotation = self.rotate_points(inputs, DEGREES=DEGREES)
            p = self.rotate_points(p, use_rotation_tensor=True)
        
        # Check if cropping parameters are provided
        if input_crop_size is not None and query_crop_size is not None and grid_reso is not None:
            raise NotImplementedError
            # Perform cropping
        vol_bound_all = self.get_crop_bound(p.unsqueeze(0))
        # nx = resolution0
        n_crop = vol_bound_all['n_crop']
        n_crop_axis = vol_bound_all['axis_n_crop']

        # occupancy in each direction
        # r = nx * 2 ** self.upsampling_steps
        # occ_values = np.array([]).reshape(r, r, 0)
        # occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])
        # occ_values_x = np.array([]).reshape(0, r * n_crop_axis[1], r * n_crop_axis[2])
        
        def filter_input(inputs, vol_bound):
            mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) & \
                     (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) & \
                     (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) & \
                     (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z

            return inputs[mask], mask
        
        total_loss = 0.0 
        
        for i in trange(n_crop):
            # encode the current crop
            vol_bound = {}
            vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
            vol_bound['input_vol'] = self.vol_bound['input_vol'][i]
            
            print(f'Crop {i}: {vol_bound}')
                        
            c, input_cur = self.encode_crop(inputs, device, vol_bound=vol_bound)
                        
            p_filtered, mask = filter_input(p, vol_bound)#points in the current crop filtered from p
            occ_filtered = occ[mask] #points in the current crop filtered from occ
            
            p_filtered = torch.from_numpy(p_filtered).to(device)
            occ_filtered = torch.from_numpy(occ_filtered).to(device)
            
            get_bounding_box(input_cur)
            get_bounding_box(p_filtered)
            
            print(p_filtered.shape, occ_filtered.shape)
                            
            kwargs = {}
            # General points
            logits = self.model.decode(p_filtered, c, **kwargs).logits
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ_filtered, reduction='none')
            total_loss += loss_i.item()  # Accumulate loss for current crop

        # Compute mean loss over all crops
        mean_loss = total_loss / n_crop

        return mean_loss


    def rotate_points(self, pointcloud_model, DEGREES=0, query_points=False, use_rotation_tensor=False,
                      save_rotation_tensor=False):
        ## https://en.wikipedia.org/wiki/Rotation_matrix
        """
            Function for rotating points
            Args:
                pointcloud_model (numpy 3d array) - batch_size x pointcloud_size x 3d channel sized numpy array which presents pointcloud
                DEGREES (int) - range of rotations to be used
                query_points (boolean) - used for rotating query points with already existing rotation matrix
                use_rotation_tensor (boolean) - asking whether DEGREES should be used for generating new rotation matrix, or use the already established one
                save_rotation_tensor (boolean) - asking to keep rotation matrix in a pytorch .pt file
        """
        if (use_rotation_tensor != True):
            angle_range = DEGREES
            x_angle = radians(random.uniform(-1,1) * 5)
            y_angle = radians(random.uniform(-1,1) * 180)
            z_angle = radians(random.uniform(-1,1) * 5)

            rot_x = torch.Tensor(
                [[1, 0, 0, 0], [0, cos(x_angle), -sin(x_angle), 0], [0, sin(x_angle), cos(x_angle), 0], [0, 0, 0, 1]])
            rot_y = torch.Tensor(
                [[cos(y_angle), 0, sin(y_angle), 0], [0, 1, 0, 0], [-sin(y_angle), 0, cos(y_angle), 0], [0, 0, 0, 1]])
            rot_z = torch.Tensor(
                [[cos(z_angle), -sin(z_angle), 0, 0], [sin(z_angle), cos(z_angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            rotation_matrix = torch.mm(rot_z, rot_y)
            rotation_matrix = torch.mm(rot_x, rotation_matrix)
            rotation_matrix = torch.transpose(rotation_matrix, 0, 1)

            batch_size, point_cloud_size, _ = pointcloud_model.shape
            pointcloud_model = torch.cat(
                [pointcloud_model, torch.ones(batch_size, point_cloud_size, 1).to(self.device)], dim=2)

            pointcloud_model_rotated = torch.matmul(pointcloud_model, rotation_matrix.to(self.device))
            self.rotation_matrix = rotation_matrix

            if (save_rotation_tensor):
                torch.save(rotation_matrix, 'rotation_matrix.pt')  # used for plane prediction, change it at your will
            return pointcloud_model_rotated[:, :, 0:3], (x_angle, y_angle, z_angle)
        else:
            batch_size, point_cloud_size, _ = pointcloud_model.shape
            #pointcloud_model = pointcloud_model / sqrt(0.55 ** 2 + 0.55 ** 2 + 0.55 ** 2)
            pointcloud_model = torch.cat(
                [pointcloud_model, torch.ones(batch_size, point_cloud_size, 1).to(self.device)], dim=2)
            pointcloud_model_rotated = torch.matmul(pointcloud_model, self.rotation_matrix.to(self.device))
            return pointcloud_model_rotated[:, :, 0:3]

