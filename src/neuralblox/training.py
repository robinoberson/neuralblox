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
                 vis_dir=None, threshold=0.5, eval_sample=False, vol_range = [[-0.55, -0.55, -0.55], [0.55, 0.55, 0.55]]):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.vol_bound = vol_bound
        self.grid_reso = vol_bound['reso']
        self.unet = None
        self.vol_range = vol_range

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
    def train_step(self, data, DEGREES = 0):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
            DEGREES (integer): degree range in which object is going to be rotated
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, DEGREES = DEGREES)
        # loss = self.compute_loss_voxels(data, DEGREES = DEGREES)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def validate_step(self, data, DEGREES = 0):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
            DEGREES (integer): degree range in which object is going to be rotated
        '''
        self.model.eval()
        loss = self.compute_loss(data, DEGREES = DEGREES)
        return loss.item()
    
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
        inputs_occ = data.get('inputs.occ').to(device).unsqueeze(-1)
        inputs = torch.cat((inputs, inputs_occ), dim=-1)  # Concatenate along the last dimension

        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        # inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # # add pre-computed normalized coordinates
        # points_normalized = normalize_coord(points, vol_range=self.vol_range, plane = 'grid')
        # points = add_key(points, points_normalized, 'p', 'p_n', device=device)
        
        # points_iou_normalized = normalize_coord(points_iou, vol_range=self.vol_range, plane = 'grid')
        # points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            fea_type = 'grid'
            index = {}
            ind = coord2index(inputs.clone(), self.vol_range, reso=self.grid_reso, plane=fea_type)
            index[fea_type] = ind
            inputs_3d = inputs.clone()[..., :3]
            input_cur = add_key(inputs_3d, index, 'points', 'index', device=device)
            
            fea, self.unet = self.model.encode_inputs(input_cur)
            fea_du, _ = self.unet(fea) #downsample and upsample 
            kwargs = {}
            
            # Query points
            pi_in = points_iou
            pi_in = {'p': pi_in}
            p_n = {}
            
            p_n[fea_type] = normalize_coord(points_iou.clone(), self.vol_range, plane=fea_type).to(self.device)
            pi_in['p_n'] = p_n
            
            c = {}
            c['grid'] = fea_du

            p_out = self.model.decode(pi_in, c, **kwargs)

        
        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()

        return iou

    def compute_loss(self, data, DEGREES = 0):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs').to(device)
        inputs_occ = data.get('inputs.occ').to(device).unsqueeze(-1)
        
        inputs = torch.cat((inputs, inputs_occ), dim=-1)  # Concatenate along the last dimension

        # import open3d as o3d
        # pcd_occ = o3d.geometry.PointCloud()
        # pcd_unocc = o3d.geometry.PointCloud()
        
        # pcd_occ.points = o3d.utility.Vector3dVector(inputs[0, inputs_occ[0, :, 0] == 1, :3].cpu().numpy())
        # pcd_unocc.points = o3d.utility.Vector3dVector(inputs[0, inputs_occ[0, :, 0] == 0, :3].cpu().numpy())
        # pcd_occ.paint_uniform_color([1, 0, 0])
        # pcd_unocc.paint_uniform_color([0, 1, 0])
        
        # o3d.visualization.draw_geometries([pcd_occ, pcd_unocc])
        
        if (DEGREES != 0):
            inputs, rotation = self.rotate_points(inputs, DEGREES=DEGREES)
            p = self.rotate_points(p, use_rotation_tensor=True)
        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)
        
        vol_range = self.vol_range
        vol_bound = {}
        vol_bound['input_vol'] = vol_range
        fea = 'grid'
        index = {}
        ind = coord2index(inputs.clone(), vol_bound['input_vol'], reso=self.grid_reso, plane=fea)
        index[fea] = ind
        inputs_3d = inputs.clone()[..., :3]
        input_cur = add_key(inputs_3d, index, 'points', 'index', device=device)

        if self.unet == None:
            fea, self.unet = self.model.encode_inputs(input_cur)
        else:
            fea, _ = self.model.encode_inputs(input_cur)
        
        fea_du, _ = self.unet(fea) #downsample and upsample 
        
        #save fea_du
        # torch.save(fea_du, '/home/roberson/MasterThesis/master_thesis/Playground/BackboneEmpty/fea_du.pt')
        # torch.save(inputs_3d, '/home/roberson/MasterThesis/master_thesis/Playground/BackboneEmpty/inputs.pt')

        kwargs = {}
        # General points
        pi_in = p
        pi_in = {'p': pi_in}
        p_n = {}
        for key in self.vol_bound['fea_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(p.clone(), vol_bound['input_vol'], plane=key).to(self.device)
        pi_in['p_n'] = p_n
        
        c = {}
        c['grid'] = fea_du

        logits = self.model.decode(pi_in, c, **kwargs).logits
            
        occ_hat = logits.detach().cpu().numpy()[0]
        values = np.exp(occ_hat) / (1 + np.exp(occ_hat))
        
        points_occ = p[0, values >= 0.5, :3].cpu().numpy()
        points_unocc = p[0, values < 0.5, :3].cpu().numpy()
        
        bb_min = np.min(points_unocc, axis=0)
        bb_max = np.max(points_unocc, axis=0)
        
        print(f'len points occ {len(points_occ)}')
        
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_occ)
        pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd])
        
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()
        
        # export pi_in and input_cur as pickle 
        # pi_in['occ'] = occ
        # pi_in['logits'] = logits
        # input_cur['occ'] = inputs_occ
        # import pickle
        # with open('pi_in.pkl', 'wb') as f:
        #     pickle.dump(pi_in, f)
            
        # with open('input_cur.pkl', 'wb') as f:
        #     pickle.dump(input_cur, f)
        
        return loss

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

