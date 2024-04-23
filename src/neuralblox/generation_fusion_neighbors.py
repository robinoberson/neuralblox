import torch
import numpy as np
from tqdm import trange
import trimesh
from src.utils import libmcubes
from src.common import normalize_coord, add_key, coord2index
import time
from memory_profiler import profile


class Generator3DNeighbors(object):
    '''  Generator class for scene generation and latent code fusion.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        model_merge (nn.Module): trained fusion network
        sample points (torch.Tensor): sampled points to define maximum boundaries of scene
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value to define {occupied, free} and for mesh generation
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): query point density (resolution0 points/voxel)
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
        voxel_threshold (float): threshold to define whether or not to encode points to latent code
        boundary_interpolation (bool): whether boundary interpolation is performed
    '''

    def __init__(self, model, model_merge, 
                 points_batch_size=3000000,
                 threshold=0.05, 
                 device=None,
                 resolution0=16, 
                 upsampling_steps=3,
                 padding=0.1,
                 vol_bound=None,
                 voxel_threshold = 0.01,
                 unet_hdim = 32,
                 unet_depth = 2,
                 ):
        
        self.model = model.to(device)
        self.model_merge = model_merge.to(device)
        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.device = device
        self.resolution0 = 16
        self.upsampling_steps = upsampling_steps
        self.padding = padding

        # Merge all scans
        self.latent = None
        self.unet = None
        self.grid_reso = None
        self.voxel_threshold = voxel_threshold
        self.init_get_bound = None
        self.n_crop = None
        self.n_crop_axis = None
        self.crop_with_change_count = None
        self.unet_hdim = unet_hdim
        self.unet_depth = unet_depth

        # for pointcloud_crop
        self.vol_bound_all = None
        self.vol_bound = vol_bound
        self.grid_reso = vol_bound['reso']
        self.input_crop_size = vol_bound['input_crop_size']
        self.query_crop_size = vol_bound['query_crop_size']
        self.reso = vol_bound['reso']
        self.factor = 2**unet_depth
        self.hdim = unet_hdim
        
    def generate_latent(self, data):
        ''' Generates voxels of latent codes from input point clouds.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
        '''
        inputs = data.get('inputs', [])

        full_points = torch.from_numpy(np.concatenate(inputs, axis=0)).to(self.device)
        self.vol_bound_all = self.get_crop_bound(full_points.view(-1, 3), self.input_crop_size, self.query_crop_size)
        
        vol_bound = self.get_crop_bound(full_points.view(-1, 3), self.input_crop_size, self.query_crop_size, padding=False)
        if self.vol_bound is not None:
            self.vol_bound['axis_n_crop'] = vol_bound['axis_n_crop']
            self.vol_bound['n_crop'] = vol_bound['n_crop']
            self.vol_bound['input_vol'] = vol_bound['input_vol']
            self.vol_bound['query_vol'] = vol_bound['query_vol']
        
        inputs_distributed, occupied_voxels, vol_bound_valid = self.distribute_inputs(inputs, self.vol_bound_all['n_crop'], self.vol_bound_all['query_vol'])
        latent_map = self.encode_latent_map(inputs_distributed, occupied_voxels, vol_bound_valid)
        latent_map_stacked = self.stack_latents(latent_map, self.vol_bound_all)
        latent_map_merged = self.merge_latent_map(latent_map_stacked) 
        
        H, W, D = self.vol_bound['axis_n_crop'][0], self.vol_bound['axis_n_crop'][1], self.vol_bound['axis_n_crop'][2]
        
        return latent_map_merged.reshape(H, W, D, *latent_map_merged.shape[1:])

    def generate_latent_no_merge(self, data):
        ''' Generates voxels of latent codes from input point clouds.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
        '''
        inputs = data.get('inputs', [])

        full_points = torch.from_numpy(np.concatenate(inputs, axis=0)).to(self.device)
        self.vol_bound_all = self.get_crop_bound(full_points.view(-1, 3), self.input_crop_size, self.query_crop_size)
        
        vol_bound = self.get_crop_bound(full_points.view(-1, 3), self.input_crop_size, self.query_crop_size, padding=False)
        if self.vol_bound is not None:
            self.vol_bound['axis_n_crop'] = vol_bound['axis_n_crop']
            self.vol_bound['n_crop'] = vol_bound['n_crop']
            self.vol_bound['input_vol'] = vol_bound['input_vol']
            self.vol_bound['query_vol'] = vol_bound['query_vol']
        
        inputs_distributed, occupied_voxels, vol_bound_valid = self.distribute_inputs(inputs, self.vol_bound_all['n_crop'], self.vol_bound_all['query_vol'])
        latent_map = self.encode_latent_map(inputs_distributed, occupied_voxels, vol_bound_valid)
        latent_map = self.remove_padding(latent_map.squeeze(0))
        return latent_map, inputs_distributed
    
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
        
        return latent_map
    def get_input_crop(self, p_input, vol_bound):
        
        mask_x = (p_input[..., 0] >= vol_bound[0][0]) & \
                    (p_input[..., 0] < vol_bound[1][0])
        mask_y = (p_input[..., 1] >= vol_bound[0][1]) & \
                    (p_input[..., 1] < vol_bound[1][1])
        mask_z = (p_input[..., 2] >= vol_bound[0][2]) & \
                    (p_input[..., 2] < vol_bound[1][2])
        mask = mask_x & mask_y & mask_z
        
        return p_input[mask]
    
    def distribute_inputs(self, p_input, n_crop, vol_bound_all):
        n_inputs = len(p_input)
        occpied_voxels = torch.zeros(n_inputs, n_crop, dtype=torch.bool).to(self.device)
        inputs_croped_list = [[] for i in range(n_inputs)]
        vol_bound_valid = [[] for i in range(n_inputs)]
        
        for idx_pin in range(n_inputs):
            for i in range(n_crop):
                inputs_crop = self.get_input_crop(p_input[idx_pin], vol_bound_all[i])
                
                if inputs_crop.shape[0] > 0:
                    if isinstance(inputs_crop, np.ndarray):
                        inputs_crop = torch.from_numpy(inputs_crop).to(self.device)
                    inputs_croped_list[idx_pin].append(inputs_crop)
                    vol_bound_valid[idx_pin].append(vol_bound_all[i])
                    occpied_voxels[idx_pin, i] = True
                    
        return inputs_croped_list, occpied_voxels, vol_bound_valid
    
    def get_crop_bound(self, inputs, input_crop_size, query_crop_size, padding = True):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        '''

        vol_bound = {}

        if padding:
            lb = torch.min(inputs, dim=0).values.cpu().numpy() - 0.01 - query_crop_size #Padded once
            ub = torch.max(inputs, dim=0).values.cpu().numpy() + 0.01 + query_crop_size #Padded once
        else:
            lb = torch.min(inputs, dim=0).values.cpu().numpy() - 0.01 
            ub = torch.max(inputs, dim=0).values.cpu().numpy() + 0.01  
            
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
    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        occ_hat = pi.new_empty((pi.shape[0]))

        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {'p': pi_in}
        p_n = {}
        for key in self.vol_bound['fea_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(pi.clone(), vol_bound['query_vol'], plane=key).unsqueeze(0).to(self.device)
        pi_in['p_n'] = p_n
        
        min_vals, _ = torch.min(p_n['grid'], dim=1)
        max_vals, _ = torch.max(p_n['grid'], dim=1)

        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.model.decode(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)

        return occ_hat

    def eval_points(self, p, c=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound, **kwargs)
            occ_hats.append(occ_hat)

        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat
    # @profile
    def generate_mesh_from_neural_map(self, latent_all, return_stats=True):
        occ_values_x = self.generate_occupancy(latent_all)
        
        return self.generate_mesh_from_occ(occ_values_x, return_stats=return_stats)

    # @profile
    def generate_occupancy(self, latent_all):
        self.model.eval()
        device = self.device
        decoder_unet = self.unet.decoders

        n_crop = self.vol_bound['n_crop']
        print("Decoding latent codes from {} voxels".format(n_crop))
        # acquire the boundary for every crops
        kwargs = {}

        n_crop_axis = self.vol_bound['axis_n_crop']
        max_x, max_y, max_z = n_crop_axis[0], n_crop_axis[1], n_crop_axis[2]
                
        n = self.resolution0

        print(f'max_x: {max_x}, max_y: {max_y}, max_z: {max_z}, new resolution: {n}')
        value_grid = np.zeros((max_x, max_y, max_z, n, n, n), dtype = np.float32)

        for i in trange(n_crop):
            # 1D to 3D index
            x = i // (max_y * max_z)
            remainder_x = i % (max_y * max_z)
            y = remainder_x // max_z
            z = remainder_x % max_z

            latent = latent_all[x, y, z].unsqueeze(0)

            if torch.sum(latent) == 0.0:
                continue
            else:
                c = {}
                c['grid'] = self.unet3d_decode(latent, decoder_unet)
                c['grid'] = self.unet.final_conv(c['grid'])

                # encode the current crop
                vol_bound = {}
                vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
                vol_bound['input_vol'] = self.vol_bound['input_vol'][i]

                bb_min = self.vol_bound['query_vol'][i][0]
                bb_max = bb_min + self.vol_bound['query_crop_size']

                t = (bb_max - bb_min) / n  # interval
                
                pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]]
                pp = pp.reshape(3, -1).T
                pp = torch.from_numpy(pp).to(device)

                values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                values = values.reshape(n, n, n)

                # convert to probability here
                values = np.exp(values) / (1 + np.exp(values))
                value_grid[x, y, z] = values

        print("Organize voxels for mesh generation")

        r = n * 2 ** self.upsampling_steps
        occ_values = np.array([]).reshape(r, r, 0)
        occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])
        occ_values_x = np.array([]).reshape(0, r * n_crop_axis[1], r * n_crop_axis[2])

        for i in trange(n_crop):
            index_x = i // (max_y * max_z)
            remainder_x = i % (max_y * max_z)
            index_y = remainder_x // max_z
            index_z = remainder_x % max_z

            values = value_grid[index_x][index_y][index_z]

            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i + 1) % n_crop_axis[2] == 0:
                occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i + 1) % (n_crop_axis[2] * n_crop_axis[1]) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
                occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])
        del occ_values, occ_values_y
        
        occ_values_x = occ_values_x.astype(np.float32)
        
        return occ_values_x
    # @profile
    def generate_mesh_from_occ(self, value_grid, return_stats=True, stats_dict=dict()):
        print(f'Step 1')
        value_grid[np.where(value_grid == 1.0)] = 0.9999999
        print(f'Step 2')
        value_grid[np.where(value_grid == 0.0)] = 0.0000001
        print(f'Step 3')
        np.divide(value_grid, 1 - value_grid, out=value_grid)
        np.log(value_grid, out=value_grid)

        print("Generating mesh")
        t0 = time.time()
        mesh = self.extract_mesh(value_grid, stats_dict=stats_dict)
        t1 = time.time()
        generate_mesh_time = t1 - t0
        print("Mesh generated in {:.2f}s".format(generate_mesh_time))
        if return_stats:
            return mesh, stats_dict, value_grid
        else:
            return mesh

    def unet3d_decode(self, z, decoder_unet):
        ''' Decode latent code into feature volume

        Args:
            z (torch.Tensor): latent code
            decoder_unet (torch model): decoder
        '''

        batch_size, num_channels, h, w, d = z.size()

        with torch.no_grad():
            for depth in range(len(decoder_unet)):
                up = (depth + 1) * 2
                dummy_shape_tensor = torch.zeros(1, 1, h * up, w * up, d * up)
                z = decoder_unet[depth].upsampling(dummy_shape_tensor, z)
                z = decoder_unet[depth].basic_module(z)

        return z

    def extract_mesh(self, occ_hat, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (
                        self.vol_bound['axis_n_crop'].max() * self.resolution0 * 2 ** self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else:
            # Normalize to bounding box
            vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
            vertices = box_size * (vertices - 0.5)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        return mesh
