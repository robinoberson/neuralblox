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

    def __init__(self, model, model_merge, trainer,
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
        self.trainer = trainer
        self.model = model.to(device)
        self.model_merge = model_merge.to(device)
        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
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
        self.limited_gpu = True
        # for pointcloud_crop
        self.vol_bound = vol_bound
        self.grid_reso = vol_bound['reso']
        self.input_crop_size = vol_bound['input_crop_size']
        self.query_crop_size = vol_bound['query_crop_size']
        self.reso = vol_bound['reso']
        self.factor = 2**unet_depth
        self.hdim = unet_hdim
                
    def generate_latent_no_merge(self, data: torch.Tensor, verbose: bool = False):
        self.model.eval()
        self.model_merge.eval()
        self.trainer.model.eval()
        self.trainer.model_merge.eval()
        
        with torch.no_grad():
        
            if data.shape[-1] != 4 or len(data.shape) != 3:
                raise ValueError('data should be a tensor with shape (m, n, 4)')
            
            self.trainer.vol_bound_all = self.trainer.get_crop_bound(data.view(-1, 4), self.trainer.input_crop_size, self.trainer.query_crop_size)
            latent_map, inputs_distributed = self.trainer.get_latent_gt(data)

        self.unet = self.trainer.unet
        self.vol_bound = self.trainer.vol_bound_all
        return latent_map, inputs_distributed

    def generate_latent(self, inputs: torch.Tensor, verbose: bool = False):
        self.model.eval()
        self.model_merge.eval()
        self.trainer.model.eval()
        self.trainer.model_merge.eval()

        with torch.no_grad():
        
            if inputs.shape[-1] != 4 or len(inputs.shape) != 3 or inputs.shape[0] != 2:
                raise ValueError('inputs should be a tensor with shape (2, n, 4)')
            
            self.trainer.vol_bound_all = self.trainer.get_crop_bound(inputs.view(-1, 4), self.trainer.input_crop_size, self.trainer.query_crop_size)

            inputs_distributed = self.trainer.distribute_inputs(inputs.unsqueeze(1), self.trainer.vol_bound_all)
            latent_map_sampled = self.trainer.encode_latent_map(inputs_distributed, self.trainer.vol_bound_all['input_vol'])
            latent_map_sampled_stacked = self.trainer.stack_latents_safe(latent_map_sampled, self.trainer.vol_bound_all)
            latent_map_sampled_merged = self.trainer.merge_latent_map(latent_map_sampled_stacked) 
            
        self.unet = self.trainer.unet
        self.vol_bound = self.trainer.vol_bound_all
        
        return latent_map_sampled_merged

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
            p_n[key] = normalize_coord(pi.clone(), vol_bound['input_vol'], plane=key).unsqueeze(0).to(self.device)
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
        # return occ_values_x
        return self.generate_mesh_from_occ(occ_values_x, return_stats=return_stats)

    # @profile
    def generate_occupancy(self, latent_map_full):
        self.model.eval()
        device = self.device
        decoder_unet = self.unet.decoders

        n_crop = latent_map_full.shape[0]
        print("Decoding latent codes from {} voxels".format(n_crop))
        # acquire the boundary for every crops
        kwargs = {}

        vol_bound = self.vol_bound['query_vol'].copy()
        vol_bound_inner = self.trainer.remove_padding_single_dim(vol_bound)
        vol_bound_inner_reshaped = vol_bound_inner.reshape(-1, *vol_bound_inner.shape[3:])
        
        n = self.resolution0
        
        pp_full = np.zeros((vol_bound_inner_reshaped.shape[0], n**3, 3))
        pp_n_full = np.zeros((vol_bound_inner_reshaped.shape[0], n**3, 3))
        
        for i in range(vol_bound_inner_reshaped.shape[0]):
            bb_min = vol_bound_inner_reshaped[i, 0]
            bb_max = vol_bound_inner_reshaped[i, 1]
            t = (bb_max - bb_min) / n
            # print(t)
            
            centers = (bb_min + bb_max) / 2.0
            
            pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]]
            pp = pp[:n, :n, :n]
            # print(pp.shape)
            # print(bb_max - bb_min)
            pp = pp.reshape(3, -1).T

            bb_size = bb_max - bb_min
            pp_n = (pp-centers) / bb_size + 0.5
            
            pp_full[i] = pp
            pp_n_full[i] = pp_n
            
        if self.limited_gpu:
            n_batch_max = 10
        else:
            n_batch_max = 1000
                
        n_batch = int(np.ceil(n_crop / n_batch_max))
        logits_stacked = None
        for i in range(n_batch):
            start = i * n_batch_max
            end = min((i + 1) * n_batch_max, n_crop)

            p_stacked_batch = torch.from_numpy(pp_full[start:end]).to(self.device)
            p_n_stacked_batch = torch.from_numpy(pp_n_full[start:end]).to(self.device)
            latent_map_full_batch = latent_map_full[start:end]

            p_n_dict = {'grid': p_n_stacked_batch}

            pi_in = {'p': p_stacked_batch}
            pi_in['p_n'] = p_n_dict
            # Get latent codes of valid crops
            kwargs = {}
            fea = {}
            fea['unet3d'] = self.unet
            fea['latent'] = latent_map_full_batch

            p_r = self.model.decode(pi_in, fea, **kwargs)
            logits_decoded = p_r.logits

            if logits_stacked is None:
                logits_stacked = logits_decoded.clone()  # Initialize logits directly with the first batch
            else:
                logits_stacked = torch.cat((logits_stacked, logits_decoded), dim=0)  # Concatenate logits

        values = logits_stacked.detach().cpu().numpy().reshape(-1, n, n, n)
        values = np.exp(values) / (1 + np.exp(values))
        
        n_axis_crop = self.vol_bound['axis_n_crop']
        max_x, max_y, max_z = n_axis_crop[0] - 2, n_axis_crop[1] - 2, n_axis_crop[2] - 2
        
        if latent_map_full.shape[0] == 1:
            max_x, max_y, max_z = 1, 1, 1
        
        value_grid = values.reshape(max_x, max_y, max_z, n, n, n)

        print("Organize voxels for mesh generation")
        
        r = n * 2 ** self.upsampling_steps
        occ_values = np.array([]).reshape(r, r, 0)
        occ_values_y = np.array([]).reshape(r, 0, r * max_z)
        occ_values_x = np.array([]).reshape(0, r * max_y, r * max_z)

        for i in trange(n_crop):
            index_x = i // (max_y * max_z)
            remainder_x = i % (max_y * max_z)
            index_y = remainder_x // max_z
            index_z = remainder_x % max_z

            values = value_grid[index_x][index_y][index_z]

            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i + 1) % max_z == 0:
                occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i + 1) % (max_z * max_y) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
                occ_values_y = np.array([]).reshape(r, 0, r * max_z)
        del occ_values, occ_values_y
        
        occ_values_x = occ_values_x.astype(np.float32)
        
        return occ_values_x
    # @profile
    def generate_mesh_from_occ(self, value_grid, return_stats=True, stats_dict=dict()):
        value_grid[np.where(value_grid == 1.0)] = 0.9999999
        value_grid[np.where(value_grid == 0.0)] = 0.0000001
        np.divide(value_grid, 1 - value_grid, out=value_grid)
        np.log(value_grid, out=value_grid)

        # print("Generating mesh")
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
                dummy_shape_tensor = torch.zeros(1, 1, h * up, w * up, d * up).to(self.device)
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
            vol_bound = self.trainer.remove_padding_single_dim(self.vol_bound['query_vol'].copy()).reshape(-1, 2, 3)
            bb_min = vol_bound[:, 0].min(axis=0)
            bb_max = vol_bound[:, 1].max(axis=0)
            # bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            # bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            axis_n_crop = self.vol_bound['axis_n_crop'] - np.array([2, 2, 2])
            mc_unit = max(bb_max - bb_min) / (
                        axis_n_crop.max() * self.resolution0 * 2 ** self.upsampling_steps)
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
