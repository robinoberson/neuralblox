import torch
import numpy as np
from tqdm import trange
import trimesh
from src.utils import libmcubes
from src.common import normalize_coord, add_key, coord2index
import time
from memory_profiler import profile


class Generator3DSequential(object):
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
                 unet_hdim = 32,
                 unet_depth = 2,
                 ):
        self.trainer = trainer
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
        
    def generate_sequence(self, batch):
        p_in = self.get_inputs(batch)
        n_sequence = p_in.shape[0]
        mesh_list = []
        inputs_frame_list = []
        
        for i in range(n_sequence):
        
            inputs_frame = p_in[i]
            inputs_frame_list.append(inputs_frame)
            
            if i == 0:
                self.trainer.voxel_grid.reset()
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.trainer.fuse_cold_start(inputs_frame)
                print(f'Voxel grid is empty, start with cold start')
            else:
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.trainer.fuse_inputs(inputs_frame)
                # print(f'Perform latent fusion')
            stacked_latents, centers = self.stack_latents_all()
            mesh = self.generate_mesh_from_neural_map(stacked_latents, centers, crop_size = self.input_crop_size, return_stats=True)
            mesh_list.append(mesh)
        return mesh_list, inputs_frame_list
    
    def generate_mesh_at_index(self, batch, index):
        p_in = self.get_inputs(batch)
        n_sequence = p_in.shape[0]
        mesh_list = []
        inputs_frame_list = []
        
        for i in range(n_sequence):
            if i > index:
                break
            inputs_frame = p_in[i]
            inputs_frame_list.append(inputs_frame)
            
            if i == 0:
                self.trainer.voxel_grid.reset()
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.trainer.fuse_cold_start(inputs_frame)
                print(f'Voxel grid is empty, start with cold start')
            else:
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.trainer.fuse_inputs(inputs_frame)
                print(f'Perform latent fusion')
                
            stacked_latents, centers = self.stack_latents_all()
            mesh, _ = self.generate_mesh_from_neural_map(stacked_latents, centers, crop_size = self.input_crop_size, return_stats=False)
            mesh_list.append(mesh)
        
        return mesh_list, inputs_frame_list
    def get_inputs(self, batch):
        p_in_3D = batch.get('inputs').to(self.device).squeeze(0)
        p_in_occ = batch.get('inputs.occ').to(self.device).squeeze(0).unsqueeze(-1)
        
        p_in = torch.cat((p_in_3D, p_in_occ), dim=-1)
        return p_in

    def fuse_inputs(self, inputs_frame):
        with torch.no_grad():
            if self.trainer.voxel_grid.is_empty():
                print(f'Voxel grid is empty, start with cold start')
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.trainer.fuse_cold_start(inputs_frame)
            else:
                print(f'Perform latent fusion')
                latent_map_stacked_merged, centers_frame_occupied, inputs_frame_distributed = self.trainer.fuse_inputs(inputs_frame)
            
        return latent_map_stacked_merged, centers_frame_occupied

    def stack_latents_region(self, x_limit, y_limit, z_limit):
        stacked_latents = None
        centers = None
        for x in range(x_limit[0], x_limit[1]):
            for y in range(y_limit[0], y_limit[1]):
                for z in range(z_limit[0], z_limit[1]):
                    center = (x, y, z)
                    latent = self.trainer.voxel_grid.get_latent(center)
                    
                    if latent is not None:
                        if stacked_latents is None:
                            stacked_latents = latent
                            centers = torch.tensor(center, device = self.device)
                        else:
                            stacked_latents = torch.cat((stacked_latents, latent), 0)
                            centers = torch.cat((centers, center), 0)
                            
        return stacked_latents, centers
    
    def stack_latents_all(self):
        stacked_latents = None
        centers_list = list(self.trainer.voxel_grid.centers_table.values())
        centers = None
        for center in centers_list:
            if centers is None:
                centers = center.unsqueeze(0)
            else:
                centers = torch.cat((centers, center.unsqueeze(0)), 0)
        
        for center in centers:
            latent = self.trainer.voxel_grid.get_latent(center).unsqueeze(0)
            
            if latent is not None:
                if stacked_latents is None:
                    stacked_latents = latent
                else:
                    stacked_latents = torch.cat((stacked_latents, latent), 0)
                    
        return stacked_latents, centers
        
    def generate_logits(self, latent_map_stacked_merged, centers_frame_occupied, p_query_distributed, centers_query, visualize=False):
        with torch.no_grad():
            p_stacked, latents, centers, occ = self.trainer.prepare_data_logits(latent_map_stacked_merged, centers_frame_occupied, p_query_distributed, centers_query)

            logits_sampled = self.trainer.get_logits(p_stacked, latents, centers)
            inputs_distributed = self.get_inputs_map(centers_frame_occupied)
            
            if visualize:
                self.trainer.visualize_logits(logits_sampled, p_stacked, inputs_distributed, force_viz = visualize)
        return logits_sampled, p_stacked
    
    def get_inputs_map(self, centers):
        inputs_distributed = None
        for center in centers:
            inputs = self.trainer.voxel_grid.get_points(center)
            if inputs is not None:
                if inputs_distributed is None:
                    inputs_distributed = inputs.unsqueeze(0)
                else:
                    inputs_distributed = torch.cat((inputs_distributed, inputs.unsqueeze(0)), 0)

        return inputs_distributed
            
    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        p_query_distributed = self.trainer.distribute_query(p_query, self.trainer.vol_bound_all)
        logits_sampled = self.trainer.get_logits(latent_map_sampled_merged, p_query_distributed, torch.tensor(self.trainer.vol_bound_all['input_vol'], device = self.trainer.device), remove_padding=True)

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
    def generate_mesh_from_neural_map(self, latent_map_full, centers, crop_size, return_stats=True):
        return self.generate_occupancy(latent_map_full, centers, crop_size)
        

    # @profile
    def generate_occupancy(self, latent_map_full, centers, crop_size):
        centers = np.array(centers.cpu())
        n_crop = latent_map_full.shape[0]
        print("Decoding latent codes from {} voxels".format(n_crop))
        # acquire the boundary for every crops
        kwargs = {}

        n_voxels = latent_map_full.shape[0]
        
        n = self.resolution0
        
        pp_full = np.zeros((n_voxels, n**3, 3))
        pp_n_full = np.zeros((n_voxels, n**3, 3))
        
        lb = centers - crop_size / 2.0
        ub = centers + crop_size / 2.0
        
        for i in range(n_voxels):
            center = centers[i]
            bb_min = lb[i]
            bb_max = ub[i]
            t = (bb_max - bb_min) / n
            
            pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]]
            pp = pp[:, :n, :n, :n]

            pp = pp.reshape(3, -1).T
            
            bb_size = bb_max - bb_min
            pp_centered = (pp-center)
            pp_n = pp_centered / bb_size 
            pp_n = pp_n + 0.5
            
            pp_full[i] = pp
            pp_n_full[i] = pp_n
            
        if self.limited_gpu:
            n_batch_max = 20
        else:
            n_batch_max = 1000
                
        n_batch = int(np.ceil(n_crop / n_batch_max))
        logits_stacked = None
        for i in range(n_batch):
            start = i * n_batch_max
            end = min((i + 1) * n_batch_max, n_crop)

            p_stacked_batch = torch.from_numpy(pp_full[start:end]).to(self.trainer.device)
            p_n_stacked_batch = torch.from_numpy(pp_n_full[start:end]).to(self.trainer.device)
            latent_map_full_batch = latent_map_full[start:end]

            kwargs = {}
            pi_in = p_stacked_batch[..., :3].clone()
            pi_in = {'p': pi_in}
            p_n = {}
            p_n['grid'] = p_n_stacked_batch[..., :3].clone()
            pi_in['p_n'] = p_n
            c = {}
            latent_map_full_batch_decoded = self.trainer.unet.decode(latent_map_full_batch, self.trainer.features_shapes)
            
            c['grid'] = latent_map_full_batch_decoded.clone()
            logits_decoded = self.trainer.model.decode(pi_in, c, **kwargs).logits
            
            if logits_stacked is None:
                logits_stacked = logits_decoded.clone()  # Initialize logits directly with the first batch
            else:
                logits_stacked = torch.cat((logits_stacked, logits_decoded), dim=0)  # Concatenate logits

        values = logits_stacked.detach().cpu().numpy()
        values = np.exp(values) / (1 + np.exp(values))
        
        pp_full = pp_full.reshape(-1, 3)
        values = values.reshape(-1)
        
        scaling_factor = self.resolution0/self.trainer.query_crop_size # Adjust this factor based on your desired resolution increase

        min_coords = np.min(pp_full, axis=0)
        shift_vector = -min_coords + 1  # Shift enough to make all coordinates positive

        # Scale the coordinates
        scaled_points = (pp_full + shift_vector) * scaling_factor

        # Round scaled coordinates to nearest integers for indexing
        x = np.round(scaled_points[:, 0]).astype(int)
        y = np.round(scaled_points[:, 1]).astype(int)
        z = np.round(scaled_points[:, 2]).astype(int)

        # Determine the maximum extents of scaled x, y, z
        x_max = x.max() + 1 if x.size > 0 else 0
        y_max = y.max() + 1 if y.size > 0 else 0
        z_max = z.max() + 1 if z.size > 0 else 0
        z_max = z.max() + 1 if z.size > 0 else 0

        # Determine original shape based on maximum extents
        original_shape = (x_max, y_max, z_max)
        
        values_reconstructed = np.zeros(original_shape)
        
        values_reconstructed[x, y, z] = values
        threshold = 0.2

        vertices, triangles = libmcubes.marching_cubes(values_reconstructed, threshold)
        mesh = trimesh.Trimesh(vertices/scaling_factor - shift_vector, triangles, vertex_normals=None, process=False)

        return mesh, pp_full
        
        # self.lb_overall = np.min(centers, axis=0) - self.trainer.query_crop_size / 2
        # self.ub_overall = np.max(centers, axis=0) + self.trainer.query_crop_size / 2
        
        # n_axis_crop = np.ceil((self.ub_overall - self.lb_overall) / self.trainer.query_crop_size).astype(int)
        # self.axis_n_crop = n_axis_crop
        # n_crop = n_axis_crop[0] * n_axis_crop[1] * n_axis_crop[2]
        # max_x, max_y, max_z = n_axis_crop[0], n_axis_crop[1], n_axis_crop[2] 
        # value_grid = np.zeros((max_x, max_y, max_z, n, n, n))

        # for idx_center in range(len(centers)):
        #     center = centers[idx_center, :]
        #     #convert to x,y,z 
        #     center = center - self.lb_overall
        #     center = center / self.trainer.query_crop_size
        #     center = center - 0.5
            
        #     x, y, z = int(center[0]), int(center[1]), int(center[2])
        #     value_grid[x,y,z] = values[idx_center]
        
        # if latent_map_full.shape[0] == 1:
        #     max_x, max_y, max_z = 1, 1, 1
        
        # # print("Organize voxels for mesh generation")
        
        # r = n * 2 ** self.upsampling_steps
        # occ_values = np.array([]).reshape(r, r, 0)
        # occ_values_y = np.array([]).reshape(r, 0, r * max_z)
        # occ_values_x = np.array([]).reshape(0, r * max_y, r * max_z)

        # for i in trange(n_crop):
        #     index_x = i // (max_y * max_z)
        #     remainder_x = i % (max_y * max_z)
        #     index_y = remainder_x // max_z
        #     index_z = remainder_x % max_z

        #     values = value_grid[index_x][index_y][index_z]

        #     occ_values = np.concatenate((occ_values, values), axis=2)
        #     # along y axis
        #     if (i + 1) % max_z == 0:
        #         occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
        #         occ_values = np.array([]).reshape(r, r, 0)
        #     # along x axis
        #     if (i + 1) % (max_z * max_y) == 0:
        #         occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
        #         occ_values_y = np.array([]).reshape(r, 0, r * max_z)
        # del occ_values, occ_values_y
        
        # occ_values_x = occ_values_x.astype(np.float32)
            
        # return occ_values_x
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
        # print("Mesh generated in {:.2f}s".format(generate_mesh_time))
        if return_stats:
            return mesh, stats_dict, value_grid
        else:
            return mesh

    def extract_mesh(self, occ_hat, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1
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
            
            # bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            # bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            
            mc_unit = max(self.ub_overall - self.lb_overall) / (
                        self.axis_n_crop.max() * self.resolution0 * 2 ** self.upsampling_steps)
            vertices = vertices * mc_unit + self.lb_overall
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
