import torch
import numpy as np
from tqdm import trange
import trimesh
from src.utils import libmcubes
from src.common import normalize_coord, add_key, coord2index
import src.neuralblox.helpers.visualization_utils as vis_utils
import src.neuralblox.helpers.sequential_trainer_utils as st_utils
from src.neuralblox.helpers.voxel_grid import VoxelGrid
import time


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
                 prob_threshold=0.3, 
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
        self.prob_threshold = prob_threshold
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
        self.voxel_grid = VoxelGrid()
        self.points_threshold = self.trainer.points_threshold
        
    def generate_sequence(self, batch, idx_batch):
        self.trainer.model.eval()
        self.trainer.model_merge.eval()
        self.voxel_grid.reset()
        
        p_in, _ = st_utils.get_inputs_from_scene(batch, self.device)
        n_sequence = p_in.shape[1]
        mesh_list = []
        inputs_frame_list = []
        
        for idx_sequence in range(n_sequence):        
            inputs_frame = p_in[idx_batch][idx_sequence]
            inputs_frame_list.append(inputs_frame)
            
            if idx_sequence == 0:
                merged_latents, centers_frame, inputs_frame = self.fuse_cold_start(inputs_frame)
            else:
                merged_latents, centers_frame, inputs_frame = self.fuse(inputs_frame)
                # print(f'Perform latent fusion')
                
            for merged_latent, center, inputs_frame in zip(merged_latents, centers_frame, inputs_frame):
                self.voxel_grid.add_voxel_wi(center, merged_latent, inputs_frame, overwrite=True, threshold=self.points_threshold)
                
            stacked_latents, centers, pcds = self.stack_latents_all()
            mesh, _ = self.generate_mesh_from_neural_map(stacked_latents, centers, crop_size = self.trainer.query_crop_size, return_stats=False)
            mesh_list.append(mesh)
            
        return mesh_list, inputs_frame_list
    
    def generate_mesh_at_index(self, batch, index, idx_batch, generate_mesh = False, generate_logits = False, memory_keep = False):
        self.trainer.model.eval()
        self.trainer.model_merge.eval()
        
        p_in_full, p_query_full = st_utils.get_inputs_from_scene(batch, self.device)
        p_in = p_in_full[idx_batch]
        p_query = p_query_full[idx_batch]
        
        n_sequence = p_in.shape[0]
        mesh_list = []
        inputs_frame_list = []
        times = []
        logits = []
        
        inputs_frame = None
        
        for idx_sequence in range(n_sequence):
            t0 = time.time()
            if idx_sequence > index:
                break
            if inputs_frame is None:
                inputs_frame = p_in[idx_sequence]
            elif memory_keep:
                inputs_frame = torch.cat((inputs_frame, p_in[idx_sequence]), 0)
            else:
                inputs_frame = p_in[idx_sequence]

            inputs_frame_list.append(inputs_frame)
            
            if idx_sequence == 0:
                merged_latents, centers_frame, inputs_frame = self.fuse_cold_start(inputs_frame)
            else:
                merged_latents, centers_frame, inputs_frame = self.fuse(inputs_frame)
            
            times.append(time.time() - t0)
            
            for merged_latent, center, inputs_frame_lat in zip(merged_latents, centers_frame, inputs_frame):
                self.voxel_grid.add_voxel_wi(center, merged_latent, inputs_frame_lat, overwrite=True, threshold=self.points_threshold)
            
            occupied_voxels = inputs_frame[..., 3].sum(dim = -1) > 10

            if generate_logits:
                query_frame_distributed_padded, centers_frame_query, vol_bound_frame_query = self.trainer.get_distributed_inputs(p_query[idx_sequence], self.trainer.n_max_points_query, self.trainer.occ_per_query, return_empty = True, isquery = True, padding = True)
        
                query_mask = st_utils.compute_mask_occupied(centers_frame_query, centers_frame)
                centers_frame_query = centers_frame_query[query_mask].reshape(-1, 3)
                query_frame_distributed = query_frame_distributed_padded[query_mask].reshape(-1, self.trainer.n_max_points_query, 4)

                # print(f'Are centers the same? {torch.equal(centers_frame, centers_frame_query)}')
                
                p_stacked = query_frame_distributed
                centers = centers_frame_query
                occ = query_frame_distributed[..., 3]
            
                logits_sampled = self.trainer.get_logits(p_stacked, merged_latents, centers)
                logits.append([logits_sampled[occupied_voxels], p_stacked[occupied_voxels], inputs_frame[occupied_voxels], centers[occupied_voxels]])

            if generate_mesh or idx_sequence == index:
                print(f'Generating mesh at index {idx_sequence}')
                stacked_latents, centers, pcds = self.stack_latents_all()

                mesh, _ = self.generate_mesh_from_neural_map(stacked_latents, centers, crop_size = self.trainer.query_crop_size, return_stats=False)
                mesh_list.append(mesh)
            
            times.append(time.time() - t0)
        
        return mesh_list, inputs_frame_list, times, logits
    
    def generate_logits_at_index(self, batch, index, idx_batch, n_points_query=8192):
        self.trainer.model.eval()
        self.trainer.model_merge.eval()
        
        p_in_full, p_query_full = st_utils.get_inputs_from_scene(batch, self.device)
        p_in = p_in_full[idx_batch]
        p_query = p_query_full[idx_batch]
        
        n_sequence = p_in.shape[0]
        
        logits = []
        
        for idx_sequence in range(n_sequence):
            if idx_sequence > index:
                break
            
            inputs_frame = p_in[idx_sequence]
            if idx_sequence == 0:
                merged_latents, centers_frame, inputs_frame = self.fuse_cold_start(inputs_frame)
            else:
                merged_latents, centers_frame, inputs_frame = self.fuse(inputs_frame)
                
            for merged_latent, center, inputs_frame_lat in zip(merged_latents, centers_frame, inputs_frame):
                self.voxel_grid.add_voxel_wi(center, merged_latent, inputs_frame_lat, overwrite=True, threshold=self.points_threshold)
                
            stacked_latents, centers, pcds = self.stack_latents_all()
            
            occupied_voxels = inputs_frame[..., 3].sum(dim = -1) > self.points_threshold
            
            query_points = st_utils.get_empty_inputs(centers, crop_size = self.trainer.query_crop_size, n_max_points = n_points_query)
            
            logits_sampled = self.trainer.get_logits(query_points, stacked_latents, centers)
            
            logits.append([logits_sampled, query_points, inputs_frame, centers])
        
        return logits

    def get_inputs(self, batch):
        p_in_3D = batch.get('inputs').to(self.device).squeeze(0)
        p_in_occ = batch.get('inputs.occ').to(self.device).squeeze(0).unsqueeze(-1)
        
        p_in = torch.cat((p_in_3D, p_in_occ), dim=-1)
        return p_in

    def stack_latents_region(self, x_limit, y_limit, z_limit):
        stacked_latents = None
        centers = None
        for x in range(x_limit[0], x_limit[1]):
            for y in range(y_limit[0], y_limit[1]):
                for z in range(z_limit[0], z_limit[1]):
                    center = (x, y, z)
                    latent = self.voxel_grid.get_latent(center)
                    
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
        centers_list = list(self.voxel_grid.centers_table.values())
        centers = None
        pcds = []
        for center in centers_list:
            if centers is None:
                centers = center.unsqueeze(0)
            else:
                centers = torch.cat((centers, center.unsqueeze(0)), 0)
        
        for center in centers:
            latent = self.voxel_grid.get_latent(center).unsqueeze(0)
            pcd = self.voxel_grid.get_pcd(center)
            pcds.append(pcd)
            
            if latent is not None:
                if stacked_latents is None:
                    stacked_latents = latent
                else:
                    stacked_latents = torch.cat((stacked_latents, latent), 0)
                    
        return stacked_latents, centers, pcds
        

    def get_inputs_map(self, centers):
        inputs_distributed = None
        for center in centers:
            inputs = self.voxel_grid.get_points(center)
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
            pp_n = pp_centered / self.trainer.input_crop_size
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

        vertices, triangles = libmcubes.marching_cubes(values_reconstructed, self.prob_threshold)
        mesh = trimesh.Trimesh(vertices/scaling_factor - shift_vector, triangles, vertex_normals=None, process=False)

        return mesh, pp_full
    
    def fuse_cold_start(self, inputs_frame_raw):
        inputs_frame_padded, centers_frame_padded, vol_bound_frame_padded = self.trainer.get_distributed_inputs(inputs_frame_raw, self.trainer.n_max_points_input, 1.0, return_empty = True, isquery = False, padding = True)
        n_x, n_y, n_z = vol_bound_frame_padded['axis_n_crop'] #padded

        centers_frame = centers_frame_padded.reshape(n_x, n_y, n_z, 3)[1:-1, 1:-1, 1:-1, :].reshape(-1, 3)
        inputs_frame = inputs_frame_padded.reshape(n_x, n_y, n_z, self.trainer.n_max_points_input, 4)[1:-1, 1:-1, 1:-1, :, :].reshape(-1, self.trainer.n_max_points_input, 4)

        grid_latents_frame_current = self.trainer.encode_occupied_inputs_frame(inputs_frame, centers_frame, vol_bound_frame_padded, threshold = self.points_threshold) #contains padded empty latents
        centers_lookup = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame), 1).to(self.device)
        grid_shapes = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame), 1).to(self.device)

        centers_frame_idx = st_utils.centers_to_grid_indexes(centers_frame, vol_bound_frame_padded['lb'], self.trainer.query_crop_size).int().reshape(-1, 3)

        distributed_latents = st_utils.get_distributed_voxel(centers_frame_idx, grid_latents_frame_current, grid_shapes, centers_lookup, self.trainer.shifts)
        c, h, w, d = self.trainer.empty_latent_code.shape

        distributed_latents = distributed_latents.reshape(-1, c, 3*h, 3*w, 3*d)
        stacked_frame = torch.cat((distributed_latents, distributed_latents), dim = 1)
        merged_latents = self.trainer.merge_latent_map(stacked_frame)
        
        return merged_latents, centers_frame, inputs_frame
    
    def init_timing(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()
        
    def print_timing(self, description):
        self.end.record()
        torch.cuda.synchronize()
        print(f'{description}: {self.start.elapsed_time(self.end)} ms')
        self.start.record()
        
    def fuse(self, inputs_frame_raw):
        self.init_timing()
        
        inputs_frame_padded, centers_frame_padded, vol_bound_frame_padded = self.trainer.get_distributed_inputs(inputs_frame_raw, self.trainer.n_max_points_input, 1.0, return_empty = True, isquery = False, padding = True)
        n_x, n_y, n_z = vol_bound_frame_padded['axis_n_crop'] #padded

        centers_frame = centers_frame_padded.reshape(n_x, n_y, n_z, 3)[1:-1, 1:-1, 1:-1, :].reshape(-1, 3)
        inputs_frame = inputs_frame_padded.reshape(n_x, n_y, n_z, self.trainer.n_max_points_input, 4)[1:-1, 1:-1, 1:-1, :, :].reshape(-1, self.trainer.n_max_points_input, 4)

        # self.print_timing('get_distributed_inputs')
        
        grid_latents_frame_current = self.trainer.encode_occupied_inputs_frame(inputs_frame, centers_frame, vol_bound_frame_padded, self.points_threshold) #contains padded empty latents

        # self.print_timing('encode_occupied_inputs_frame')
        
        centers_lookup = torch.tensor([0, (n_x * n_y * n_z)]).repeat(len(centers_frame), 1).to(self.device)
        grid_shapes = torch.tensor([n_x, n_y, n_z]).repeat(len(centers_frame), 1).to(self.device)

        centers_frame_idx = st_utils.centers_to_grid_indexes(centers_frame, vol_bound_frame_padded['lb'], self.trainer.query_crop_size).int().reshape(-1, 3)

        distributed_latents = st_utils.get_distributed_voxel(centers_frame_idx, grid_latents_frame_current, grid_shapes, centers_lookup, self.trainer.shifts)
        c, h, w, d = self.trainer.empty_latent_code.shape
        distributed_latents = distributed_latents.reshape(-1, c, 3*h, 3*w, 3*d)
        
        # self.print_timing('get_distributed_voxel')
        
        ### retrieve existing latents 
        grid_latents_frame_existing = grid_latents_frame_current.clone()
        grid_latents_frame_existing = grid_latents_frame_existing.reshape(n_x, n_y, n_z, *self.trainer.empty_latent_code.shape)
        
        existing_latent = 0
        for center_idx, center_coords in zip(centers_frame_idx, centers_frame):
            merged_latent = self.voxel_grid.get_latent(center_coords)
            if merged_latent is not None:
                grid_latents_frame_existing[center_idx[0], center_idx[1], center_idx[2]] = merged_latent
            existing_latent += 1
        
        grid_latents_frame_existing = grid_latents_frame_existing.reshape(-1, c, h, w, d)
        distributed_latents_existing = st_utils.get_distributed_voxel(centers_frame_idx, grid_latents_frame_existing, grid_shapes, centers_lookup, self.trainer.shifts)
        distributed_latents_existing = distributed_latents_existing.reshape(-1, c, 3*h, 3*w, 3*d)
        
        # self.print_timing('get_distributed_voxel_existing')
        
        stacked_frame = torch.cat((distributed_latents, distributed_latents_existing), dim = 1)
        merged_latents = self.trainer.merge_latent_map(stacked_frame)
        
        self.print_timing('merge_latent_map')
        
        print('*' * 50)
        print('')
        # print(f'found {existing_latent} existing latents out of {(n_x-2)*(n_y-2)*(n_z-2)}')
        # print(f'should be {(n_x-2)*(n_y-2)*(n_z-2)}, is {merged_latents.shape[0]}')
        return merged_latents, centers_frame, inputs_frame
