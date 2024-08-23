import torch
import copy
import open3d as o3d
import numpy as np

class VoxelGrid:
    def __init__(self):
        self.centers_table = {}
        self.latents_table = {}
        self.pcd_table = {}
        self.latent_shape = None
        self.verbose = False
    def compute_hash(self, center):
        # Assuming center is a tuple or a tensor with (x, y, z) coordinates
        return f"{torch.round(center[0]*1000)/1000:.3f}_{torch.round(center[1]*1000)/1000:.3f}_{torch.round(center[2]*1000)/1000:.3f}"
    
    def add_voxel(self, center, latent, overwrite=False):
        h = self.compute_hash(center)
        list_keys = list(self.centers_table.keys())
        if overwrite or h not in list(self.centers_table.keys()):
            if self.verbose:
            #print if overwriting has been requested
                if overwrite and h in list(self.centers_table.keys()):
                    print('Overwriting', h)
                
            self.centers_table[h] = center
            self.latents_table[h] = latent
            
        if self.latent_shape is None:
            self.latent_shape = latent.shape    
    def add_voxel_wi(self, center, latent, inputs, center_frame, overwrite_distance, overwrite=False, threshold=0):
        h = self.compute_hash(center)
        # print(h)
        list_keys = list(self.centers_table.keys())
        # print(f'list_keys: {list_keys}')
        n_occ_inputs = inputs[..., 3].sum()
        # if n_occ_inputs < threshold and overwrite:
            # print(f'Not enough points in the occ input, skipping {h}, threshold={threshold}, n_occ_inputs={n_occ_inputs}')
        # print(f'Adding {h}, threshold={threshold}, n_occ_inputs={n_occ_inputs}')
        h_in_list_bool = h in list(self.centers_table.keys())
        if (not h_in_list_bool and n_occ_inputs >= threshold) or (overwrite and h_in_list_bool and n_occ_inputs >= 2 * threshold):
            # print(f'Adding {h}, threshold={threshold}, n_occ_inputs={n_occ_inputs}')
            
            block_overwrite = False
            
            if h_in_list_bool: 
                current_occ_inputs = len(self.pcd_table.get(h, None).points)
                if n_occ_inputs < current_occ_inputs*0.75: #if the number of points is less than 75% of the current number of points, do not overwrite
                    # print(f'Not enough points in the occ input, skipping {h}, threshold={threshold}, n_occ_inputs={n_occ_inputs}, current_occ_inputs={current_occ_inputs}')
                    block_overwrite = True

                if torch.norm(center - center_frame) > overwrite_distance:
                    # print(f'Center too far from frame, skipping {h}, distance = {torch.norm(center - center_frame)}, overwrite_distance = {overwrite_distance}')
                    block_overwrite = True #if the distance is greater than the overwrite distance, do not overwrite
            
            if not block_overwrite:
                pcd = o3d.geometry.PointCloud()
                points = inputs.cpu().detach().numpy().astype(np.float64)[..., :3]
                occ = inputs.cpu().detach().numpy().astype(np.float64)[..., 3]
                pcd.points = o3d.utility.Vector3dVector(points[occ == 1])
                pcd.paint_uniform_color([1.0, 0.5, 0.0])
                
                self.pcd_table[h] = pcd
                self.centers_table[h] = center
                self.latents_table[h] = latent
            
    def add_pcd(self, center, inputs):
        h = self.compute_hash(center)
        pcd = o3d.geometry.PointCloud()
        points = torch.from_numpy(inputs.cpu().detach().numpy())[..., :3]
        occ = torch.from_numpy(inputs.cpu().detach().numpy())[..., 3]
        pcd.points = o3d.utility.Vector3dVector(points[occ == 1].cpu().detach().numpy())
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.pcd_table[h] = pcd
        
    def get_voxel(self, center):
        h = self.compute_hash(center)
        center = self.centers_table.get(h, None)
        latent = self.latents_table.get(h, None)
        return center, latent
    
    def get_voxel_wi(self, center):
        h = self.compute_hash(center)
        center = self.centers_table.get(h, None)
        latent = self.latents_table.get(h, None)
        pcd = self.pcd_table.get(h, None)
        return center, latent, pcd
    
    def get_latent(self, center):
        h = self.compute_hash(center)
        latent = self.latents_table.get(h, None)
        return latent
    
    def get_pcd(self, center):
        h = self.compute_hash(center)
        pcd = self.pcd_table.get(h, None)
        return pcd
    
    def paint_pcd(self):
        for pcd in self.pcd_table.values():
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            
    def detach_latents(self):
        self.latents_table = {k: v.detach() for k, v in self.latents_table.items()}
        
    def reset(self):
        self.centers_table = {}
        self.latents_table = {}
        self.latent_shape = None
        
    def copy(self):
        return copy.deepcopy(self)
    
    def is_empty(self):
        return len(self.latents_table) == 0