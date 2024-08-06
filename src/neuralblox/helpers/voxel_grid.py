import torch
import copy
import open3d as o3d

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
    def add_voxel_wi(self, center, latent, inputs, overwrite=False, threshold=20):
        h = self.compute_hash(center)
        list_keys = list(self.centers_table.keys())
        n_occ_inputs = inputs[..., 3].sum()
        # if n_occ_inputs < threshold and overwrite:
            # print(f'Not enough points in the occ input, skipping {h}, threshold={threshold}, n_occ_inputs={n_occ_inputs}')
        if (h not in list(self.centers_table.keys()) and n_occ_inputs > threshold) or (overwrite and h in list(self.centers_table.keys()) and n_occ_inputs > 4 * threshold):
            # print(f'Adding {h}, threshold={threshold}, n_occ_inputs={n_occ_inputs}')
            self.centers_table[h] = center
            self.latents_table[h] = latent
            
            pcd = o3d.geometry.PointCloud()
            points = torch.from_numpy(inputs.cpu().detach().numpy())[..., :3]
            occ = torch.from_numpy(inputs.cpu().detach().numpy())[..., 3]
            pcd.points = o3d.utility.Vector3dVector(points[occ == 1].cpu().detach().numpy())
            pcd.paint_uniform_color([1.0, 0.5, 0.0])
            self.pcd_table[h] = pcd
            
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