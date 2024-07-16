class PrecomputedStorage:
    def __init__(self):
        self.storage_latents = {}   
        self.storage_indexes = {}
        self.storage_centers_occupied = {}
        self.storage_dim = {}
        self.storage_pcd = {}
        
        self.scene_idx = 0
        self.sequence_idx = 1
        
    def reset(self):
        print('Reset storage')
        self.storage_latents = {}   
        self.storage_indexes = {}
        self.storage_centers_occupied = {}
        self.storage_dim = {}
        
        self.scene_idx = 0
        self.sequence_idx = 1
        
    def compute_hash(self, scene_idx, sequence_idx):
        return f'{scene_idx}_{sequence_idx}'
               
    def add_sequence_idx(self, torch_latents, torch_indexes, torch_centers_occupied, grid_dim):
        h = self.compute_hash(self.scene_idx, self.sequence_idx)
        self.storage_latents[h] = torch_latents.detach().clone()
        self.storage_indexes[h] = torch_indexes.detach().clone()
        self.storage_centers_occupied[h] = torch_centers_occupied.detach().clone()
        self.storage_dim[h] = grid_dim.detach().clone()
        
        # print(f'Added scene {self.scene_idx} sequence {self.sequence_idx}')
        self.sequence_idx += 1
    
    def create_pcd_full(self, centers, voxel_grid, color):
        pcd_full = []
        for center in centers:
            pcd = voxel_grid.get_pcd(center)
            pcd.paint_uniform_color(color)
            pcd_full.append(pcd)

        self.add_pcd(pcd_full)

    def add_pcd(self, pcd_full):
        h = self.compute_hash(self.scene_idx, self.sequence_idx)
        self.storage_pcd[h] = pcd_full
    
    def get_pcd(self, scene_idx, sequence_idx):
        h = self.compute_hash(scene_idx, sequence_idx)
        pcd_full = self.storage_pcd.get(h, None)
        return pcd_full
    
    def visualize_pcd(self):
        pcd_full = self.get_pcd(self.scene_idx, self.sequence_idx + 1)
        o3d.visualization.draw_geometries(pcd_full)
        
    def load_idx(self, scene_idx, sequence_idx):
        h = f'{scene_idx}_{sequence_idx}'
        latents = self.storage_latents.get(h, None)
        indexes = self.storage_indexes.get(h, None)
        centers_occupied = self.storage_centers_occupied.get(h, None)
        
        return latents, indexes, centers_occupied