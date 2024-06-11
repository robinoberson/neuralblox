import torch
import torch.nn as nn
from src.layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from src.common import coordinate2index, normalize_3d_coordinate, \
    map2local, positional_encoding
from src.encoder.unet3d_noskipconnection import UNet3D
from src.encoder.unet3d_noskipconnection_latent import UNet3D_noskipconnection_latent
import math
import time


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'grid' - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max',
                 unet3d=False, unet3d_kwargs=None,
                 grid_resolution=None, plane_type='grid', padding=0.1, n_blocks=5,
                 pos_encoding=True, unit_size=0.1):
        super().__init__()
        self.c_dim = c_dim

        if pos_encoding == True:
            dim = 60
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet3d:
            self.unet3d = UNet3D_noskipconnection_latent(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid,
                                    self.reso_grid)  # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid, latent_code = self.unet3d(fea_grid)
            return fea_grid, latent_code
        else:
            return fea_grid, None

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)

            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        
        fea = {}

        if 'grid' in self.plane_type:
            fea['grid'], latent_code = self.generate_grid_features(p, c)

        return fea, net, c, latent_code

class PatchLocalPoolPointnetLatent(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
        First transform input points to local system based on the given voxel size.
        Support non-fixed number of point cloud, but need to precompute the index

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'grid' - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        local_coord (bool): whether to use local coordinate
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        unit_size (float): defined voxel unit size for local system
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max',
                 unet3d=False, unet3d_kwargs=None, grid_resolution=None, plane_type='grid', padding=0.1, n_blocks=5,
                 local_coord=False, pos_encoding='linear', unit_size=0.1):
        super().__init__()
        self.c_dim = c_dim

        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if unet3d:
            self.unet3d = UNet3D_noskipconnection_latent(**unet3d_kwargs)
        else:
            self.unet3d = None

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2 * hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

    def generate_grid_features(self, index, c):
        # scatter grid features from points
        c = c.permute(0, 2, 1)
        # fea_grid = c.new_zeros(c.size(0), self.c_dim, 2 * self.reso_grid ** 3) # 27648
        # test = scatter_mean(c, index) #27649
        # fea_grid = scatter_mean(c, index, out=fea_grid)  # B x c_dim x reso^3
        # current: 
        # fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        #option 1: 
        # fea_grid = fea_grid.reshape(c.size(0), 2* self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        # option 2:
        # fea_grid = fea_grid.reshape(c.size(0), 2, self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        # # option 3: 
        # new_reso = math.ceil(((self.reso_grid ** 3) * 2) ** (1/3))
        # fea_grid = fea_grid.reshape(c.size(0), self.c_dim, new_reso, new_reso, new_reso)

        if index.max() < 2*self.reso_grid ** 3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, 2 * self.reso_grid ** 3)
            fea_grid = scatter_mean(c, index, out=fea_grid)  # B x c_dim x reso^3
        else:
            fea_grid = scatter_mean(c, index)  # B x c_dim x reso^3
            if fea_grid.shape[-1] > 2 * self.reso_grid ** 3:  # deal with outliers
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), 2*self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)


        return fea_grid
    def pool_local(self, index, c, limited_gpu=False):
        # Memory efficient version
        
        indexes = index['grid']
        n_batch, n_points, n_features = c.shape
        
        unique_indexes, inverse_indices = torch.unique(indexes.reshape(-1), return_inverse=True)
        n_unique_indexes = len(unique_indexes)
        
        sum_tensor_flat = torch.zeros(n_batch, n_features, n_unique_indexes, dtype=torch.float).reshape(-1).to(c.device)
        count_tensor_flat = torch.zeros(n_batch, n_features, n_unique_indexes, dtype=torch.float).reshape(-1).to(c.device)
        
        feature_indices = torch.arange(n_features).repeat_interleave(n_points).to(c.device)
        base_indices = feature_indices * n_unique_indexes
        
        inverse_indices_expanded = inverse_indices.view(n_batch, 1, n_points).expand(-1, n_features, -1)
        base_indices_expanded = base_indices.view(1, n_features, n_points).expand(n_batch, -1, -1).to(c.device)
        batch_offset = (torch.arange(n_batch) * n_features * n_unique_indexes).view(n_batch, 1, 1).to(c.device)
        
        inverse_indices_flat = (inverse_indices_expanded + base_indices_expanded + batch_offset).reshape(-1)
        
        sum_tensor_flat.index_add_(0, inverse_indices_flat, c.permute(0, 2, 1).reshape(-1))
        count_tensor_flat.index_add_(0, inverse_indices_flat, torch.ones_like(c.permute(0, 2, 1)).reshape(-1))
        
        mean_values = sum_tensor_flat / count_tensor_flat
        final_tensor = mean_values[inverse_indices_flat].view(n_batch, n_features, n_points)

        return final_tensor.permute(0, 2, 1)
                
    def pool_local_fast(self, index, c, limited_gpu = False):
        self.t0 = time.time()
        indexes = index['grid']
            
        bs, fea_dim = c.size(0), c.size(2)
        
        fea = torch.empty(bs, fea_dim, 2 * self.reso_grid ** 3, requires_grad=True).to(c.device)

        if self.scatter == scatter_max:
            fea, _ = self.scatter(c.permute(0, 2, 1), indexes, out = fea)
        else: 
            fea = self.scatter(c.permute(0, 2, 1), indexes, out = fea)
                
        fea = fea.gather(dim=2, index=indexes.expand(-1, fea_dim, -1))
                
        return fea.permute(0, 2, 1)

    def print_time(self, description):
        t1 = time.time()
        print(f'Time elapsed in pooled: {t1 - self.t0:.3f}, {description}')
        self.t0 = time.time()
        
    def forward(self, inputs, limited_gpu = False):
        self.t0 = time.time()
        
        p = inputs['points']
        index = inputs['index']
        
        # bb_min_p = torch.min(p.reshape(-1, 3), dim=0).values
        # bb_max_p = torch.max(p.reshape(-1, 3), dim=0).values
        
        # print(bb_min_p, bb_max_p)
        
        # dir = 'training_output'
        dir = 'generator_output'
        
        fea = {}

        if self.map2local:
            pp = self.map2local(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
            
        
        net = self.blocks[0](net)
        for idx_block, block in enumerate(self.blocks[1:]):
            
            pooled = self.pool_local(index, net, limited_gpu = limited_gpu)
            net = torch.cat([net, pooled], dim=2)

            net = block(net)     
        c = self.fc_c(net)
                
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(index['grid'], c)
            
        unet = self.unet3d

        if limited_gpu:
            del c, net, pooled, pp
        return fea['grid'], unet