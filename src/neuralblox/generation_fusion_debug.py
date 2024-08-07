import torch
import numpy as np
from tqdm import trange
import trimesh
import sys
import os
sys.path.append(os.getcwd())
from src.utils import libmcubes
from src.common import normalize_coord, add_key, coord2index
import src.neuralblox.helpers.visualization_utils as vis_utils
import src.neuralblox.helpers.sequential_trainer_utils as st_utils
from src.neuralblox.helpers.voxel_grid import VoxelGrid
import time


voxel_grid = VoxelGrid()

centers = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,1]])
inputs = torch.randn(4, 1000, 3)
occ = torch.ones(4, 1000, 1)

inputs = torch.cat((inputs, occ), 2)
latents = torch.randn(4, 128, 6, 6, 6)

for i in range(4):
    center = centers[i]
    latent = latents[i]
    inputs_frame = inputs[i]
    voxel_grid.add_voxel_wi(center, latent, inputs_frame)
    
def stack_latents_all():
    stacked_latents = None
    centers_list = list(voxel_grid.centers_table.values())
    centers = None
    pcds = []
    for center in centers_list:
        if centers is None:
            centers = center.unsqueeze(0)
        else:
            centers = torch.cat((centers, center.unsqueeze(0)), 0)
    
    for center in centers:
        latent = voxel_grid.get_latent(center).unsqueeze(0)
        pcd = voxel_grid.get_pcd(center)
        pcds.append(pcd)
        
        if latent is not None:
            if stacked_latents is None:
                stacked_latents = latent
            else:
                stacked_latents = torch.cat((stacked_latents, latent), 0)
                
    return stacked_latents, centers, pcds

stacked_latents, centers, pcds = stack_latents_all()

print(centers)