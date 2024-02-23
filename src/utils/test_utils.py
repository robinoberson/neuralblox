from pathlib import Path
from os.path import join
import numpy as np
import torch
from src.neuralblox.generation_fusion import Generator3D

def process_and_generate_latent(pcl_unaligned, align_matrix, generator : Generator3D, device, export_pc, out_dir, file_name):

    N = pcl_unaligned.shape[0]
    one = np.ones((N, 1))
    pcl_unaligned = np.hstack((pcl_unaligned, one))
    
    pcl_unaligned = torch.from_numpy(pcl_unaligned).to(device).float()
    pcl_world = torch.matmul(align_matrix, pcl_unaligned.T).T
    pcl_world = pcl_world[:, :3]

    if export_pc:
        sampled_index = np.random.permutation(len(pcl_world))[:int(0.5 * len(pcl_world))]
        sampled_pcl = pcl_world[sampled_index]
    
        Path(join(out_dir, 'pcl')).mkdir(parents=True, exist_ok=True)
        pcl_out_file = join(out_dir, 'pcl', f'{file_name}.npy')
        np.save(pcl_out_file, sampled_pcl.detach().cpu().numpy())

    input_data = {'inputs': pcl_world.unsqueeze(0)}
    latent = generator.generate_latent(input_data)
    
    return latent

def generate_and_save_mesh(generator, latent, out_dir, mesh_name):
    # Update latent representation
    latent_merged = generator.update_all(latent)
    
    # Generate mesh from neural map
    mesh = generator.generate_mesh_from_neural_map(latent_merged)[0]
    
    # Create directory if it doesn't exist
    Path(join(out_dir, 'mesh')).mkdir(parents=True, exist_ok=True)
    
    # Define mesh output file path
    mesh_out_file = join(out_dir, 'mesh', f'{mesh_name}.off')
    
    # Print info
    print(f'Saving mesh to {mesh_out_file}')
    
    # Export mesh
    mesh.export(mesh_out_file)