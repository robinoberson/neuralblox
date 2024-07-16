import open3d as o3d
import matplotlib.pyplot as plt
import os 
import yaml
import numpy as np
import torch 

def visualize_weights(weights, p_query, inputs_distributed):
    """
    Visualize point cloud `p_query` with colors based on `weights`.

    Parameters:
    weights (torch.Tensor): Weights for each point in `p_query`, shape (n_points,)
    p_query (torch.Tensor): Point cloud data, shape (n_points, 3)

    Returns:
    None
    """
    # Convert torch tensors to numpy arrays
    inputs_distributed_np = inputs_distributed.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    p_query_np = p_query[..., :3].detach().cpu().numpy()
    
    n_skip = 20
    geos = []
    for i in range(p_query_np.shape[0]):

        # Create Open3D PointCloud
        pcd_query = o3d.geometry.PointCloud()
        pcd_query.points = o3d.utility.Vector3dVector(p_query_np[i][::n_skip])
        
        pcd_in = o3d.geometry.PointCloud()
        inputs_points = inputs_distributed_np[i]
        pcd_in.points = o3d.utility.Vector3dVector(inputs_points[inputs_points[:, 3] == 1][:, :3])
        pcd_in.paint_uniform_color([.0, 1., 0.])

        # Normalize weights to [0, 1] for colormap
        colors = plt.cm.jet(weights_np[i])[::n_skip]

        # Assign colors to PointCloud
        pcd_query.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Use RGB values from colormap
        geos.append(pcd_query)
        geos.append(pcd_in)
        
        # o3d.visualization.draw_geometries([pcd_query, pcd_in])
        # Visualize using Open3D
    o3d.visualization.draw_geometries(geos)
    
def visualize_logits(logits_sampled, p_query, location,weights = None, inputs_distributed=None, force_viz = False):
    geos = []
    
    current_dir = os.getcwd()
        
    file_path = f'configs/simultaneous/train_simultaneous_{location}.yaml'
    # file_path = '/home/robin/Dev/MasterThesis/GithubRepos/master_thesis/neuralblox/configs/fusion/train_fusion_home.yaml'

    try:
        with open(os.path.join(current_dir, file_path), 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(e)
        return
        
    if not(force_viz or config['visualization']):
        return    
    
    if weights is not None:
        visualize_weights(weights, p_query, inputs_distributed)
    
    p_stacked = p_query[..., :3]
    
    p_full = p_stacked.detach().cpu().numpy().reshape(-1, 3)

    occ_sampled = logits_sampled.detach().cpu().numpy()

    values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
    
    values_sampled = values_sampled.reshape(-1)

    threshold = 0.1

    values_sampled[values_sampled < threshold] = 0
    values_sampled[values_sampled >= threshold] = 1
    
    values_gt = p_query[..., -1].reshape(-1).detach().cpu().numpy()

    both_occ = np.logical_and(values_gt, values_sampled)
    
    pcd = o3d.geometry.PointCloud()
    colors = np.zeros((values_gt.shape[0], 3))
    colors[values_gt == 1] = [1, 0, 0] # red
    colors[values_sampled == 1] = [0, 0, 1] # blue
    colors[both_occ == 1] = [0, 1, 0] # green
    
    mask = np.any(colors != [0, 0, 0], axis=1)
    # print(mask.shape, values_gt.shape, values_sampled.shape, colors.shape)
    if inputs_distributed is not None:
        pcd_inputs = o3d.geometry.PointCloud()
        inputs_reshaped = inputs_distributed.reshape(-1, 4).detach().cpu().numpy()
        pcd_inputs.points = o3d.utility.Vector3dVector(inputs_reshaped[inputs_reshaped[..., -1] == 1, :3])
        pcd_inputs.paint_uniform_color([1., 0.5, 1.0]) # blue
        geos += [pcd_inputs]
        
    colors = colors[mask]
    pcd.points = o3d.utility.Vector3dVector(p_full[mask])
    bb_min_points = np.min(p_full[mask], axis=0)
    bb_max_points = np.max(p_full[mask], axis=0)
    # print(bb_min_points, bb_max_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    geos += [pcd, base_axis]
    o3d.visualization.draw_geometries(geos)