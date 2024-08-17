import yaml 
import numpy as np
import open3d as o3d
import torch


def filter_points(points, margin, margin_x_p):
    # Calculate original bounds
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    
    # Calculate filtered bounds
    filtered_xmin = xmin + margin
    filtered_xmax = xmax - margin_x_p
    filtered_ymin = ymin + margin
    filtered_ymax = ymax - margin
    
    # Create a mask for filtering
    mask = (
        (points[:, 0] >= filtered_xmin) & 
        (points[:, 0] <= filtered_xmax) & 
        (points[:, 1] >= filtered_ymin) & 
        (points[:, 1] <= filtered_ymax)
    )
    
    # Apply the mask to filter points
    filtered_points = points[mask]
    
    # Print the bounding box
    print("Bounding box of filtered points:")
    print(f"X: [{filtered_xmin:.2f}, {filtered_xmax:.2f}]")
    print(f"Y: [{filtered_ymin:.2f}, {filtered_ymax:.2f}]")
    print(f"Z: [{zmin:.2f}, {zmax:.2f}]")  # Z bounds remain the same
    
    return filtered_points

def load_config(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)
    
def compute_accuracy(scene, predicted_points):
    predicted_points_tensor = o3d.core.Tensor(predicted_points, dtype=o3d.core.Dtype.Float32)
    closest_points = scene.compute_closest_points(predicted_points_tensor)
    distances = np.linalg.norm(predicted_points - closest_points['points'].numpy(), axis=-1)
    return distances.mean()

def compute_completeness(ground_truth_mesh_points, predicted_points):
    # Sample points from the ground truth mesh
    random_indices = np.random.choice(len(ground_truth_mesh_points), size=10000, replace=False)
    query_points = ground_truth_mesh_points[random_indices]
    
    # Compute distances from each ground truth point to the nearest predicted point
    distances = np.linalg.norm(query_points[:, None, :] - predicted_points[None, :, :], axis=-1)
    
    # Find the minimum distance for each ground truth point
    min_distances = np.min(distances, axis=1)
    
    # Return the mean of these minimum distances (MAD)
    return min_distances.mean()


def compute_recall(ground_truth_mesh_points, predicted_points, tau_r=0.05):
    # Sample points from the ground truth mesh
    random_indices = np.random.choice(len(ground_truth_mesh_points), size=10000, replace=False)
    query_points = ground_truth_mesh_points[random_indices]
    
    # Compute distances from each ground truth point to the nearest predicted point
    distances = np.linalg.norm(query_points[:, None, :] - predicted_points[None, :, :], axis=-1)
    
    # Find the minimum distance for each ground truth point
    min_distances = np.min(distances, axis=1)
    
    # Compute recall as the fraction of points within the threshold tau_r
    recall = np.mean(min_distances <= tau_r)
    
    return recall

import torch
import numpy as np

def process_batch(batch, config_metrics):
    """
    Processes the input batch according to the specified configuration metrics.

    Parameters:
    - batch: dict, the input batch containing tensors.
    - config_metrics: dict, configuration parameters including 'generation' settings.

    Returns:
    - batch_subsampled_reduced: dict, processed and subsampled batch.
    """
    
    # Initialize subsampled batch
    batch_subsampled = {}
    
    # Subsample the batch
    for key in batch:
        batch_subsampled[key] = batch[key].squeeze(0)[0, ::config_metrics['generation']['skip_frames'], ...]
    
    # Maximum number of inputs
    n_max_inputs = config_metrics['generation']['n_max_inputs']
    
    # Create a reduced version of the subsampled batch
    batch_subsampled_reduced = batch_subsampled.copy()
    batch_subsampled_reduced['inputs'] = torch.zeros(batch_subsampled['inputs'].shape[0], n_max_inputs, 3)
    batch_subsampled_reduced['inputs.occ'] = torch.ones(batch_subsampled['inputs.occ'].shape[0], n_max_inputs)
    
    # Process each input in the subsampled batch
    for i in range(batch_subsampled['inputs'].shape[0]):
        inputs_i_occ = batch_subsampled['inputs'][i][batch_subsampled['inputs.occ'][i] == 1]
        indices = np.random.choice(inputs_i_occ.shape[0], size=n_max_inputs, replace=False)
        batch_subsampled_reduced['inputs'][i] = inputs_i_occ[indices]
    
    # Unsqueeze each key to add a new dimension
    for key in batch_subsampled_reduced:
        batch_subsampled_reduced[key] = batch_subsampled_reduced[key].unsqueeze(0)
        print(f'reduced batch: {key}: {batch_subsampled_reduced[key].shape}')
    
    return batch_subsampled_reduced

# Example usage:
# batch_subsampled_reduced = process_batch(batch, config_metrics)
