import yaml 
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
import os

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

def process_batch(batch, config_metrics, mode = 'evaluation'):
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
        if key in ['transform', 'model_infos']:
            batch_subsampled[key] = batch[key]
        else:
            batch_subsampled[key] = batch[key].squeeze(0)[0, ::config_metrics[mode]['skip_frames'], ...]
    
    # Maximum number of inputs
    n_max_inputs = config_metrics[mode]['n_max_inputs']
    
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
        if key not in ['transform', 'model_infos']:
            batch_subsampled_reduced[key] = batch_subsampled_reduced[key].unsqueeze(0)
            # print(f'reduced batch: {key}: {batch_subsampled_reduced[key].shape}')
    
    return batch_subsampled_reduced


def generate_logits(generator_robot, batch_subsampled_reduced, cfg, idx):
    with torch.no_grad():
        logits_tup = generator_robot.generate_logits_at_index(batch_subsampled_reduced, idx, 0, cfg['evaluation']['n_logits'])
    logits_sampled, query_points, inputs_frame, centers_frame = logits_tup[-1]
    query_points = query_points.reshape(-1, 3).cpu().numpy()
    logits_sampled = logits_sampled.reshape(-1).cpu().numpy()
    return logits_sampled, query_points

def load_ground_truth(cfg, terrain):
    scene_folder = cfg['data']['scene_folder']
    gt_mesh_path = os.path.join(scene_folder, terrain, 'terrain.obj')
    gt_mesh_points_path = os.path.join(scene_folder, terrain, 'surface_points.npy')
    
    gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
    gt_mesh_points = np.load(gt_mesh_points_path)
    
    return gt_mesh_o3d, gt_mesh_points


def apply_transformations(batch_subsampled_reduced, gt_mesh_o3d, gt_mesh_points):
    R_gravity = R.from_euler('xyz', [-np.pi/2, 0., 0.])
    angles_deg, rand_translation = batch_subsampled_reduced['transform']
    angles_deg = angles_deg.squeeze(0).cpu().numpy()
    rand_translation = rand_translation.squeeze(0).cpu().numpy()
    rand_rot = R.from_euler('xyz', angles_deg, degrees=True)
    
    gt_mesh_points = R_gravity.apply(gt_mesh_points)
    gt_mesh_points = rand_rot.apply(gt_mesh_points) + rand_translation
    
    gt_mesh_o3d.rotate(R_gravity.as_matrix(), center=(0, 0, 0))
    gt_mesh_o3d.rotate(rand_rot.as_matrix(), center=(0, 0, 0)).translate(rand_translation)
    
    return gt_mesh_o3d, gt_mesh_points

def filter_ground_points(gt_mesh_points, query_points, logits_sampled, batch):
    inputs = batch['inputs'].reshape(-1, 3).cpu().numpy()
    bb_min_inputs = np.min(inputs, axis=0)
    bb_max_inputs = np.max(inputs, axis=0)
    
    bb_min_gt = np.min(gt_mesh_points, axis=0) + np.array([0,0.1,0])
    bb_max_gt = np.max(gt_mesh_points, axis=0)
    
    bb_min = np.max([bb_min_inputs, bb_min_gt], axis=0)
    bb_max = np.min([bb_max_inputs, bb_max_gt], axis=0)
    
    # print(f"Bounding box: {bb_min_gt} - {bb_max_gt}")
    
    mask_within_bounds_gt = np.all((gt_mesh_points >= bb_min) & (gt_mesh_points <= bb_max), axis=1)
    mask_within_bounds_query = np.all((query_points >= bb_min) & (query_points <= bb_max), axis=1)
    
    # print(f'Number of points within bounds query: {mask_within_bounds_query.sum()}, {mask_within_bounds_query.shape[0]}')
    # print(f'Number of points within bounds gt: {mask_within_bounds_gt.sum()}, {mask_within_bounds_gt.shape[0]}')
    
    gt_mesh_points = gt_mesh_points[mask_within_bounds_gt]
    query_points = query_points[mask_within_bounds_query]
    logits_sampled = logits_sampled[mask_within_bounds_query]
    
    return gt_mesh_points, query_points, logits_sampled

def evaluate_points(query_points, logits_sampled, cfg, gt_mesh_o3d, gt_mesh_points):
    n_max_query_points = cfg['evaluation']['n_max_query_points']
    visualize = cfg['evaluation']['visualize']
    
    scene = o3d.t.geometry.RaycastingScene()
    
    gt_mesh = o3d.t.geometry.TriangleMesh.from_legacy(gt_mesh_o3d)
    _ = scene.add_triangles(gt_mesh)
    
    if visualize:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_mesh_points)
        pcd_gt.paint_uniform_color([0, 1, 0])

        pcd_query = o3d.geometry.PointCloud()

    best_params = {'Accuracy': np.inf, 'Accuracy.threshold': 0,
                   'Completeness': np.inf, 'Completeness.threshold': 0,
                   'Recall': -np.inf, 'Recall.threshold': 0}
    
    for idx, thresh in enumerate([0.1, 0.2, 0.3, 0.4]):
        occ_sampled = 1 / (1 + np.exp(-logits_sampled))
        occ_sampled = (occ_sampled > thresh)
        
        query_pts_occ = query_points[occ_sampled]
        
        if query_pts_occ.shape[0] > n_max_query_points:
            print(f'Number of query points: {query_pts_occ.shape[0]}, limiting to {n_max_query_points}')
            query_pts_occ = query_pts_occ[torch.randperm(query_pts_occ.shape[0])[:n_max_query_points]]
        
        if visualize and idx == 0:
            pcd_query.points = o3d.utility.Vector3dVector(query_pts_occ)
            pcd_query.paint_uniform_color([0, 0, 1])
            
            o3d.visualization.draw_geometries([pcd_query, pcd_gt, gt_mesh_o3d])
        
        accuracy = compute_accuracy(scene, query_pts_occ)
        completeness = compute_completeness(gt_mesh_points, query_pts_occ)
        recall = compute_recall(gt_mesh_points, query_pts_occ, tau_r=cfg['evaluation']['tau_r'])
        
        if accuracy < best_params['Accuracy']:
            best_params['Accuracy'] = accuracy
            best_params['Accuracy.threshold'] = thresh
        if completeness < best_params['Completeness']:
            best_params['Completeness'] = completeness
            best_params['Completeness.threshold'] = thresh
        if recall > best_params['Recall']:
            best_params['Recall'] = recall
            best_params['Recall.threshold'] = thresh
    
    print('')
    print(f'Best accuracy: {best_params["Accuracy"]}, threshold: {best_params["Accuracy.threshold"]}')
    print(f'Best completeness: {best_params["Completeness"]}, threshold: {best_params["Completeness.threshold"]}')
    print(f'Best recall: {best_params["Recall"]}, threshold: {best_params["Recall.threshold"]}')
    
    return best_params["Accuracy"], best_params["Completeness"], best_params["Recall"]