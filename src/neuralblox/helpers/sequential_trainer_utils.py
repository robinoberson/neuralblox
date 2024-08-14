import time
import torch

def print_timing(t0, operation):
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'Time elapsed: {t1 - t0:.3f}, {operation}')

    t0 = time.time()

def compute_gaussian_weights(gt_points_batch, input_points_batch, sigma=1.0):
    
    batch_size = gt_points_batch.shape[0]
    n_gt = gt_points_batch.shape[1]
    
    weights_batch = torch.ones(batch_size, n_gt, device=gt_points_batch.device)

    for b in range(batch_size):
        gt_points = gt_points_batch[b]
        input_points = input_points_batch[b]
        
        # Flatten to handle each batch element independently
        gt_points_flat = gt_points[..., :3].reshape(-1, 3)
        inputs_flat = input_points[input_points[..., 3] == 1, :3].reshape(-1, 3)
        
        if inputs_flat.shape[0] == 0:
            continue
        # Compute pairwise distances
        distances = torch.cdist(gt_points_flat, inputs_flat)
        
        # Find the minimum distance for each gt point
        min_distances, _ = distances.min(dim=1)
        
        # Compute Gaussian weights
        weights = torch.exp(-min_distances ** 2 / (2 * sigma ** 2))
        weights_batch[b] = weights
        
    return weights_batch

def generate_points(n, lb = [0.0, 0.0, 0.0], ub = [1.0, 1.0, 1.0]):
    """
    Generate n points within the bounds lb and ub.

    Args:
    - n (int): Number of points to generate.
    - lb (list of float): Lower bound for each dimension.
    - ub (list of float): Upper bound for each dimension.

    Returns:
    - torch.Tensor: Tensor of shape (n, 3) containing the generated points.
    """
    lb = torch.tensor(lb)
    ub = torch.tensor(ub)
    
    # Generate n points with each dimension in the range [0, 1)
    points = torch.rand(n, 3)
    
    # Scale points to the range [lb, ub)
    points = lb + points * (ub - lb)
    
    return points

def get_empty_inputs(centers, crop_size, n_max_points = 2048):
    lb_input = centers - crop_size / 2
    ub_input = centers + crop_size / 2
    
    vol_bound_inputs = torch.stack([lb_input, ub_input], axis=1)
    
    n_crops = vol_bound_inputs.shape[0]
    
    bb_min = vol_bound_inputs[:, 0, :].unsqueeze(1) # Shape: (n_crops, 3, 1)
    bb_max = vol_bound_inputs[:, 1, :].unsqueeze(1)  # Shape: (n_crops, 3, 1)
    bb_size = bb_max - bb_min  # Shape: (n_crops, 3, 1)

    random_points = generate_points(n_max_points)
    random_points = random_points.repeat(n_crops, 1, 1).to(device=vol_bound_inputs.device)
    random_points *= bb_size  # Scale points to fit inside each bounding box
    random_points += bb_min  # Translate points to be within each bounding box

    return random_points

def get_distributed_voxel(centers_idx, grids, grid_shapes, centers_lookup, shifts):

    # Convert centers and centers_lookup to appropriate dimensions
    center_x, center_y, center_z = centers_idx[:, 0], centers_idx[:, 1], centers_idx[:, 2]
    grid_offsets = centers_lookup[:, 0]

    # Create index matrices for shifts
    base_indices = grid_offsets + (center_x * grid_shapes[:, 1] + center_y) * grid_shapes[:, 2] + center_z

    # Calculate shifted indices
    shift_dx, shift_dy, shift_dz = shifts[:, 0], shifts[:, 1], shifts[:, 2]
    shifted_indices = base_indices.unsqueeze(1) + (shift_dx.unsqueeze(0) * grid_shapes[:, 1].unsqueeze(1) + shift_dy.unsqueeze(0)) * grid_shapes[:, 2].unsqueeze(1) + shift_dz.unsqueeze(0)

    shifted_indices_min = torch.min(shifted_indices)
    shifted_indices_max = torch.max(shifted_indices)
    
    # Check if any indices are out of bounds
    if (shifted_indices_min < 0).any() or (shifted_indices_max >= grids.shape[0]).any():
        raise ValueError("Some indices are out of bounds")
    # Fetch the shifted values
    shifted_values = grids[shifted_indices.to(grids.device)]

    return shifted_values
def get_inputs_from_scene(batch, device):
        
    p_in_3D = batch.get('inputs').to(device)
    p_in_occ = batch.get('inputs.occ').to(device).unsqueeze(-1)
            
    p_query_3D = batch.get('points').to(device)
    p_query_occ = batch.get('points.occ').to(device).unsqueeze(-1)
    
    # print(f'p_in_3D: {p_in_3D.shape}, p_in_occ: {p_in_occ.shape}, p_query_3D: {p_query_3D.shape}, p_query_occ: {p_query_occ.shape}')
    p_in = torch.cat((p_in_3D, p_in_occ), dim=-1)
    p_query = torch.cat((p_query_3D, p_query_occ), dim=-1)
    
    return p_in, p_query

def get_grid_from_centers(centers, crop_size):
    lb = centers - crop_size / 2
    ub = centers + crop_size / 2
    vol_bounds = torch.stack([lb, ub], dim=1)
    
    return vol_bounds

def compute_vol_bound(inputs, query_crop_size, input_crop_size, padding = False):
    # inputs must have shape (n_points, 3)
    assert inputs.shape[1] == 3 and inputs.shape[0] > 0 and len(inputs.shape) == 2
    device = inputs.device

    vol_bound = {}

    lb_p = torch.min(inputs, dim=0).values - torch.tensor([0.01, 0.01, 0.01], device=device)
    ub_p = torch.max(inputs, dim=0).values
    
    # print(lb_p, ub_p)

    lb = torch.round((lb_p - lb_p % query_crop_size) * 1e6) / 1e6
    ub = torch.round((((ub_p - ub_p % query_crop_size) / query_crop_size) + 1) * query_crop_size * 1e6) / 1e6

    if padding:
        lb -= query_crop_size
        ub += query_crop_size
    
    lb_query = torch.stack(torch.meshgrid(
        torch.arange(lb[0], ub[0] - 0.01, query_crop_size, device=device),
        torch.arange(lb[1], ub[1] - 0.01, query_crop_size, device=device),
        torch.arange(lb[2], ub[2] - 0.01, query_crop_size, device=device),
    ), dim=-1).reshape(-1, 3)

    ub_query = lb_query + query_crop_size
    centers = (lb_query + ub_query) / 2

    # Number of crops alongside x, y, z axis
    vol_bound['axis_n_crop'] = torch.ceil((ub - lb - 0.01) / query_crop_size).int()

    # Total number of crops
    num_crop = torch.prod(vol_bound['axis_n_crop']).item()
    vol_bound['n_crop'] = num_crop
    vol_bound['input_vol'] = get_grid_from_centers(centers, input_crop_size)
    vol_bound['query_vol'] = get_grid_from_centers(centers, query_crop_size)
    vol_bound['lb'] = lb
    vol_bound['ub'] = ub

    return vol_bound, centers

def remove_nearby_points(points, occupied_inputs, thresh):
    if points.numel() == 0 or occupied_inputs.numel() == 0:
        # If either is empty, no points can be removed
        return points, 0
    
    # Calculate the squared Euclidean distances to avoid the cost of square root computation
    dist_squared = torch.cdist(points[..., :3], occupied_inputs[..., :3], p=2).pow(2)
    
    # Find the minimum distance for each point in `points` to any point in `occupied_inputs`
    min_dist_squared, _ = dist_squared.min(dim=1)
    
    # Keep only the points where the minimum distance is greater than or equal to `thresh` squared
    mask = min_dist_squared >= thresh**2
    filtered_points = points[mask]
    
    return filtered_points, mask.sum().item()

def maintain_n_sample_points(centers, crop_size, random_points, occupied_inputs, n_sample, thresh):
    device = centers.device
    # Remove points from `random_points` that are within `thresh` distance to any `occupied_inputs`
    filtered_points, n_filtered = remove_nearby_points(random_points, occupied_inputs, thresh)
    n_removed = n_sample - n_filtered
    # print("Number of points removed initially:", n_removed)
    n_iter = 0
    # Loop for removing and resampling points
    while n_filtered < n_sample:
        # Calculate the number of points to sample
        n_to_sample = int(1.5 * n_removed)

        # Sample new points inside the box
        new_points = get_empty_inputs(centers, crop_size, n_max_points = n_to_sample).squeeze(0)
        new_points = torch.cat((new_points, torch.zeros(new_points.shape[0], 1, device=device)), dim=1)
        # Remove points from new_points that are within thresh distance to any occupied_inputs
        new_points_filtered, n_new_filtered = remove_nearby_points(new_points, occupied_inputs, thresh)

        # Calculate how many points we need to add to reach n_sample
        n_to_add = min(n_removed, n_new_filtered)

        # Add the new points to filtered_points
        filtered_points = torch.cat((filtered_points, new_points_filtered[:n_to_add]), dim=0)
        n_filtered += n_to_add
        n_removed -= n_to_add
        
        n_iter += 1
        if n_iter > 1000:
            print("Failed to maintain n_sample points")
            break
    # Ensure we have exactly n_sample points
    filtered_points = filtered_points[:n_sample]
    # print(f'N iter {n_iter}')

    return filtered_points

def get_elevation_mask(inputs_frame_occupied):
    device = inputs_frame_occupied.device
    elevation_mask = torch.rand(len(inputs_frame_occupied), device=device) < 0.5

    # Mask for occupied points
    occupied_mask = inputs_frame_occupied[:, :, -1] == 1

    # Compute bounding boxes for all frames
    bb_min = torch.min(torch.where(occupied_mask.unsqueeze(-1), inputs_frame_occupied[:, :, :3], float('inf')), dim=1)[0]
    bb_max = torch.max(torch.where(occupied_mask.unsqueeze(-1), inputs_frame_occupied[:, :, :3], float('-inf')), dim=1)[0]
    
    # Compute y_diff
    y_diff = bb_max[:, 1] - bb_min[:, 1]
    
    # print(y_diff)
    # Update elevation_mask based on y_diff
    elevation_mask = elevation_mask | (y_diff > 0.1) #max elevation is query_crop_size[1]
    
    # Count the number of flat and elevated frames
    n_flat = (~elevation_mask).sum()
    n_elevated = elevation_mask.sum()
    
    
    return elevation_mask

def compute_mask_occupied(centers_grid, centers_occupied):
    centers_grid_expanded = centers_grid.unsqueeze(1)  
    centers_occupied_full_expanded = centers_occupied.unsqueeze(0)  
    
    error = 0.1

    matches = (torch.norm(centers_grid_expanded - centers_occupied_full_expanded, dim=2) <= error)

    mask_centers_grid_occupied = matches.any(dim=1)
    
    return mask_centers_grid_occupied

def centers_to_grid_indexes(centers, lb, query_crop_size):
    centers_shifted = torch.round((centers - (lb + query_crop_size / 2)) / query_crop_size * 10e4) / 10e4

    return centers_shifted

def print_gradient_norms(iteration, model, print_every=100):
    if iteration % print_every == 0:
        mean_norm = torch.mean(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
        max_norm = torch.max(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
        print(f'Iteration {iteration}, Mean Gradient Norm: {mean_norm.item()}, Max Gradient Norm: {max_norm.item()}')

def create_batch_groups(train_loader, group_size): #TODO include this as a standard data loader
    batch_groups = []
    current_group = []
    
    for batch in train_loader:
        current_group.append(batch)
        if len(current_group) == group_size:
            batch_groups.append(current_group)
            current_group = []
    
    # Add any remaining batches
    if current_group:
        batch_groups.append(current_group)
    
    return batch_groups