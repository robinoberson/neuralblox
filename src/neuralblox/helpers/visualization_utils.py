import open3d as o3d
import matplotlib.pyplot as plt
import os 
import yaml
import numpy as np
import torch 
import src.neuralblox.helpers.sequential_trainer_utils as st_utils
import matplotlib.cm as cm
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
    
def create_boxes(centers, query_crop_size, loss_batch = None, color = None):
    boxes = []
    
    if loss_batch is not None:
        loss_batch_sum = loss_batch.sum(dim=1)
        
        normalized_loss = (loss_batch_sum - loss_batch_sum.min()) / (loss_batch_sum.max() - loss_batch_sum.min())
        jet_cmap = cm.get_cmap('jet')
        jet_colors = jet_cmap(normalized_loss.detach().cpu().numpy())
    
    if color is None:
        color = np.random.rand(3)
    
    bbox_min = centers - query_crop_size / 2
    bbox_max = centers + query_crop_size / 2

    # print(loss_batch.shape, loss_batch_sum)
    for i in range(len(centers)):
        # box = o3d.geometry.TriangleMesh.create_box(width=query_crop_size, height=query_crop_size, depth=query_crop_size)
        # box.translate(centers[i].detach().cpu().numpy())
        # box.paint_uniform_color(jet_colors[i, :3])
        # boxes.append(box)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min[i].detach().cpu().numpy(), max_bound=bbox_max[i].detach().cpu().numpy())
        if loss_batch is not None: 
            bbox.color = jet_colors[i, :3]
        else:
            bbox.color = color
        # bbox.LineSet = 7
        boxes.append(bbox)
    return boxes

def create_pcd(points3d, occupancy, color, shift = None):
    pcd = o3d.geometry.PointCloud()
    points3d = points3d.detach().cpu().numpy()[..., :3].reshape(-1, 3)
    if shift is not None:
        points3d = points3d + shift
    occupancy = occupancy.detach().cpu().numpy().reshape(-1)
    pcd.points = o3d.utility.Vector3dVector(points3d[occupancy == 1])
    pcd.paint_uniform_color(color)
    
    return pcd

def create_pcd_logits(p_query, logits_sampled, threshold = 0.01):
    p_stacked = p_query[..., :3].detach().cpu().numpy().reshape(-1, 3)
    values_gt = p_query[..., 3].detach().cpu().numpy().reshape(-1)
    
    occ_sampled = logits_sampled.detach().cpu().numpy()
    values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
    values_sampled[values_sampled < threshold] = 0
    values_sampled[values_sampled >= threshold] = 1
    
    both_occ = np.logical_and(values_gt, values_sampled)
    
    colors = np.zeros((values_gt.shape[0], 3))
    colors[values_gt == 1] = [1,0,0] # red
    colors[values_sampled == 1] = [0,0,1] # blue
    colors[both_occ == 1] = [0,1,0] # purple
    mask = np.any(colors != [0, 0, 0], axis=1)
    colors = colors[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p_stacked[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
def visualize_logits_voxel(logits_sampled, p_query, centers_current_int, centers_prev_int, centers_current, centers_prev, loss, inputs_distributed_current, inputs_distributed_prev, location, query_crop_size = 1.0, threshold = 0.1):    
    current_dir = os.getcwd()
        
    file_path = f'configs/simultaneous/train_simultaneous_{location}.yaml'
    # file_path = '/home/robin/Dev/MasterThesis/GithubRepos/master_thesis/neuralblox/configs/fusion/train_fusion_home.yaml'

    try:
        with open(os.path.join(current_dir, file_path), 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(e)
        return
        
    if not(config['visualization']):
        return    
    
    geos = []
    
    bboxes_loss = create_boxes(centers_current_int, query_crop_size, loss)
    summed_loss = loss.sum(dim=1)
    
    k = int(config['k_value'] * len(summed_loss))
    thresh = torch.topk(summed_loss, k).values[-1]
    
    pcd_inputs_full_current = create_pcd(inputs_distributed_current, inputs_distributed_current[..., 3], [1, 1, 0.278]) #light yellow
    pcd_inputs_full_prev = create_pcd(inputs_distributed_prev, inputs_distributed_prev[..., 3], [0.58, 0.58, 0.337]) #dark yellow
    pcd_query_full = create_pcd(p_query[..., :3], p_query[..., 3], [0.49, 0.176, 0.341]) #light pink
    
    for i in range(len(centers_current)):
        if summed_loss[i] < thresh:
            continue
        # geos = [pcd_inputs_full_current.voxel_down_sample(voxel_size=0.1), pcd_inputs_full_prev.voxel_down_sample(voxel_size=0.1)]
        geos = []
        centers_current_i = centers_current[i]
        centers_prev_i = centers_prev[i]
        
        inputs_curr_i = inputs_distributed_current[i].reshape(-1, 4)
        inputs_prev_i = inputs_distributed_prev[i].reshape(-1, 4)
               
        pcd_curr_i = create_pcd(inputs_curr_i[..., :3], inputs_curr_i[..., 3], [1, 0.592, 0.973], shift = np.array([0, 0, 0.1])) #light pink
        pcd_prev_i = create_pcd(inputs_prev_i[..., :3], inputs_prev_i[..., 3], [0.58, 0.337, 0.565], shift = np.array([0, 0, 0.1])) #dark pink    
        # pcd_query = create_pcd(p_query[i][..., :3], query_points_i[..., 3], [0,1,0])

        # geos.append(pcd_query)
        geos.append(pcd_curr_i)
        geos.append(pcd_prev_i)
        geos.append(bboxes_loss[i])
        
        # bboxes_inputs = create_boxes(centers_current_i, query_crop_size, color = [0,0,0])
        # geos += bboxes_inputs
        
        pcd = create_pcd_logits(p_query[i], logits_sampled[i], threshold = threshold)
        
        base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        geos += [pcd, base_axis]
        # geos += [base_axis]
        o3d.visualization.draw_geometries(geos)
def visualize_logits(logits_sampled, p_query, centers, loss, location, weights = None, inputs_distributed=None, force_viz = False, threshold = 0.01, query_crop_size = 1.0):
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
    
    # if weights is not None:
    #     visualize_weights(weights, p_query, inputs_distributed)
    
    p_stacked = p_query[..., :3]
    
    p_full = p_stacked.detach().cpu().numpy().reshape(-1, 3)

    occ_sampled = logits_sampled.detach().cpu().numpy()

    values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
    
    values_sampled = values_sampled.reshape(-1)

    values_sampled[values_sampled < threshold] = 0
    values_sampled[values_sampled >= threshold] = 1
    
    values_gt = p_query[..., -1].reshape(-1).detach().cpu().numpy()

    both_occ = np.logical_and(values_gt, values_sampled)
    
    pcd = o3d.geometry.PointCloud()
    colors = np.zeros((values_gt.shape[0], 3))
    colors[values_gt == 1] = [0.7372549019607844, 0.2784313725490196, 0.28627450980392155] # red
    colors[values_sampled == 1] = [0.231372549019607850, 0.95686274509803930, 0.9843137254901961] # blue
    colors[both_occ == 1] = [0.8117647058823529, 0.8196078431372549, 0.5254901960784314] # purple
    # colors[values_gt == 1] = [0.7372549019607844, 0.2784313725490196, 0.28627450980392155] # red
    # colors[values_sampled == 1] = [0.7372549019607844, 0.2784313725490196, 0.28627450980392155] # blue
    # colors[both_occ == 1] = [0.7372549019607844, 0.2784313725490196, 0.28627450980392155] # purple
    
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
    
    boxes_loss = create_boxes(centers, query_crop_size, loss)
    geos += boxes_loss
    o3d.visualization.draw_geometries(geos)
    
def visualize_batch(batch, idx, device):
    p_in_full, p_query_full = st_utils.get_inputs_from_scene(batch, device)
    p_query_full = p_query_full[idx].detach().cpu().numpy().reshape(-1, 4)
    p_in = p_in_full[idx]
    points_full = None
    for i in range(p_in.shape[0]):
        geos = []
        pcd = o3d.geometry.PointCloud()
        pcd_full = o3d.geometry.PointCloud()
        # p_query_full = p_query[i].detach().cpu().numpy().reshape(-1, 4)

        # print(f'sum: {p_in[i, ..., -1].sum()}')
        
        points = p_in[i].detach().cpu().numpy().reshape(-1, 4)
                
        # pcd.points = o3d.utility.Vector3dVector(points[..., :3][points[..., -1] == 1])
        pcd.points = o3d.utility.Vector3dVector(points[..., :3])
        pcd.paint_uniform_color(np.random.rand(3))
        geos.append(pcd)

        if points_full is None:
            points_full = points
        else:
            # pcd_full.points = o3d.utility.Vector3dVector(points_full[..., :3])
            pcd_full.points = o3d.utility.Vector3dVector(points_full[..., :3][points_full[..., -1] == 1])
            points_full = np.concatenate([points_full, points])
            pcd_full.paint_uniform_color([0.5, 0.5, 0.5])
            geos.append(pcd_full)
            
    base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    pcd_query = o3d.geometry.PointCloud()
    pcd_query.points = o3d.utility.Vector3dVector(p_query_full[p_query_full[:, -1] == 1, :3])
    pcd_query.paint_uniform_color([1.0, 0.5, 0.5])
    geos.append(pcd_query)
    o3d.visualization.draw_geometries(geos + [base_axis])
    return pcd_query
    
def visualize_debug(self):
    #### visualize
    grid_inputs_frame_current = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_current2 = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_current = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_existing = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_existing2 = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_existing[occupied_existing_mask_padded] = inputs_existing.to(torch.device('cpu')).detach()
    
    grid_inputs_frame_current2[mask_complete_existing.to(torch.device('cpu'))] = grid_inputs_frame_existing[mask_complete_existing.to(torch.device('cpu'))]
    grid_inputs_frame_existing2[mask_complete_current.to(torch.device('cpu'))] = grid_inputs_frame_current[mask_complete_current.to(torch.device('cpu'))]
    
    print(grid_inputs_frame_current2[..., 3].sum())
    print(grid_inputs_frame_existing2[..., 3].sum())
    
    import open3d as o3d
    pcd_existing = o3d.geometry.PointCloud()
    pcd_current = o3d.geometry.PointCloud()
    
    pcd_existing2 = o3d.geometry.PointCloud()
    pcd_current2 = o3d.geometry.PointCloud()
    
    points_existing = grid_inputs_frame_existing.reshape(-1, 4).detach().cpu().numpy()
    points_current = grid_inputs_frame_current.reshape(-1, 4).detach().cpu().numpy()
    
    points_existing2 = grid_inputs_frame_existing2.reshape(-1, 4).detach().cpu().numpy()
    points_current2 = grid_inputs_frame_current2.reshape(-1, 4).detach().cpu().numpy()
    
    pcd_existing.points = o3d.utility.Vector3dVector(points_existing[points_existing[:, 3] == 1, :3])
    pcd_current.points = o3d.utility.Vector3dVector(points_current[points_current[:, 3] == 1, :3])
    
    pcd_existing2.points = o3d.utility.Vector3dVector(points_existing2[points_existing2[:, 3] == 1, :3])
    pcd_current2.points = o3d.utility.Vector3dVector(points_current2[points_current2[:, 3] == 1, :3])
            
    pcd_existing.paint_uniform_color([0, 0, 1]) #blue
    pcd_current.paint_uniform_color([1, 0, 0]) #red
    
    pcd_existing2.paint_uniform_color([0, 1, 1]) #yellow
    pcd_current2.paint_uniform_color([0, 1, 0]) #magenta
    
    o3d.visualization.draw_geometries([pcd_existing, pcd_current, pcd_existing2, pcd_current2])       
    
    ## end visualize 
    
def visu_debug_2(self):
    grid_inputs_frame_current = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_current2 = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_current[occupied_current_mask_padded] = occupied_inputs_interior_current.to(torch.device('cpu')).detach()
    grid_inputs_frame_existing = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_existing2 = torch.zeros((n_x, n_y, n_z, self.n_max_points_input, 4)).to(torch.device('cpu'))
    grid_inputs_frame_existing[occupied_existing_mask_padded] = inputs_existing.to(torch.device('cpu')).detach()
    
    
    grid_inputs_frame_existing2[mask_complete_existing.to(torch.device('cpu'))] = grid_inputs_frame_current[mask_complete_existing.to(torch.device('cpu'))]
    grid_inputs_frame_current2[mask_complete_current.to(torch.device('cpu'))] = grid_inputs_frame_existing[mask_complete_current.to(torch.device('cpu'))]

    grid_inputs_frame_current_temp = grid_inputs_frame_current.to(torch.device(self.device))[centers_current_shifted[..., 0], centers_current_shifted[..., 1], centers_current_shifted[..., 2]]
    grid_inputs_frame_existing_temp = grid_inputs_frame_existing.to(torch.device(self.device))[centers_current_shifted[..., 0], centers_current_shifted[..., 1], centers_current_shifted[..., 2]]

    import open3d as o3d
    pcd_existing = o3d.geometry.PointCloud()
    pcd_current = o3d.geometry.PointCloud()

    points_existing = grid_inputs_frame_existing_temp.reshape(-1, 4).detach().cpu().numpy()
    points_current = grid_inputs_frame_current_temp.reshape(-1, 4).detach().cpu().numpy()
    
    pcd_existing.points = o3d.utility.Vector3dVector(points_existing[points_existing[:, 3] == 1, :3])
    pcd_current.points = o3d.utility.Vector3dVector(points_current[points_current[:, 3] == 1, :3])
    
    pcd_current.paint_uniform_color([1, 0, 0]) #red
    pcd_existing.paint_uniform_color([0, 0, 1]) #blue
    
    pcd_existing_2 = o3d.geometry.PointCloud()
    pcd_current_2 = o3d.geometry.PointCloud()
        
    points_existing_2 = grid_inputs_frame_existing2.reshape(-1, 4).detach().cpu().numpy()
    points_current_2 = grid_inputs_frame_current2.reshape(-1, 4).detach().cpu().numpy()
    
    pcd_existing_2.points = o3d.utility.Vector3dVector(points_existing_2[points_existing_2[:, 3] == 1, :3])
    pcd_current_2.points = o3d.utility.Vector3dVector(points_current_2[points_current_2[:, 3] == 1, :3])
    
    pcd_current_2.paint_uniform_color([1, 1, 0]) #yellow
    pcd_existing_2.paint_uniform_color([0, 1, 0]) #green

    pcd_existing_i = o3d.geometry.PointCloud()
    pcd_current_i = o3d.geometry.PointCloud()
    
    for i in range(len(grid_inputs_frame_existing_temp)):
        points_existing_i = grid_inputs_frame_existing_temp[i].reshape(-1, 4).detach().cpu().numpy()
        points_current_i = grid_inputs_frame_current_temp[i].reshape(-1, 4).detach().cpu().numpy()
        pcd_existing_i.points = o3d.utility.Vector3dVector(points_existing_i[points_existing_i[:, 3] == 1, :3])
        pcd_current_i.points = o3d.utility.Vector3dVector(points_current_i[points_current_i[:, 3] == 1, :3])
        pcd_existing_i.paint_uniform_color([1, 0, 1]) #cyan
        pcd_current_i.paint_uniform_color([0, 1, 1]) #purple
        
    o3d.visualization.draw_geometries([pcd_existing, pcd_current, pcd_existing_2, pcd_current_2, pcd_existing_i, pcd_current_i])       
    

def visualize_mesh_and_points(inputs_frame_list, mesh_list, points_perc = 1.0):
    """
    Visualize a 3D mesh and a point cloud.

    Parameters:
    - inputs_frame (torch.Tensor): The input frame tensor of shape (N, M).
    - mesh (object): The mesh object containing vertices and faces.
    - point_threshold (int): The threshold value for points to be included.
    - point_interval (int): Interval for selecting points.

    Returns:
    None
    """
    points_full = None
    for i in range(len(inputs_frame_list)):
        if points_full is None:
            points_full = inputs_frame_list[i].detach().cpu().numpy().reshape(-1, 4)
        else:
            points_full = np.concatenate([points_full, inputs_frame_list[i].detach().cpu().numpy().reshape(-1, 4)])
            
    inputs_frame = inputs_frame_list[-1]
    mesh = mesh_list[-1]

    # Extract points where the last dimension is equal to the threshold value
    points_unt = inputs_frame.detach().cpu().numpy()
    points = points_unt[points_unt[:, -1] == 1, :3]
    
    # Select percentage of points
    random_indices = np.random.choice(points_full.shape[0], int(points_full.shape[0] * points_perc), replace=False)
    points_full = points_full[random_indices]
    points_full = points_full[points_full[:, -1] == 1, :3]

    # Create the mesh for visualization
    mesho3d = o3d.geometry.TriangleMesh()
    mesho3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesho3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesho3d.compute_vertex_normals()

    # Create a coordinate frame for reference
    base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Create a point cloud for the selected points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points + np.array([0, 0.05, 0.0]))
    pcd.paint_uniform_color([1, 0, 0])
    
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points_full + np.array([0, 0.05, 0.0]))
    pcd_full.paint_uniform_color([0,0,1])

    # Visualize the mesh, point cloud, and coordinate frame
    o3d.visualization.draw_geometries([mesho3d, base_axis, pcd_full, pcd])