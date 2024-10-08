import torch
# from src.utils.libkdtree import KDTree
import numpy as np
import math

def define_align_matrix(align_config):
    ''' Get transformation matrix to align scene

        Args:
            align_config : aligning configuration
        '''
    deg_x, deg_y, deg_z = align_config['deg_x'], align_config['deg_y'], align_config['deg_z']
    shift_vertical = align_config['shift_vertical']
    is_flip = align_config['is_flip']

    cos_theta, sin_theta = math.cos(math.radians(deg_x)), math.sin(math.radians(deg_x))
    cos_alpha, sin_alpha = math.cos(math.radians(deg_y)), math.sin(math.radians(deg_y))
    cos_beta, sin_beta = math.cos(math.radians(deg_z)), math.sin(math.radians(deg_z))

    flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    rot_1 = np.array([[1, 0, 0, 0],
                   [0, cos_theta, -sin_theta, 0],
                   [0, sin_theta, cos_theta, 0],
                   [0, 0, 0, 1]])

    rot_2 = np.array([[cos_alpha, 0, sin_alpha, 0],
                   [0, 1, 0, 0],
                   [-sin_alpha, 0, cos_alpha, 0],
                   [0, 0, 0, 1]])

    rot_3 = np.array([[cos_beta, -sin_beta, 0, 0],
                   [sin_beta, cos_beta, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    shift_mat = np.array([[1, 0, 0, 0],
                   [0, 1, 0, shift_vertical],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    if is_flip == True:
        align_matrix = np.matmul(rot_1, flip)
    else:
        align_matrix = rot_1

    align_matrix = np.matmul(rot_2, align_matrix)
    align_matrix = np.matmul(rot_3, align_matrix)
    align_matrix = np.matmul(shift_mat, align_matrix)

    return align_matrix

def get_shift_noise(std_x, std_y, std_z):
    ''' Get shift noises along x, y and z axes

        Args:
            std_x, std_y, std_z : standard deviations in meter
        '''
    noise_x = np.random.normal(scale=std_x)
    noise_y = np.random.normal(scale=std_y)
    noise_z = np.random.normal(scale=std_z)

    return noise_x, noise_y, noise_z

def get_yaw_noise_matrix(std, device):
    ''' Get rotation noise along gravity direction

        Args:
            std : standard deviation in degree
            device : pytorch device
        '''

    degree = np.random.normal(scale=std)
    cos_alpha = math.cos(math.radians(degree))
    sin_alpha = math.sin(math.radians(degree))

    rot_2 = torch.tensor([[cos_alpha, 0, sin_alpha],
                      [0, 1, 0],
                      [-sin_alpha, 0, cos_alpha]], device=device)

    return rot_2

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    
    mask_zeros = np.sum(occ1, axis=-1) > 0
    occ1 = occ1[mask_zeros]
    occ2 = occ2[mask_zeros]
    if occ1.shape[0] == 0:
        # print("occ1 is empty")
        return np.array([-1])
    
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = np.maximum(1, (occ1 | occ2).astype(np.float32).sum(axis=-1))
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


# def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
#     ''' Returns the chamfer distance for the sets of points.

#     Args:
#         points1 (numpy array): first point set
#         points2 (numpy array): second point set
#         use_kdtree (bool): whether to use a kdtree
#         give_id (bool): whether to return the IDs of nearest points
#     '''
#     if use_kdtree:
#         return chamfer_distance_kdtree(points1, points2, give_id=give_id)
#     else:
#         return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def chamfer_distance_kdtree(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


# def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
#     ''' Returns the nearest neighbors for point sets batchwise.

#     Args:
#         points_src (numpy array): source points
#         points_tgt (numpy array): target points
#         k (int): number of nearest neighbors to return
#     '''
#     indices = []
#     distances = []

#     for (p1, p2) in zip(points_src, points_tgt):
#         kdtree = KDTree(p2)
#         dist, idx = kdtree.query(p1, k=k)
#         indices.append(idx)
#         distances.append(dist)

#     return indices, distances


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert(Rt_new.size() == (batch_size, 3, 4))
    return Rt_new

def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    
    # p_vis = p.view(-1, 3)
    # p_bbox_min = torch.min(p_vis, dim=0)[0]
    # p_bbox_max = torch.max(p_vis, dim=0)[0]
    
    p_nor = p / (1 + padding) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    
    # p_nor_vis = p_nor.view(-1, 3)
    # p_nor_bbox_min = torch.min(p_nor_vis, dim=0)[0]
    # p_nor_bbox_max = torch.max(p_nor_vis, dim=0)[0]    
    
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        # print(f'Problem with p_nor max: {p_nor.max()}, should be < 1, check padding')
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        # print(f'Problem with p_nor min: {p_nor.min()}, should be >= 0, check padding')
        p_nor[p_nor < 0] = 0.0
    return p_nor

def normalize_coord(p, vol_range, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments

    Args:
        p (tensor): point
        vol_range (numpy array): volume boundary
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    # Extract coordinates and occupancy flag

        
    if p.shape[-1] == 4:
        coord = p[..., :3].clone()
        occ = p[..., 3].unsqueeze(-1).clone()
    else:
        coord = p.clone()
        
    if not isinstance(vol_range, torch.Tensor):
        vol_range = torch.tensor(vol_range).to(p.device)
        
    if len(vol_range.shape) == 2:
        batch_size = coord.shape[0]
        vol_range = vol_range.unsqueeze(0).repeat(batch_size, 1, 1).clone()
        
    # Normalize coordinates #TODO check     coord = (coord - vol_range[0]) / (vol_range[1] - vol_range[0])

    for dim in range(3):      
        range_diff = vol_range[:, 1] - vol_range[:, 0]  # Shape: [batch_size, 3]
        # coord[:, :, dim] = coord[:, :, dim] * range_diff[:, dim].unsqueeze(1) + vol_range[:, 0, dim].unsqueeze(1)
        coord[..., dim] = (coord[..., dim] - vol_range[:, 0, dim].unsqueeze(1)) / (range_diff[:, dim].unsqueeze(1))

    bb_min_coord = torch.min(coord, dim=-2)[0]
    bb_max_coord = torch.max(coord, dim=-2)[0]
    
    if torch.max(coord) > 1.0 or torch.min(coord) < 0.0:
        max_coord = torch.max(coord)
        min_coord = torch.min(coord)
        print(f'Problem with coord, max: {max_coord}, min: {min_coord}')
        # print('Problem with coord')
        # a_reshaped = coord.reshape(-1, 3)
        # mask_bigger = torch.sum(a_reshaped > 1.0, 1)
        # mask_smaller = torch.sum(a_reshaped < 0.0, 1)

        # mask = (mask_bigger + mask_smaller) > 0
        # print(mask.sum())
        
        coord[coord > 1.0] = 1.0 - 10e-6
        coord[coord < 0.0] = 0.0
    
    if torch.max(coord) < 0.8 or torch.min(coord) > 0.2:
        print('Problem with coord, not in [0.2, 0.8]')
        

    if p.shape[-1] == 4:
        # Concatenate normalized coordinates and occupancy flag
        x = torch.cat((coord, occ), dim=-1)
    else:
        x = coord
    
    return x

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def coord2index(p, vol_range, reso=None, plane='grid', normalize_coords = True):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): points
        vol_range (numpy array): volume boundary
        reso (int): defined resolution
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    # normalize to [0, 1]
    if normalize_coords:
        temp = normalize_coord(p, vol_range, plane=plane)
    else:
        temp = p
        
    bb_min_temp = torch.min(temp, dim=-2)[0]
    bb_max_temp = torch.max(temp, dim=-2)[0]
    
    x = temp[..., :3]
    if p.shape[-1] == 4:
        occ = temp[..., 3]
    
    if isinstance(x, np.ndarray):
        x = np.floor(x * reso).astype(int)
    else:  # Assuming it's a torch tensor
        x = (x * reso).long()
    
    index = x[:, :, 0] + reso * x[:, :, 1] + reso**2 * x[:, :, 2] 
    
    if index.max() >= reso**3:
        # sum_idxes = torch.sum(index.reshape(-1) > reso**3, dim=-1)
        index[index >= reso**3] = 0 #set it as unoccupied
        
    if index.min() < 0:
        # sum_idxes = torch.sum(index.reshape(-1) < 0, dim=-1)
        index[index < 0] = 0
    index += reso**3 * occ.long()

    return index[:, None, :]

def update_reso(reso, depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    '''
    base = 2**(int(depth) - 1)
    if ~(reso / base).is_integer(): # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break    
    return reso

def decide_total_volume_range(query_vol_metric, recep_field, unit_size, unet_depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        query_vol_metric (numpy array): query volume size
        recep_field (int): defined the receptive field for U-Net
        unit_size (float): the defined voxel size
        unet_depth (int): U-Net number of layers
    '''
    reso = query_vol_metric / unit_size + recep_field - 1
    reso = update_reso(int(reso), unet_depth) # make sure input reso can be processed by UNet
    input_vol_metric = reso * unit_size
    p_c = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    lb_input_vol, ub_input_vol = p_c - input_vol_metric/2, p_c + input_vol_metric/2
    lb_query_vol, ub_query_vol = p_c - query_vol_metric/2, p_c + query_vol_metric/2
    input_vol = [lb_input_vol, ub_input_vol]
    query_vol = [lb_query_vol, ub_query_vol]

    # handle the case when resolution is too large
    if reso > 10000:
        reso = 1
    
    return input_vol, query_vol, reso

def add_key(base, new, base_name, new_name, device=None):
    ''' Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    '''
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base,
                new_name: new}
    return base

class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''
    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p): 
        p = torch.remainder(p, self.s) / self.s # always positive
        p = self.pe(p)
        return p

class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0 # change to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p
