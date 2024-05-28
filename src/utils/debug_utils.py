from src import config, layers
from src.checkpoints import CheckpointIO
from src import data as data_src
import torch 
import os 
import torch.optim as optim
import random
from src.layers import Conv3D_one_input

def get_trainer(cfg_fusion_training):
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    model = config.get_model(cfg_fusion_training, device=device)
    checkpoint_io = CheckpointIO(cfg_fusion_training['training']['out_dir'], model=model)
    checkpoint_io.load(cfg_fusion_training['training']['backbone_file'])
    
    model_merging = layers.Conv3D_one_input().to(device)

    model_merging_dir = cfg_fusion_training['training']['out_dir']

    checkpoint_io_merging = CheckpointIO(model_merging_dir, model=model_merging)
    try:
        _ = checkpoint_io_merging.load(os.path.join(model_merging_dir, cfg_fusion_training['test']['model_file']))
    except FileExistsError as e:
        print(f'No checkpoint file found! {e}')
        
    optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=0.01)
    trainer = config.get_trainer_sequence(model, model_merging, optimizer, cfg_fusion_training, device=device)
    
    return trainer, checkpoint_io_merging

def get_train_loader(cfg_path_fusion_training, cfg_path_default):
    cfg_fusion_training = config.load_config(cfg_path_fusion_training, cfg_path_default)

    batch_size_fusion = cfg_fusion_training['training']['batch_size']
    train_dataset_fusion = config.get_dataset('train', cfg_fusion_training)

    batch_sampler = data_src.CategoryBatchSampler(train_dataset_fusion, batch_size=batch_size_fusion, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_fusion, num_workers=cfg_fusion_training['training']['n_workers'],
        collate_fn=data_src.collate_remove_none,
        batch_sampler=batch_sampler,
        worker_init_fn=data_src.worker_init_fn)
    
    return train_loader

def plot_batch():
    p_in = batch.get('inputs').to(device)        
    batch_size, T, D = p_in.size() 
    if batch_size < cfg_fusion_training['training']['batch_size']:
        print('Batch size too small, skipping batch')

    categories = batch.get('category')
    unique_cat = np.unique(categories)
    if len(unique_cat) > 1:
        print('Multiple categories found in batch, skipping batch')

    category = unique_cat[0]
    
    path_gt_points = os.path.join(cfg_fusion_training['data']['path_gt'], category, '000000', cfg_fusion_training['data']['gt_file_name'])
    
    points_gt_npz = np.load(path_gt_points)
    points_gt_3D = points_gt_npz['points']
    points_gt_3D = torch.from_numpy(points_gt_3D).to(device).float()
    
    points_gt_occ = np.unpackbits(points_gt_npz['occupancies'])
    points_gt_occ = torch.from_numpy(points_gt_occ).to(device).float()
    
    random_indices = np.random.choice(points_gt_3D.shape[0], size=T, replace=False)
    points_gt_3D = points_gt_3D[random_indices]
    points_gt_occ = points_gt_occ[random_indices]

    points_gt = torch.concatenate([points_gt_3D, points_gt_occ.unsqueeze(-1)], dim = -1)

    print(f"points_gt.shape: {points_gt.shape}, points_gt_occ.shape: {points_gt_occ.shape}")
    print(f"batch['inputs'].shape: {batch['inputs'].shape}, batch['inputs.occ'].shape: {batch['inputs.occ'].shape}")
    
    points_full = batch['inputs'].reshape(-1, 3).cpu().numpy()
    points_occ_full = batch['inputs.occ'].reshape(-1).cpu().numpy()
    points_gt_full = points_gt[..., :3].reshape(-1, 3).cpu().numpy()

    print(points_full.shape, points_occ_full.shape, points_gt_full.shape)

    import open3d as o3d

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points_full)
    colors = np.zeros((points_full.shape[0], 3))
    colors[points_occ_full == 0] = [0, 1, 0]
    colors[points_occ_full == 1] = [0, 0, 1]
    pcd_full.colors = o3d.utility.Vector3dVector(colors)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt_full)
    colors = np.zeros((points_gt_full.shape[0], 3))
    print(points_gt.shape)
    colors[points_gt[:, 3].cpu().numpy() == 0] = [0, 1, 0]
    colors[points_gt[:, 3].cpu().numpy() == 1] = [1, 0, 0]
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd_full, pcd_gt])
    
    
def create_model(num_blocks, num_channels):
    return Conv3D_one_input(num_blocks=num_blocks, num_channels=num_channels)

def generate_random_configurations(num_samples):
    configurations = []
    for _ in range(num_samples):
        num_blocks = random.randint(1, 5)
        num_channels = [256]
        for i in range(num_blocks+1):
            if i < (num_blocks+1) // 2:  # First half: increment
                channel_count = random.randint(num_channels[-1], 512)
            else:  # Second half: decrement
                channel_count = random.randint(128, num_channels[-1])
            
            # Ensure channel_count is a multiple of 32
            channel_count = int(round(channel_count / 32)) * 32
            num_channels.append(channel_count)

        num_channels.append(128)
        configurations.append((num_blocks, num_channels))
    return configurations
