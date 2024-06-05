from src import config, layers
from src.checkpoints import CheckpointIO
from src import data as data_src
import torch 
import torch.nn as nn

import os 
import torch.optim as optim
import random
# from src.layers import Conv3D_one_input, CombinedLoss
import numpy as np
def get_trainer(cfg_fusion_training, device):
    

    model = config.get_model(cfg_fusion_training, device=device)
    checkpoint_io = CheckpointIO(cfg_fusion_training['training']['out_dir'], model=model)
    checkpoint_io.load(cfg_fusion_training['training']['backbone_file'])
    
    # model_merging = layers.Conv3D_one_input(num_blocks=num_blocks, num_channels=num_channels).to(device)
    model_merging = layers.Conv3D_one_input().to(device)

    model_merging_dir = cfg_fusion_training['training']['out_dir']

    checkpoint_io_merging = CheckpointIO(model_merging_dir, model=model_merging)
    try:
        path_model = os.path.join(model_merging_dir, cfg_fusion_training['training']['starting_model'])
        _ = checkpoint_io_merging.load(path_model)
    except FileExistsError as e:
        print(f'No checkpoint file found! {e}')
        
    optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=0.01)
    trainer = config.get_trainer_sequence(model, model_merging, optimizer, cfg_fusion_training, device=device)
    
    trainer.model_merge.to(device)
    trainer.unet = trainer.model.encoder.unet3d
    
    encoder_feature_shapes = [torch.Size([1, 128, 6, 6, 6]), torch.Size([1, 64, 12, 12, 12]), torch.Size([1, 32, 24, 24, 24])] #TODO Move this to config

    trainer.features_shapes = encoder_feature_shapes
    
    return trainer, checkpoint_io_merging

def get_train_loader(cfg_path_fusion_training, cfg_path_default, shuffle_bool = False):
    cfg_fusion_training = config.load_config(cfg_path_fusion_training, cfg_path_default)

    batch_size_fusion = cfg_fusion_training['training']['batch_size']
    train_dataset_fusion = config.get_dataset('train', cfg_fusion_training)

    batch_sampler = data_src.CategoryBatchSampler(train_dataset_fusion, batch_size=batch_size_fusion, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_fusion, num_workers=cfg_fusion_training['training']['n_workers'],
        collate_fn=data_src.collate_remove_none,
        batch_sampler=batch_sampler,
        worker_init_fn=data_src.worker_init_fn,
        shuffle=shuffle_bool)
    
    return train_loader

def get_gt_points(batch, device, cfg):
    p_in = batch.get('inputs').to(device)        
    batch_size, T, D = p_in.size() 
    if batch_size < cfg['training']['batch_size']:
        print('Batch size too small, skipping batch')

    categories = batch.get('category')
    unique_cat = np.unique(categories)
    if len(unique_cat) > 1:
        print('Multiple categories found in batch, skipping batch')
        
    category = unique_cat[0]
    path_gt_points = os.path.join(cfg['data']['path_gt'], category, '000000', cfg['data']['gt_file_name'])
    
    points_gt_npz = np.load(path_gt_points)
    points_gt_3D = points_gt_npz['points']
    points_gt_3D = torch.from_numpy(points_gt_3D).to(device).float()

    points_gt_occ = np.unpackbits(points_gt_npz['occupancies'])
    points_gt_occ = torch.from_numpy(points_gt_occ).to(device).float().unsqueeze(-1)
    
    random_indices = np.random.choice(points_gt_3D.shape[0], size=T, replace=False)
    points_gt_3D = points_gt_3D[random_indices]
    points_gt_occ = points_gt_occ[random_indices]
    
    points_gt = torch.cat((points_gt_3D, points_gt_occ), dim=-1)
    
    return points_gt

def plot_batch(batch, device, cfg_fusion_training):
    
    points_gt = get_gt_points(batch, device, cfg_fusion_training)
    
    points_full = batch['inputs'].reshape(-1, 3).cpu().numpy()
    points_occ_full = batch['inputs.occ'].reshape(-1).cpu().numpy()
    points_gt_full = points_gt[..., :3].reshape(-1, 3).cpu().numpy()

    import open3d as o3d
    
    
    pcd_0 = o3d.geometry.PointCloud()
    pcd_0.points = o3d.utility.Vector3dVector(batch['inputs'][0])
    colors_0 = np.zeros((batch['inputs'][0].shape[0], 3))
    colors_0[batch['inputs.occ'][0] == 0] = [1, 1, 0]
    colors_0[batch['inputs.occ'][0] == 1] = np.random.rand(3)
    pcd_0.colors = o3d.utility.Vector3dVector(colors_0)
    
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(batch['inputs'][1])
    colors_1 = np.zeros((batch['inputs'][1].shape[0], 3))
    colors_1[batch['inputs.occ'][1] == 0] = [0, 1, 1]
    colors_1[batch['inputs.occ'][1] == 1] = np.random.rand(3)
    pcd_1.colors = o3d.utility.Vector3dVector(colors_1)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt_full)
    colors = np.zeros((points_gt_full.shape[0], 3))
    print(points_gt.shape)
    colors[points_gt[:, 3].cpu().numpy() == 0] = [1, 0, 1]
    colors[points_gt[:, 3].cpu().numpy() == 1] = np.random.rand(3)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd_1, pcd_0, pcd_gt])
    
def get_batch_pcd(batch, device, cfg_fusion_training):
    
    points_gt = get_gt_points(batch, device, cfg_fusion_training)
    
    points_full = batch['inputs'].reshape(-1, 3).cpu().numpy()
    points_occ_full = batch['inputs.occ'].reshape(-1).cpu().numpy()
    points_gt_full = points_gt[..., :3].reshape(-1, 3).cpu().numpy()

    import open3d as o3d

    pcd_0 = o3d.geometry.PointCloud()
    pcd_0.points = o3d.utility.Vector3dVector(batch['inputs'][0])
    colors_0 = np.zeros((batch['inputs'][0].shape[0], 3))
    colors_0[batch['inputs.occ'][0] == 0] = [1, 1, 0]
    colors_0[batch['inputs.occ'][0] == 1] = np.random.rand(3)
    pcd_0.colors = o3d.utility.Vector3dVector(colors_0)
    
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(batch['inputs'][1])
    colors_1 = np.zeros((batch['inputs'][1].shape[0], 3))
    colors_1[batch['inputs.occ'][1] == 0] = [0, 1, 1]
    colors_1[batch['inputs.occ'][1] == 1] = np.random.rand(3)
    pcd_1.colors = o3d.utility.Vector3dVector(colors_1)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt_full)
    colors = np.zeros((points_gt_full.shape[0], 3))
    print(points_gt.shape)
    colors[points_gt[:, 3].cpu().numpy() == 0] = [1, 0, 1]
    colors[points_gt[:, 3].cpu().numpy() == 1] = np.random.rand(3)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)

    return pcd_0, pcd_1, pcd_gt    
    
def create_model(num_blocks, num_channels):
    return layers.Conv3D_one_input(num_blocks=num_blocks, num_channels=num_channels)

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

def plot_4D_points(points, plot_unocc=True):
    import open3d as o3d
    points_full = points.reshape(-1, 4).cpu().numpy()
    points = points_full[:, :3]
    occ = points_full[:, 3]
    
    pcd_occ = o3d.geometry.PointCloud()
    pcd_occ.points = o3d.utility.Vector3dVector(points[occ == 1])
    pcd_occ.paint_uniform_color([1, 0, 0])
    
    pcd_unocc = o3d.geometry.PointCloud()
    pcd_unocc.points = o3d.utility.Vector3dVector(points[occ == 0])
    pcd_unocc.paint_uniform_color([0, 1, 0])
    
    if plot_unocc:
        o3d.visualization.draw_geometries([pcd_occ, pcd_unocc])
    else:
        o3d.visualization.draw_geometries([pcd_occ])

def save_visualization(trainer, dataloader, idx_config, cfg):
    with torch.no_grad():
        trainer.model_merge.eval()
        trainer.model.eval()
        
        device = trainer.device
        output_dir = cfg['training']['out_dir']
        
        vis_dir = os.path.join(output_dir, cfg['training']['vis_dir'], str(idx_config))
        if os.path.exists(vis_dir):
            #remove old files
            os.system(f'rm -rf {vis_dir}')
        os.makedirs(vis_dir)        
        print(f'Visualization directory: {vis_dir}')

        for batch_idx, (input_tensor, output_tensor, p_stacked, p_n_stacked, p_in) in enumerate(dataloader):

            input_tensor = input_tensor.to(device).squeeze(0)
            output_tensor = output_tensor.to(device).squeeze(0)
            p_stacked = p_stacked.to(device).squeeze(0)
            p_n_stacked = p_n_stacked.to(device).squeeze(0)
            p_in = p_in.to(device).squeeze(0)
                
            # Forward pass
            latent_map_sampled_merged = trainer.merge_latent_map(input_tensor) 
            outputs = trainer.get_logits(latent_map_sampled_merged, p_stacked, )
                        
            torch.save(outputs, os.path.join(vis_dir, f'logits_sampled_{batch_idx}.pt'))
            torch.save(output_tensor, os.path.join(vis_dir, f'logits_gt_{batch_idx}.pt'))
            torch.save(p_stacked, os.path.join(vis_dir, f'p_stacked_{batch_idx}.pt'))
            torch.save(p_in, os.path.join(vis_dir, f'p_in_{batch_idx}.pt'))
        


def visualize_logits(logits_gt, logits_sampled, p_stacked, inputs_distributed=None, show=False):
    import open3d as o3d

    p_full = p_stacked.detach().cpu().numpy().reshape(-1, 3)

    occ_gt = logits_gt.detach().cpu().numpy()
    occ_sampled = logits_sampled.detach().cpu().numpy()

    values_gt = np.exp(occ_gt) / (1 + np.exp(occ_gt))
    values_sampled = np.exp(occ_sampled) / (1 + np.exp(occ_sampled))
    
    values_gt = values_gt.reshape(-1)
    values_sampled = values_sampled.reshape(-1)

    threshold = 0.5

    values_gt[values_gt < threshold] = 0
    values_gt[values_gt >= threshold] = 1

    values_sampled[values_sampled < threshold] = 0
    values_sampled[values_sampled >= threshold] = 1

    both_occ = np.logical_and(values_gt, values_sampled)
    
    pcd = o3d.geometry.PointCloud()
    colors = np.zeros((values_gt.shape[0], 3))
    colors[values_gt == 1] = [0.7372549019607844, 0.2784313725490196, 0.28627450980392155] # red
    colors[values_sampled == 1] = [0.231372549019607850, 0.95686274509803930, 0.9843137254901961] # blue
    colors[both_occ == 1] = [0.8117647058823529, 0.8196078431372549, 0.5254901960784314] # purple
    
    mask = np.any(colors != [0, 0, 0], axis=1)
    # print(mask.shape, values_gt.shape, values_sampled.shape, colors.shape)
    colors = colors[mask]
    pcd.points = o3d.utility.Vector3dVector(p_full[mask])
    bb_min_points = np.min(p_full[mask], axis=0)
    bb_max_points = np.max(p_full[mask], axis=0)
    # print(bb_min_points, bb_max_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    pcd_inputs = o3d.geometry.PointCloud()
    if inputs_distributed is not None:
        points_inputs_dist = inputs_distributed.reshape(-1, 4).detach().cpu().numpy()
        points_inputs_dist = points_inputs_dist[points_inputs_dist[:, 3] == 1][:, :3]
        pcd_inputs.points = o3d.utility.Vector3dVector(points_inputs_dist)
        pcd_inputs.paint_uniform_color([0, 1, 1])  # cyan
        
    base_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    if show: o3d.visualization.draw_geometries([pcd, base_axis, pcd_inputs])
    return pcd, pcd_inputs

def train_and_evaluate(trainer, dataloader, cfg, idx_config, num_iter=15000, lr=0.001, save_checkpoint=False, checkpoint_io_merging=None, experiment = None, debug = False):
    # Move model to the device
    log_comet = cfg['training']['log_comet']
    device = trainer.device
    
    # Define the loss function and optimizer
    criterion = data_src.CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(trainer.model_merge.parameters(), lr=lr, weight_decay=0)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=300, factor=0.9)
    
    # Initial learning rate
    init_lr = lr
    
    iteration = 0
    # Training loop
    for epoch in range(int(num_iter)):
        epoch_loss = 0.0
        num_batches = 0

        trainer.model_merge.train()  # Set the model to training mode
        
        for batch_idx, (input_tensor, output_tensor, p_stacked, p_n_stacked, batch) in enumerate(dataloader):
            if debug and batch_idx > 0:
                break
            # print(f'Epoch {epoch}, batch {batch_idx}')
            input_tensor = input_tensor.to(device).squeeze(0)
            output_tensor = output_tensor.to(device).squeeze(0)
            p_stacked = p_stacked.to(device).squeeze(0)
            p_n_stacked = p_n_stacked.to(device).squeeze(0)
            
            # Forward pass
            latent_map_sampled_merged = trainer.merge_latent_map(input_tensor) 
            outputs = trainer.get_logits(latent_map_sampled_merged, p_stacked, trainer.vol_bound_all['input_vol'], device = trainer.device)

            loss = criterion(outputs, output_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear the gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the model parameters

            # Step the scheduler based on the validation loss
            scheduler.step(loss)
            epoch_loss += loss.item()
            num_batches += 1

            if log_comet: experiment.log_metric(f'train_loss_{idx_config}', loss, step=iteration)
            
            # Check if learning rate has changed
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != init_lr:
                print(f'Learning rate changed to {current_lr:.6f} at epoch {epoch+1}')
                init_lr = current_lr
            
            # if (iteration+1) % 50 == 0:
            #     print(f'Epoch [{iteration+1}/{num_iter}], Loss: {loss.item():.4f}')
            # if (iteration+1) > 10 and (iteration+1) % 00 == 0:
            #     # trainer.visualize_logits(output_tensor, outputs, p_stacked, p_n_stacked)
            #     if save_checkpoint and checkpoint_io_merging is not None:
            #         print(f'Saving model at epoch {iteration+1}')
            #         checkpoint_io_merging.save(f'model_merging_{idx_config}.pt', epoch_it=epoch, it=iteration)
    
            iteration += 1
            
            if iteration >= num_iter:
                break
        if iteration >= num_iter:
            break
        
        mean_epoch_loss = epoch_loss / num_batches
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch + 1}, {iteration}/{num_iter}], Mean Loss: {mean_epoch_loss:.4f}')
        if (epoch+1) % 100 == 0:
            save_visualization(trainer, dataloader, idx_config, cfg)

        if log_comet: experiment.log_metric(f'epoch_loss_{idx_config}', mean_epoch_loss, step=epoch)
    
    #save the model
    if save_checkpoint: 
        checkpoint_io_merging.save(f'model_merging_{idx_config}.pt', epoch_it=epoch, it=iteration)
        print(f'Saving model at epoch {epoch+1}, iteration {iteration}, loss {mean_epoch_loss:.4f}, model_merging_{idx_config}.pt')
    
    return loss.item()