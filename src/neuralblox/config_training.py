from torch import nn
import os
from src.encoder import encoder_dict
from src.neuralblox import models, training, training_fusion, training_fusion_old, training_sequential, training_sequential_shuffled
from src import data, config, layers
from src.common import update_reso
from src.checkpoints import CheckpointIO
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    # update the feature volume/plane resolution
    if cfg['data']['input_type'] == 'pointcloud_crop':
        raise NotImplementedError
    
    if cfg['data']['input_type'] == 'pointcloud_merge' or cfg['data']['input_type'] == 'pointcloud_sequential':
        raise NotImplementedError

    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    model = models.ConvolutionalOccupancyNetwork(
        decoder, encoder, device=device
    )

    if cfg['data']['input_type'] == 'pointcloud_sequential':
        return model
    else:
        return model

def get_trainer_sequential_shuffled(model, model_merge, optimizer_backbone, optimizer_merging, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    input_type = cfg['data']['input_type']
    query_n = cfg['data']['points_subsample']
    unet_hdim = cfg['model']['encoder_kwargs']['unet3d_kwargs']['f_maps']
    unet_depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] - 1
    limited_gpu = cfg['training']['limited_gpu']
    input_crop_size = cfg['data']['input_vol']
    query_crop_size = cfg['data']['query_vol']
    n_max_points = cfg['training']['n_max_points']
    n_max_points_query = cfg['training']['n_max_points_query']
    n_voxels_max = cfg['training']['n_voxels_max']
    n_batch = cfg['training']['n_batch']
    return_flat = cfg['training']['return_flat']
    sigma = cfg['training']['sigma']
    
    
    trainer = training_sequential_shuffled.SequentialTrainerShuffled(
        model, model_merge, optimizer_backbone, optimizer_merging,
        cfg = cfg,
        device=device, 
        input_type=input_type,
        threshold=threshold,
        query_n = query_n,
        unet_hdim = unet_hdim,
        unet_depth = unet_depth,
        limited_gpu = limited_gpu,
        input_crop_size = input_crop_size,
        query_crop_size = query_crop_size,
        n_voxels_max = n_voxels_max,
        n_max_points = n_max_points,
        n_batch = n_batch,
        n_max_points_query = n_max_points_query,
        return_flat=return_flat,
        sigma = sigma
    )

    return trainer

def get_trainer_sequential(model, model_merge, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    input_type = cfg['data']['input_type']
    query_n = cfg['data']['points_subsample']
    unet_hdim = cfg['model']['encoder_kwargs']['unet3d_kwargs']['f_maps']
    unet_depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] - 1
    limited_gpu = cfg['training']['limited_gpu']
    input_crop_size = cfg['data']['input_vol']
    query_crop_size = cfg['data']['query_vol']
    n_max_points = cfg['training']['n_max_points']
    n_max_points_query = cfg['training']['n_max_points_query']
    n_voxels_max = cfg['training']['n_voxels_max']
    return_flat = cfg['training']['return_flat']
    sigma = cfg['training']['sigma']
    
    
    trainer = training_sequential.SequentialTrainer(
        model, model_merge, optimizer, 
        cfg = cfg,
        device=device, 
        input_type=input_type,
        vis_dir=vis_dir, 
        threshold=threshold,
        query_n = query_n,
        unet_hdim = unet_hdim,
        unet_depth = unet_depth,
        limited_gpu = limited_gpu,
        input_crop_size = input_crop_size,
        query_crop_size = query_crop_size,
        n_voxels_max = n_voxels_max,
        n_max_points = n_max_points,
        n_max_points_query = n_max_points_query,
        return_flat=return_flat,
        sigma = sigma
    )

    return trainer

def get_trainer_overfit(model, model_merge, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    input_type = cfg['data']['input_type']
    query_n = cfg['data']['points_subsample']
    unet_hdim = cfg['model']['encoder_kwargs']['unet3d_kwargs']['f_maps']
    unet_depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] - 1
    limited_gpu = cfg['training']['limited_gpu']
    input_crop_size = cfg['data']['input_vol']
    query_crop_size = cfg['data']['query_vol']

    trainer = training_fusion_old.TrainerOld(
        model, model_merge, optimizer, 
        cfg = cfg,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        query_n = query_n,
        unet_hdim = unet_hdim,
        unet_depth = unet_depth,
        limited_gpu = limited_gpu,
        input_crop_size = input_crop_size,
        query_crop_size = query_crop_size
    )

    return trainer


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )

    return fields

def get_transform(mode, cfg):
    specific_angle = cfg['data']['transform']['specific_angle']
    angle_x = cfg['data']['transform']['angle_x']
    angle_y = cfg['data']['transform']['angle_y']
    angle_z = cfg['data']['transform']['angle_z']
    
    specific_translation = cfg['data']['transform']['specific_translation']
    translation_x = cfg['data']['transform']['translation_x']
    translation_y = cfg['data']['transform']['translation_y']
    translation_z = cfg['data']['transform']['translation_z']
    
    if specific_angle:
        rot_low = [angle_x, angle_y, angle_z]
        rot_high = [angle_x, angle_y, angle_z]
    else:
        rot_low=[-angle_x, -angle_y, -angle_z]
        rot_high=[angle_x, angle_y, angle_z]
    
    if specific_translation:
        trans_low = [translation_x, translation_y, translation_z]
        trans_high = [translation_x, translation_y, translation_z]
    else:  
        trans_low=[-translation_x, -translation_y, -translation_z]
        trans_high=[translation_x, translation_y, translation_z]
        
    def transform(data_list):
        for data in data_list:
            # Generate random rotation angles
            angles_deg = np.random.uniform(low=rot_low, high=rot_high)
            # print(f'angles_deg: {angles_deg}')
            rand_rot = R.from_euler('xyz', angles_deg, degrees=True)
            
            rand_translation = np.random.uniform(low=trans_low, high=trans_high, size=(3,)).astype(np.float32)
            # print(f'rand_translation: {rand_translation}')
            # Apply transformation to relevant keys
            for key in data:
                if key == 'points' or key == 'inputs':
                    shape = data[key].shape
                    # Ensure data[key] is a numpy array
                    if isinstance(data[key], np.ndarray):
                        data[key] = rand_rot.apply(data[key].reshape(-1, 3)).reshape(shape).astype(np.float32) + rand_translation
                    else:
                        raise TypeError(f"Expected numpy array for key '{key}', but got {type(data[key])}")
        return [angles_deg, rand_translation]
    return transform
