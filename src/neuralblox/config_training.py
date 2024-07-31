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

def get_trainer_sequential_shuffled(model, model_merge, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
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
    
    
    trainer = training_sequential_shuffled.SequentialTrainerShuffled(
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

def get_trainer_sequential(model, model_merge, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
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
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
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
    angle_x = cfg['data']['transform']['angle_x']
    angle_y = cfg['data']['transform']['angle_y']
    angle_z = cfg['data']['transform']['angle_z']
    
    def transform(data):
        angles_deg = np.random.uniform(low=[-angle_x, -angle_y, -angle_z], high=[angle_x, angle_y, angle_z])
        rand_trans = R.from_euler('xyz', angles_deg, degrees=True)

        for key in data:
            if key == 'points' or key == 'inputs':
                shape = data[key].shape
                data[key] = rand_trans.apply(data[key].reshape(-1, 3)).reshape(shape).astype(np.float32)

    return transform
