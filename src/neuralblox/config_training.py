from torch import nn
import os
from src.encoder import encoder_dict
from src.neuralblox import models, training, training_fusion, training_fusion_old, training_sequential
from src import data, config, layers
from src.common import update_reso
from src.checkpoints import CheckpointIO

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
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        if dataset is not None:
            if (dataset.split == 'train') or (cfg['generation']['sliding_window']):
                recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
                reso = cfg['data']['query_vol_size'] + recep_field - 1
                if 'grid' in fea_type:
                    encoder_kwargs['grid_resolution'] = update_reso(reso, dataset.depth)
                    encoder_kwargs['grid_resolution'] = cfg['data']['grid_resolution']
                if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                    encoder_kwargs['plane_resolution'] = update_reso(reso, dataset.depth)
            # if dataset.split == 'val': #TODO run validation in room level during training
            else:
                if 'grid' in fea_type:
                    encoder_kwargs['grid_resolution'] = dataset.total_reso
                if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                    encoder_kwargs['plane_resolution'] = dataset.total_reso
        else:
            encoder_kwargs['grid_resolution'] = cfg['data']['grid_resolution']

    if cfg['data']['input_type'] == 'pointcloud_merge' or cfg['data']['input_type'] == 'pointcloud_sequential':
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2 ** (cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']

        grid_reso = cfg['data']['grid_resolution']
        input_vol_size = cfg['data']['input_vol']
        query_vol_size = cfg['data']['query_vol']

        if 'grid' in fea_type:
            if cfg['data']['input_type'] == 'pointcloud_sequential':
                encoder_kwargs['grid_resolution'] = grid_reso
            else:
                encoder_kwargs['grid_resolution'] = grid_reso
                encoder_kwargs['input_crop_size'] = input_vol_size
                encoder_kwargs['query_crop_size'] = query_vol_size

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


def get_trainer(model, optimizer, cfg, device, **kwargs):
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
    grid_reso = cfg['model']['encoder_kwargs']['grid_resolution']
    input_vol_size = cfg['data']['input_vol']
    query_vol_size = cfg['data']['query_vol']
    input_type = cfg['data']['input_type']
    vol_bound = {'query_crop_size': query_vol_size,
                     'input_crop_size': input_vol_size,
                     'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                     'reso': grid_reso
                    }
    vol_range = cfg['data']['vol_range']
    trainer = training.Trainer(
        model, optimizer,
        vol_bound=vol_bound,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        vol_range=vol_range
    )

    return trainer

def get_trainer_sequence(model, model_merge, optimizer, cfg, device, **kwargs):
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

    trainer = training_fusion.Trainer(
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
        n_max_points_query = n_max_points_query
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
