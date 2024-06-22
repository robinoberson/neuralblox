from torch import nn
import os
from src.encoder import encoder_dict
from src.neuralblox import models, training, training_fusion, training_fusion_old, training_sequential
from src.neuralblox import generation, generation_fusion, generation_fusion_neighbors, generation_fusion_sequential
from src import data, config, layers
from src.common import decide_total_volume_range, update_reso
from src.checkpoints import CheckpointIO
import torch.optim as optim

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

        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)

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

def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.
    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2 ** (cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']

        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)

        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else:
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type=cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info=vol_info,
        vol_bound=vol_bound,
        vol_range = cfg['data']['vol_range'],
        grid_reso=cfg['model']['encoder_kwargs']['grid_resolution'],
    )
    return generator


def get_generator_fusion(model, model_merge, trainer, cfg, device, sample_points=None, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): the backbone encoder and decoder which are used
        model_merge : the fusion network
        sample_points : points sampled to define scene ranges
        cfg (dict): config dictionary
        device (device): pytorch device
    '''

    if cfg['data']['input_type'] == 'pointcloud_crop' or cfg['data']['input_type'] == 'pointcloud_sequential':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2 ** (cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']

        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)

        grid_reso = cfg['data']['grid_resolution']
        input_vol_size = cfg['data']['input_vol']
        query_vol_size = cfg['data']['query_vol']
        voxel_threshold = cfg['generation']['voxel_threshold']
        boundary_interpolation = cfg['generation'].get("boundary_interpolation", True)

        unet_hdim = cfg['model']['encoder_kwargs']['unet3d_kwargs']['f_maps']
        unet_depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] - 1

        vol_bound = {'query_crop_size': query_vol_size,
                     'input_crop_size': input_vol_size,
                     'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                     'reso': grid_reso}          
    else:
        vol_bound = None
        vol_info = None

    if cfg['generation']['generator_type'] == 'neighbors':
        generator = generation_fusion_neighbors.Generator3DNeighbors(
            model,
            model_merge,
            trainer,
            threshold=cfg['generation']['threshold'],
            device=device,
            resolution0=cfg['generation']['resolution_0'],
            upsampling_steps=cfg['generation']['upsampling_steps'],
            padding=cfg['data']['padding'],
            vol_bound=vol_bound,
            voxel_threshold=voxel_threshold,
        )
        
    else:
        generator = generation_fusion.Generator3D(
            model,
            model_merge,
            sample_points,
            device=device,
            threshold=cfg['test']['threshold'],
            resolution0=cfg['generation']['resolution_0'],
            upsampling_steps=cfg['generation']['upsampling_steps'],
            refinement_step=cfg['generation']['refinement_step'],
            input_type=cfg['data']['input_type'],
            padding=cfg['data']['padding'],
            vol_info=vol_info,
            vol_bound=vol_bound,
            voxel_threshold=voxel_threshold,
            boundary_interpolation=boundary_interpolation,
            unet_hdim = unet_hdim,
            unet_depth = unet_depth,
    
        )
        
    return generator

def get_generator_simultaneous(cfg, device):
    ''' Returns the generator object.

    Args:
        model (nn.Module): the backbone encoder and decoder which are used
        model_merge : the fusion network
        sample_points : points sampled to define scene ranges
        cfg (dict): config dictionary
        device (device): pytorch device
    '''

    grid_reso = cfg['data']['grid_resolution']
    input_vol_size = cfg['data']['input_vol']
    query_vol_size = cfg['data']['query_vol']

    vol_bound = {'query_crop_size': query_vol_size,
                    'input_crop_size': input_vol_size,
                    'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                    'reso': grid_reso}          
    model = config.get_model(cfg, device=device)
    model_merging = layers.Conv3D_one_input().to(device)

    checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=model)
    checkpoint_io_merging = CheckpointIO(cfg['training']['out_dir'], model=model_merging)

    try:
        checkpoint_io.load(cfg['generation']['model_backbone_file'])
        checkpoint_io_merging.load(cfg['generation']['model_merging_file'])
        
    except FileExistsError as e:
        print(f'No checkpoint file found! {e}')
        return None
    
    optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=cfg['training']['lr'])
    trainer = config.get_trainer_sequence(model, model_merging, optimizer, cfg, device=device)
    
    generator = generation_fusion_neighbors.Generator3DNeighbors(
            model,
            model_merging,
            trainer,
            threshold=cfg['generation']['threshold'],
            device=device,
            resolution0=cfg['generation']['resolution_0'],
            upsampling_steps=cfg['generation']['upsampling_steps'],
            padding=cfg['data']['padding'],
            vol_bound=vol_bound,
        )
        
    return generator
def get_generator_sequential(cfg, device):
    ''' Returns the generator object.

    Args:
        model (nn.Module): the backbone encoder and decoder which are used
        model_merge : the fusion network
        sample_points : points sampled to define scene ranges
        cfg (dict): config dictionary
        device (device): pytorch device
    '''

    grid_reso = cfg['data']['grid_resolution']
    input_vol_size = cfg['data']['input_vol']
    query_vol_size = cfg['data']['query_vol']

    vol_bound = {'query_crop_size': query_vol_size,
                    'input_crop_size': input_vol_size,
                    'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                    'reso': grid_reso}          
    model = config.get_model(cfg, device=device)
    model_merging = layers.Conv3D_one_input().to(device)

    checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=model)
    checkpoint_io_merging = CheckpointIO(cfg['training']['out_dir'], model=model_merging)

    try:
        checkpoint_io.load(cfg['generation']['model_backbone_file'])
        checkpoint_io_merging.load(cfg['generation']['model_merging_file'])
        
    except FileExistsError as e:
        print(f'No checkpoint file found! {e}')
        return None
    
    optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=cfg['training']['lr'])
    trainer = config.get_trainer_sequential(model, model_merging, optimizer, cfg, device=device)
    
    generator = generation_fusion_sequential.Generator3DSequential(
            model,
            model_merging,
            trainer,
            threshold=cfg['generation']['threshold'],
            device=device,
            resolution0=cfg['generation']['resolution_0'],
            upsampling_steps=cfg['generation']['upsampling_steps'],
            padding=cfg['data']['padding'],
            vol_bound=vol_bound,
        )
        
    return generator
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
