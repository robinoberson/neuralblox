import yaml
from torchvision import transforms
from src import data
from src import neuralblox
import os

method_dict = {
    'neuralblox': neuralblox
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config_training.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config_training.get_trainer(
        model, optimizer, cfg, device)
    return trainer

def get_trainer_sequence(model, model_merge, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config_training.get_trainer_sequence(
        model, model_merge, optimizer, cfg, device)
    return trainer

def get_trainer_sequential(model, model_merge, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config_training.get_trainer_sequential(
        model, model_merge, optimizer, cfg, device)
    return trainer

def get_trainer_overfit(model, model_merge, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config_training.get_trainer_overfit(
        model, model_merge, optimizer, cfg, device)
    return trainer



# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.
    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config_training.get_generator(model, cfg, device)
    return generator

def get_generator_fusion(model, model_merge, trainer, cfg, sample_points = None, device = None):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the backbone encoder and decoder which are used
        model_merge : the fusion network
        sample_points : points sampled to define scene ranges
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config_training.get_generator_fusion(model, model_merge, trainer, cfg, device, sample_points=sample_points)
    return generator

def get_generator_simultaneous(cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the backbone encoder and decoder which are used
        model_merge : the fusion network
        sample_points : points sampled to define scene ranges
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config_training.get_generator_simultaneous(cfg, device)
    return generator
def get_generator_sequential(cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the backbone encoder and decoder which are used
        model_merge : the fusion network
        sample_points : points sampled to define scene ranges
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config_training.get_generator_sequential(cfg, device)
    return generator
# Datasets
def get_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    
    if mode == 'val':
        categories = cfg['data']['val_classes']
    elif mode == 'test':
        categories = cfg['data']['test_classes']
    elif mode == 'train':
        categories = cfg['data']['train_classes']
    else:
        raise ValueError('Unknown mode: {}'.format(mode))
        
    if categories is None:
        print(f'Using all classes in {dataset_folder}, splitting into train/val')
        categories = sorted(os.listdir(dataset_folder))
        categories = [c for c in categories
                        if os.path.isdir(os.path.join(dataset_folder, c))]
        if len(categories) > 1:
            if mode == 'train':
                categories = categories[:int(0.8*len(categories))]
            elif mode == 'val':
                categories = categories[int(0.8*len(categories)):]
            #if mode is test, then use all categories
        
    split = None
    
    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config_training.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg,
            transform=method_dict[method].config_training.get_transform(mode, cfg)
        )
    elif dataset_type == 'Scenes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        # fields = method_dict[method].config.get_data_fields(mode, cfg)
        fields = {}
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg=cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset

def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        raise ValueError(
            'No input type specified')
        
    elif input_type == 'pointcloud' or input_type == 'pointcloud_merge' or input_type == 'pointcloud_sequential':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform=transform, 
            unpackbits=cfg['data']['points_unpackbits'],
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PartialPointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
    
        inputs_field = data.PatchPointCloudField(
            cfg['data']['pointcloud_file'], 
            transform,
            multi_files= cfg['data']['multi_files'],
        )
    
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field