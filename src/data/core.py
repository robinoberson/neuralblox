import os
import logging
from torch.utils import data
import numpy as np
import yaml
from src.common import decide_total_volume_range, update_reso
import torch
import random
import torch.nn as nn

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError

# class CategoryBatchSampler(data.BatchSampler):
#     def __init__(self, dataset, batch_size, drop_last=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         super().__init__(dataset, batch_size, drop_last)
    
#     def __iter__(self):
#         categories = list(self.dataset.metadata.keys())
#         random.shuffle(categories)
#         for category in categories:
#             batch = []
#             category_indices = [idx for idx, model in enumerate(self.dataset.models) if model['category'] == category]
#             random.shuffle(category_indices)
#             batch.extend(category_indices)
#             while len(batch) >= self.batch_size:                
#                 yield batch[:self.batch_size]
#                 batch = batch[self.batch_size:]
        
            
class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None, cfg=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg
        
        self.initialize_data(categories, split)

    def initialize_data(self, categories, split=None):

        # Read metadata file
        metadata_file = os.path.join(self.dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            if split is None:
                self.models += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]
                
        self.batch_groups = self.create_batch_groups()

    def create_batch_groups(self):
        ''' Create batch groups from the dataset, ensuring no duplicate categories in each group.

        Returns:
            list: List of batch groups
        '''
        batch_groups = []
        current_group = []
        current_categories = set()
        group_size = self.cfg['training']['batch_group_size']
        #randomly shuffle models
        models_grouping = random.sample(self.models, len(self.models))
        
        if group_size > len(models_grouping):
            print(f'group size {group_size} is larger than number of models {len(self.models)}')
            while len(current_group) < group_size:
                current_group += random.sample(self.models, len(self.models))[:group_size - len(current_group)]

            batch_groups.append(current_group)
            print(f'added group, size: {len(current_group)}, total groups: {len(batch_groups)}')
        else:
            print(f'group size: {group_size}, total models: {len(self.models)}')
            it = 0
            while len(models_grouping) > 0 and it < 1000:
                for idx_model, model_info in enumerate(models_grouping): #models_grouping[0]
                    category = model_info['category']
                    if category not in current_categories:
                        
                        current_categories.add(category)
                        current_group.append(model_info)
                        models_grouping.pop(idx_model)
                    
                        # Check if adding this model would exceed the batch group size or duplicate a category
                        if len(current_group) == group_size or len(models_grouping) == 0 or it > 1000:
                            print(f'added group, size: {len(current_group)}, total groups: {len(batch_groups)}')
                            batch_groups.append(current_group)
                            current_group = []
                            current_categories = set()
                            it = 0
                        break
                it += 1
        return batch_groups
                
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.batch_groups)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        if idx >= len(self.batch_groups) or idx < 0:
            raise IndexError('Index out of range.')
        
        batch_group = self.batch_groups[idx]
        data_batch = []
        
        for model_info in batch_group:

            category = model_info['category']
            model = model_info['model']
            c_idx = self.metadata[category]['idx']
        

            model_path = os.path.join(self.dataset_folder, category, model)
            data = {}

            info = c_idx
            
            for field_name, field in self.fields.items():
                try:
                    field_data = field.load(model_path, info)
                except Exception as e:
                    if self.no_except:
                        logger.warn(
                            'Error occured when loading field %s of model %s, skipping. Error: %s'
                            % (field_name, model, str(e))
                        )
                        return None
                    else:
                        raise

                if isinstance(field_data, dict):
                    for k, v in field_data.items():
                        if k is None:
                            data[field_name] = v
                        else:
                            data['%s.%s' % (field_name, k)] = v
                else:
                    data[field_name] = field_data

            data_batch.append(data)
            
        if self.transform is not None:
            self.transform(data_batch)
            
        # Initialize a dictionary to store lists of tensors
        array_dict = {key: [] for key in data_batch[0].keys()}

        # Collect tensors for each key
        for item in data_batch:
            for key in item.keys():
                array_dict[key].append(item[key])

        # Stack tensors for each key
        data_batch = {key: np.stack(array_dict[key], axis=0) for key in array_dict}
        
        return data_batch
       
    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    
    

    batch = list(filter(lambda x: x is not None, batch))
        
    # for i, item in enumerate(batch):
    #     if isinstance(item, dict):
    #         for key, value in item.items():
    #             if isinstance(value, torch.Tensor):
    #                 batch[i][key] = value.float()  # Convert tensor to float

    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    def set_num_threads(nt):
        try: 
            import mkl; mkl.set_num_threads(nt)
        except: 
            pass
            torch.set_num_threads(1)
            os.environ['IPC_ENABLE']='1'
            for o in ['OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

class OptiFusionDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.batch_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    def __len__(self):
        return len(self.batch_dirs)

    def __getitem__(self, idx):
        batch_dir = self.batch_dirs[idx]
        latent_map_sampled_stacked = torch.load(os.path.join(batch_dir, 'latent_map_sampled_stacked.pt'))
        logits_gt = torch.load(os.path.join(batch_dir, 'logits_gt.pt'))
        p_stacked = torch.load(os.path.join(batch_dir, 'p_stacked.pt'))
        p_n_stacked = torch.load(os.path.join(batch_dir, 'p_n_stacked.pt'))
        p_in = torch.load(os.path.join(batch_dir, 'p_in.pt'))
        return latent_map_sampled_stacked, logits_gt, p_stacked, p_n_stacked, p_in
    

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha  # Weight for the L1 loss component
    
    def forward(self, input, target):
        mse_loss = self.mse(input, target)
        l1_loss = self.l1(input, target)
        combined_loss = (1 - self.alpha) * mse_loss + self.alpha * l1_loss
        return combined_loss