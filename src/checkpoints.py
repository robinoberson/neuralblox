import os
import urllib
import torch
from torch.utils import model_zoo


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''
    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)
        # print('Saved checkpoint to %s' % filename)

    def load(self, filename):
        '''Loads a module dictionary from local file or url.
        
        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def load_file(self, filename):
        '''Loads a module dictionary from file.
        
        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print(filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename)
            
            if 'model' in state_dict and isinstance(state_dict['model'], dict):
                loaded_state_dict = state_dict['model']
                model_state_dict = self.module_dict['model'].state_dict()
                
                # Create a new state dict with updated keys
                new_state_dict = {}
                for key in model_state_dict.keys():
                    if key in loaded_state_dict:
                        new_state_dict[key] = loaded_state_dict[key]
                    else:
                        print(f'Warning: Key {key} not found in the loaded state dict.')

                # Load the new state dict into the model
                self.module_dict['model'].load_state_dict(new_state_dict, strict=False)

                # Load other scalars
                scalars = {k: v for k, v in state_dict.items() if k not in ['model', 'optimizer']}
                return scalars
            else:
                print('Warning: No model found in checkpoint state dict!')
                raise ValueError('Invalid checkpoint format.')
        else:
            print('Warning: Could not find %s in checkpoint!' % filename)
            raise FileExistsError('Checkpoint file not found.')


    def load_url(self, url):
        '''Load a module dictionary from url.
        
        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.
        
        Args:
            state_dict (dict): State dict of model
    '''

        for k, v in self.module_dict.items():
            if k in state_dict:
                temp = state_dict[k]
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict.items()
                    if k not in self.module_dict}
        return scalars

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')