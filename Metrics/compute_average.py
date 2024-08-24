import os 
import torch 
import yaml

def load_config(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)

if 'robin' in os.getcwd():
    bool_location = 1
    cfg_path = '/home/robin/Dev/MasterThesis/GithubRepos/master_thesis/neuralblox/configs/evaluation_cfg.yaml'
    print(f'On home')

elif 'cluster' in os.getcwd():
    bool_location = 2
    print(f'On euler')
else:
    bool_location = 0
    cfg_path = 'configs/evaluation_cfg_local.yaml'
    print(f'On local')
    
cfg = load_config(cfg_path)

if cfg['evaluation']['is_neuralblox']:
    processed_data_dir = cfg['evaluation']['processed_data_dir_neuralblox']
else:
    processed_data_dir = cfg['evaluation']['processed_data_dir']
    
path = os.path.join(processed_data_dir, f'full_metrics_og{cfg["evaluation"]["is_neuralblox"]}_ground{cfg["evaluation"]["discard_ground"]}.pth')
path = os.path.join('/media/robin/T7/neuralblox/evaluation/processed_data_neuralblox', 'full_metrics.pth')

full_metrics = torch.load(path)

acc = 0
completeness = 0
recall = 0
thresh = cfg['evaluation']['thresh']
for key in full_metrics.keys():
    n_sum = 0

    for key2 in full_metrics[key].keys():
        print(f'\t{key2}')
        for thr in full_metrics[key][key2]['metrics'].keys():
            print(f'\t\t{full_metrics[key][key2]["metrics"][thr]}')
        acc += full_metrics[key][key2]['metrics'][thresh]['Accuracy']
        completeness += full_metrics[key][key2]['metrics'][thresh]['Completeness']
        recall += full_metrics[key][key2]['metrics'][thresh]['Recall']
        n_sum += 1
    print(f'{key}')
    print(f'Accuracy: {acc/n_sum}')
    print(f'Completeness: {completeness/n_sum}')
    print(f'Recall: {recall/n_sum}')
    print()