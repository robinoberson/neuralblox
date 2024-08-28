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
    
path = os.path.join(processed_data_dir, f'full_metrics_with_ground.pth')
# path = os.path.join(processed_data_dir, 'full_metrics.pth')

metrics = torch.load(path)
import pandas as pd

# Initialize lists to store data
discard_patches_list = []
is_neuralblox_list = []
n_points = []
accuracies = []
completenesses = []
recalls = []

# Loop through full_metrics to collect the data
for discard_patch in [False, True]:
    for is_neuralblox in [False, True]:
        if is_neuralblox:
            thresh = 0.01
        else:
            thresh = 0.1
            
        full_metrics = metrics[discard_patch][is_neuralblox]
        for n_inputs in full_metrics.keys():
            n_sum = 0
            acc = 0
            completeness = 0
            recall = 0
            
            for file_name in full_metrics[n_inputs].keys():
                acc += full_metrics[n_inputs][file_name]['metrics'][thresh]['Accuracy']
                completeness += full_metrics[n_inputs][file_name]['metrics'][thresh]['Completeness']
                recall += full_metrics[n_inputs][file_name]['metrics'][thresh]['Recall']
                n_sum += 1
            
            # Calculate averages
            avg_acc = acc / n_sum
            avg_completeness = completeness / n_sum
            avg_recall = recall / n_sum
            
            # Append data to lists
            discard_patches_list.append(discard_patch)
            is_neuralblox_list.append(is_neuralblox)
            n_points.append(n_inputs)
            accuracies.append(avg_acc)
            completenesses.append(avg_completeness)
            recalls.append(avg_recall)

# Create a DataFrame
df = pd.DataFrame({
    'discard_patches': discard_patches_list,
    'is_neuralblox': is_neuralblox_list,
    'n_points': n_points,
    'Accuracy': accuracies,
    'Completeness': completenesses,
    'Recall': recalls
})

name_df = f'metrics_og_with_ground.csv'
# Display the DataFrame
print(df)
#save 
df.to_csv(os.path.join(processed_data_dir, name_df))
