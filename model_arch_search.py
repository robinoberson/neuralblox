print('Launching script for model architecture search...')
import numpy as np
from src import layers
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import itertools
import random
from scipy.stats import truncnorm
import datetime
import os 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)

# def generate_random_tensor(shape, mean, std, lower_bound, upper_bound, device):

#     samples = []
#     while len(samples) < np.prod(shape):
#         # Generate random samples from the Gaussian distribution
#         random_samples = truncnorm.rvs((lower_bound - mean) / std, (upper_bound - mean) / std, loc=mean, scale=std, size=1000)
#         # Filter samples within the desired range
#         filtered_samples = random_samples[(random_samples >= lower_bound) & (random_samples <= upper_bound)]
#         samples.extend(filtered_samples.tolist())

#     # Convert the samples to a numpy array and reshape
#     random_array = np.array(samples)[:np.prod(shape)].reshape(shape)

#     # Convert the numpy array to a PyTorch tensor and move to the specified device
#     random_tensor = torch.tensor(random_array, dtype=torch.float32, device=device)

#     return random_tensor


# # Parameters obtained from the Kolmogorov-Smirnov test
# mean = 15.5506935 
# std = 39.65122

# # Define the lower and upper bounds of the truncated range
# lower_bound = 0
# upper_bound = 1500

# input_tensor = generate_random_tensor((1, 256, 18, 18, 18), mean, std, lower_bound, upper_bound, device)
# output_tensor = generate_random_tensor((1, 128, 6, 6, 6), mean, std, lower_bound, upper_bound, device)

latent_map_sampled_stacked = torch.load('../Playground/Training/debug/fea_down/latent_map_sampled_merged.pt').reshape(-1, 256, 18, 18, 18)

input_tensor = latent_map_sampled_stacked[7].unsqueeze(0)
output_tensor = latent_map_sampled_stacked[7,:128, 6:12, 6:12, 6:12].unsqueeze(0)
print(input_tensor.shape, output_tensor.shape)

class Conv3D_one_input(nn.Module):
    def __init__(self, num_layers, num_channels):
        super(Conv3D_one_input, self).__init__()
        
        assert num_layers >= 3 and num_layers <= 10, "Number of layers must be between 3 and 7"
        assert num_channels[0] == 256 and num_channels[-1] == 128, "Input must have 256 channels and output must have 128 channels"

        self.conv_layers = nn.ModuleList()

        # Define the initial three layers
        self.conv_layers.append(nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, padding=0))
        # print(f'Adding layer 0, {num_channels[0]} -> {num_channels[1]}')
        self.conv_layers.append(nn.Conv3d(num_channels[1], num_channels[2], kernel_size=3, padding=1, stride=2))
        # print(f'Adding layer 1, {num_channels[1]} -> {num_channels[2]}')
        self.conv_layers.append(nn.Conv3d(num_channels[2], num_channels[3], kernel_size=3, padding=0))
        # print(f'Adding layer 2, {num_channels[2]} -> {num_channels[3]}')

        # Define the additional layers that preserve input/output sizes
        for i in range(3, num_layers-2):
            # print(f'Adding layer {i}, {num_channels[i]} -> {num_channels[i+1]}')
            self.conv_layers.append(nn.Conv3d(num_channels[i], num_channels[i+1], kernel_size=3, padding=1))
        
        # print(f'Adding final layer {num_layers-2}, {num_channels[-2]} -> {num_channels[-1]}')
        # Define the final layer to ensure output has 128 channels
        self.conv_layers.append(nn.Conv3d(num_channels[-2], num_channels[-1], kernel_size=3, padding=1))

        # Initialize weights
        for conv_layer in self.conv_layers:
            init.xavier_uniform_(conv_layer.weight)
        
        self.activation = nn.LeakyReLU()

    def forward(self, fea):
        z = fea['latent']
        
        for conv_layer in self.conv_layers:
            z = conv_layer(z)
            z = self.activation(z)

        return z

# Function to create a model with a varying number of layers and channels
def create_model(num_layers, num_channels):
    return Conv3D_one_input(num_layers=num_layers, num_channels=num_channels)

def generate_random_configurations(num_samples):
    configurations = []
    for _ in range(num_samples):
        num_layers = random.randint(5, 10)
        num_channels = [256]
        for i in range(num_layers - 2):
            if i < (num_layers - 2) // 2:  # First half: increment
                channel_count = random.randint(num_channels[-1], 512)
            else:  # Second half: decrement
                channel_count = random.randint(128, num_channels[-1])
            
            # Ensure channel_count is a multiple of 32
            channel_count = int(round(channel_count / 32)) * 32
            num_channels.append(channel_count)

        num_channels.append(128)
        configurations.append((num_layers, num_channels))
    return configurations
    
def train_and_evaluate(model, input_tensor, output_tensor, device, num_epochs=10000, lr=0.001):
    # Move model to the device
    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        # Forward pass
        outputs = model({'latent': input_tensor})
        
        if outputs.shape != output_tensor.shape:
            print(f'Output shape: {outputs.shape}, Random Output shape: {output_tensor.shape}')
            return None
        loss = criterion(outputs, output_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the model parameters

        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss.item()
num_samples = 30
configurations = generate_random_configurations(num_samples)
print(configurations)

best_config = None
best_loss = float('inf')

results = []
# Specify the file path
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Specify the file path with the date included
results_file = f'out_fusion/redwood/logs/results_{current_datetime}.txt'
#create dir if not exist
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# Open the file in write mode
with open(results_file, 'w') as f:
    for num_layers, num_channels in configurations:
        print(f'Testing configuration: num_layers={num_layers}, num_channels={num_channels}')
        model = create_model(num_layers=num_layers, num_channels=num_channels)
        final_loss = train_and_evaluate(model, input_tensor, output_tensor, device)
        results.append((num_layers, num_channels, final_loss))
        print(f'Configuration: num_layers={num_layers}, num_channels={num_channels}, Final Loss: {final_loss:.4f}')
        print('')
        print('*' * 50)
        print('')

        # Write the results to the file
        f.write(f'Testing configuration: num_layers={num_layers}, num_channels={num_channels}\n')
        f.write(f'Configuration: num_layers={num_layers}, num_channels={num_channels}, Final Loss: {final_loss:.4f}\n')
        f.write('*' * 50)
        f.write('\n')

        # Update best configuration if necessary
        if final_loss < best_loss:
            best_loss = final_loss
            best_config = (num_layers, num_channels)

        # Write the best configuration to the file
        f.write("\nBest Configuration:\n")
        f.write(f'num_layers={best_config[0]}, num_channels={best_config[1]}, Final Loss={best_loss:.4f}\n')

