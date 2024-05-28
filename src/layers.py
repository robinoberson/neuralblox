import torch.nn as nn
import torch.nn.init as init

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Conv3D_one_input(nn.Module):
    def __init__(self, num_blocks = 1, num_channels = [256, 416, 416, 128]):
        super(Conv3D_one_input, self).__init__()

        assert num_channels[0] == 256 and num_channels[-1] == 128, "Input must have 256 channels and output must have 128 channels"

        self.initial_conv = nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, padding=0)
        self.initial_bn = nn.BatchNorm3d(num_channels[1])
        self.initial_relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        for i in range(0, num_blocks):
            stride = 1
            if i == 0:
                stride = 2
            
            self.layers.append(ResidualBlock(num_channels[i+1], num_channels[i + 2], stride=stride))

        self.final_conv = nn.Conv3d(num_channels[-2], num_channels[-1], kernel_size=3, padding=0)
        self.final_bn = nn.BatchNorm3d(num_channels[-1])
        self.final_relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._initialize_weights()

    def forward(self, fea):
        z = fea['latent']

        z = self.initial_conv(z)
        z = self.initial_bn(z)
        z = self.initial_relu(z)

        for layer in self.layers:
            z = layer(z)

        z = self.final_conv(z)
        z = self.final_bn(z)
        z = self.final_relu(z)

        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
# class Conv3D_one_input(nn.Module):
#     ''' 2-layer 3D convolutional networks.

#     Args:
#         num_channels (list of int): number of channels for each layer
#     '''

#     def __init__(self, num_channels=[256, 224, 192, 160, 128]):
#         super(Conv3D_one_input, self).__init__()
        
#         self.conv3d0 = nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, padding=0)
#         self.conv3d1 = nn.Conv3d(num_channels[1], num_channels[2], kernel_size=3, padding=1, stride=2)
#         self.conv3d2 = nn.Conv3d(num_channels[2], num_channels[3], kernel_size=3, padding=0)
#         self.conv3d3 = nn.Conv3d(num_channels[3], num_channels[4], kernel_size=3, padding=1)
        
#         self.conv_layers = nn.ModuleList([self.conv3d0, self.conv3d1, self.conv3d2, self.conv3d3])
#         self.activation = nn.ReLU()

#     def forward(self, fea):
#         z = fea['latent']
        
#         for conv_layer in self.conv_layers:
#             z = conv_layer(z)
#             z = self.activation(z)

#         return z
    
# class Conv3D_one_input(nn.Module):
#     ''' Single fully connected layer network.

#     Args:
#         input_size (int): input size
#         output_size (int): output size
#         output_shape (tuple): shape of the output tensor
#     '''

#     def __init__(self, num_channels=None):
#         input_size = 256 * 8 * 8 * 8  # Example input size, calculated from the input tensor shape
#         output_size = 128 * 6 * 6 * 6  # Example output size, calculated from the desired output tensor shape
#         output_shape = (128, 6, 6, 6) 
        
#         super(Conv3D_one_input, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.pool = nn.MaxPool3d(kernel_size=3, stride=2)  # Max pooling with kernel size 3 and stride 3
#         self.fc = nn.Linear(input_size, output_size)
#         self.activation = nn.ReLU()
#         self.output_shape = output_shape

#     def forward(self, fea):
#         x = fea['latent']
#         x = self.pool(x)  # Max pooling to reduce spatial dimensions
#         x = x.view(x.size(0), -1)  # Flatten the pooled input tensor
#         x = self.fc(x)
#         x = self.activation(x)
#         x = x.view(x.size(0), *self.output_shape)  # Reshape the output tensor
#         return x
