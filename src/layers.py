import torch.nn as nn

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

class Conv3D_one_input(nn.Module):
    ''' 2-layer 3D convolutional networks.

    Args:
        num_channels (list of int): number of channels for each layer
    '''

    def __init__(self, num_channels=[256, 224, 192, 160, 128]):
        super(Conv3D_one_input, self).__init__()
        
        self.conv3d0 = nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, padding=0)
        self.conv3d1 = nn.Conv3d(num_channels[1], num_channels[2], kernel_size=3, padding=1, stride=2)
        self.conv3d2 = nn.Conv3d(num_channels[2], num_channels[3], kernel_size=3, padding=0)
        self.conv3d3 = nn.Conv3d(num_channels[3], num_channels[4], kernel_size=3, padding=1)
        
        self.conv_layers = nn.ModuleList([self.conv3d0, self.conv3d1, self.conv3d2, self.conv3d3])
        self.activation = nn.ReLU()

    def forward(self, fea):
        z = fea['latent']
        
        for conv_layer in self.conv_layers:
            z = conv_layer(z)
            z = self.activation(z)

        return z
