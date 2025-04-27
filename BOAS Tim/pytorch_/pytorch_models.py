import numpy as np
import torch
from torch import nn
from torch.nn import init



class AudioClassifierNoSlice(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, num_classes, hybrid_et=False):
        super().__init__()
        
        self.num_classes = num_classes
        starting_channels = 2 if hybrid_et else 1
        conv_layers = []

        # General 
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.dropout = nn.Dropout2d(p=0.2)

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.nchannels1 = 64
        self.conv1 = nn.Conv2d(starting_channels, self.nchannels1, kernel_size=(5, 11), stride=(2, 2), padding=2)
        self.bn1 = nn.BatchNorm2d(self.nchannels1)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu, self.bn1, self.pooling, self.dropout]

        # Second Convolution Block
        self.nchannels2 = 64
        self.conv2 = nn.Conv2d(self.nchannels1, self.nchannels2, kernel_size=(3, 5), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(self.nchannels2)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu, self.bn2]#, self.pooling, self.dropout]

        # Third Convolution Block
        self.nchannels3 = 128
        self.conv3 = nn.Conv2d(self.nchannels2, self.nchannels3, kernel_size=(3, 5), stride=(1, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(self.nchannels3)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu, self.bn3, self.pooling, self.dropout]

        # Fourth Convolution Block
        self.nchannels4 = 128
        self.conv4 = nn.Conv2d(self.nchannels3, self.nchannels4, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(self.nchannels4)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu, self.bn4]#, self.pooling, self.dropout]

        # Fifth Convolution Block
        self.nchannels5 = 256
        self.conv5 = nn.Conv2d(self.nchannels4, self.nchannels5, kernel_size=(3, 3), stride=(1, 2), padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(self.nchannels5)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu, self.bn5, self.pooling, self.dropout]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=self.nchannels5, out_features=self.nchannels5)
        self.lin2 = nn.Linear(in_features=self.nchannels5, out_features=self.num_classes)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = self.flatten(x)

        # Linear layer
        """ x = self.lin1(x)
        x = self.relu(x) """
        x = self.lin2(x)

        # Final output
        return x
    


class AudioClassifierSlice (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, num_classes, hybrid_et=False):
        super().__init__()
        
        self.num_classes = num_classes
        starting_channels = 2 if hybrid_et else 1
        conv_layers = []

        # General 
        self.act = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.dropout = nn.Dropout2d(p=0.1)

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(starting_channels, 16, kernel_size=(5, 15), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.act, self.bn1, self.pooling, self.dropout]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3, 7), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.act, self.bn2, self.pooling, self.dropout]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.act]#, self.bn3, self.pooling, self.dropout]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.act]#, self.bn4, self.pooling, self.dropout]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.act, self.bn5, self.pooling, self.dropout]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=128, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=self.num_classes)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = self.flatten(x)

        # Linear layer
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)

        # Final output
        return x