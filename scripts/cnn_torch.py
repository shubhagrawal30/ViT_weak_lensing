# a simple CNN model for weak lensing mass map reconstruction
# in PyTorch, with n_filters filers, kernel size k_size, n_layers layers
# with Conv2d, BatchNorm, AveragePooling2d, and then a Flatten and a single
# linear layer to output the mass map parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_channels, num_classes, dropout=0.1, \
                 n_filters=32, k_size=3, n_layers=7):
        super(CNN, self).__init__()
        self.n_filters = n_filters
        self.k_size = k_size
        self.n_layers = n_layers
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv_laters = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.conv_laters.append(nn.Conv2d(num_channels, n_filters, k_size, \
                                                  padding="same"))
            else:
                self.conv_laters.append(nn.Conv2d(n_filters, n_filters, k_size, \
                                                  padding="same"))
            self.bn_layers.append(nn.BatchNorm2d(n_filters))
            self.pool_layers.append(nn.AvgPool2d(2))
            self.dropout_layers.append(nn.Dropout2d(dropout))

        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(n_filters)
        self.dropout_lin = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters, num_classes)
        
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.conv_laters[i](x)
            x = F.relu(x)
            x = self.dropout_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.pool_layers[i](x)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout_lin(x)
        x = self.fc(x)
        return x

        