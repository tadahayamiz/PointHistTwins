# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

histogram encoder with convolutional neural network

@author: tadahaya
"""
import math
import torch
import torch.nn as nn

# HistEncoder
class Hist1dConv(nn.Module):
    def __init__(self, output_channels=64):
        super(Hist1dConv, self).__init__()
        # convolutional layer
        self.conv = nn.Conv1d(in_channels=1, # input channel is 1 because of histogram
                              out_channels=output_channels,
                              kernel_size=3,  # filter size 3x3
                              stride=1,       # stride 1
                              padding=1)      # padding 'same'
        # global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # finally (1, 1)

    def forward(self, x):
        # convolution
        x = self.conv(x)
        # global average pooling
        x = self.global_avg_pool(x)
        # flatten as embedding vector
        x = x.view(x.size(0), -1)
        return x


class Hist2dConv(nn.Module):
    def __init__(self, output_channels=64):
        super(Hist2dConv, self).__init__()
        # convolutional layer
        self.conv = nn.Conv2d(in_channels=1, # input channel is 1 because of histogram
                              out_channels=output_channels,
                              kernel_size=3,  # filter size 3x3
                              stride=1,       # stride 1
                              padding=1)      # padding 'same'
        # global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # finally (1, 1)

    def forward(self, x):
        # convolution
        x = self.conv(x)
        # global average pooling
        x = self.global_avg_pool(x)
        # flatten as embedding vector
        x = x.view(x.size(0), -1)
        return x


class Hist3dConv(nn.Module):
    def __init__(self, output_channels=64):
        super(Hist3dConv, self).__init__()
        # convolutional layer
        self.conv = nn.Conv3d(in_channels=1, # input channel is 1 because of histogram
                              out_channels=output_channels,
                              kernel_size=3,  # filter size 3x3
                              stride=1,       # stride 1
                              padding=1)      # padding 'same'
        # global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # finally (1, 1)

    def forward(self, x):
        # convolution
        x = self.conv(x)
        # global average pooling
        x = self.global_avg_pool(x)
        # flatten as embedding vector
        x = x.view(x.size(0), -1)
        return x
    

# HistEncoder
class HistEncoder(nn.Module):
    def __init__(self, hist_dim:int, hidden_dim:int, output_dim:int, prob_dropout=0.3):
        super(HistEncoder, self).__init__()
        # convolutional layer
        if hist_dim == 1:
            self.hist_conv = Hist1dConv(output_channels=hidden_dim)
        elif hist_dim == 2:
            self.hist_conv = Hist2dConv(output_channels=hidden_dim)
        elif hist_dim == 3:
            self.hist_conv = Hist3dConv(output_channels=hidden_dim)
        else:
            raise ValueError("!! hist_dim should be 1, 2 or 3 !!")
        # MLP for output
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Linear
            nn.ReLU(),                         # ReLU
            nn.Dropout(prob_dropout),          # Dropout
            nn.Linear(hidden_dim, output_dim)  # Linear for output
        )

    def forward(self, x):
        # convolution

        print(x.size())

        x = self.hist_conv(x)

        print(x.size())

        # MLP
        x = self.mlp(x)
        return x