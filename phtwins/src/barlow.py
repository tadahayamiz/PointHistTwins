# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

Point-Histogram Twins, inspired from Barlow Twins
This code is based on the following repositories:

- [origin](https://github.com/facebookresearch/barlowtwins)  
- [simple version](https://github.com/MaxLikesMath/Barlow-Twins-Pytorch/tree/main)  

@author: tadahaya
"""
import torch
import torch.nn as nn

from .models.hist_encoder import HistEncoder
from .models.point_encoder import PointEncoder

def off_diagonal(x):
    """ return a flattened view of the off-diagonal elements of a square matrix """
    n, m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    """
    single GPU version based on https://github.com/facebookresearch/barlowtwins

    """
    def __init__(
            self, input_dim, hidden_mlp, hidden_attn, dropout_mlp, dropout_attn, # for PointEncoder
            hist_dim, hidden_hist, dropout_hist, # for HistEncoder
            latent_dim, hidden_proj, output_proj, num_proj=2, lambd=0.005, scale_factor=1, # for BarlowTwins
            ):
        """
        Parameters
        ----------
        latent_dim: dimension of latent representation from encoders

        hidden_proj: dimension of hidden layer

        output_proj: dimension of output data

        num_proj: number of projection layers
        
        lambd: tradeoff function

        scale_factor: factor to scale loss by

        """
        super().__init__()
        # encoder
        self.point_encoder = PointEncoder(input_dim, hidden_mlp, latent_dim, hidden_attn, dropout_mlp, dropout_attn)
        self.hist_encoder = HistEncoder(hist_dim, hidden_hist, latent_dim, dropout_hist)
        # projector
        layers = []
        in_features = latent_dim
        for i in range(num_proj):
            layers.append(nn.Linear(in_features, hidden_proj, bias=False)) # bias=False, due to BN
            layers.append(nn.BatchNorm1d(hidden_proj))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_proj
        layers.append(nn.Linear(hidden_proj, output_proj, bias=False)) # bias=False, due to BN
        self.projector = nn.Sequential(*layers)
        # normalization layer for z1 and z2
        self.bn = nn.BatchNorm1d(output_proj, affine=False) # no learnable parameters
        self.lambd = lambd
        self.scale_factor = scale_factor


    def forward(self, point, hist):  # input two views
        # encode views
        z1, weight = self.point_encoder(point)  # returns embedding and attention weights
        z2 = self.hist_encoder(hist)
        # project representations
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        batch_size = z1.shape[0]
        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2)) / batch_size
        # scaling
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() / batch_size
        off_diag = off_diagonal(c).pow_(2).sum() / (batch_size * (batch_size - 1))  # off-diagonal
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return (z1, z2), loss

        
class FeatureExtractor(nn.Module):
    def __init__(self, twins, frozen:bool=False):
        super().__init__()
        if frozen:
            for param in twins.parameters():
                param.requires_grad = False
        self.model = twins


    def forward(self, point, hist):
        (z1, z2), loss = self.model(point, hist)
        return (z1 + z2) / 2, loss  # average two features


class LinearHead(nn.Module):
    def __init__(
            self, pretrained, latent_dim:int, num_classes:int, num_layers:int=2,
            hidden_head:int=512, dropout_head:float=0.3, frozen:bool=False
            ):
        """
        Parameters
        ----------
        pretrained: pre-trained model

        latent_dim: dimension of the representation

        num_classes: number of classes

        num_layers: number of layers in MLP

        hidden_head: number of hidden units in MLP
            int or list of int

        dropout_head: dropout rate

        """
        super().__init__()
        self.model = FeatureExtractor(pretrained, frozen=frozen)
        # MLP
        layers = []
        if isinstance(hidden_head, int):
            hidden_head = [hidden_head] * num_layers
        in_features = latent_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_head[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_head))
            in_features = hidden_head[i]
        layers.append(nn.Linear(hidden_head[i], num_classes))  # output layer
        self.linear_head = nn.Sequential(*layers)


    def forward(self, point, hist):
        output, bt_loss = self.model(point, hist) # feature extraction
        return self.linear_head(output), bt_loss # classification