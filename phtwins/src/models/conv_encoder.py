# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

histogram encoder with convolutional neural network

models file

@author: tadahaya
"""
import math
import torch
import torch.nn as nn

# HistEncoder
class Hist1dEncoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(Hist1dEncoder, self).__init__()
        # 畳み込み層
        self.conv = nn.Conv1d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=3,  # フィルターサイズ
                              stride=1,       # ストライド 1
                              padding=1)      # パディング 'same' パディング
        # グローバル平均プーリング
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 最終的に (1, 1) にする

    def forward(self, x):
        # 畳み込み
        x = self.conv(x)
        # グローバル平均プーリング
        x = self.global_avg_pool(x)
        # 埋め込みベクトルとしてフラット化
        x = x.view(x.size(0), -1)
        return x


class Hist2dEncoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(Hist2dEncoder, self).__init__()
        # 畳み込み層
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=3,  # フィルターサイズ 3x3
                              stride=1,       # ストライド 1
                              padding=1)      # パディング 'same' パディング
        # グローバル平均プーリング
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 最終的に (1, 1) にする

    def forward(self, x):
        # 畳み込み
        x = self.conv(x)
        # グローバル平均プーリング
        x = self.global_avg_pool(x)
        # 埋め込みベクトルとしてフラット化
        x = x.view(x.size(0), -1)
        return x


class Hist3dEncoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(Hist3dEncoder, self).__init__()
        # 畳み込み層
        self.conv = nn.Conv3d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=3,  # フィルターサイズ 3x3x3
                              stride=1,       # ストライド 1
                              padding=1)      # パディング 'same' パディング
        # グローバル平均プーリング
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # 最終的に (1, 1) にする

    def forward(self, x):
        # 畳み込み
        x = self.conv(x)
        # グローバル平均プーリング
        x = self.global_avg_pool(x)
        # 埋め込みベクトルとしてフラット化
        x = x.view(x.size(0), -1)
        return x