# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

Point encoder with shared MLP and self-attention pooling

models file

@author: tadahaya
"""
import torch
import torch.nn as nn

# Shared MLPの定義
class SharedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prob_dropout=0.5):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Linear層
            nn.LayerNorm(hidden_dim),          # Layer Normalization
            nn.ReLU(),                         # ReLU活性化関数
            nn.Dropout(prob_dropout),          # Dropout
            nn.Linear(hidden_dim, output_dim)  # 出力層
        )

    # 各点に対して同じweightのMLPを当てる
    def forward(self, x):
        return self.mlp(x) # (batch, num_points, input_dim) -> (batch, num_points, output_dim)

# Self-Attention Poolingの定義
class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, prob_dropout=0.3):
        super(SelfAttentionPooling, self).__init__()
        # attention層（2層の線形層とReLU）
        self.attn_layer1 = nn.Linear(input_dim, hidden_dim)  # 入力次元から隠れ層の次元への線形変換
        self.attn_relu = nn.ReLU()  # ReLU活性化関数
        self.dropout = nn.Dropout(prob_dropout)  # Dropout層
        self.attn_layer2 = nn.Linear(hidden_dim, input_dim)  # 隠れ層から出力次元への線形変換
        self.softmax = nn.Softmax(dim=-1)  # ソフトマックスで注意重みを計算

    def forward(self, x):
        # xは(batch_size, num_points, feature_dim)の形状
        # Attention重みの計算
        attn_weights = self.attn_layer1(x)  # (batch_size, num_points, hidden_dim)
        attn_weights = self.attn_relu(attn_weights)  # ReLUを適用
        attn_weights = self.dropout(attn_weights)  # Dropoutを適用
        attn_weights = self.attn_layer2(attn_weights)  # (batch_size, num_points, feature_dim)
        # ソフトマックスを使ってattention weightを計算
        attn_weights = self.softmax(attn_weights)  # (batch_size, num_points, feature_dim)
        # 特徴量の集約
        aggregated_features = torch.sum(attn_weights * x, dim=1)  # (batch_size, feature_dim)
        return aggregated_features, attn_weights


class LinearSelfAttentionPooling(nn.Module):
    """ 完全線形版 """
    def __init__(self, input_dim, hidden_dim):
        super(LinearSelfAttentionPooling, self).__init__()
        self.attn = nn.Linear(input_dim, hidden_dim)  # 入力次元からhidden_dimへの線形変換
        self.softmax = nn.Softmax(dim=-1)  # ソフトマックスで注意重みを計算
        # attention weightsをfeature_dimに調整するための線形層
        self.attn_adjustment = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # xは(batch_size, num_points, feature_dim)の形状
        # Attention重みの計算
        attn_weights = self.softmax(self.attn(x))  # (batch_size, num_points, hidden_dim)
        # attn_weightsをfeature_dimに合わせて調整
        attn_weights = self.attn_adjustment(attn_weights)  # (batch_size, num_points, feature_dim)
        # 特徴量の集約
        aggregated_features = torch.sum(attn_weights * x, dim=1)  # (batch_size, feature_dim)
        return aggregated_features, attn_weights


# PointEncoder
class PointEncoder(nn.Module):
    def __init__(self, input_dim, hidden_mlp, hidden_attn, dropout_mlp=0.5, dropout_attn=0.3):
        super(PointEncoder, self).__init__()
        self.shared_mlp = SharedMLP(input_dim, hidden_mlp, hidden_mlp, dropout_mlp)
        self.self_attention_pooling = SelfAttentionPooling(hidden_mlp, hidden_attn, dropout_mlp)
        # self.self_attention_pooling = LinearSelfAttentionPooling(input_dim, hidden_dim)

    def forward(self, x):
        out = self.shared_mlp(x) # (batch, num_points, input_dim) -> (batch, num_points, hidden_mlp)
        out, weights = self.self_attention_pooling(out) # (batch, num_points, hidden_mlp) -> (batch, hidden_mlp)
        return out, weights