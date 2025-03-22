# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def calc_hist(X, bins=16) -> np.ndarray:
    try:
        s = X.shape[1]
    except IndexError:
        s = 1
    if s == 1:
        hist, _ = np.histogram(X, bins=bins)
    elif s == 2:
        hist, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    elif s == 3:
        hist, _ = np.histogramdd(X, bins=bins)
    else:
        raise ValueError("!! Input array must be 1D, 2D, or 3D. !!")
    return hist


class PointHistDataset(Dataset):
    def __init__(self, df, key_identify, key_data, key_label, num_points=768, bins=16, transform=None):
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the data and label

        key_identify: str
            the key to identify the data

        key_data: list
            the keys for the data

        key_label: int
            the key for the label
            note that the label should be integer

        bins: int
            the number of bins for the histogram

        num_points: int
            the number of points to be sampled

        transform: callable
            the transform function to be applied to the data
        
        """
        self.df = df
        self.bins = bins
        self.num_points = num_points
        self.transform = transform
        # prepare meta data
        self.key_identify = key_identify
        identifier = list(df[key_identify].unique())
        self.idx2id = {k: v for k, v in enumerate(identifier)} # index to identifier
        self.num_data = len(identifier) # number of data
        # prepare data
        self.key_data = key_data
        # prepare label
        self.key_label = key_label
        self.id2label = dict(zip(df[key_identify], df[key_label]))

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx) -> tuple:
        # get data
        key = self.idx2id[idx]
        data = self.df[self.df[self.key_identify] == key]
        # get point cloud
        pointcloud = np.array(data[self.key_data], dtype=np.float32)
        # limit the number of points if necessary (random sampling)
        if pointcloud.shape[0] > self.num_points:
            idxs = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
            pointcloud = pointcloud[idxs, :]
        # transform
        if self.transform:
            pointcloud = self.transform(pointcloud)
        # prepare histogram
        hist = calc_hist(pointcloud, bins=self.bins)
        hist = torch.tensor(hist, dtype=torch.float32)
        # prepare point cloud
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        # prepare label
        label = self.id2label[key]
        try:
            label = torch.tensor(label, dtype=torch.long)
        except ValueError:
            pass # if label is None

        print(f"pointcloud: {pointcloud.shape}, hist: {hist.shape}, label: {label}")

        return (pointcloud, hist), label


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)