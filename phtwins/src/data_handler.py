# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import random
import numpy as np
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageOps, ImageFilter

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class MyDataset(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, input=None, output=None, transform=None):
        if input is None:
            raise ValueError('!! Give input !!')
        if output is None:
            raise ValueError('!! Give output !!')
        if type(transform) == list:
            if len(transform) != 0:
                if transform[0] is None:
                    self.transform = []
                else:
                    self.transform = transform
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = []
            else:
                self.transform = [transform]
        self.input = input
        self.output = output
        self.datanum = len(self.input)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        input = self.input[idx]
        output = self.output[idx]
        if len(self.transform) > 0:
            for t in self.transform:
                input = t(input)
        return input, output


class SSLTransform:
    def __init__(self, transform=None, transform_prime=None) -> None:
        """
        transform for self-supervised learning

        Parameters
        ----------
        transform: torchvision.transforms
            transform for the original image

        transform_prime: torchvision.transforms
            transform to be applied to the second
        
        """
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                    p=0.8
                    ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.5),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        if transform_prime is None:
            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                    p=0.8
                    ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.5),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform_prime = transform_prime
        

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def prep_dataset(image_path:str, transform=None) -> torch.utils.data.Dataset:
    """
    prepare dataset using ImageFolder
    
    Parameters
    ----------
    image_path: str
        the path to the image folder
    
    transform: a list of transform functions
        each function should return torch.tensor by __call__ method
    
    """
    mydataset = datasets.ImageFolder(
        root=image_path, transform=transform
        )
    return mydataset


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
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


def generate_subset(dataset:torch.utils.data.Dataset, ratio:float=0.1, random_seed:int=0):
    """
    generate a subset of the dataset
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        the dataset to be subset
    
    ratio: float
        the ratio of the subset to the original dataset
    
    random_seed: int
        the random seed for reproducibility
    
    """
    # set seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # generate sizes
    n_samples = len(dataset)
    size_test = int(n_samples * ratio)
    size_train = n_samples - size_test
    # shuffle and split
    ds_train, ds_test = torch.utils.data.random_split(
        dataset, [size_train, size_test]
        )
    return ds_train, ds_test


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _default_transform():
    """ return default transforms """
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (32, 32), scale=(0.8, 1.0),
                ratio=(0.75, 1.3333), interpolation=2
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return train_transform, test_transform


def prep_data(
    image_path=(None, None), batch_size:int=4,
    transform=(None, None), shuffle=(True, False),
    num_workers:int=2, pin_memory:bool=True, 
    ratio:float=0.1, random_seed:int=0
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    
    Parameters
    ----------
    image_path: (str, str)
        the path to the training and test image folders, respectively
            
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing

    """
    # check transform
    if transform[0] is None:
        transform = _default_transform()
    # dataset and dataloader preparation
    if image_path[1] is not None:
        train_dataset = prep_dataset(image_path[0], transform[0])
        train_loader = prep_dataloader(
            train_dataset, batch_size, shuffle[0], num_workers, pin_memory
            )
        test_dataset = prep_dataset(image_path[1], transform[1])
        test_loader = prep_dataloader(
            test_dataset, batch_size, shuffle[1], num_workers, pin_memory
            )
        # class names in string
        classes = train_dataset.classes
    else:
        dataset = prep_dataset(image_path[0], transform[0])
        train_dataset, test_dataset = generate_subset(
            dataset, ratio, random_seed
            )
        train_loader = prep_dataloader(
            train_dataset, batch_size, shuffle[0], num_workers, pin_memory
            )
        test_loader = prep_dataloader(
            test_dataset, batch_size, shuffle[1], num_workers, pin_memory
            )    
        # class name dict
        classes = dataset.class_to_idx
    return train_loader, test_loader, classes


def prep_test(
        batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None
        ):
    """ test data preparation """
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (32, 32), scale=(0.8, 1.0),
                ratio=(0.75, 1.3333), interpolation=2
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./", train=False, download=True, transform=test_transform
        )
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    classes = trainset.class_to_idx
    return trainloader, testloader, classes