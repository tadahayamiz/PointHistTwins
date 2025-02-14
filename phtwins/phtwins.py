# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

ihvit module

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from typing import Tuple
import yaml

from tqdm.auto import tqdm

from .src.models import *
from .src.utils import visualize_images, visualize_attention
from .src.trainer import Trainer
from .src.data_handler import prep_dataset, prep_data, prep_test


class IhBT:
    """ IhVitをモジュールとして使うためのクラス """
    def __init__(
            self, config_path: str
            ):
        # configの読み込み
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.config["config_path"] = config_path
        self.input_path = None
        self.input_path2 = None
        self.model = None


    def load_model(self, model_path: str, config_path: str=None):
        """ モデルの読み込み """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.model = VitForClassification(self.config)
        self.model.load_state_dict(torch.load(model_path))


    def load_data(self, input_path: str, transform=None):
        """
        prepare dataset using ImageFolder
        
        Parameters
        ----------
        image_path: str
            the path to the image folder
        
        transform: a list of transform functions
            each function should return torch.tensor by __call__ method
        
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((self.config["image_size"], self.config["image_size"])),
                transforms.ToTensor()
            ])
        mydataset = datasets.ImageFolder(
            root=input_path, transform=transform
        )
        return mydataset


    def prep_data(
            self, exp_name: str=None, input_path: str=None, input_path2: str=None,
            transform: Tuple[transforms.Compose, transforms.Compose]=(None, None)
            ):
        """ dataの読み込み """
        if exp_name is None:
            exp_name = "exp"
        self.config["exp_name"] = exp_name
        self.input_path = input_path
        self.input_path2 = input_path2
        train_loader, test_loader, classes = prep_data(
            image_path=(input_path, input_path2), 
            batch_size=self.config["batch_size"], transform=transform, shuffle=(True, False)
            )
        return train_loader, test_loader, classes


    def prep_test(self, exp_name: str=None):
        """ CIFAR10を使ったテスト用 """
        if exp_name is None:
            exp_name = "exp"
        self.config["exp_name"] = exp_name
        train_loader, test_loader, classes = prep_test(batch_size=self.config["batch_size"])
        return train_loader, test_loader, classes


    def fit(self, train_loader, test_loader, classes=None):
        """ training """
        # モデル等の準備
        self.model = VitForClassification(self.config)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            self.config, self.model, optimizer, loss_fn, self.config["exp_name"], device=self.config["device"]
            )
        # training
        trainer.train(
            train_loader, test_loader, classes, save_model_evry_n_epochs=self.config["save_model_every"]
            )
        if self.input_path2 is None:
            accuracy, avg_loss = trainer.evaluate(test_loader)
            print(f"Accuracy: {accuracy} // Average Loss: {avg_loss}")


    def predict(self, data_loader=None):
        """ prediction """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.model.eval()
        preds = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.config["device"])
                output = self.model(data)[0]
                preds.append(output.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)


    def check_images(self, indices:list=[], output:str="", nrow:int=3, ncol:int=4):
        """ check images """
        if self.input_path is None:
            raise ValueError("!! Give input_path !!")
        mydataset = prep_dataset(self.input_path)
        visualize_images(mydataset, indices, output, nrow, ncol)
 

    def get_attentions(
        self, indices:list=[], output:str="", nrow:int=2, ncol:int=3
        ):
        """
        visualize the attention of the images in the given dataset
        
        """
        if self.input_path is None:
            raise ValueError("!! Give input_path !!")
        mydataset = prep_dataset(self.input_path)
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        visualize_attention(
            self.model, mydataset, self.config,
            indices, output, nrow, ncol, self.config["device"]
            )