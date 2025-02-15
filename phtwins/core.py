# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

ihvit module

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

from .src.barlow import BarlowTwins, LinearHead
from .src.trainer import PreTrainer, Trainer
from .src.data_handler import PointHistDataset, prep_dataloader

class PHTwins:
    """ class for training and prediction """
    def __init__(
            self, config_path: str, df: pd.DataFrame, test_df:pd.DataFrame=None,
            outdir: str=None, exp_name: str=None
            ):
        self.df = df # DataFrame containing the point data and label
        self.test_df = test_df # DataFrame containing the point data and label
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.config["config_path"] = config_path
        if exp_name is None:
            exp_name = f"exp-{datetime.today().strftime('%y%m%d')}"
        self.config["exp_name"] = exp_name
        self.outdir = outdir
        self.pretrained_model = None
        self.finetuned_model = None


    def prep_data(self, key_identify:str, key_data:list, key_label:str):
        """ prepare data """
        train_dataset = PointHistDataset(
            self.df, key_identify, key_data, key_label, self.config["num_points"], self.config["bins"]
        )
        train_loader = prep_dataloader(
            train_dataset, self.config["batch_size"], True, self.config["num_workers"], self.config["pin_memory"]
            )
        if self.test_df is None:
            return train_loader, None
        else:
            test_dataset = PointHistDataset(
                self.test_df, key_identify, key_data, key_label, self.config["num_points"], self.config["bins"]
            )
            test_loader = prep_dataloader(
                test_dataset, self.config["batch_size"], False, self.config["num_workers"], self.config["pin_memory"]
                )
            return train_loader, test_loader


    # ToDo: implement this
    def pretrain(self, train_loader, test_loader, classes=None):
        """ training """
        # prepare model
        self.pretrained = BarlowTwins(
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["hidden_proj"], # the dimension of the hidden layer
            self.config["output_proj"], # the dimension of the output layer
            self.config["num_proj"], # the number of the projection MLPs
            self.config["lambd"], # tradeoff parameter
            self.config["scale_factor"] # factor to scale the loss by
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
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


    # ToDo: implement this
    def finetune(self, train_loader, test_loader, classes=None):
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
        probs = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.config["device"])
                output = self.model(data)[0] # ToDo: check this
                preds.append(output.argmax(dim=1).cpu().numpy())
                probs.append(output.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)


    def get_representation(self, data_loader=None):
        """ get representation """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.model.eval()
        reps = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.config["device"])
                output = self.model(data)[1]# ToDo: check this
                reps.append(output.cpu().numpy())
        return np.concatenate(reps)



    def check_data(self, indices:list=[], output:str="", nrow:int=3, ncol:int=4):
        """ check data """
        raise NotImplementedError("!! Not implemented yet !!")
    

    def load_pretrained(self, model_path: str, config_path: str=None):
        """ load pretrained model """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.pretrained = BarlowTwins(
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["hidden_dim"], # the dimension of the hidden layer
            self.config["output_dim"], # the dimension of the output layer
            self.config["num_projection"], # the number of the projection MLPs
            self.config["lambd"], # tradeoff parameter
            self.config["scale_factor"] # factor to scale the loss by
        )
        self.pretrained.load_state_dict(torch.load(model_path))


    def load_finetuned(self, model_path: str, config_path: str=None):
        """ load model with linear head """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.finetuned = LinearHead(
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["num_classes"], # the number of classes
            self.config["num_layers"], # the number of layers in the MLP
            self.config["hidden_head"], # the number of hidden units in the MLP
            self.config["dropout_head"], # the dropout rate
            self.config["frozen"] # whether the pretrained model is frozen
        )
        self.finetuned.load_state_dict(torch.load(model_path))