# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

core module

note 250320
- which is better to use: self.model or self.pretrained/self.finetuned?

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
from .src.data_handler import PointHistDataset, prep_dataloader, plot_hist

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
        if outdir is not None:
            self.config["outdir"] = outdir
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


    def pretrain(self, train_loader, test_loader):
        """ pretraining """
        # prepare model
        self.pretrained_model = BarlowTwins(
            self.config["input_dim"], # the dimension of the input
            self.config["hidden_mlp"], # the dimension of the hidden layer
            self.config["hidden_attn"], # the dimension of the hidden layer
            self.config["dropout_mlp"], # the dropout rate
            self.config["dropout_attn"], # the dropout rate
            self.config["hidden_hist"], # the dimension of the hidden layer
            self.config["dropout_hist"], # the dropout rate
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["hidden_proj"], # the dimension of the hidden layer
            self.config["output_proj"], # the dimension of the output layer
            self.config["num_proj"], # the number of the projection MLPs
            self.config["lambd"], # tradeoff parameter
            self.config["scale_factor"] # factor to scale the loss by
        )
        optimizer = optim.AdamW(
            self.pretrained_model.parameters(), lr=float(self.config["lr"]), weight_decay=float(self.config["weight_decay"])
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["epochs"])
        trainer = PreTrainer(
            self.config, self.pretrained_model, optimizer, scheduler=scheduler, device=self.config["device"]
            )
        # training
        trainer.train(train_loader, test_loader)
        print("> Pretraining is done.")


    def finetune(self, train_loader, test_loader):
        """ finetuning """
        # prepare model
        self.finetuned_model = LinearHead(
            self.pretrained_model, # the pre-trained model
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["num_classes"], # the number of classes
            self.config["num_layers"], # the number of layers in the MLP
            self.config["hidden_head"], # the number of hidden units in the MLP
            self.config["dropout_head"], # the dropout rate
            self.config["frozen"] # whether the pretrained model is frozen
        )
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.finetuned_model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"]
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["epochs"])
        trainer = Trainer(
            self.config, self.finetuned_model, loss_fn, optimizer, scheduler, self.config["device"]
            )
        # training
        trainer.train(train_loader, test_loader)


    # ToDo: implement this
    def predict(self, data_loader=None):
        """ prediction """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.finetuned_model.eval()
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
        """
        get representation
        note: pretrained model weight is changed after finetuning.
        
        """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.pretrained_model is None:
            raise ValueError("!! fit or load_model first !!")
        self.pretrained_model.eval()
        reps = []
        with torch.no_grad():
            for data, _ in data_loader:
                point, hist = (x.to(self.device) for x in data)
                (z1, z2), _ = self.pretrained_model(point, hist)
                output = (z1 + z2) / 2 # average two features
                reps.append(output.cpu().numpy())
        return np.concatenate(reps)


    # ToDo: implement this
    def check_data(self, dataloader, indices:list=[], output:str="", nrow:int=3, ncol:int=4):
        """ check data """
        point_list = []
        for i, (data, _) in enumerate(dataloader):
            if i in indices:
                point_list.append(data[0].numpy())
        # plot
        plot_hist(point_list, output, nrow, ncol)


    def load_pretrained(self, model_path: str, config_path: str=None):
        """ load pretrained model """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.pretrained_model = BarlowTwins(
            self.config["input_dim"], # the dimension of the input
            self.config["hidden_mlp"], # the dimension of the hidden layer
            self.config["hidden_attn"], # the dimension of the hidden layer
            self.config["dropout_mlp"], # the dropout rate
            self.config["dropout_attn"], # the dropout rate
            self.config["hidden_hist"], # the dimension of the hidden layer
            self.config["dropout_hist"], # the dropout rate
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["hidden_dim"], # the dimension of the hidden layer
            self.config["output_dim"], # the dimension of the output layer
            self.config["num_projection"], # the number of the projection MLPs
            self.config["lambd"], # tradeoff parameter
            self.config["scale_factor"] # factor to scale the loss by
        )
        self.pretrained_model.load_state_dict(torch.load(model_path))


    def load_finetuned(self, model_path: str, config_path: str=None):
        """ load model with linear head """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        init_bt_model = BarlowTwins(
            self.config["input_dim"], # the dimension of the input
            self.config["hidden_mlp"], # the dimension of the hidden layer
            self.config["hidden_attn"], # the dimension of the hidden layer
            self.config["dropout_mlp"], # the dropout rate
            self.config["dropout_attn"], # the dropout rate
            self.config["hidden_hist"], # the dimension of the hidden layer
            self.config["dropout_hist"], # the dropout rate
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["hidden_proj"], # the dimension of the hidden layer
            self.config["output_proj"], # the dimension of the output layer
            self.config["num_proj"], # the number of the projection MLPs
            self.config["lambd"], # tradeoff parameter
            self.config["scale_factor"] # factor to scale the loss by
        )
        self.finetuned_model = LinearHead(
            init_bt_model, # initialized model
            self.config["latent_dim"], # the dimension of the latent representation
            self.config["num_classes"], # the number of classes
            self.config["num_layers"], # the number of layers in the MLP
            self.config["hidden_head"], # the number of hidden units in the MLP
            self.config["dropout_head"], # the dropout rate
            self.config["frozen"] # whether the pretrained model is frozen
        )
        self.finetuned_model.load_state_dict(torch.load(model_path))