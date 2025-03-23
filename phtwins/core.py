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
from schedulefree import RAdamScheduleFree
import numpy as np
import pandas as pd
import os, yaml
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
        self.train_dataset = None
        self.test_dataset = None
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
        self.train_dataset = PointHistDataset(
            self.df, key_identify, key_data, key_label, self.config["num_points"], self.config["bins"]
        )
        train_loader = prep_dataloader(
            self.train_dataset, self.config["batch_size"], True, self.config["num_workers"], self.config["pin_memory"]
            )
        if self.test_df is None:
            return train_loader, None
        else:
            self.test_dataset = PointHistDataset(
                self.test_df, key_identify, key_data, key_label, self.config["num_points"], self.config["bins"]
            )
            test_loader = prep_dataloader(
                self.test_dataset, self.config["batch_size"], False, self.config["num_workers"], self.config["pin_memory"]
                )
            return train_loader, test_loader


    def pretrain(self, train_loader, test_loader):
        """ pretraining """
        # prepare model
        self.pretrained_model = BarlowTwins(
            input_dim=self.config["hist_dim"], # the dimension of the input
            hidden_hist=self.config["hidden_hist"], # the dimension of the hidden layer
            dropout_hist=self.config["dropout_hist"], # the dropout rate
            num_blocks=self.config["num_blocks"], # the number of blocks in the ResNet
            latent_dim=self.config["latent_dim"], # the dimension of the latent representation
            hidden_proj=self.config["hidden_proj"], # the dimension of the hidden layer
            output_proj=self.config["output_proj"], # the dimension of the output layer
            num_proj=self.config["num_proj"], # the number of the projection MLPs
            lambd=self.config["lambd"], # tradeoff parameter
            scale_factor=self.config["scale_factor"] # factor to scale the loss by
        )
        optimizer = RAdamScheduleFree(self.pretrained_model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
        trainer = PreTrainer(
            self.config, self.pretrained_model, optimizer, device=self.config["device"]
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
        optimizer = RAdamScheduleFree(self.pretrained_model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
        trainer = Trainer(
            self.config, self.finetuned_model, loss_fn, optimizer, self.config["device"]
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


    def get_representation(self, dataset=None, indices:list=[]):
        """
        get representation
        note: pretrained model weight is changed after finetuning.
        
        """
        if dataset is None:
            dataset = self.train_dataset
        if self.pretrained_model is None:
            raise ValueError("!! fit or load_model first !!")
        self.pretrained_model.eval()
        num_data = len(dataset)
        if len(indices) == 0:
            indices = list(range(num_data))
        reps = []
        with torch.no_grad():
            for i in indices:
                data, _ = dataset[i]
                hist0, hist1 = (x.to(self.config["device"]).unsqueeze(0) for x in data)  # add batch dimension
                (z1, z2), _ = self.pretrained_model(hist0, hist1)
                output = (z1 + z2) / 2  # average two features
                reps.append(output.cpu().numpy().reshape(1, -1))  # del batch dimension
        return np.vstack(reps)


    def check_data(self, dataset, indices:list=[], bins=16, nrow:int=3, ncol:int=4, output:str=""):
        """
        check data
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            the PHTwins dataset
        indices: list
            the list of indices to be checked
        
        """
        hist_list = [dataset[i][0][0].numpy().resahpe(1, -1) for i in indices] # ((hist, hist), label)
        plot_hist(hist_list, bins, nrow, ncol, output)


    def qual_eval(self, dataset, query_indices, bins=16, outdir:str=""):
        """
        qualitative evaluation
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            the PHTwins dataset

        indices: list
            the list of indices to be checked
        
        """
        # get representations
        reps = self.get_representation(dataset) # default: train dataset
        # query data
        query_reps = reps[query_indices]
        # calculate cosine similarity
        norm_query = np.linalg.norm(query_reps, axis=1, keepdims=True)
        norm_reps = np.linalg.norm(reps, axis=1)
        norm_query[norm_query == 0] = 1e-10
        norm_reps[norm_reps == 0] = 1e-10
        sim_matrix = np.dot(query_reps, reps.T) / (norm_query * norm_reps)
        # plot query, most similar, and least similar
        for i, idx in enumerate(query_indices):
            output = os.path.join(outdir, f"qual_eval_{i}.tif")
            indices = np.argsort(sim_matrix[i])[::-1]
            self.check_data(dataset, [idx] + [indices[0]] + [indices[-1]], self.config["bins"], 1, 3, output)
            # nrows, ncols = 1, 3 (query / most similar / least similar)


    def load_pretrained(self, model_path: str, config_path: str=None):
        """ load pretrained model """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.pretrained_model = BarlowTwins(
            self.config["hist_dim"], # the dimension of the input
            self.config["hidden_hist"], # the dimension of the hidden layer
            self.config["dropout_hist"], # the dropout rate
            self.config["num_blocks"], # the number of blocks in the ResNet
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
            self.config["hist_dim"], # the dimension of the input
            self.config["hidden_hist"], # the dimension of the hidden layer
            self.config["dropout_hist"], # the dropout rate
            self.config["num_blocks"], # the number of blocks in the ResNet
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