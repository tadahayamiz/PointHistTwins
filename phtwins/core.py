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
import inspect
from datetime import datetime

from .src.barlow import BarlowTwins, LinearHead
from .src.trainer import PreTrainer, Trainer
from .src.data_handler import PointHistDataset, prep_dataloader, plot_hist
from .src.utils import fix_seed

class PHTwins:
    """ class for training and prediction """
    def __init__(
            self, config_path: str, df: pd.DataFrame=None, test_df:pd.DataFrame=None,
            outdir: str=None, exp_name: str=None, seed: int=42
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
        self.pretrainer = None
        self.trainer = None
        # fix seed
        g, seed_worker = fix_seed(seed, fix_cuda=True)
        self._seed = {"seed": seed, "g": g, "seed_worker": seed_worker}
        # prepare model
        self._prep_model()


    def _prep_model(self):
        """
        prepare model
        hard coded parameters
        
        """
        # prepare pretraining model
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
        optimizer0 = RAdamScheduleFree(self.pretrained_model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
        self.pretrainer = PreTrainer(
            self.config, self.pretrained_model, optimizer0, device=self.config["device"]
            )
        # prepare linear head
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
        optimizer1 = RAdamScheduleFree(self.finetuned_model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
        self.trainer = Trainer(
            self.config, self.finetuned_model, loss_fn, optimizer1, self.config["device"]
            )


    def prep_data(self, key_identify:str, key_data:list, key_label:str):
        """ prepare data """
        self.train_dataset = PointHistDataset(
            self.df, key_identify, key_data, key_label, self.config["num_points"], self.config["bins"]
        )
        train_loader = prep_dataloader(
            self.train_dataset, self.config["batch_size"], True, self.config["num_workers"],
            self.config["pin_memory"], self._seed["g"], self._seed["seed_worker"]
            )
        if self.test_df is None:
            return train_loader, None
        else:
            self.test_dataset = PointHistDataset(
                self.test_df, key_identify, key_data, key_label, self.config["num_points"], self.config["bins"]
            )
            test_loader = prep_dataloader(
                self.test_dataset, self.config["batch_size"], False, self.config["num_workers"],
                self.config["pin_memory"], self._seed["g"], self._seed["seed_worker"]
                )
            return train_loader, test_loader


    def pretrain(self, train_loader, test_loader):
        """ pretraining """
        self.pretrainer.train(train_loader, test_loader)
        print("> Pretraining is done.")


    def finetune(self, train_loader, test_loader):
        """ finetuning """
        self.trainer.train(train_loader, test_loader)
        print("> Finetuning is done.")


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


    def check_data(self, dataset, indices:list=[], output:str="", **plot_params):
        """
        check data
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            the PHTwins dataset
        indices: list
            the list of indices to be checked
        output: str
            the output path
        plot_params: dict
            the parameters for the plot
            default_params = {
                "nrow": 1,
                "ncol": 3,
                "xlabel": "x",
                "ylabel": "y",
                "title_list": None,
                "cmap": "viridis",
                "aspect": "equal",
                "color": "royalblue",
                "alpha": 0.7
            }
        
        """
        hist_list = [dataset[i][0][0].numpy()[0] for i in indices] # ((hist, hist), label)
        plot_hist(hist_list, output, **plot_params)


    def qual_eval(self, dataset, query_indices, outdir:str=""):
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
            plot_indices = [idx] + [indices[0]] + [indices[-1]]
            plot_params = {
                "title_list": ["query", "most similar", "least similar"],
                "nrow": 1,
                "ncol": 3,
                }
            self.check_data(dataset, plot_indices, output, **plot_params)
            # nrows, ncols = 1, 3 (query / most similar / least similar)
        return sim_matrix


    def load_pretrained(self, model_path: str, config_path: str=None):
        """ load pretrained model """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        # initialize BarlowTwins model
        model_params = inspect.signature(BarlowTwins.__init__).parameters
        model_args = {k: self.config[k] for k in model_params if k in self.config}
        model_args["input_dim"] = self.config["hist_dim"] # hard coded
        self.pretrained_model = BarlowTwins(**model_args)
        self.pretrained_model.load_state_dict(torch.load(model_path))


    def load_finetuned(self, model_path: str, config_path: str=None):
        """ load model with linear head """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        # initialize BarlowTwins model
        bt_params = inspect.signature(BarlowTwins.__init__).parameters
        bt_args = {k: self.config[k] for k in bt_params if k in self.config}
        bt_args["input_dim"] = self.config["hist_dim"] # hard coded
        init_bt_model = BarlowTwins(**bt_args)
        # initialize LinearHead model
        lh_params = inspect.signature(LinearHead.__init__).parameters
        lh_args = {k: self.config[k] for k in lh_params if k in self.config}
        lh_args["pretrained"] = init_bt_model # hard coded
        self.finetuned_model = LinearHead(**lh_args)
        self.finetuned_model.load_state_dict(torch.load(model_path))


    def register_hook(self, event_name, hook_fn, mode="pretraining"):
        """
        register hook for the model

        Parameters
        ----------
        event_name: str
            the name of the event to register the hook

        hook_fn: function
            the function to be called when the event occurs

        trainer: str
            the trainer to register the hook
            "pretraining" or "finetuning"

        """
        if mode == "pretraining":
            if self.pretrainer is None:
                raise ValueError("!! Pretrainer is not initialized !!")
            self.pretrainer.register_hook(event_name, hook_fn)
        elif mode == "trainer":
            if self.trainer is None:
                raise ValueError("!! Trainer is not initialized !!")
            self.trainer.register_hook(event_name, hook_fn)
        else:
            raise ValueError("!! trainer should be 'pretraining' or 'finetuning' !!")
        print(f">> registered hook: {event_name} to {mode}")