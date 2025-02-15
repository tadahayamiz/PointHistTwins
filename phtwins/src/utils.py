# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from .models import VitForClassification

def save_experiment(
        experiment_name, config, model, train_losses, test_losses,
        accuracies, classes, base_dir=""
        ):
    """ save the experiment: config, model, metrics, and progress plot """
    if len(base_dir) == 0:
        base_dir = os.path.dirname(config["config_path"])
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # save config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # save metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'classes': classes,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # plot progress
    plot_progress(
        experiment_name, train_losses, test_losses, config["epochs"], base_dir=base_dir
        )

    # save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


def load_experiments(
        experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"
        ):
    outdir = os.path.join(base_dir, experiment_name)
    # load config
    configfile = os.path.join(outdir, "config.json")
    with open(configfile, 'r') as f:
        config = json.load(f)
    # load metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data["train_losses"]
    test_losses = data["test_losses"]
    accuracies = data["accuracies"]
    classes = data["classes"]
    # load model
    model = VitForClassification(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile)) # checkpointを読み込んでから
    return config, model, train_losses, test_losses, accuracies, classes


def plot_progress(
        experiment_name:str, train_loss:list, test_loss:list, num_epoch:int,
        base_dir:str="experiments", xlabel="epoch", ylabel="loss"
        ):
    """ plot learning progress """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    epochs = list(range(1, num_epoch + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14
    ax.plot(epochs, train_loss, c='navy', label='train')
    ax.plot(epochs, test_loss, c='darkgoldenrod', label='test')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + f'/progress_{ylabel}.tif', dpi=300, bbox_inches='tight')