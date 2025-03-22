# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json, os, time
import matplotlib.pyplot as plt
import torch


def save_experiment(config, model, history, plot_progress=True):
    """
    save the experiment: config, model, metrics, and progress plot
    
    outdir
    ├── experiment_name
        ├── config.json
        ├── history.json
        ├── progress_loss.tif
        ├── model_final.pt
        ├── model_1.pt
        ├── model_2.pt
        ├── ...
    
    """
    outdir = config["outdir"]
    experiment_name = config["exp_name"]
    resdir = os.path.join(outdir, experiment_name)
    os.makedirs(resdir, exist_ok=True)
    # save config
    configfile = os.path.join(resdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    # save history
    historyfile = os.path.join(resdir, 'history.json')
    with open(historyfile, 'w') as f:
        json.dump(history, f, sort_keys=True, indent=4)
    # save the model
    save_checkpoint(model=model, name="final", outdir=outdir)
    # plot progress
    if plot_progress:
        progress_plot(
            outdir=resdir,
            train_values=history["train_loss"],
            test_values=history["test_loss"]
        )


def save_checkpoint(model, name, outdir):
    """
    save the model checkpoint
    
    """
    cpfile = os.path.join(outdir, f"model_{name}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": model.optimizer.state_dict(),
        },
        cpfile
    )


def load_experiments(model, optimizer, resdir, checkpoint_name="model_final"):
    """
    load the experiment

    Parameters
    ----------
    model: nn.Module
        initialized model

    optimizer: torch.optim
        initialized optimizer

    resdir: str
        the result directory
    
    checkpoint_name: str
        the checkpoint name, like model_final
    
    """
    # load config
    configfile = os.path.join(resdir, "config.json")
    with open(configfile, 'r') as f:
        config = json.load(f)
    # load history
    historyfile = os.path.join(resdir, 'hisotry.json')
    with open(historyfile, 'r') as f:
        history = json.load(f)
    # load model
    pkg = torch.load(os.path.join(resdir, checkpoint_name))
    model.load_state_dict(pkg["model"])
    optimizer.load_state_dict(pkg["optimizer"])
    return model, optimizer, config, history


def progress_plot(
        outdir:str, train_values:list, test_values:list=[],
        xlabel="epoch", ylabel="loss"
        ):
    """ plot learning progress """
    fileout = os.path.join(outdir, f"progress_{ylabel}.tif")
    x = list(range(1, len(train_values) + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14
    ax.plot(x, train_values, c='navy', label='train')
    if len(test_values) > 0:
        ax.plot(x, test_values, c='darkgoldenrod', label='test')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(fileout, dpi=300, bbox_inches='tight')


def calc_elapsed_time(start_time):
    """ calculate elapsed time """
    elapsed_time = time.time() - start_time
    h = int(elapsed_time // 3600)
    m = int((elapsed_time % 3600) // 60)
    s = elapsed_time % 60
    return f"{h}h {m}m {s}s"