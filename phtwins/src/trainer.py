# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

@author: tadahaya
"""
import os, time
import torch

from .utils import save_experiment, save_checkpoint, calc_elapsed_time

class PreTrainer:
    def __init__(self, config, model, optimizer, device):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        # config contents
        self.exp_name = config["exp_name"]
        self.outdir = config["outdir"]
        self.save_model_every = config["save_model_every"]
        # I/O
        self.resdir = os.path.join(self.outdir, self.exp_name)
        os.makedirs(self.resdir, exist_ok=True)
        # early stopping
        self.patience = config["patience"]
        self.best_loss = float("inf")
        self.early_stop_count = 0
        self.best_model_path = os.path.join(self.resdir, "model_best.pth")
        # loggings
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "early_stop_epoch": None,
            "elapsed_time": None
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        start_time = time.time()
        # training
        for i in range(self.config["epochs"]):
            train_loss = self.train_epoch(trainloader)
            test_loss = self.evaluate(testloader)
            # logging
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            print(
                f"Epoch: {i + 1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}"
                )
            # early stopping
            if self.patience is not None:
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    self.early_stop_count = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                else:
                    self.early_stop_count += 1
                if self.early_stop_count >= self.patience:
                    print("> Early stopping")
                    self.history["early_stop_epoch"] = i + 1
                    break
            # save the model
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0:
                save_checkpoint(model=self.model, name=f"epoch_{i + 1}", outdir=self.resdir)
        # save the experiment
        elapsed_time = calc_elapsed_time(start_time)
        self.history["elapsed_time"] = elapsed_time
        save_experiment(config=self.config, model=self.model, history=self.history)


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        self.optimizer.train()
        total_loss = 0.0
        total_samples = 0 # for averaging the loss
        for data, label in trainloader:
            # data = (point, hist)
            point, hist = (x.to(self.device) for x in data)
            label = label.to(self.device)
            # initialize the gradients
            self.optimizer.zero_grad()
            # forward/loss calculation
            _, loss = self.model(point, hist) # output, bt_loss
            # note: loss is averaged over the batch
            # backpropagation
            loss.backward()
            # update the parameters
            self.optimizer.step()
            # Loss accumulation
            batch_size = point.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
        return total_loss / total_samples


    def evaluate(self, testloader):
        self.model.eval()
        self.optimizer.eval()
        total_loss = 0.0
        total_samples = 0 # for averaging the loss
        with torch.no_grad():
            for data, label in testloader:
                # data = (point, hist)
                point, hist = (x.to(self.device) for x in data)
                label = label.to(self.device)
                # forward/loss calculation
                _, loss = self.model(point, hist) # output, bt_loss
                # Loss accumulation
                batch_size = point.shape[0]
                total_loss += loss.item() * batch_size # detach() is not necessary
                total_samples += batch_size
        return total_loss / total_samples


# ToDo: implement Trainer class
class Trainer:
    def __init__(self, model, config, loss_fn, optimizer, device):
        self.config = config
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        # config contents
        self.exp_name = config["exp_name"]
        self.outdir = config["outdir"]
        self.save_model_every = config["save_model_every"]
        if config["frozen"]:
            self.use_pretrain_loss = False # if the model is frozen, pretrain loss is never used
        else:
            self.use_pretrain_loss = config["use_pretrain_loss"]
        # I/O
        self.resdir = os.path.join(self.outdir, self.exp_name)
        os.makedirs(self.resdir, exist_ok=True)
        # early stopping
        self.patience = config["patience"]
        self.best_loss = float("inf")
        self.early_stop_count = 0
        self.best_model_path = os.path.join(self.resdir, "model_best.pth")
        # loggings
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "early_stop_epoch": None,
            "elapsed_time": None
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        start_time = time.time()
        # training
        for i in range(self.config["epochs"]):
            train_loss, train_acc = self.train_epoch(trainloader)
            test_loss, test_acc = self.evaluate(testloader)
            # logging
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_acc)
            print(f"Epoch: {i + 1}")
            print(f"  Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
            print(f"  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
            # early stopping
            if self.patience is not None:
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    self.early_stop_count = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                else:
                    self.early_stop_count += 1
                if self.early_stop_count >= self.patience:
                    print("> Early stopping")
                    self.history["early_stop_epoch"] = i + 1
                    break
            # save the model
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0:
                save_checkpoint(model=self.model, optimizer=self.optimizer, name=f"epoch_{i + 1}", outdir=self.resdir)
        # save the experiment
        elapsed_time = calc_elapsed_time(start_time)
        self.history["elapsed_time"] = elapsed_time
        save_experiment(config=self.config, model=self.model, optimizer=self.optimizer, history=self.history)


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        self.optimizer.train()
        total_loss = 0.0
        total_samples = 0 # for averaging the loss
        correct = 0
        for data, label in trainloader:
            # data = (point, hist)
            point, hist = (x.to(self.device) for x in data)
            label = label.to(self.device)
            # initialize the gradients
            self.optimizer.zero_grad()
            # forward calculation
            output, pt_loss = self.model(point, hist)
            ft_loss = self.loss_fn(output, label)
            loss = pt_loss + ft_loss if self.use_pretrain_loss else ft_loss
            # backpropagation
            loss.backward()
            # update the parameters
            self.optimizer.step()
            # Loss accumulation
            batch_size = point.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            # Accuracy calculation (disable gradients for efficiency)
            with torch.no_grad():
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == label).sum().item()
        return total_loss / total_samples, correct / total_samples
            

    def evaluate(self, testloader):
        """Evaluate the model on the test set"""
        self.model.eval()
        self.optimizer.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for data, label in testloader:
                # Move data to device
                point, hist = (x.to(self.device) for x in data)
                label = label.to(self.device)
                # Forward pass
                output, pt_loss = self.model(point, hist)
                ft_loss = self.model.loss_fn(output, label)
                loss = pt_loss + ft_loss if self.use_pretrain_loss else ft_loss
                # Loss accumulation
                batch_size = point.shape[0]
                total_loss += loss.item() * batch_size # detach() is not necessary
                total_samples += batch_size
                # Accuracy calculation
                predictions = torch.argmax(output, dim=1)
                correct += int((predictions == label).sum())
        return total_loss / total_samples, correct / total_samples