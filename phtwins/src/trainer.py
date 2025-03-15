# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

@author: tadahaya
"""
import os
import torch

from .utils import save_experiment, save_checkpoint

# ToDo: implement Trainer class
class PreTrainer:
    def __init__(self, config, model, optimizer, scheduler, device):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        # training
        for i in range(self.config["epochs"]):
            train_loss = self.train_epoch(trainloader)
            test_loss = self.evaluate(testloader)
            # logging
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}"
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
            # scheduler
            if self.scheduler is not None:
                self.scheduler.step(test_loss)
            # save the model
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0:
                save_checkpoint(model=self.model, name=f"epoch_{i + 1}", outdir=self.resdir)
        # save the experiment
        save_experiment(model=self.model, config=self.config, history=self.history)


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for data, label in trainloader:
            # data = (point, hist)
            point, hist = data
            point, hist, label = point.to(self.device), hist.to(self.device), label.to(self.device)
            # initialize the gradients
            self.optimizer.zero_grad()
            # forward/loss calculation
            loss = self.model(point, hist)
            # backpropagation
            loss.backward()
            # update the parameters
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(trainloader.dataset)
            

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        for data, label in testloader:
            # data = (point, hist)
            point, hist = data
            point, hist, label = point.to(self.device), hist.to(self.device), label.to(self.device)
            # forward/loss calculation
            loss = self.model(point, hist)
            # accumulate the loss
            total_loss += loss.item()
        return total_loss / len(testloader.dataset)


# ToDo: implement Trainer class
class Trainer:
    def __init__(self, config, model, optimizer, loss_fn, exp_name, device):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device


    def train(self, trainloader, testloader, classes:dict=None, save_model_evry_n_epochs=0):
        """
        train the model for the specified number of epochs.
        
        """
        # configの確認
        config = self.config
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        assert config["intermediate_size"] == 4 * config["hidden_size"]
        assert config["image_size"] % config["patch_size"] == 0
        # keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # training
        for i in range(config["epochs"]):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
            if save_model_evry_n_epochs > 0 and (i + 1) % save_model_evry_n_epochs == 0 and i + 1 != config["epochs"]:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        # save the experiment
        save_experiment(
            self.exp_name, config, self.model, train_losses, test_losses, accuracies, classes
            )


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for data, label in trainloader:
            # batchをdeviceへ
            data, label = data.to(self.device), label.to(self.device)
            # 勾配を初期化
            self.optimizer.zero_grad()
            # forward
            output = self.model(data)[0] # attentionもNoneで返るので
            # loss計算
            loss = self.loss_fn(output, label)
            # backpropagation
            loss.backward()
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item() * len(data) # loss_fnがbatch内での平均の値になっている模様
        return total_loss / len(trainloader.dataset) # 全データセットのうちのいくらかという比率になっている
    

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in testloader:
                # batchをdeviceへ
                data, label = data.to(self.device), label.to(self.device)
                # 予測
                output, _ = self.model(data)
                # lossの計算
                loss = self.loss_fn(output, label)
                total_loss += loss.item() * len(data)
                # accuracyの計算
                predictions = torch.argmax(output, dim=1)
                correct += torch.sum(predictions == label).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss