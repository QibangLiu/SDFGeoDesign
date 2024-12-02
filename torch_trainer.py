# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader as GeoDataLoader
import timeit
import os
import json

# %%


class ModelCheckpoint:
    def __init__(
        self, filepath, monitor="val_loss", verbose=0, save_best_only=False, mode="min"
    ):
        if not isinstance(filepath,(list, tuple)):
            filepath=[filepath]

        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best = None
        self.mode = mode

        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        if self.mode == "min":
            self.monitor_op = lambda x, y: x < y
            self.best = float("inf")
        else:
            self.monitor_op = lambda x, y: x > y
            self.best = float("-inf")

    def __call__(self, epoch, logs=None, models=None):
        if len(self.filepath) !=len(models):
            raise ValueError("Number of models is not equal to number of filepaths")

        logs = logs or {}
        current = logs.get(self.monitor)
        current = current[-1]

        if current is None:
            raise ValueError(
                f"Monitor value '{self.monitor}' not found in logs")

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        f"Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current}, saving model to {self.filepath}"
                    )
                self.best = current
                for model, filepath in zip(models, self.filepath):
                    self._save_model(model, filepath)
            else:
                if self.verbose > 1:
                    print(
                        f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best}"
                    )
        else:
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: saving model to {self.filepath}")
            for model, filepath in zip(models, self.filepath):
                self._save_model(model, filepath)

    def _save_model(self, model, filepath):
        torch.save(model.state_dict(), filepath)


class TorchTrainer():
    def __init__(self, models, device):
        if not isinstance(models, (list, tuple)):
            models = [models]
        for model in models:
            model.to(device)
        self.models = models
        self.logs = {}
        self.device = device
        self.epoch_start = 0

    def compile(
        self,
        optimizer,
        lr=1e-3,
        loss=None,
        loss_names=None,
        lr_scheduler=None,
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        combined_params = []
        for model in self.models:
            combined_params += list(model.parameters())
        self.optimizer = optimizer(combined_params, lr=lr)


        # TODO: Add multiple loss function
        self.loss_fn = loss
        # TODO:
        self.metrics = metrics
        self.decay = decay
        self.loss_weights = loss_weights
        self.external_trainable_variables = external_trainable_variables
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            self.logs["lr"] = []

    def collect_logs(self, losses_vals={}, batch_size=1):
        for key in losses_vals:
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(sum(losses_vals[key]) / batch_size)

    def print_logs(self, epoch, time):
        print(f"Epochs {epoch + 1} took {time:.2f}s", end=", ")
        for key, val in self.logs.items():
            if val:
                print(f"{key}: {val[-1]:.4e}", end=", ")
        print()

    def evaluate_losses(self, data):
        inputs_, y_true = data[0].to(self.device), data[1].to(self.device)
        y_pred = self.model[0](inputs_)
        # TODO: Add multiple loss function
        loss = self.loss_fn(y_pred, y_true)
        loss_dic = {"loss": loss.item()}
        return loss, loss_dic

    def train_step(self, data):
        self.optimizer.zero_grad()
        loss, loss_dic = self.evaluate_losses(data)
        loss.backward()
        self.optimizer.step()
        return loss_dic

    def validate_step(self, data):
        _, loss_dic = self.evaluate_losses(data)
        val_loss = {}
        for key, value in loss_dic.items():
            val_loss["val_" + key] = value
        return val_loss

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        callbacks=None,
        print_freq=20,
    ):
        ts = timeit.default_timer()
        loss_vals = {}
        for epoch in range(self.epoch_start, self.epoch_start + epochs):
            # train
            for model in self.models:
                model.train()
            loss_vals = {}
            for data in train_loader:
                # if isinstance(data, (tuple, list)):
                #     data = [d.to(self.device) for d in data]
                # else:
                #     data = data.to(self.device)
                loss = self.train_step(data)
                for key, value in loss.items():
                    if key not in loss_vals:
                        loss_vals[key] = []
                    loss_vals[key].append(value)
            self.collect_logs(loss_vals, len(train_loader))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.logs["lr"].append(self.lr_scheduler.get_last_lr()[0])
                # validate
            if val_loader is not None:
                for model in self.models:
                    model.eval()
                loss_vals = {}
                with torch.no_grad():
                    for data in val_loader:
                        # if isinstance(data, (tuple, list)):
                        #     data = [d.to(self.device) for d in data]
                        # else:
                        #     data = data.to(self.device)
                        loss = self.validate_step(data)
                        for key, value in loss.items():
                            if key not in loss_vals:
                                loss_vals[key] = []
                            loss_vals[key].append(value)
                self.collect_logs(loss_vals, len(val_loader))
            # callbacks at end of epoch
            if callbacks is not None:
                callbacks(epoch, self.logs, self.models)

            te = timeit.default_timer()
            if (epoch + 1) % print_freq == 0:
                self.print_logs(epoch, (te - ts))
        print("Total training time:.%2f s" % (te - ts))
        self.epoch_start = epoch + 1
        return self.logs

    def save_logs(self, filebase):
        if self.logs is not None:
            if not os.path.exists(filebase):
                os.makedirs(filebase, exist_ok=True)
            his_file = os.path.join(filebase, "logs.json")
            with open(his_file, "w") as f:
                json.dump(self.logs, f)

    def load_logs(self, filebase):
        his_file = os.path.join(filebase, "logs.json")
        if os.path.exists(his_file):
            with open(his_file, "r") as f:
                self.logs = json.load(f)
        return self.logs

    def save_weights(self, filepath):
        if not isinstance(filepath, (list, tuple)):
            filepath = [filepath]
        for model, path in zip(self.models, filepath):
            torch.save(model.state_dict(), path)

    def load_weights(self, filepath, device="cpu"):
        if not isinstance(filepath, (list, tuple)):
            filepath = [filepath]
        for model, path in zip(self.models, filepath):
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()


# %%
