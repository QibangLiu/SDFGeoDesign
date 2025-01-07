# %%
import torch
import timeit
import os
import json
import numpy as np

# %%


class ModelCheckpoint:
    def __init__(
        self, filepaths=None, monitor="val_loss", verbose=0, save_best_only=False, mode="min"
    ):
        # if not isinstance(filepath,(list, tuple)):
        #     filepath=[filepath]

        # self.filepaths = filepath
        self.filepaths = filepaths
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
        if len(self.filepaths) != len(models):
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
                        f"Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current}, saving model to {self.filepaths}"
                    )
                self.best = current
                for model, filepath in zip(models, self.filepaths):
                    self._save_model(model, filepath)
            else:
                if self.verbose > 1:
                    print(
                        f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best}"
                    )
        else:
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: saving model to {self.filepaths}")
            for model, filepath in zip(models, self.filepaths):
                self._save_model(model, filepath)

    def _save_model(self, model, filepath):
        torch.save(model.state_dict(), filepath)


class TorchTrainer():
    def __init__(self, models, device, filebase='./saved_models'):
        if isinstance(models, dict):
            self.model_names = list(models.keys())
            models = list(models.values())
        elif isinstance(models, (list, tuple)):
            self.model_names = [f"model{i}" for i in range(len(models))]
            models = models
        else:
            self.model_names = [""]
            models = [models]

        for model in models:
            model.to(device)
        self.models = models
        self.logs = {}
        self.device = device
        self.epoch_start = 0
        self.filebase = filebase
        for m_name in self.model_names:
            os.makedirs(os.path.join(filebase, m_name), exist_ok=True)

    def compile(
        self,
        optimizer,
        lr=1e-3,
        weight_decay=0,
        loss=None,
        checkpoint=None,
        lr_scheduler=None,
        metrics=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        combined_params = []
        for model in self.models:
            combined_params += list(model.parameters())
        self.optimizer = optimizer(
            combined_params, lr=lr, weight_decay=weight_decay)

        self.checkpoint = checkpoint
        if checkpoint is not None and self.checkpoint.filepaths is None:
            self.checkpoint.filepaths = [os.path.join(
                self.filebase, m_name, 'model.ckpt') for m_name in self.model_names]


        # TODO: Add multiple loss function
        self.loss_fn = loss
        # TODO:
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.external_trainable_variables = external_trainable_variables
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler["scheduler"](
                self.optimizer, **lr_scheduler["params"])
            if "lr" not in self.logs:
                self.logs["lr"] = []
            if "metric_name" in lr_scheduler:
                self.metric_name = lr_scheduler["metric_name"]
            else:
                self.metric_name = "val_loss"
        else:
            self.lr_scheduler = None

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
        y_pred = self.models[0](inputs_)
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

    def learning_rate_decay(self, epoch):
        if self.lr_scheduler is None:
            return

        if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
            if self.metric_name in self.logs:
                self.lr_scheduler.step(self.logs[self.metric_name][-1])
        else:
            self.lr_scheduler.step()

        self.logs["lr"].append(self.lr_scheduler.get_last_lr()[0])

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
                loss = self.train_step(data)
                for key, value in loss.items():
                    if key not in loss_vals:
                        loss_vals[key] = []
                    loss_vals[key].append(value)
            self.collect_logs(loss_vals, len(train_loader))
                # validate
            if val_loader is not None:
                for model in self.models:
                    model.eval()
                loss_vals = {}
                with torch.no_grad():
                    for data in val_loader:
                        loss = self.validate_step(data)
                        for key, value in loss.items():
                            if key not in loss_vals:
                                loss_vals[key] = []
                            loss_vals[key].append(value)
                self.collect_logs(loss_vals, len(val_loader))
            # callbacks at end of epoch
            if callbacks is not None:
                callbacks(epoch, self.logs, self.models)
            # learning rate decay
            self.learning_rate_decay(epoch)

            te = timeit.default_timer()
            if (epoch + 1) % print_freq == 0:
                self.print_logs(epoch, (te - ts))
        print("Total training time:.%2f s" % (te - ts))
        self.epoch_start = epoch + 1
        return self.logs

    def save_logs(self, filebase=None):
        if filebase is None:
            filebase = self.filebase
        if self.logs is not None:
            if not os.path.exists(filebase):
                os.makedirs(filebase, exist_ok=True)
            his_file = os.path.join(filebase, "logs.json")
            with open(his_file, "w") as f:
                json.dump(self.logs, f)

    def load_logs(self, filebase=None):
        if filebase is None:
            filebase = self.filebase
        his_file = os.path.join(filebase, "logs.json")
        if os.path.exists(his_file):
            with open(his_file, "r") as f:
                self.logs = json.load(f)
        return self.logs

    def save_weights(self, filepaths=None):
        if filepaths is None:
            if self.checkpoint is None or self.checkpoint.filepaths is None:
                raise ValueError("No filepaths provided")
            else:
                filepaths = self.checkpoint.filepaths
        elif not isinstance(filepaths, (list, tuple)):
            filepaths = [filepaths]
        for model, path in zip(self.models, filepaths):
            torch.save(model.state_dict(), path)

    def load_weights(self, filepaths=None, device="cpu"):
        if filepaths is None:
            if self.checkpoint is None or self.checkpoint.filepaths is None:
                raise ValueError("No filepaths provided")
            else:
                filepaths = self.checkpoint.filepaths
        elif not isinstance(filepaths, (list, tuple)):
            filepaths = [filepaths]

        for model, path in zip(self.models, filepaths):
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

    def predict(self, data_loader):
        y_pred = []
        y_true = []
        self.models[0].eval()
        with torch.no_grad():
            for data in data_loader:
                inputs = data[0].to(self.device)
                pred = self.models[0](inputs)
                pred = pred.cpu().detach().numpy()
                y_pred.append(pred)
                y_true.append(data[1].cpu().detach().numpy())
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        return y_pred, y_true

# %%
