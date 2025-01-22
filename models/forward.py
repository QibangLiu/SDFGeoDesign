# %%
import os
from geoencoder import LoadGeoEncoderModel
from modules.UNets import UNet
import argparse
import trainer.torch_trainer as torch_trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from configs import models_configs, LoadData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

class ForwardModel(nn.Module):
    def __init__(
        self,
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_out=51,
        num_res_blocks=1,
        norm_groups=8,
    ):
        super().__init__()
        self.unet = UNet(
            img_shape,
            first_conv_channels,
            channel_mutipliers,
            has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
        )
        in_channels = channel_mutipliers[0] * first_conv_channels

        self.cov = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=2, padding=1)
        sz = [50, 50, 50, 50]
        self.mlp = nn.ModuleList()
        in_sz = img_shape[1]*img_shape[2]//4
        for i in range(len(sz)):
            self.mlp.append(nn.Linear(in_sz, sz[i]))
            self.mlp.append(nn.BatchNorm1d(sz[i]))
            self.mlp.append(nn.SiLU())
            in_sz = sz[i]
        self.mlp.append(nn.Linear(in_sz, num_out))
        # self.mlp.append(nn.ReLU())

    def forward(self, x):
        x = self.unet(x)
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        for layer in self.mlp:
            x = layer(x)
        return x

    def predict(self, data, geo_encoder):
        # TODO: add predict function
        pass


def ForwardModelDefinition(img_shape=(1, 128, 128),
                           channel_mutipliers=[1, 2, 4, 8],
                           has_attention=[False, False, True, True],
                           first_conv_channels=8, num_res_blocks=1):
    num_out = 51
    fwd_model = ForwardModel(
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_out=num_out,
        num_res_blocks=num_res_blocks,
        norm_groups=None,
    )
    trainable_params = sum(p.numel()
                           for p in fwd_model.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of fwd model: {trainable_params}")
    return fwd_model


# %%


def LoadForwardModel(filebase, model_params):
    model_path = os.path.join(filebase, "model.ckpt")
    fwd_model = ForwardModelDefinition(**model_params)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    fwd_model.load_state_dict(state_dict)
    fwd_model.to(device)
    fwd_model.eval()
    return fwd_model


def TrainForwardModel(fwd_model, geo_encoder, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, _, _, stress_inv_scaler = LoadData(
        seed=42)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    geo_encoder = geo_encoder.to(device)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase, geo_encoder):
            super().__init__(models, device, filebase)
            self.geo_encoder = geo_encoder
            for param in geo_encoder.parameters():
                param.requires_grad = False
            self.geo_encoder.eval()

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            y_true = data[2].to(self.device)
            latent = self.geo_encoder(pc)
            latent = latent[:, None, :, :]
            y_pred = self.models[0](latent)

            loss = nn.MSELoss()(y_true, y_pred)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data_loader):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    inputs = data[0].to(self.device)  # pc,sdf,stress
                    latent = self.geo_encoder(inputs)
                    pred = self.models[0](latent)
                    pred = pred.cpu().detach().numpy()
                    y_pred.append(pred)
                    y_true.append(data[2].cpu().detach().numpy())
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            return y_pred, y_true

    trainer = TRAINER(
        fwd_model, device, filebase, geo_encoder)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="val_loss", save_best_only=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=40)
    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="val_loss",
    )
    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_loader, val_loader=test_loader,
                    epochs=epochs, print_freq=1)
    trainer.save_logs()

    EvaluateForwardModel(trainer, test_loader, stress_inv_scaler)
    return trainer


def EvaluateForwardModel(trainer, test_loader, stress_inv_scal, plot_flag=False):
    trainer.load_weights(device=device)
    s_pred, s_true = trainer.predict(test_loader)
    s_pred = stress_inv_scal(s_pred)
    s_true = stress_inv_scal(s_true)
    error_s = np.linalg.norm(s_pred-s_true, axis=1) / \
        np.linalg.norm(s_true, axis=1)

    sort_idx = np.argsort(error_s)
    idx_best = sort_idx[0]
    idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
    idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
    idx_95perc = sort_idx[int(len(sort_idx)*0.95)]
    index_list = [idx_best, idx_32perc, idx_63perc, idx_95perc]
    labels = ["Best", "32th percentile", "63th percentile", "95th percentile"]
    for label, idx in zip(labels, index_list):
        print(f"{label} L2 error: {error_s[idx]}")

    print(f"Mean L2 error: {np.mean(error_s)}, std: {np.std(error_s)}")
    if plot_flag:
        nr, nc = 1, len(index_list)
        fig = plt.figure(figsize=(nc*4.8, nr*3.6))
        strain = np.linspace(0, 0.2, 51)
        for i, index in enumerate(index_list):

            ax = plt.subplot(nr, nc, i+1)
            s_pred_i = s_pred[index]
            s_true_i = s_true[index]
            ax.plot(strain*100, s_true_i, 'r', label="True")
            ax.plot(strain*100, s_pred_i, '--b', label="Pred.")
            ax.legend()
            ax.set_xlabel(r"$\varepsilon~[\%]$")
            ax.set_ylabel(r"$\sigma~[MPa]$")
            ax.set_title(f"{labels[i]}")

            plt.tight_layout()


# %%
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training arguments")
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs = models_configs()

    filebase = configs["ForwardModel"]["filebase"]
    model_params = configs["ForwardModel"]["model_params"]
    geo_encoder_filebase = configs["GeoEncoder"]["filebase"]
    geo_encoder_model_params = configs["GeoEncoder"]["model_params"]
    print(f"\n\nForwardModel Filebase: {filebase}, model_params:")
    print(model_params)
    print(f"\n\nGeoEncoder Filebase: {geo_encoder_filebase}, model_params:")
    print(geo_encoder_model_params)

    fwd_model = ForwardModelDefinition(**model_params)
    geo_encoder, _ = LoadGeoEncoderModel(
        geo_encoder_filebase, geo_encoder_model_params)
    trainer = TrainForwardModel(fwd_model, geo_encoder, filebase, args.train_flag,
                                epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")
