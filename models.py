# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import timeit
import os
import pickle
from sklearn.model_selection import train_test_split
import torch_trainer
from skimage import measure
import nn_modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely.geometry import Polygon
import argparse

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
        self.unet = nn_modules.UNet(
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


def ForwardModelDefinition():
    img_shape = (1, 120, 120)
    channel_mutipliers = [1, 2, 4, 8]
    has_attention = [False, False, True, True]
    num_out = 51
    first_conv_channels = 8

    fwd_model = ForwardModel(
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_out=num_out,
        num_res_blocks=1,
        norm_groups=None,
    )
    trainable_params = sum(p.numel()
                           for p in fwd_model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")
    return fwd_model

# %%


data_file = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift4_0-10000_aug.npz"


def LoadData(data_file=data_file, test_size=0.2, seed=42):
    data = np.load(data_file)
    sdf = data["sdf"].astype(np.float32)
    stress = data["stress"].astype(np.float32)
    strain = data["strain"].astype(np.float32)
    sdf = sdf.reshape(-1, 120 * 120)

    # sdf_scaler = MinMaxScaler((-1, 1))
    # sdf_norm = sdf_scaler.fit_transform(sdf)
    sdf_shift, sdf_scale = np.mean(sdf), np.std(sdf)
    sdf_norm = (sdf - sdf_shift) / sdf_scale

    sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)

    # stress_scaler = MinMaxScaler()
    # stress_scaler = StandardScaler()
    # stress_norm = stress_scaler.fit_transform(stress)

    stress_shift, stress_scale = np.mean(stress), np.std(stress)
    stress_norm = (stress - stress_shift) / stress_scale

    sdf_train, sdf_test, stress_train, stress_test = train_test_split(
        sdf_norm, stress_norm, test_size=test_size, random_state=seed
    )
    sdf_train = torch.tensor(sdf_train)
    stress_train = torch.tensor(stress_train)
    sdf_test = torch.tensor(sdf_test)
    stress_test = torch.tensor(stress_test)

    def y_inv_trans(y):
       # return stress_scaler.inverse_transform(y)
       return y * stress_scale + stress_shift

    def x_inv_trans(x):
        # x = x.reshape(-1, 120 * 120)
        # return sdf_scaler.inverse_transform(x).reshape(-1, 120, 120)
        return x * sdf_scale + sdf_shift

    sdf_inv_scaler = x_inv_trans
    stress_inv_scaler = y_inv_trans

    return sdf_train, stress_train, sdf_test, stress_test, sdf_inv_scaler, stress_inv_scaler, strain


# %%
fwd_filebase = "/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/saved_models/fwd_sig12-92_aug10000_SressSignleGaussianNorm_SDFSignleGaussianNorm"
fwd_model_path = f"{fwd_filebase}/model.ckpt"


def LoadForwardModel(model_path=fwd_model_path):
    fwd_model = ForwardModelDefinition()
    state_dict = torch.load(model_path, map_location=device)
    fwd_model.load_state_dict(state_dict)
    fwd_model.to(device)
    fwd_model.eval()
    return fwd_model


def TrainForwardModel(fwd_model, filebase, train_flag, train_loader, test_loader, epochs=300, lr=5e-4):
    trainer = torch_trainer.TorchTrainer(
        fwd_model, device, filebase)
    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="val_loss", save_best_only=True
    )
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "params": {"factor": 0.7, "patience": 40},
        "metric_name": "val_loss",
    }
    trainer.compile(optimizer=torch.optim.Adam, lr=lr, lr_scheduler=lr_scheduler,
                    loss=nn.MSELoss(), checkpoint=checkpoint)
    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_loader, val_loader=test_loader,
                    epochs=epochs, callbacks=checkpoint, print_freq=1)

    trainer.load_weights(device=device)

    return trainer


def EvaluateForwardModel(trainer, test_loader, sdf_inv_scaler, stress_inv_scal, plot_flag=False):
    h = trainer.load_logs()
    if h is not None:
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(h["loss"], label="loss")
        ax.plot(h["val_loss"], label="val_loss")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")

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
"""Inverse model """
diffu_inv_filebase = "/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/saved_models/inv_sig12-92_aug10000"
diffu_model_path = f"{diffu_inv_filebase}/model.ckpt"


def DiffusionInverseModelDefinition():
    img_shape = (1, 120, 120)
    channel_multpliers = [1, 2, 4, 8]
    has_attention = [False, False, False, False]
    num_heads = 4
    num_res_blocks = 1
    norm_groups = 16
    fist_conv_channels = 32
    label_dim = 51
    inv_Unet = nn_modules.UNetTimeStep(
        img_shape=img_shape,
        label_dim=label_dim,
        one_hot=False,
        first_conv_channels=fist_conv_channels,
        channel_mutipliers=channel_multpliers,
        has_attention=has_attention,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
    )
    trainable_params = sum(p.numel()
                           for p in inv_Unet.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of inverse Unet: {trainable_params}")

    total_timesteps = 500
    gaussian_diffusion = nn_modules.GaussianDiffusion(
        timesteps=total_timesteps)

    return inv_Unet, gaussian_diffusion


def TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, filebase, train_flag, train_loader, test_loader, epochs=300, lr=5e-4):
    total_timesteps = gaussian_diffusion.timesteps

    class TorchTrainer(torch_trainer.TorchTrainer):
        def __init__(self, model, device, filebase):
            super().__init__(model, device, filebase)

        def evaluate_losses(self, data):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            batch_size = images.shape[0]
            # random generate mask
            z_uncound = torch.rand(batch_size)
            batch_mask = (z_uncound > 0.2).int().to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, total_timesteps, (batch_size,),
                              device=device).long()

            loss = gaussian_diffusion.train_losses(
                self.models[0], images, t, labels, batch_mask
            )
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="loss", save_best_only=True)

    trainer = TorchTrainer(inv_Unet, device, filebase)
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "params": {"factor": 0.7, "patience": 40},
        "metric_name": "loss",
    }
    trainer.compile(
        optimizer=torch.optim.Adam,
        lr=lr,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
    )

    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(
        train_loader, val_loader=None, epochs=epochs, callbacks=checkpoint, print_freq=1
    )
    trainer.save_logs()
    trainer.load_weights(device=device)

    return trainer


def LoadDiffusionInverseModel(model_path=diffu_model_path):
    inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition()
    state_dict = torch.load(model_path, map_location=device)
    inv_Unet.load_state_dict(state_dict)
    inv_Unet.to(device)
    inv_Unet.eval()
    return inv_Unet, gaussian_diffusion


def EvaluateDiffusionInverseModel(fwd_model, inv_Unet, gaussian_diffusion, Ytarget, sdf_inv_scaler, stress_inv_scaler, num_sol=10, plot_flag=False):
    Ytarget = Ytarget.to(device)
    labels = Ytarget.repeat(num_sol, 1)
    Xpred = gaussian_diffusion.sample(
        inv_Unet, (1, 120, 120), labels, w=2, clip_denoised=False, conditioning=True
    )
    Xpred = torch.tensor(Xpred[-1]).to(device)
    with torch.no_grad():
        Ypred = fwd_model(Xpred)
    Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
    Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
    Xpred_inv = sdf_inv_scaler(Xpred.cpu().detach().numpy())
    L2error = np.linalg.norm(Ypred_inv - Ytarg_inv, axis=1) / \
        np.linalg.norm(Ytarg_inv, axis=1)
    sorted_idx = np.argsort(L2error)
    evl_ids = np.array([
        sorted_idx[0],
        sorted_idx[int(len(sorted_idx) * 0.33)],
        sorted_idx[int(len(sorted_idx) * 0.66)],
        sorted_idx[-1],
    ], dtype=int)
    for i, idx in enumerate(evl_ids):
        print(f"ID: {idx}, L2 error: {L2error[idx]}")

    if plot_flag:
        strain = np.linspace(0, 0.2, 51)
        legends = ["best", "33\%", "66\%", "worst"]
        fig = plt.figure(figsize=(4.8 * 5, 3.6))
        ax = plt.subplot(1, 5, 1)
        ax.plot(strain, Ytarg_inv[0], label="target")
        for i, v in enumerate(evl_ids):
            ax.plot(strain, Ypred_inv[v], label=legends[i])
        ax.legend()
        ax.set_xlabel(r"$\varepsilon~[\%]$")
        ax.set_ylabel(r"$\sigma~[MPa]$")
        for i, v in enumerate(evl_ids):
            contours = measure.find_contours(
                Xpred_inv[v], 0, positive_orientation="high")
            ax = plt.subplot(1, 5, i + 2)
            l_style = ["r-", "b--"]
            holes = []
            for j, contour in enumerate(contours):
                contour = (contour - 10) / 100
                x, y = contour[:, 1], contour[:, 0]
                if j == 0:
                    ax.fill(
                        x,
                        y,
                        alpha=1.0,
                        edgecolor="black",
                        facecolor="cyan",
                        label="Outer Boundary",
                    )
                else:
                    ax.fill(x, y, alpha=1.0, edgecolor="black",
                            facecolor="white", label="Hole")
                # ax.grid(True)
                ax.axis("off")
                ax.axis("equal")  # Keep aspect ratio square


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training arguments")
    parser.add_argument(
        "--model", type=str, default="forward")
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    args, unknown = parser.parse_known_args()
    if args.model == "forward":
        seed = 42
    elif args.model == "inverse":
        seed = 52
    sdf_train, stress_train, sdf_test, stress_test, sdf_inv_scaler, stress_inv_scaler, _ = LoadData(
        seed=seed)
    train_dataset = TensorDataset(sdf_train, stress_train)
    test_dataset = TensorDataset(sdf_test, stress_test)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False)

    if args.model == "forward":
        filebase = fwd_filebase
        fwd_model = ForwardModelDefinition()
        trainer = TrainForwardModel(fwd_model, filebase, args.train_flag, train_loader,
                                    test_loader, epochs=args.epochs, lr=args.learning_rate)
        EvaluateForwardModel(trainer, test_loader,
                             sdf_inv_scaler, stress_inv_scaler)
    elif args.model == "inverse":
        filebase = diffu_inv_filebase
        inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition()
        trainer = TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, filebase, args.train_flag,
                                             train_loader, test_loader, epochs=args.epochs, lr=args.learning_rate)
        fwd_model = LoadForwardModel()
        id = 23
        Ytarget = test_dataset[id][1].unsqueeze(0)
        EvaluateDiffusionInverseModel(
            fwd_model, inv_Unet, gaussian_diffusion, Ytarget, sdf_inv_scaler, stress_inv_scaler)

    print(filebase, " training finished")
