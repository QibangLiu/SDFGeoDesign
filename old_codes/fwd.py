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
from sklearn.preprocessing import minmax_scale
import torch_trainer
from skimage import measure
import nn_modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely.geometry import Polygon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filebase = "./saved_models/fwd_sig12-92_aug10000"
print("case: ", filebase)
train_flag = "evaluate"  # "evaluate" "continue" "train"
# %%
data_file = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift4_0-10000_aug.npz"
# data_file = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92.npz"
# data_file = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift3_0-28781_aug.npz"
data = np.load(data_file)
sdf = data['sdf'].astype(np.float32)
stress = data['stress'].astype(np.float32)
strain = data['strain'].astype(np.float32)

sdf = sdf.reshape(-1, 120*120)
# sdf_shift, sdf_scale = sdf.mean, sdf.std()
# sdf_norm = (sdf-sdf.mean())/sdf.std()
sdf_scaler = MinMaxScaler((-1, 1))
sdf_norm = sdf_scaler.fit_transform(sdf)
sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)

# stress_shift, stress_scale = stress.mean(), stress.std()
# stress_norm = (stress-stress.mean())/stress.std()
stress_scaler = MinMaxScaler()
stress_norm = stress_scaler.fit_transform(stress)

sdf_train, sdf_test, stress_train, stress_test = train_test_split(
    sdf_norm, stress_norm, test_size=0.2, random_state=42)


sdf_train = torch.tensor(sdf_train)
stress_train = torch.tensor(stress_train)
sdf_test = torch.tensor(sdf_test)
stress_test = torch.tensor(stress_test)

train_dataset = TensorDataset(sdf_train, stress_train)
test_dataset = TensorDataset(sdf_test, stress_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=512, shuffle=False)


def y_inv_trans(y):
    return stress_scaler.inverse_transform(y)


def x_inv_trans(x):
    x = x.reshape(-1, 120*120)
    return sdf_scaler.inverse_transform(x).reshape(-1, 120, 120)
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

        self.cov = nn.Conv2d(in_channels, 1, kernel_size=3, stride=2, padding=1)
        sz = [50, 50, 50, 50]
        self.mlp = nn.ModuleList()
        in_sz = img_shape[1]*img_shape[2]//4
        for i in range(len(sz)):
            self.mlp.append(nn.Linear(in_sz, sz[i]))
            self.mlp.append(nn.BatchNorm1d(sz[i]))
            self.mlp.append(nn.SiLU())
            in_sz = sz[i]
        self.mlp.append(nn.Linear(in_sz, num_out))
        self.mlp.append(nn.ReLU())

    def forward(self, x):
        x = self.unet(x)
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        for layer in self.mlp:
            x = layer(x)
        return x


# %%
img_shape = tuple(sdf_train.shape[1:])
channel_mutipliers = [1, 2, 4, 8]
has_attention = [False, False, True, True]
num_out = stress_train.shape[1]
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

# %%

checkpoint = torch_trainer.ModelCheckpoint(
    monitor="val_loss", save_best_only=True
)
trainer = torch_trainer.TorchTrainer(fwd_model, device, filebase)
lr_scheduler = {
    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "params": {"factor": 0.7, "patience": 40},
    "metric_name": "val_loss",
}
trainer.compile(optimizer=torch.optim.Adam, lr=5e-4, lr_scheduler=lr_scheduler,
                loss=nn.MSELoss(), checkpoint=checkpoint)
# %%
if train_flag == "evaluate" or train_flag == "continue":
    trainer.load_weights(device=device)
    h = trainer.load_logs()
if train_flag == "train" or train_flag == "continue":
    h = trainer.fit(train_loader, val_loader=test_loader,
                    epochs=300, callbacks=checkpoint, print_freq=1)
    trainer.save_logs(filebase)

# %%
trainer.load_weights(device=device)
h = trainer.load_logs()
if h is not None:
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(h["loss"], label="loss")
    ax.plot(h["val_loss"], label="val_loss")
    ax.legend()
    ax.set_yscale("log")
# %%

s_pred, s_true = trainer.predict(test_loader)
s_pred = y_inv_trans(s_pred)
s_true = y_inv_trans(s_true)
error_s = np.linalg.norm(s_pred-s_true, axis=1) / \
    np.linalg.norm(s_true, axis=1)

fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(error_s, bins=20)
ax.set_xlabel("L2 relative error")
ax.set_ylabel("Frequency")
# %%
sort_idx = np.argsort(error_s)
idx_best = sort_idx[0]
idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
idx_95perc = sort_idx[int(len(sort_idx)*0.95)]

index_list = [idx_best, idx_32perc, idx_63perc, idx_95perc]
labels = ["Best", "32th percentile", "63th percentile", "95th percentile"]
for label, idx in zip(labels, index_list):
    print(f"{label} L2 error: {error_s[idx]}")
nr, nc = 1, len(index_list)
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
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
sdf_test_inv = x_inv_trans(sdf_test.numpy())

nr, nc = 1, len(index_list)
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
for i, idx in enumerate(index_list):
    contours = measure.find_contours(
        sdf_test_inv[idx], 0, positive_orientation='high')
    ax = plt.subplot(nr, nc, i+1)
    l_style = ['r-', 'b--']
    holes = []
    for j, contour in enumerate(contours):
        contour = (contour-10)/100
        x, y = contour[:, 1], contour[:, 0]
        if j == 0:
            ax.fill(x, y, alpha=1.0, edgecolor="black",
                    facecolor="cyan", label="Outer Boundary")
        else:
            ax.fill(x, y, alpha=1.0, edgecolor="black",
                    facecolor="white", label="Hole")
        # ax.grid(True)
        ax.axis("off")
        ax.axis("equal")  # Keep aspect ratio square


# %%
