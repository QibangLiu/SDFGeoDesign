# %%
from skimage import measure
import torch_trainer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from abc import abstractmethod
import torch.nn.functional as nnF
from tqdm import tqdm
import numpy as np
import time
from torch.utils.data import Dataset, TensorDataset, DataLoader
import nn_modules

# %%
train_flag = "train"
filebase = "./saved_models/inv_sig12-92_aug10000"
data_file = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift4_0-10000_aug.npz"
fwd_model_path = "/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/saved_models/fwd_sig12-92_aug10000/model.ckpt"
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load(data_file)
sdf = data["sdf"].astype(np.float32)
stress = data["stress"].astype(np.float32)
strain = data["strain"].astype(np.float32)

sdf = sdf.reshape(-1, 120 * 120)
sdf_scaler = MinMaxScaler((-1, 1))
sdf_norm = sdf_scaler.fit_transform(sdf)
sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)

stress_scaler = MinMaxScaler((-1, 1))
stress_norm = stress_scaler.fit_transform(stress)

sdf_train, sdf_test, stress_train, stress_test = train_test_split(
    sdf_norm, stress_norm, test_size=0.2, random_state=42
)


sdf_train = torch.tensor(sdf_train)
stress_train = torch.tensor(stress_train)
sdf_test = torch.tensor(sdf_test)
stress_test = torch.tensor(stress_test)

train_dataset = TensorDataset(sdf_train, stress_train)
test_dataset = TensorDataset(sdf_test, stress_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


def y_inv_trans(y):
    return stress_scaler.inverse_transform(y)


def x_inv_trans(x):
    x = x.reshape(-1, 120 * 120)
    return sdf_scaler.inverse_transform(x).reshape(-1, 120, 120)


# %%
# load forward model


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
        in_sz = img_shape[1] * img_shape[2] // 4
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

state_dict = torch.load(fwd_model_path, map_location=device)
fwd_model.load_state_dict(state_dict)
fwd_model.to(device)
fwd_model.eval()
# %%
img_shape = tuple(sdf_train.shape[1:])
channel_multpliers = [1, 2, 4, 8]
has_attention = [False, False, False, False]
num_heads = 4
num_res_blocks = 1
norm_groups = None
fist_conv_channels = 16
label_dim = stress_train.shape[1]

inv_model = nn_modules.UNetTimeStep(
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
trainable_params = sum(p.numel() for p in inv_model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {trainable_params}")

total_timesteps = 500
gaussian_diffusion = nn_modules.GaussianDiffusion(timesteps=total_timesteps)
# %%


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
        t = torch.randint(0, total_timesteps, (batch_size,), device=device).long()

        loss = gaussian_diffusion.train_losses(
            self.models[0], images, t, labels, batch_mask
        )
        loss_dic = {"loss": loss.item()}
        return loss, loss_dic


checkpoint = torch_trainer.ModelCheckpoint(monitor="loss", save_best_only=True)

trainer = TorchTrainer(inv_model, device, filebase)
lr_scheduler = {
    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "params": {"factor": 0.7, "patience": 40},
    "metric_name": "loss",
}
trainer.compile(
    optimizer=torch.optim.Adam,
    lr=5e-4,
    lr_scheduler=lr_scheduler,
    checkpoint=checkpoint,
)

# %%

if train_flag == "evaluate" or train_flag == "continue":
    trainer.load_weights(device=device)
    h = trainer.load_logs()

if train_flag == "train" or train_flag == "continue":
    h = trainer.fit(
        train_loader, val_loader=None, epochs=150, callbacks=checkpoint, print_freq=1
    )
    trainer.save_logs(filebase)

trainer.load_weights(device=device)
h = trainer.load_logs()
if h is not None:
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(h["loss"], label="loss")
    ax.legend()
    ax.set_yscale("log")


# %%
# design
id = 0
num_sol = 10
Xtarg = test_dataset[id][0].unsqueeze(0).to(device)
Ytarg = test_dataset[id][1].unsqueeze(0).to(device)

labels = Ytarg.repeat(num_sol, 1)
Xpred = gaussian_diffusion.sample(
    inv_model, img_shape, labels, w=2, clip_denoised=False, conditioning=True
)
Xpred = torch.tensor(Xpred[-1]).to(device)
with torch.no_grad():
    Ypred = fwd_model(Xpred)

Ypred_inv = y_inv_trans(Ypred.cpu().detach().numpy())
Ytarg_inv = y_inv_trans(labels.cpu().detach().numpy())
Xpred_inv = x_inv_trans(Xpred.cpu().detach().numpy())
L2error = np.linalg.norm(Ypred_inv - Ytarg_inv) / np.linalg.norm(Ytarg_inv)
sorted_idx = np.argsort(L2error)

# %%
evl_ids = [
    sorted_idx[0],
    sorted_idx[int(len(sorted_idx) * 0.33)],
    sorted_idx[int(len(sorted_idx) * 0.66)],
    sorted_idx[-1],
]
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
    contours = measure.find_contours(Xpred_inv[v], 0, positive_orientation="high")
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
            ax.fill(x, y, alpha=1.0, edgecolor="black", facecolor="white", label="Hole")
        # ax.grid(True)
        ax.axis("off")
        ax.axis("equal")  # Keep aspect ratio square
# %%
