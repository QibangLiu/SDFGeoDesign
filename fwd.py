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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

data = np.load(
    '/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_1-120.npz')
sdf = data['sdf'].astype(np.float32)
stress = data['stress'].astype(np.float32)
strain = data['strain'].astype(np.float32)

sdf = sdf.reshape(-1, 120*120)
# sdf_shift, sdf_scale = sdf.mean, sdf.std()
# sdf_norm = (sdf-sdf.mean())/sdf.std()
sdf_scaler = StandardScaler()
sdf_norm = sdf_scaler.fit_transform(sdf)
sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)

# stress_shift, stress_scale = stress.mean(), stress.std()
# stress_norm = (stress-stress.mean())/stress.std()
stress_scaler = StandardScaler()
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
    test_dataset, batch_size=1024, shuffle=False)


def y_inv_trans(y):
    return stress_scaler.inverse_transform(y)
# %%


class ForwardModel(nn.Module):
    def __init__(self, img_shape, channel_list, has_attention, num_out=51, first_conv_channels=16, num_res_blocks=1, norm_groups=8):
        super().__init__()
        self.unet = nn_modules.UNet(img_shape, channel_list,
                                    has_attention, first_conv_channels=first_conv_channels, num_res_blocks=num_res_blocks, norm_groups=norm_groups)
        self.cov = nn.Conv2d(channel_list[0], 1, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(img_shape[1]*img_shape[2], 10),
            nn.SiLU(),
            nn.Linear(10, 50),
            nn.SiLU(),
            nn.Linear(50, 50),
            # nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(50, num_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.unet(x)
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


# %%
img_shape = tuple(sdf_train.shape[1:])
channel_list = [8, 16, 32, 64]
has_attention = [False, False, True, True]
num_out = stress_train.shape[1]
fwd_model = ForwardModel(img_shape, channel_list,
                         has_attention, num_out=num_out, first_conv_channels=8, num_res_blocks=1, norm_groups=8)
trainable_params = sum(p.numel()
                       for p in fwd_model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {trainable_params}")

# %%
filebase = "./saved_models/fwd_model"
checkpoint = torch_trainer.ModelCheckpoint(
    monitor="val_loss", save_best_only=True
)
trainer = torch_trainer.TorchTrainer(fwd_model, device, filebase)
trainer.compile(optimizer=torch.optim.Adam, lr=1e-4,
                loss=nn.MSELoss(), checkpoint=checkpoint)
h = trainer.fit(train_loader, val_loader=test_loader,
                epochs=500, callbacks=checkpoint, print_freq=1)
trainer.save_logs(filebase)

# %%
trainer.load_weights(device=device)
h = trainer.load_logs()
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")
# %%


# def predict(self, data_loader):
#     y_pred = []
#     y_true = []
#     with torch.no_grad():
#         for data in data_loader:
#             inputs = data[0].to(device)
#             pred = fwd_model(inputs)
#             pred = pred.cpu().detach().numpy()
#             y_pred.append(pred)
#             y_true.append(data[1].cpu().detach().numpy())
#     y_true = np.vstack(y_true)
#     y_pred = np.vstack(y_pred)
#     return y_pred, y_true


s_pred, s_true = trainer.predict(test_loader)
s_pred = y_inv_trans(s_pred)
s_true = y_inv_trans(s_true)
error_s = np.linalg.norm(s_pred-s_true, axis=1) / \
    np.linalg.norm(s_true, axis=1)
fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(error_s, bins=20)

# %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[int(len(sort_idx)*0.97)]
median_index = sort_idx[int(len(sort_idx)*0.2)]
# # Print the indexes
print("Index for minimum geo:", min_index,
      "with error", error_s[min_index])
print("Index for maximum geo:", max_index,
      "with error", error_s[max_index])
print("Index for median geo:", median_index,
      "with error", error_s[median_index])
min_median_max_index = np.array([min_index, median_index, max_index])
nr, nc = 1, 3
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
for i, index in enumerate(min_median_max_index):

    ax = plt.subplot(nr, nc, i+1)
    s_pred_i = s_pred[index]
    s_true_i = s_true[index]
    ax.plot(strain, s_true_i, label="true")
    ax.plot(strain, s_pred_i, label="pred")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()

# %%
