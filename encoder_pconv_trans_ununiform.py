# %%
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from models.modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
from models.modules.point_encoding import PointSetEmbedding, SimplePerceiver
from models.modules.transformer import Transformer
from models.modules.params_proj import ChannelsParamsProj
import torch
import math
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import timeit
import os
import pickle
from sklearn.model_selection import train_test_split
import trainer.torch_trainer as torch_trainer
from skimage import measure
import math
from typing import Optional
# import itertools
# from my_collections import AttrDict
from functools import lru_cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

filename = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_all.pkl'
with open(filename, "rb") as f:
    geo_data = pickle.load(f)
vertices_all = geo_data['vertices']
inner_loops_all = geo_data['inner_loops']
out_loop_all = geo_data['out_loop']
points_cloud_all = geo_data['points_cloud']
sdf_all = geo_data['sdf']
x_grid = geo_data['x_grids']
y_grid = geo_data['y_grids']

# %%


class PointCloudPerceiverChannelsEncoder(nn.Module):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(self,
                 input_channels: int = 2,
                 out_c: int = 128,
                 width: int = 128,
                 latent_ctx: int = 128,
                 n_point: int = 128,
                 n_sample: int = 8,
                 radius: float = 0.2,
                 patch_size: int = 8,
                 padding_mode: str = "circular",
                 d_hidden: List[int] = [128, 128],
                 fps_method: str = 'first',
                 num_heads: int = 4,
                 cross_attn_layers: int = 1,
                 self_attn_layers: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Args:
            input_channels (int): 2 or 3
            width (int): hidden dimension
            latent_ctx (int): number of context points
            n_point (int): number of points in the point set embedding
            n_sample (int): number of samples in the point set embedding
            radius (float): radius for the point set embedding
            patch_size//2 (int): padding size of dim 1 of conv in the point set embedding
            padding_mode (str): padding mode of the conv in the point set embedding
            d_hidden (list): hidden dimensions for the conv in the point set embedding
            fps_method (str): method for point sampling in the point set embedding, 'fps' or 'first', 'fps' has issue
            out_c (int): output channels
            final out shape: [B, out_c*latent_ctx]
        """
        self.width = width
        self.latent_ctx = latent_ctx
        self.n_point = n_point
        self.out_c = out_c
        # position embeding + linear layer
        self.pos_emb_linear = PosEmbLinear("nerf", input_channels, self.width)

        d_input = self.width
        self.point_set_embedding \
            = PointSetEmbedding(ndim=input_channels, radius=radius, n_point=self.n_point,
                                n_sample=n_sample, d_input=d_input,
                                d_hidden=d_hidden, patch_size=patch_size,
                                padding_mode=padding_mode,
                                fps_method=fps_method)

        self.register_parameter(
            "output_tokens",
            nn.Parameter(torch.randn(self.latent_ctx, self.width)),
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)

        self.encoder = SimplePerceiver(
            width=self.width, heads=num_heads, layers=cross_attn_layers)

        self.processor = Transformer(
            width=self.width, heads=num_heads, layers=self_attn_layers)
        self.output_proj = nn.Linear(
            self.width, self.out_c)

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): [B, C, N]
                   C =2 or 3, or >3 if has other features
        Returns:
            torch.Tensor: [B, out_c*latent_ctx]
        """
        xyz = points
        # [B,C1,N] -> [B,N,C1]
        points = points.permute(0, 2, 1)
        # [B, N, C1] -> [B, N, C2], C2=self.width
        dataset_emb = self.pos_emb_linear(points)  # [B, N, C]
        # [B, N, C2] -> [B, C2, N]
        points = dataset_emb.permute(0, 2, 1)
        # [B, C2, N] -------------> [B, C3, No], No=n_point
        #      \ pointNet             / mean (dim=2)
        #       \ permute            / Conv, C3=d_hidden[-1]
        #       [B, C2+ndim,  n_sample, n_point]
        data_tokens = self.point_set_embedding(xyz, points)
        # [B, Co, No] -> [B, No, Co]
        data_tokens = data_tokens.permute(0, 2, 1)
        batch_size = points.shape[0]
        latent_tokens = self.output_tokens.unsqueeze(
            0).repeat(batch_size, 1, 1)  # [B, latent_ctx, width]
        # [B, n_point+latent_ctx, width]
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.n_point +
                           self.latent_ctx, self.width)
        # [B, n_point+latent_ctx, width] -> [B,  n_point+latent_ctx, width]
        h = self.encoder(h, dataset_emb)
        h = self.processor(h)
        # [B,  n_point+latent_ctx, width] -> [B, latent_ctx, width]
        # -> [B, latent_ctx, out_c]
        h = self.output_proj(self.ln_post(h[:, -self.latent_ctx:]))
        h = h.view(batch_size, -1)
        return h

# %%


pc = torch.randn(7, 2, 484).to(device)

geo_encoder = PointCloudPerceiverChannelsEncoder()
geo_encoder = geo_encoder.to(device)
h = geo_encoder(pc)
print(h.shape)
# %%
print("Total number of parameters of encoder: ", sum(p.numel()
      for p in geo_encoder.parameters()))
# %%


class implicit_sdf(nn.Module):
    def __init__(self, latent_ctx=64, ndim=2, emd_version="nerf"):

        super().__init__()
        # Create a list of (weight size, bias size, activation function) tuples
        self.params_pre_name = 'projed_mlp_'
        self.emd_version = emd_version
        d = position_encoding_channels(emd_version)
        weight_shapes = [torch.Size([latent_ctx//4, d*ndim]), torch.Size(
            [latent_ctx//4, latent_ctx//4]),  torch.Size([latent_ctx//4, latent_ctx//4]), torch.Size([latent_ctx//4, latent_ctx//4])]

        self.param_shapes = {}

        for i, v in enumerate(weight_shapes):
            self.param_shapes[self.params_pre_name+str(i)+'_weight'] = v
            self.register_parameter(
                self.params_pre_name+str(i)+'_bias', nn.Parameter(torch.randn(v[0])))

        self.d_latent = latent_ctx
        learned_scale = 0.0625
        use_ln = True
        self.params_proj = ChannelsParamsProj(
            device=device,
            param_shapes=self.param_shapes,
            d_latent=self.d_latent,
            learned_scale=learned_scale,
            use_ln=use_ln,
        )
        l1 = nn.Linear(latent_ctx//4, 100)  # , bias=False
        l2 = nn.Linear(100, 100)
        l3 = nn.Linear(100, 1)
        self.nn_layers = nn.ModuleList([l1, nn.SiLU(), l2, nn.SiLU(), l3])

    def forward(self, x, latent):
        x = x[None].repeat(latent.shape[0], 1, 1)
        x = encode_position('nerf', position=x)
        latent = latent.view(latent.shape[0], self.d_latent, -1)
        proj_params = self.params_proj(latent)
        for i, kw in enumerate(proj_params.keys()):
            w = proj_params[kw]
            b = getattr(self, self.params_pre_name+str(i)+'_bias')
            x = torch.einsum("bpi,boi->bpo", x, w)
            x = torch.add(x, b)
            x = F.silu(x)

        for layer in self.nn_layers:
            x = layer(x)
        return x.squeeze()


# %%
latent_ctx = 128
output_dim_encoder = latent_ctx*latent_ctx
sdf_NN = implicit_sdf(latent_ctx=latent_ctx)
sdf_NN = sdf_NN.to(device)
x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).to(device)
params = torch.randn(3, output_dim_encoder).to(device)
sdf_pred = sdf_NN(x, params)
print("Total number of parameters of sdf_NN: ", sum(p.numel()
      for p in sdf_NN.parameters()))

# %%
x_grids = geo_data['x_grids'].astype(np.float32)
y_grids = geo_data['y_grids'].astype(np.float32)
points_cloud_all = [torch.tensor(x[:, :2], dtype=torch.float32)
                    for x in points_cloud_all]
points_cloud_all = pad_sequence(
    points_cloud_all, batch_first=True, padding_value=-1.0)
sdf_all = np.array(geo_data['sdf'], dtype=np.float32)

sdf_shift, sdf_scale = np.mean(sdf_all), np.std(sdf_all)
sdf_all_norm = (sdf_all-sdf_shift)/sdf_scale
grid_coor = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
SDFs = torch.tensor(sdf_all_norm)
grid_coor = torch.tensor(grid_coor).to(device)
points_cloud = torch.tensor(points_cloud_all).permute(0, 2, 1)

pc_train, pc_test, SDF_train, SDF_test = train_test_split(
    points_cloud[:], SDFs[:], test_size=0.2, random_state=42)
dataset_train = TensorDataset(pc_train, SDF_train)
dataset_test = TensorDataset(pc_test, SDF_test)
train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
test_loader = DataLoader(
    dataset_test, batch_size=1024, shuffle=False)

# %%


class TRAINER(torch_trainer.TorchTrainer):
    def __init__(self, models, device, filebase):
        super().__init__(models, device, filebase)

    def evaluate_losses(self, data):
        pc = data[0].to(self.device)
        SDF = data[1].to(self.device)
        # num=0
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         num+=param.numel()
        # print("number of parameters with grad:",num)
        params = self.models[0](pc)
        sdf_pred = self.models[1](grid_coor, params)
        loss = self.loss_fn(sdf_pred, SDF)

        loss_dic = {"loss": loss.item()}
        return loss, loss_dic


filebase = "./saved_models/geo_pointconv_embpoint128_sequence"
trainer = TRAINER({"encoder": geo_encoder, "sdf_NN": sdf_NN}, device, filebase)
optimizer = torch.optim.Adam(trainer.parameters(), lr=5e-4)
checkpoint = torch_trainer.ModelCheckpoint(
    monitor="val_loss", save_best_only=True)
trainer.compile(optimizer, loss_fn=nn.MSELoss(), checkpoint=checkpoint)


# %%
# trainer.load_weights(device=device)
# h = trainer.load_logs()
h = trainer.fit(train_loader, val_loader=test_loader,
                epochs=500,  print_freq=1)
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
sd_pred_test, sd_ture_test = [], []
with torch.no_grad():
    for test_data in test_loader:
        pc_test = test_data[0].to(device)
        sdf_test = test_data[1].to(device)
        para_test = geo_encoder(pc_test)
        sd_pred = trainer.models[1](
            grid_coor, para_test).cpu().detach().numpy()
        sd_true = sdf_test.view(-1, grid_coor.shape[0]).cpu().numpy()
        sd_pred = sd_pred*sdf_scale+sdf_shift
        sd_true = sd_true*sdf_scale+sdf_shift
        sd_pred_test.append(sd_pred)
        sd_ture_test.append(sd_true)
sd_pred_test = np.vstack(sd_pred_test)
sd_ture_test = np.vstack(sd_ture_test)
error_s = np.linalg.norm(sd_pred_test-sd_ture_test, axis=1) / \
    np.linalg.norm(sd_ture_test, axis=1)

fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)

_ = ax.hist(error_s, bins=20, color="skyblue", edgecolor="black")
ax.set_xlabel("L2 relative error")
ax.set_ylabel("Frequency")
# %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[int(len(sort_idx)*0.97)]
median_index = sort_idx[len(sort_idx) // 2]
titles = ["best", "50% percentile", "97% percentile"]
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
    sd_pred_i = sd_pred_test[index].reshape(x_grids.shape)
    sd_true_i = sd_ture_test[index].reshape(x_grids.shape)
    pred_geo = measure.find_contours(sd_pred_i, 0, positive_orientation='high')
    true_geo = measure.find_contours(sd_true_i, 0, positive_orientation='high')
    for c, contour in enumerate(true_geo):
        if c == 0:
            ax.plot(contour[:, 1], contour[:, 0],
                    'r', linewidth=2, label="Truth")
        else:
            ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
    for c, contour in enumerate(pred_geo):
        if c == 0:
            ax.plot(contour[:, 1], contour[:, 0], '--b',
                    linewidth=2, label="Predicted")
        else:
            ax.plot(contour[:, 1], contour[:, 0], '--b', linewidth=2)

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(f"{titles[i]}")

    plt.tight_layout()

# %%
fig = plt.figure()
ax = fig.add_subplot(111)
pc = dataset_test[median_index][0].cpu().numpy()
ax.scatter(pc[0], pc[1], c='b', s=2, marker='o',)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
ax.axis('equal')
ax.axis('off')

plt.show()
# %%
