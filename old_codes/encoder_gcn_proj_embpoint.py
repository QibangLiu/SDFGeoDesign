# %%
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
from functools import lru_cache
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import pickle
from skimage import measure

from sklearn.model_selection import train_test_split
import torch_trainer
import os
import matplotlib.pyplot as plt

import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

with open('./training_data/geo_sdf_randv_1.pkl', "rb") as f:
    geo_data = pickle.load(f)
vertices_all = geo_data['vertices']
inner_loops_all = geo_data['inner_loops']
out_loop_all = geo_data['out_loop']
points_cloud_all = geo_data['points_cloud']
sdf_all = np.array(geo_data['sdf'], dtype=np.float32)
x_grids = geo_data['x_grids'].astype(np.float32)
y_grids = geo_data['y_grids'].astype(np.float32)
# %%


def generate_graph(vertices, inner_loops, out_loop):
    # Create edges between consecutive points, closing the loop
    edges = []
    for i in range(len(out_loop)-1):
        edges.append((out_loop[i], out_loop[i+1]))
    for loop in inner_loops:
        for i in range(len(loop)-1):
            edges.append((loop[i], loop[i+1]))

    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Create node features (e.g., (x, y) coordinates)
    node_features = torch.tensor(vertices[:, :2], dtype=torch.float)
    return Data(x=node_features, edge_index=edges)


# %%


@lru_cache
def get_scales(
    min_deg: int,
    max_deg: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return 2.0 ** torch.arange(min_deg, max_deg, device=device, dtype=dtype)


def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://arxiv.org/pdf/2210.04628.pdf
    """
    if min_deg == max_deg:
        return x
    scales = get_scales(min_deg, max_deg, x.dtype, x.device)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)


def encode_position(position: torch.Tensor, version="nerf") -> torch.Tensor:
    if version == "v1":
        freqs = get_scales(0, 10, position.dtype, position.device).view(1, -1)
        freqs = position.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(*position.shape[:-1], -1)
    elif version == "nerf":
        return posenc_nerf(position, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def flatten_param_shapes(param_shapes: Dict[str, Tuple[int]]):
    flat_shapes = {
        name: (int(np.prod(shape)) // shape[-1], shape[-1])
        for name, shape in param_shapes.items()
    }
    return flat_shapes


def _sanitize_name(x: str) -> str:
    # return x
    return x.replace(".", "_")


class ParamsProj(nn.Module, ABC):
    def __init__(self, *, device: torch.device, param_shapes: Dict[str, Tuple[int]], d_latent: int):
        super().__init__()
        self.device = device
        self.param_shapes = param_shapes
        self.d_latent = d_latent

    @abstractmethod
    def forward(self, x: torch.Tensor, options: Optional[Dict] = None) -> Dict:
        pass


class ChannelsProj(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        vectors: int,
        channels: int,
        d_latent: int,
        init_scale: float = 1.0,
        learned_scale: Optional[float] = None,
        use_ln: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(d_latent, vectors * channels, device=device)
        self.use_ln = use_ln
        self.learned_scale = learned_scale
        if use_ln:
            self.norm = nn.LayerNorm(
                normalized_shape=(channels,), device=device)
            if learned_scale is not None:
                self.norm.weight.data.fill_(learned_scale)
            scale = init_scale / math.sqrt(d_latent)
        elif learned_scale is not None:
            gain = torch.ones((channels,), device=device) * learned_scale
            self.register_parameter("gain", nn.Parameter(gain))
            scale = init_scale / math.sqrt(d_latent)
        else:
            scale = init_scale / math.sqrt(d_latent * channels)
        # nn.init.normal_(self.proj.weight, std=scale)
        # nn.init.zeros_(self.proj.bias)
        self.d_latent = d_latent
        self.vectors = vectors
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bvd = x
        w_vcd = self.proj.weight.view(
            self.vectors, self.channels, self.d_latent)
        b_vc = self.proj.bias.view(1, self.vectors, self.channels)
        h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        if self.use_ln:
            h = self.norm(h)
        elif self.learned_scale is not None:
            h = h * self.gain.view(1, 1, -1)
        h = h + b_vc
        return h


class ChannelsParamsProj(ParamsProj):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        d_latent: int,
        init_scale: float = 1.0,
        learned_scale: Optional[float] = None,
        use_ln: bool = False,
    ):
        super().__init__(device=device, param_shapes=param_shapes, d_latent=d_latent)
        self.param_shapes = param_shapes
        self.projections = nn.ModuleDict({})
        self.flat_shapes = flatten_param_shapes(param_shapes)
        self.learned_scale = learned_scale
        self.use_ln = use_ln
        for k, (vectors, channels) in self.flat_shapes.items():
            self.projections[_sanitize_name(k)] = ChannelsProj(
                device=device,
                vectors=vectors,
                channels=channels,
                d_latent=d_latent,
                init_scale=init_scale,
                learned_scale=learned_scale,
                use_ln=use_ln,
            )

    def forward(self, x: torch.Tensor, options: Optional[Dict] = None) -> Dict:
        out = dict()
        start = 0
        for k, shape in self.param_shapes.items():
            vectors, _ = self.flat_shapes[k]
            end = start + vectors
            x_bvd = x[:, start:end]
            out[k] = self.projections[_sanitize_name(k)](
                x_bvd).reshape(len(x), *shape)
            start = end
        return out

# %%


class implicit_sdf(nn.Module):
    def __init__(self, latent_ctx=64):
        """the output size is not 1, in order the match total number of parameters
        to the output size of the encoder.
        the average of the output is taken as the final output,
        so the final output size is 1"""
        super().__init__()
        # Create a list of (weight size, bias size, activation function) tuples
        self.params_pre_name = 'projed_mlp_'
        weight_shapes = [torch.Size([latent_ctx//4, 62]), torch.Size(
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
        x = encode_position(x)
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
# Define a simple GNN encoder model


class EncoderGNNModel(torch.nn.Module):
    def __init__(self, num_node_features, graph_hid_dims, fcn_hid_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(num_node_features, graph_hid_dims[0]))
        self.layers.append(torch.nn.ReLU())
        for i in range(1, len(graph_hid_dims)):
            self.layers.append(GCNConv(graph_hid_dims[i-1], graph_hid_dims[i]))
            BatchNorm(graph_hid_dims[i])
            self.layers.append(torch.nn.ReLU())
            self.layers.append(GATConv(
                in_channels=graph_hid_dims[i], out_channels=graph_hid_dims[i], heads=4, concat=False, dropout=0.2))
            BatchNorm(graph_hid_dims[i])
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(
            graph_hid_dims[-1], fcn_hid_dims[0]))
        self.layers.append(torch.nn.ReLU())
        for i in range(1, len(fcn_hid_dims)):
            self.layers.append(torch.nn.Linear(
                fcn_hid_dims[i-1], fcn_hid_dims[i]))
            self.layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Linear(fcn_hid_dims[-1], output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            if isinstance(layer, (GCNConv, GATConv)):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        x = global_mean_pool(x, data.batch)  # Pooling layer
        x = self.fc(x)
        return x


# %%
# Generate multiple surfaces
graphs = [generate_graph(vertices_all[i], inner_loops_all[i],
                         out_loop_all[i]) for i in range(len(vertices_all))]
# %%
# Attach sign distance and grid points
sdf_shift, sdf_scale = np.mean(sdf_all), np.std(sdf_all)
sdf_all_norm = (sdf_all-sdf_shift)/sdf_scale


grid_points = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
SDFs = torch.tensor(sdf_all_norm)
grid_coor = torch.tensor(grid_points).to(device)
for i in range(len(graphs)):
    graphs[i].y = SDFs[i]
# %%
graph_train, graph_test, SDF_train, SDF_test = train_test_split(
    graphs, SDFs, test_size=0.2, random_state=42)

for i in range(len(graph_train)):
    graph_train[i].y = SDF_train[i]
for i in range(len(graph_test)):
    graph_test[i].y = SDF_test[i]
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
test_loader = DataLoader(graph_test, batch_size=128, shuffle=False)


# train_loader = DataLoader(graphs[:1], batch_size=128, shuffle=True)
# test_loader = DataLoader(graphs[-1000:], batch_size=2048, shuffle=False)
# %%

class TRAINER(torch_trainer.TorchTrainer):
    def __init__(self, models, device):
        super().__init__(models, device)

    def evaluate_losses(self, data):
        data = data.to(self.device)
        # num=0
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         num+=param.numel()
        # print("number of parameters with grad:",num)
        params = self.models[0](data)
        sdf_pred = self.models[1](grid_coor, params)
        loss = self.loss_fn(sdf_pred, data.y.view(-1, grid_coor.shape[0]))

        loss_dic = {"loss": loss.item()}
        return loss, loss_dic


# %%
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

geo_encoder = EncoderGNNModel(num_node_features=2, graph_hid_dims=[
                              128]*4, fcn_hid_dims=[100]*4, output_dim=output_dim_encoder)  # Assuming output is a scalar
print("number parameters of geo_encoder:", sum(p.numel()
      for p in geo_encoder.parameters()))
out_p = geo_encoder(next(iter(train_loader)))
# for name, param in geo_encoder.named_parameters():
#     print(f'Parameter Name: {name}, Shape: {param.shape}')

for lay in geo_encoder.layers:
    print(lay._get_name())
    for name, param in lay.named_parameters():
        print(f'Parameter Name: {name}, Shape: {param.shape}')
# %%

trainer = TRAINER([geo_encoder, sdf_NN], device)
trainer.compile(optimizer=torch.optim.Adam, lr=1e-3, loss=nn.MSELoss())
filebase = "./saved_model/geo_gcn"
model_path = ["encoder", "sdf_NN"]
checkpoint_fnames = []
for m_path in model_path:
    m_path = os.path.join(filebase, m_path)
    os.makedirs(m_path, exist_ok=True)
    checkpoint_fnames.append(os.path.join(m_path, "model.ckpt"))
checkpoint = torch_trainer.ModelCheckpoint(
    checkpoint_fnames, monitor="val_loss", save_best_only=True
)

# %%
# trainer.load_weights(checkpoint_fnames, device)
# h = trainer.load_logs(filebase)
h = trainer.fit(test_loader, val_loader=test_loader,
                epochs=4000, callbacks=checkpoint, print_freq=1)
trainer.save_logs(filebase)

# %%
trainer.load_weights(checkpoint_fnames, device)
h = trainer.load_logs(filebase)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")


# %%
test_data = next(iter(train_loader))
test_data = test_data.to(device)
para_test = geo_encoder(test_data)
sd_pred = sdf_NN(grid_coor, para_test).cpu().detach().numpy()
sd_true = test_data.y.view(-1, grid_coor.shape[0]).cpu().numpy()

sd_pred = sd_pred*sdf_scale+sdf_shift
sd_true = sd_true*sdf_scale+sdf_shift

error_s = np.linalg.norm(sd_pred-sd_true, axis=1) / \
    np.linalg.norm(sd_true, axis=1)

fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(error_s, bins=20)

# %%

# # %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]
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
    sd_pred_i = sd_pred[index].reshape(x_grids.shape)
    sd_true_i = sd_true[index].reshape(x_grids.shape)
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
            ax.plot(contour[:, 1], contour[:, 0], 'b',
                    linewidth=2, label="Predicted")
        else:
            ax.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.tight_layout()

# %%
