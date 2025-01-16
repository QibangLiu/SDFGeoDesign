# %%
import torch.nn as nn
from typing import List, Optional, Tuple, Union

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
import torch_utils.torch_trainer as torch_trainer
from skimage import measure
import math
from typing import Optional
# import itertools
# from my_collections import AttrDict
from functools import lru_cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

filename = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_pcn_all.pkl'
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


def encode_position(position=torch.zeros(1, 1), version='nerf'):
    if version == "v1":
        freqs = get_scales(0, 10, position.dtype, position.device).view(1, -1)
        freqs = position.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(*position.shape[:-1], -1)
    elif version == "nerf":
        return posenc_nerf(position, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def position_encoding_channels(version: Optional[str] = None) -> int:
    if version is None:
        return 1
    return encode_position(torch.zeros(1, 1), version).shape[-1]


class PosEmbLinear(nn.Linear):
    def __init__(
        self, posemb_version: Optional[str], in_features: int, out_features: int, **kwargs
    ):
        super().__init__(
            in_features * position_encoding_channels(posemb_version),
            out_features,
            **kwargs,
        )
        self.posemb_version = posemb_version

    def forward(self, x: torch.Tensor):
        if self.posemb_version is not None:
            x = encode_position(version=self.posemb_version, position=x)
        return super().forward(x)


def get_act(name):
    return {
        "relu": torch.nn.functional.relu,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "swish": torch.nn.functional.silu,
        "tanh": torch.tanh,
        "gelu": torch.nn.functional.gelu,
        # "quick_gelu": torch.nn.functional.quick_gelu,
        # "torch_gelu": torch_gelu,
        # "gelu2": quick_gelu,
        # "geglu": geglu,
        "sigmoid": torch.sigmoid,
        "sin": torch.sin,
        "sin30": torch.nn.functional.SirenSin(w0=30.0),
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "identity": lambda x: x,
    }[name]


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(
            device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(
        device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint,
    radius,
    nsample,
    xyz,
    points,
    returnfps=False,
    deterministic=False,
    fps_method: str = "fps",
):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = torch.arange(npoint)[None].repeat(B, 1)

    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class PointSetEmbedding(nn.Module):
    def __init__(
        self,
        *,
        radius: float,
        n_point: int,
        n_sample: int,
        d_input: int,
        d_hidden: List[int],
        patch_size: int = 1,
        stride: int = 1,
        activation=torch.nn.functional.silu,
        group_all: bool = False,
        padding_mode: str = "zeros",
        fps_method: str = "fps",
        **kwargs,
    ):
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample
        self.mlp_convs = nn.ModuleList()
        self.act = activation
        self.patch_size = patch_size
        self.stride = stride
        last_channel = d_input + 2  # TODO: 2 is 2D 3 is 3D
        for out_channel in d_hidden:
            self.mlp_convs.append(
                nn.Conv2d(
                    last_channel,
                    out_channel,
                    kernel_size=(patch_size, 1),
                    stride=(stride, 1),
                    padding=(patch_size // 2, 0),
                    padding_mode=padding_mode,
                    **kwargs,
                )
            )
            last_channel = out_channel
        self.group_all = group_all
        self.fps_method = fps_method

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: sample points feature data, [B, d_hidden[-1], n_point]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.n_point,
                self.radius,
                self.n_sample,
                xyz,
                points,
                deterministic=not self.training,
                fps_method=self.fps_method,
            )
        # new_xyz: sampled points position data, [B, n_point, C]
        # new_points: sampled points data, [B, n_point, n_sample, C+D]
        # [B, C+D, n_sample, n_point]
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            new_points = self.act(self.apply_conv(new_points, conv))

        new_points = new_points.mean(dim=2)
        return new_points

    def apply_conv(self, points: torch.Tensor, conv: nn.Module):
        batch, channels, n_samples, _ = points.shape
        # Shuffle the representations
        if self.patch_size > 1:
            # TODO shuffle deterministically when not self.training
            _, indices = torch.rand(
                batch, channels, n_samples, 1, device=points.device).sort(dim=2)
            points = torch.gather(
                points, 2, torch.broadcast_to(indices, points.shape))
        return conv(points)


# %%
a = encode_position(version="nerf", position=torch.zeros(1, 1))
print(f"{a.numpy()[-1,-2]:.12e}")
# %%
weight_shapes = [torch.Size([32, 62]), torch.Size(
    [32, 32]),  torch.Size([32, 32]), torch.Size([32, 32])]


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        n_data: int,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 1.0,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        # self.attn = MultiheadCrossAttention(
        #     n_ctx=n_ctx,
        #     n_data=n_data,
        #     width=width,
        #     heads=heads,
        #     data_width=data_width,
        #     init_scale=init_scale,
        # )
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        kv = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), kv, kv)[0]
        x = x + self.mlp(self.ln_3(x))
        return x


class SimplePerceiver(nn.Module):
    """
    Only does cross attention
    """

    def __init__(
        self,
        n_ctx: int,
        n_data: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    n_ctx=n_ctx,
                    n_data=n_data,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    data_width=data_width,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        for block in self.resblocks:
            x = block(x, data)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        # self.attn = MultiheadAttention(
        #     n_ctx=n_ctx,
        #     width=width,
        #     heads=heads,
        #     init_scale=init_scale,
        # )
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(width)
        # self.ln_k = nn.LayerNorm(width)
        # self.ln_v = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        qkv = self.ln_1(x)
        x = x + self.attn(qkv, qkv, qkv)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointCloudPerceiverChannelsEncoder(nn.Module):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        input_channels = 2
        self.width = 128
        self.pos_emb_linear = PosEmbLinear("nerf", input_channels, self.width)
        self.data_ctx = 128
        radius = 0.2
        n_sample = 8
        padding_mode = "circular"
        fps_method = 'first'
        patch_size = 8  # ???
        d_input = 128
        d_hidden = [128, 128]
        self.point_set_embedding \
            = PointSetEmbedding(radius=radius, n_point=self.data_ctx,
                                n_sample=n_sample, d_input=d_input,
                                d_hidden=d_hidden, patch_size=patch_size,
                                padding_mode=padding_mode,
                                fps_method=fps_method)
        self.latent_ctx = sum(ws[0] for ws in weight_shapes)

        self.register_parameter(
            "output_tokens",
            nn.Parameter(torch.randn(self.latent_ctx, self.width)),
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)

        n_ctx = self.data_ctx + self.latent_ctx
        layers = 1
        self.encoder = SimplePerceiver(
            n_ctx=n_ctx,
            n_data=self.latent_ctx,
            width=self.width,
            layers=layers,
            heads=4
        )

        self.processor = Transformer(
            n_ctx=self.data_ctx + self.latent_ctx,
            layers=3,
            width=self.width,
            heads=4,
            # init_scale=init_scale,
        )
        self.output_proj = nn.Linear(
            self.width, self.data_ctx)

    def forward(self, points):
        xyz = points
        points = points.permute(0, 2, 1)
        dataset_emb = self.pos_emb_linear(points)  # [B, N, C]
        points = dataset_emb.permute(0, 2, 1)

        data_tokens = self.point_set_embedding(xyz, points)
        data_tokens = data_tokens.permute(0, 2, 1)
        batch_size = points.shape[0]
        latent_tokens = self.output_tokens.unsqueeze(
            0).repeat(batch_size, 1, 1)
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.data_ctx +
                           self.latent_ctx, self.width)
        h = self.encoder(h, dataset_emb)
        h = self.processor(h)
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
points_cloud_all = geo_data['points_cloud'][:, :, :2].astype(np.float32)
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


filebase = "./saved_model/geo_pointconv_embpoint128_simple_reduceLN"
trainer = TRAINER({"encoder": geo_encoder, "sdf_NN": sdf_NN}, device, filebase)
optimizer = torch.optim.Adam(trainer.parameters(), lr=5e-4)
checkpoint = torch_trainer.ModelCheckpoint(
    monitor="val_loss", save_best_only=True)
trainer.compile(optimizer, loss_fn=nn.MSELoss(), checkpoint=checkpoint)

# model_path = ["encoder", "sdf_NN"]
# checkpoint_fnames = []
# for m_path in model_path:
#     m_path = os.path.join(filebase, m_path)
#     os.makedirs(m_path, exist_ok=True)
#     checkpoint_fnames.append(os.path.join(m_path, "model.ckpt"))
# checkpoint = torch_trainer.ModelCheckpoint(
#     checkpoint_fnames, monitor="val_loss", save_best_only=True
# )


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
