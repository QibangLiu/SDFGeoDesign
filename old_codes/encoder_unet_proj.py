# %%
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
import torch_trainer
from skimage import measure
import math
from typing import Optional
import itertools
from my_collections import AttrDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


class AttentionBlock(nn.Module):
    """Applies self-attention.
    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8):
        super(AttentionBlock, self).__init__()
        self.units = units
        self.groups = groups

        self.norm = nn.GroupNorm(groups, units)
        self.query = nn.Linear(units, units)
        self.key = nn.Linear(units, units)
        self.value = nn.Linear(units, units)
        self.proj = nn.Linear(units, units)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        scale = self.units ** (-0.5)
        # [b, c, h, w] -> [b, h, w, c]
        inputs = self.norm(inputs)  # [b, c, h, w]
        # [b, c, h, w] -> [b, h, w, c]
        inputs = inputs.permute(0, 2, 3, 1)
        q = self.query(inputs)  # [b, h, w, c]
        k = self.key(inputs)  # [b, h, w, c]
        v = self.value(inputs)  # [b, h, w, c]

        # equivalent: [hw X C] * [hw X C]^T, eliminate the index of c
        attn_score = torch.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = attn_score.view(batch_size, height, width, height * width)

        attn_score = F.softmax(attn_score, dim=-1)  # [b, h, w, hw]
        attn_score = attn_score.view(
            batch_size, height, width, height, width)  # [b, h, w, h, w]

        # equivalent: [hw X hw] * [hw X c]
        proj = torch.einsum("bhwHW,bHWc->bhwc", attn_score, v)  # [b, h, w, c]
        proj = self.proj(proj)  # [b, h, w, c]
        # [b, h, w, c] -> [b, c, h, w]
        output = (inputs + proj).permute(0, 3, 1, 2)
        return output


inputs = torch.randn(1, 16, 64, 64)  # Example input tensor
attention_block = AttentionBlock(units=16, groups=8)
outputs = attention_block(inputs)
print(outputs.shape)
# %%


class ResidualBlock(nn.Module):
    """Residual block.

    Args:
        channel: Number of channels in the convolutional layers
        groups: Number of groups to be used for GroupNormalization layer
        activation_fn: Activation function to be used
    """

    def __init__(self, channel_in, channel_out, groups=8, activation_fn=F.silu):
        super(ResidualBlock, self).__init__()

        self.groups = groups
        self.activation_fn = activation_fn
        self.net = nn.ModuleList()
        self.net.append(nn.GroupNorm(groups, channel_in))
        self.net.append(nn.Conv2d(channel_in, channel_out,
                                  kernel_size=3, padding="same"))
        # self.net.append(nn.Dropout2d(p=0.2))
        self.net.append(nn.GroupNorm(groups, channel_out))
        self.net.append(nn.Conv2d(channel_out, channel_out,
                                  kernel_size=3, padding="same"))
        # self.net.append(nn.Dropout2d(p=0.2))
        if channel_in != channel_out:
            self.shortcut = nn.Conv2d(channel_in, channel_out, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        x = inputs
        for layer in self.net:
            x = layer(x)  # [b,ci,h,w]->[b,co,h,w]
        x = x + self.shortcut(inputs)
        return x


# %%


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.channel = channel
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        return self.conv(inputs)


class UpSample(nn.Module):
    def __init__(self, channel, interpolation="nearest"):
        super(UpSample, self).__init__()
        self.channel = channel
        self.interpolation = interpolation
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        return x

# %%


class UNet(nn.Module):
    def __init__(self, img_shape, channel_list, has_attention, num_res_blocks=1, norm_groups=8,
                 interpolation="nearest", activation_fn=F.silu, first_conv_channels=64):
        super().__init__()
        channel_in, img_size = img_shape[0], img_shape[1]
        self.activation_fn = activation_fn
        self.skip_channels = []
        encoder_skip_idx = []
        self.first_conv = nn.Conv2d(
            channel_in, first_conv_channels, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList()
        for i in range(len(channel_list)):
            if i == 0:
                in_channel = first_conv_channels
            else:
                in_channel = channel_list[i-1]
            self.encoder.append(ResidualBlock(
                in_channel, channel_list[i], groups=norm_groups, activation_fn=activation_fn))
            if has_attention[i]:
                self.encoder.append(AttentionBlock(
                    channel_list[i], groups=norm_groups))
            self.skip_channels.append(channel_list[i])
            encoder_skip_idx.append(len(self.encoder)-1)
            for _ in range(1, num_res_blocks):
                self.encoder.append(ResidualBlock(
                    channel_list[i], channel_list[i], groups=norm_groups, activation_fn=activation_fn))
                if has_attention[i]:
                    self.encoder.append(AttentionBlock(
                        channel_list[i], groups=norm_groups))
                self.skip_channels.append(channel_list[i])
                encoder_skip_idx.append(len(self.encoder)-1)
                # skip connection
            if i != len(channel_list)-1:
                self.encoder.append(DownSample(channel_list[i]))

        self.encoder_skip = torch.zeros(len(self.encoder), dtype=bool)
        self.encoder_skip[encoder_skip_idx] = True

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResidualBlock(
            channel_list[-1], channel_list[-1], groups=norm_groups, activation_fn=activation_fn))
        self.bottleneck.append(AttentionBlock(
            channel_list[-1], groups=norm_groups))
        self.bottleneck.append(ResidualBlock(
            channel_list[-1], channel_list[-1], groups=norm_groups, activation_fn=activation_fn))

        self.decoder = nn.ModuleList()
        decoder_skip_idx = []
        for i in reversed(range(len(channel_list))):
            if i == len(channel_list)-1:
                in_channel = channel_list[-1]+self.skip_channels.pop()
            else:
                in_channel = channel_list[i+1]+self.skip_channels.pop()
            self.decoder.append(ResidualBlock(
                in_channel, channel_list[i], groups=norm_groups, activation_fn=activation_fn))
            decoder_skip_idx.append(len(self.decoder)-1)
            if has_attention[i]:
                self.decoder.append(AttentionBlock(
                    channel_list[i], groups=norm_groups))
            for _ in range(1, num_res_blocks):
                in_channel = channel_list[i]+self.skip_channels.pop()
                self.decoder.append(ResidualBlock(
                    in_channel, channel_list[i], groups=norm_groups, activation_fn=activation_fn))
                decoder_skip_idx.append(len(self.decoder)-1)
                if has_attention[i]:
                    self.decoder.append(AttentionBlock(
                        channel_list[i], groups=norm_groups))

            if i != 0:
                self.decoder.append(
                    UpSample(channel_list[i], interpolation=interpolation))

        self.decoder_skip = torch.zeros(len(self.decoder), dtype=bool)
        self.decoder_skip[decoder_skip_idx] = True

    def forward(self, x):
        skips = []
        x = self.first_conv(x)
        for eskip, layer in zip(self.encoder_skip, self.encoder):
            x = layer(x)
            if eskip:
                skips.append(x)
        for layer in self.bottleneck:
            x = layer(x)
        for dskip, layer in zip(self.decoder_skip, self.decoder):
            if dskip:
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# %%


class EncoderGeo(nn.Module):
    def __init__(self, img_shape, channel_list, has_attention, num_out=11, first_conv_channels=16, num_res_blocks=1, norm_groups=8):
        super().__init__()
        self.unet = UNet(img_shape, channel_list,
                         has_attention, first_conv_channels=first_conv_channels, num_res_blocks=num_res_blocks, norm_groups=norm_groups)
        self.cov = nn.Conv2d(
            channel_list[0], 1, kernel_size=3, padding=0, stride=2)
        ims = (img_shape[1]-3)//2+1
        self.cov2 = nn.Conv2d(
            1, 1, kernel_size=3, padding=0, stride=2)
        ims = (ims-3)//2+1

        self.mlp = nn.Sequential(
            nn.Linear(ims*ims, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.SiLU(),
            nn.Linear(100, 100),
            # nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(100, num_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.unet(x)
        x = self.cov(x)
        x = self.cov2(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


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

# %%


class implicit_sdf(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, latent_ctx=64):
        """the output size is not 1, in order the match total number of parameters
        to the output size of the encoder.
        the average of the output is taken as the final output,
        so the final output size is 1"""
        super().__init__()
        # Create a list of (weight size, bias size, activation function) tuples
        self.params_pre_name = 'projed_mlp_'
        weight_shapes = [torch.Size([latent_ctx//4, 2]), torch.Size(
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

        # self.layers = []
        # self.latent_ctx = latent_ctx
        # if len(hidden_sizes) != 4:
        #     raise ValueError("The hidden sizes must have 4 layers")
        # in_size = input_size
        # self.paras_all = {}
        # self.projections = nn.ModuleList()
        # for i, hs in enumerate(hidden_sizes):
        #     proj = ChannelsProj(device=device, vectors=latent_ctx //
        #                         4, channels=in_size, d_latent=latent_ctx)
        #     self.projections.append(proj)
        #     self.register_parameter(
        #         f'proj_weights_{i}', nn.Parameter(torch.randn(latent_ctx//4, in_size, latent_ctx)))
        #     self.register_parameter(f'proj_bias_{i}',
        #                             nn.Parameter(torch.randn(1, latent_ctx//4, in_size)))
        #     # projed_weights: (nb,latent_ctx//4,in_size))
        #     self.register_parameter(
        #         f'projed_bias_{i}', nn.Parameter(torch.randn(latent_ctx//4)))
        #     self.paras_all[f'projed_weights_{i}'] = None
        #     self.paras_all[f'projed_bias_{i}'] = None
        #     in_size = hs

        # for hidden_size in hidden_sizes:
        #     weight = (hidden_size, in_size)
        #     bias = (1, hidden_size)
        #     self.layers.append((weight, bias, nn.SiLU()))
        #     in_size = hidden_size
        # weight = (output_size, in_size)
        # bias = (1, output_size)
        # self.layers.append((weight, bias, nn.Identity()))
        # # the total number of parameters must match the output size of the encoder
        # self.num_para = sum([w[0] * w[1] + b[1] for w, b, _ in self.layers])

        l1 = nn.Linear(latent_ctx//4, 100)  # , bias=False
        l2 = nn.Linear(100, 100)
        l3 = nn.Linear(100, 1)
        self.nn_layers = nn.ModuleList([l1, nn.SiLU(), l2, nn.SiLU(), l3])

    def forward(self, x, latent):
        x = x[None].repeat(latent.shape[0], 1, 1)
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
        # start = 0
        # for i, k in enumerate(self.paras_all.keys()):
        #     end = start + self.latent_ctx//4
        #     x_bvd = latent[:, start:end]
        #     w_vcd = getattr(self, f'proj_weights_{i}')
        #     b_vcd = getattr(self, f'proj_bias_{i}')
        #     h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        #     self.paras_all[k] = getattr(self, "dynamic_weight")
        #     self.projections[_sanitize_name(k)](x_bvd).reshape(len(x), *shape)
        #     start = end

        # for i in range(4):

        #     # x: np,2
        #     # params_all: nb, number of parameters of output of encoder
        # s = 0
        # x = x.expand(params_all.shape[0], -1, -1)
        # # x=x[None].repeat(params_all.shape[0], 1, 1)

        # for w_size, b_size, af in self.layers:
        #     layer_weight_size = w_size[0] * w_size[1]
        #     weight = params_all[:, s: (
        #         s + layer_weight_size)].reshape(-1, *w_size)
        #     s += layer_weight_size
        #     layer_bias_size = b_size[1]
        #     bias = params_all[:, s: (s + layer_bias_size)].reshape(-1, *b_size)
        #     s += layer_bias_size
        #     # b is the batch size
        #     # p is the number of points to evaluate sign distance
        #     # i is the input size of linear layer
        #     # o is the output size of linear layer
        #     x = torch.einsum("bpi,boi->bpo", x, weight)
        #     x = x + bias
        #     x = af(x)  # nb,np,no
        # for layer in self.nn_layers:
        #     # weight, bias = layer.weight, layer.bias
        #     # x = torch.einsum("bpi,oi->bpo", x, weight) #+ bias.view(1, 1, -1)
        #     # x = F.silu(x)
        #     x = layer(x)
        #     # x = F.silu(x)
        # # x: nb,np,1
        # # x = torch.mean(x, dim=-1)
        # return x.squeeze()


# %%
# latent_ctx = 64
# output_dim_encoder = latent_ctx*latent_ctx
# sdf_NN = implicit_sdf(input_size=2, hidden_sizes=[
#                       12]*7, output_size=4, latent_ctx=latent_ctx)
# # print("Total num parameters of implicit sdf: ", sdf_NN.num_para)

# # if output_dim_encoder != sdf_NN.num_para:
# #     raise ValueError(
# #         "output_dim_encoder must match the number of parameters of sdf_NN")
# x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).to(device)
# params = torch.randn(3, output_dim_encoder).to(device)
# sdf_pred = sdf_NN(x, params)

# params = params.view(params.shape[0], 64, 64)
# out = sdf_NN.params_proj(params)
# for name, param in sdf_NN.named_parameters():
#     print(
#         f"Parameter name: {name}, shape {param.shape} requires_grad: {param.requires_grad}")
# print("Total number of parameters of sdf_NN: ", sum(p.numel()
#       for p in sdf_NN.parameters()))

# %%


class TRAINER(torch_trainer.TorchTrainer):
    def __init__(self, models, device):
        super().__init__(models, device)

    def evaluate_losses(self, data):
        data = data[0].to(self.device)
        # num=0
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         num+=param.numel()
        # print("number of parameters with grad:",num)
        params = self.models[0](data)
        sdf_pred = self.models[1](grid_coor, params)
        loss = self.loss_fn(sdf_pred, data.view(-1, grid_coor.shape[0]))

        loss_dic = {"loss": loss.item()}
        return loss, loss_dic
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

sdf_shift, sdf_scale = np.mean(sdf_all), np.std(sdf_all)
sdf_all_norm = (sdf_all-sdf_shift)/sdf_scale

nSamples = len(sdf_all)
grid_coor = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
SDFs = torch.tensor(sdf_all_norm.reshape(-1, 1, *x_grids.shape))
grid_coor = torch.tensor(grid_coor).to(device)

SDF_train, SDF_test = train_test_split(SDFs, test_size=0.2, random_state=42)
dataset_train = TensorDataset(SDF_train)
dataset_test = TensorDataset(SDF_test)
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False)
# %%
# %%
latent_ctx = 128
output_dim_encoder = latent_ctx*latent_ctx
sdf_NN = implicit_sdf(input_size=2, hidden_sizes=[
                      12]*7, output_size=4, latent_ctx=latent_ctx)
sdf_NN = sdf_NN.to(device)
x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).to(device)
params = torch.randn(3, output_dim_encoder).to(device)
sdf_pred = sdf_NN(x, params)
print("Total number of parameters of sdf_NN: ", sum(p.numel()
      for p in sdf_NN.parameters()))
# %%

# Define the input shape and parameters
img_shape = tuple(SDF_train.shape[1:])
channel_list = [8, 16, 32, 64]
has_attention = [False, False, True, True]

geo_encoder = EncoderGeo(img_shape, channel_list,
                         has_attention, num_out=output_dim_encoder, first_conv_channels=channel_list[0], num_res_blocks=1, norm_groups=8)
img = torch.randn(1, *img_shape)
geo_encoder(img).shape
total_params = sum(p.numel() for p in geo_encoder.parameters())
trainable_params = sum(p.numel()
                       for p in geo_encoder.parameters() if p.requires_grad)
print(
    f"Total number of trainable parameters of EncoderGeo: {trainable_params}")
print(f"Total number of parameters of EncoderGeo: {total_params}")
# %%
trainer = TRAINER([geo_encoder, sdf_NN], device)
trainer.compile(optimizer=torch.optim.Adam, lr=5e-4, loss=nn.MSELoss())
filebase = "./saved_model/geo_unet"
model_path = ["encoder", "sdf_NN"]
checkpoint_fnames = []
for m_path in model_path:
    m_path = os.path.join(filebase, m_path)
    os.makedirs(m_path, exist_ok=True)
    checkpoint_fnames.append(os.path.join(m_path, "model.ckpt"))
checkpoint = torch_trainer.ModelCheckpoint(
    checkpoint_fnames, monitor="val_loss", save_best_only=True
)

# combined_parameters = list(geo_encoder.parameters()) + list(sdf_NN.parameters())
# optimizer = torch.optim.Adam(combined_parameters, lr=0.001)
# criterion = nn.MSELoss()  # Mean Squared Error for regression tasks

# sdf_NN.to(device)
# geo_encoder.to(device)
# print(sdf_NN.nn_layers[-1].bias)
# for epoch in range(6):
#     for g_data in train_loader:
#         data = g_data[0].to(device)
#         optimizer.zero_grad()
#         params = geo_encoder(data)
#         sdf_pred = sdf_NN(grid_coor, params)
#         loss = criterion(sdf_pred, data.view(-1, grid_coor.shape[0]))
#         loss.backward()

#         optimizer.step()

#         # Print loss every 10 epochs

#     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# print(sdf_NN.nn_layers[-1].bias)

# %%
trainer.load_weights(checkpoint_fnames, device)
h = trainer.load_logs(filebase)
print(sdf_NN.nn_layers[-1].bias)
# h = trainer.fit(train_loader, val_loader=test_loader,
#                 epochs=1000, callbacks=checkpoint, print_freq=1)
# trainer.save_logs(filebase)
print(sdf_NN.nn_layers[-1].bias)
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
test_data = test_data[0].to(device)
para_test = geo_encoder(test_data)
sd_pred = trainer.models[1](grid_coor, para_test).cpu().detach().numpy()
sd_true = test_data.view(-1, grid_coor.shape[0]).cpu().numpy()
sd_pred = sd_pred*sdf_scale+sdf_shift
sd_true = sd_true*sdf_scale+sdf_shift
error_s = np.linalg.norm(sd_pred-sd_true, axis=1) / \
    np.linalg.norm(sd_true, axis=1)

fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(error_s, bins=20)


# %%
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
            ax.plot(contour[:, 1], contour[:, 0], '--b',
                    linewidth=2, label="Predicted")
        else:
            ax.plot(contour[:, 1], contour[:, 0], '--b', linewidth=2)

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.tight_layout()

# %%
