# %%
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math
import numpy as np
from tqdm import tqdm

# %%


def norm_layer(channel_in, groups=None):
    if groups is None:
        return nn.BatchNorm2d(channel_in)
    else:
        return nn.GroupNorm(groups, channel_in)


class AttentionBlock(nn.Module):
    """Applies self-attention.
    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=None):
        super(AttentionBlock, self).__init__()
        self.units = units  # number of channels
        self.norm = norm_layer(units, groups)
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

        attn_score = nnF.softmax(attn_score, dim=-1)  # [b, h, w, hw]
        attn_score = attn_score.view(
            batch_size, height, width, height, width
        )  # [b, h, w, h, w]

        # equivalent: [hw X hw] * [hw X c]
        proj = torch.einsum("bhwHW,bHWc->bhwc", attn_score, v)  # [b, h, w, c]
        proj = self.proj(proj)  # [b, h, w, c]
        # [b, h, w, c] -> [b, c, h, w]
        output = (inputs + proj).permute(0, 3, 1, 2)
        return output


# %%


class ResidualBlockConv(nn.Module):
    """Residual block.

    Args:
        channel: Number of channels in the convolutional layers
        groups: Number of groups to be used for GroupNormalization layer
        activation_fn: Activation function to be used
    """

    def __init__(
        self, channel_in, channel_out, groups=8, activation_fn=nn.SiLU(), dropout=None
    ):
        super(ResidualBlockConv, self).__init__()

        self.activation_fn = activation_fn
        self.net = nn.ModuleList()
        self.norm = norm_layer(channel_in, groups)
        self.net.append(activation_fn)
        if dropout is not None:
            self.net.append(nn.Dropout2d(p=dropout))
        self.net.append(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding="same")
        )
        if groups is None:
            self.net.append(nn.BatchNorm2d(channel_out))
        else:
            self.net.append(nn.GroupNorm(groups, channel_out))
        self.net.append(activation_fn)
        if dropout is not None:
            self.net.append(nn.Dropout2d(p=dropout))
        self.net.append(
            nn.Conv2d(channel_out, channel_out, kernel_size=3, padding="same")
        )
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
    def __init__(
        self,
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention=None,
        num_res_blocks=1,
        norm_groups=None,
        interpolation="nearest",
        activation_fn=nn.SiLU(),
        dropout=None,
    ):
        super().__init__()
        channel_in, img_size = img_shape[0], img_shape[1]
        channel_list = [
            first_conv_channels * channel_mutiplier
            for channel_mutiplier in channel_mutipliers
        ]
        if has_attention is None:
            has_attention = [False] * len(channel_list)

        if len(has_attention) != len(channel_list):
            raise ValueError(
                "has_attention should have the same length as channel_mutipliers"
            )

        self.activation_fn = activation_fn
        self.skip_channels = []
        encoder_skip_idx = []
        self.first_conv = nn.Conv2d(
            channel_in, first_conv_channels, kernel_size=3, padding=1
        )

        self.encoder = nn.ModuleList()
        for i in range(len(channel_list)):
            if i == 0:
                in_channel = first_conv_channels
            else:
                in_channel = channel_list[i - 1]
            self.encoder.append(
                ResidualBlockConv(
                    in_channel,
                    channel_list[i],
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            if has_attention[i]:
                self.encoder.append(AttentionBlock(
                    channel_list[i], groups=norm_groups))
            self.skip_channels.append(channel_list[i])
            encoder_skip_idx.append(len(self.encoder) - 1)
            for _ in range(1, num_res_blocks):
                self.encoder.append(
                    ResidualBlockConv(
                        channel_list[i],
                        channel_list[i],
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                if has_attention[i]:
                    self.encoder.append(
                        AttentionBlock(channel_list[i], groups=norm_groups)
                    )
                self.skip_channels.append(channel_list[i])
                encoder_skip_idx.append(len(self.encoder) - 1)
                # skip connection
            if i != len(channel_list) - 1:
                self.encoder.append(DownSample(channel_list[i]))

        self.encoder_skip = torch.zeros(len(self.encoder), dtype=bool)
        self.encoder_skip[encoder_skip_idx] = True

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(
            ResidualBlockConv(
                channel_list[-1],
                channel_list[-1],
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )
        self.bottleneck.append(AttentionBlock(
            channel_list[-1], groups=norm_groups))
        self.bottleneck.append(
            ResidualBlockConv(
                channel_list[-1],
                channel_list[-1],
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )

        self.decoder = nn.ModuleList()
        decoder_skip_idx = []
        for i in reversed(range(len(channel_list))):
            if i == len(channel_list) - 1:
                in_channel = channel_list[-1] + self.skip_channels.pop()
            else:
                in_channel = channel_list[i + 1] + self.skip_channels.pop()
            self.decoder.append(
                ResidualBlockConv(
                    in_channel,
                    channel_list[i],
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            decoder_skip_idx.append(len(self.decoder) - 1)
            if has_attention[i]:
                self.decoder.append(AttentionBlock(
                    channel_list[i], groups=norm_groups))
            for _ in range(1, num_res_blocks):
                in_channel = channel_list[i] + self.skip_channels.pop()
                self.decoder.append(
                    ResidualBlockConv(
                        in_channel,
                        channel_list[i],
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                decoder_skip_idx.append(len(self.decoder) - 1)
                if has_attention[i]:
                    self.decoder.append(
                        AttentionBlock(channel_list[i], groups=norm_groups)
                    )

            if i != 0:
                self.decoder.append(
                    UpSample(channel_list[i], interpolation=interpolation)
                )

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


# %%


class ResidualBlockConvTimeStep(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        time_channels,
        label_emb_dim,
        groups=None,
        activation_fn=nn.SiLU(),
        dropout=None,
    ):
        super().__init__()
        lay_list1 = [norm_layer(channel_in, groups), activation_fn]
        if dropout is not None:
            lay_list1.append(nn.Dropout(dropout))
        lay_list1.append(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding="same")
        )
        self.conv1 = nn.Sequential(*lay_list1)

        self.time_emb = nn.Sequential(
            activation_fn, nn.Linear(time_channels, channel_out)
        )

        self.label_emb = nn.Sequential(
            activation_fn, nn.Linear(label_emb_dim, channel_out)
        )

        lay_list2 = [norm_layer(channel_out, groups), activation_fn]
        if dropout is not None:
            lay_list2.append(nn.Dropout(dropout))
        lay_list2.append(
            nn.Conv2d(channel_out, channel_out, kernel_size=3, padding="same")
        )
        self.conv2 = nn.Sequential(*lay_list2)

        if channel_in != channel_out:
            self.shortcut = nn.Conv2d(channel_in, channel_out, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t, c, mask):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        `c` has shape `[batch_size, label_dim]`
        `mask` has shape `[batch_size, ]`
        """
        h = self.conv1(x)
        emb_t = self.time_emb(t)
        emb_c = self.label_emb(c) * mask[:, None]
        h += emb_t[:, :, None, None] + emb_c[:, :, None, None]
        h = self.conv2(h)

        return h + self.shortcut(x)


# %%
# Multi Attention block


class MultiAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, groups=None):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels, groups)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# %%
def timestep_embedding(timesteps, embed_dim, max_period=10000.0):
    """
    use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # position = timesteps[:, None].float()  # [N, 1]
    # div_term = torch.exp(torch.arange(0, embed_dim, 2) *
    #                      (-math.log(max_period) / embed_dim))
    # # Apply sine to even indices and cosine to odd indices
    # embedding = torch.zeros(len(position), embed_dim)
    # embedding[:, 0::2] = torch.sin(position * div_term)
    # if embed_dim % 2 != 0:
    #     embedding[:, 1::2] = torch.cos(position * div_term[:-1]
    #                                    )  # Handle odd embed_dim
    # else:
    #     embedding[:, 1::2] = torch.cos(position * div_term)  # Regular case
    # return embedding
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# %%


class UNetTimeStep(nn.Module):

    def __init__(
        self,
        img_shape,
        label_dim,
        one_hot,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_heads=4,
        num_res_blocks=1,
        norm_groups=None,
        interpolation="nearest",
        activation_fn=nn.SiLU(),
        dropout=None,
    ):
        super().__init__()
        channel_in, img_size = img_shape[0], img_shape[1]
        channel_list = [
            first_conv_channels * channel_mutiplier
            for channel_mutiplier in channel_mutipliers
        ]
        if has_attention is None:
            has_attention = [False] * len(channel_list)

        if len(has_attention) != len(channel_list):
            raise ValueError(
                "has_attention should have the same length as channel_mutipliers"
            )

        self.activation_fn = activation_fn
        self.skip_channels = []
        encoder_skip_idx = []
        self.first_conv_channels = first_conv_channels
        time_emb_dim = first_conv_channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(first_conv_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # labels embedding
        label_emb_dim = first_conv_channels
        if one_hot:
            self.label_emb = nn.Embedding(label_dim, label_emb_dim)
        else:
            self.label_emb = nn.Sequential(
                nn.Linear(label_dim, 2*label_emb_dim),
                activation_fn,
                nn.Linear(2*label_emb_dim, 2*label_emb_dim),
                activation_fn,
                nn.Linear(2*label_emb_dim, 2*label_emb_dim),
                activation_fn,
                nn.Linear(2*label_emb_dim, label_emb_dim),
            )

        self.first_conv = nn.Conv2d(
            channel_in, first_conv_channels, kernel_size=3, padding=1
        )

        self.encoder = nn.ModuleList()

        for i in range(len(channel_list)):
            if i == 0:
                in_channel = first_conv_channels
            else:
                in_channel = channel_list[i - 1]
            self.encoder.append(
                ResidualBlockConvTimeStep(
                    in_channel,
                    channel_list[i],
                    time_emb_dim,
                    label_emb_dim,
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            if has_attention[i]:
                self.encoder.append(
                    MultiAttentionBlock(
                        channel_list[i], num_heads=num_heads, groups=norm_groups
                    )
                )
            self.skip_channels.append(channel_list[i])
            encoder_skip_idx.append(len(self.encoder) - 1)

            for _ in range(1, num_res_blocks):
                self.encoder.append(
                    ResidualBlockConvTimeStep(
                        channel_list[i],
                        channel_list[i],
                        time_emb_dim,
                        label_emb_dim,
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                if has_attention[i]:
                    self.encoder.append(
                        MultiAttentionBlock(
                            channel_list[i], num_heads=num_heads, groups=norm_groups
                        )
                    )
                self.skip_channels.append(channel_list[i])
                encoder_skip_idx.append(len(self.encoder) - 1)
                # skip connection
            if i != len(channel_list) - 1:
                self.encoder.append(DownSample(channel_list[i]))

        self.encoder_skip = torch.zeros(len(self.encoder), dtype=bool)
        self.encoder_skip[encoder_skip_idx] = True

        self.bottleneck = nn.ModuleList()

        self.bottleneck.append(
            ResidualBlockConvTimeStep(
                channel_list[-1],
                channel_list[-1],
                time_emb_dim,
                label_emb_dim,
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )
        self.bottleneck.append(
            MultiAttentionBlock(
                channel_list[-1], num_heads=num_heads, groups=norm_groups
            )
        )
        self.bottleneck.append(
            ResidualBlockConvTimeStep(
                channel_list[-1],
                channel_list[-1],
                time_emb_dim,
                label_emb_dim,
                groups=norm_groups,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        )

        self.decoder = nn.ModuleList()
        decoder_skip_idx = []
        for i in reversed(range(len(channel_list))):
            if i == len(channel_list) - 1:
                in_channel = channel_list[-1] + self.skip_channels.pop()
            else:
                in_channel = channel_list[i + 1] + self.skip_channels.pop()
            self.decoder.append(
                ResidualBlockConvTimeStep(
                    in_channel,
                    channel_list[i],
                    time_emb_dim,
                    label_emb_dim,
                    groups=norm_groups,
                    activation_fn=activation_fn,
                    dropout=dropout,
                )
            )
            decoder_skip_idx.append(len(self.decoder) - 1)
            if has_attention[i]:
                self.decoder.append(
                    MultiAttentionBlock(
                        channel_list[i], num_heads=num_heads, groups=norm_groups
                    )
                )
            for _ in range(1, num_res_blocks):
                in_channel = channel_list[i] + self.skip_channels.pop()
                self.decoder.append(
                    ResidualBlockConvTimeStep(
                        in_channel,
                        channel_list[i],
                        time_emb_dim,
                        label_emb_dim,
                        groups=norm_groups,
                        activation_fn=activation_fn,
                        dropout=dropout,
                    )
                )
                decoder_skip_idx.append(len(self.decoder) - 1)
                if has_attention[i]:
                    self.decoder.append(
                        MultiAttentionBlock(
                            channel_list[i], num_heads=num_heads, groups=norm_groups
                        )
                    )

            if i != 0:
                self.decoder.append(
                    UpSample(channel_list[i], interpolation=interpolation)
                )

        self.decoder_skip = torch.zeros(len(self.decoder), dtype=bool)
        self.decoder_skip[decoder_skip_idx] = True

        self.out = nn.Sequential(
            norm_layer(channel_list[0], norm_groups),
            nn.SiLU(),
            nn.Conv2d(channel_list[0], channel_in, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, c, mask):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c: a 1-D batch of classes.
        :param mask: a 1-D batch of conditioned/unconditioned.
        :return: an [N x C x ...] Tensor of outputs.
        """
        skips = []
        x = self.first_conv(x)

        # QB: N x 1 -> N x 4*model_channels
        t_emb = self.time_emb(
            timestep_embedding(timesteps, embed_dim=self.first_conv_channels)
        )
        c_emb = self.label_emb(c)  # N x num -> N x model_channels

        for eskip, layer in zip(self.encoder_skip, self.encoder):
            if isinstance(layer, ResidualBlockConvTimeStep):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
            if eskip:
                skips.append(x)

        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlockConvTimeStep):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)

        for dskip, layer in zip(self.decoder_skip, self.decoder):
            if dskip:
                x = torch.cat([x, skips.pop()], dim=1)
            if isinstance(layer, ResidualBlockConvTimeStep):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)

        return self.out(x)


# %%
"""Classifier free guidance Diffusion Model"""


def beta_scheduler(tot_timesteps, type="linear", s=0.008):
    if type == "linear":
        scale = 1000 / tot_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, tot_timesteps)
    elif type == "sigmoid":
        betas = torch.linspace(-6, 6, tot_timesteps)
        betas = (
            torch.sigmoid(betas)
            / (betas.max() - betas.min())
            * (0.02 - betas.min())
            / 10
        )
        return betas
    elif type == "cosine":
        """
        cosine schedule
        as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = tot_timesteps + 1
        x = torch.linspace(0, tot_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = (
            torch.cos(((x / tot_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"unknown beta scheduler {type}")


# %%


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule="linear"):
        self.timesteps = timesteps

        self.betas = beta_scheduler(timesteps, type=beta_schedule)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nnF.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        """
        t is time step in [0, total_time_step], shape: (batch_size, )
        take the value of a at t
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion (using the nice property): q(x_t | x_0)
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise
        t: (batch_size, )
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        # (total_timesteps), (nb, ) -> (nb, 1,...)
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        """
        q(x_t | x_0);
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise;
        mu=sqrt(\bar{alpha_t}) * x_0;
        sigma^2=1-\bar{alpha_t};
        """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        diffusion posterior: q(x_{t-1} | x_t, x_0)
        mu=coe1 * x_0 + coe2 * x_t;
        coe1=beta_t * sqrt(\bar{\alpha_{t-1}}) / (1 - \bar{\alpha_t});
        coe2=(1 - \bar{\alpha_{t-1}}) * sqrt(\bar{\alpha_t}) / (1 - \bar{\alpha_t});
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #
    def predict_start_from_noise(self, x_t, t, noise):
        """
        compute x_0 from x_t and noise
        reverse of `q_sample`:
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    #
    def p_mean_variance(
        self, model, x_t, t, c, w, clip_denoised=True, conditioning=True
    ):
        """
        compute predicted mean and variance of p(x_{t-1} | x_t)

        noise is from the model

        x0 is from the inverse of the q_sample, q(x_t | x_0) -->x0,
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise


        then diffusion posterior: q(x_{t-1} | x_t, x_0) to get the mean and variance

        """
        device = next(model.parameters()).device
        batch_size = x_t.shape[0]
        # predict noise using model

        pred_noise_none = model(x_t, t, c, torch.zeros(batch_size).int().to(device))
        if conditioning:
            pred_noise_c = model(x_t, t, c, torch.ones(batch_size).int().to(device))
            pred_noise = (1 + w) * pred_noise_c - w * pred_noise_none
        else:
            pred_noise = pred_noise_none

        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(x_recon, x_t, t)
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, c, w, clip_denoised=True, conditioning=True):
        """
        denoise_step: sample x_{t-1} from x_t and pred_noise:
        1. compute predicted mean and variance of p(x_{t-1} | x_t):
            a. noise is from the model
            b. x0 is from the inverse of the q_sample, q(x_t | x_0) -->x0,
               x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise
            c. then diffusion posterior: q(x_{t-1} | x_t, x_0) to get the mean and variance
        2. sample x_{t-1} from the predicted mean and variance
        """
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, c, w, clip_denoised=clip_denoised, conditioning=conditioning
        )
        noise = torch.randn_like(x_t)
        # no noise when t == 0 QB: to be look into???
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        image_shape,
        labels,
        w=2,
        clip_denoised=True,
        conditioning=True,
        timesteps=None,
    ):
        if timesteps is None:
            timesteps = self.timesteps

        batch_size = labels.shape[0]
        device = next(model.parameters()).device
        labels = labels.to(device)
        # start from pure noise (for each example in the batch)
        img = torch.randn((batch_size, *image_shape), device=device)
        imgs = []
        for i in tqdm(
            reversed(range(0, timesteps)),
            desc="sampling loop time step",
            total=timesteps,
        ):
            img = self.p_sample(
                model,
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                labels,
                w,
                clip_denoised,
                conditioning=conditioning,
            )
            imgs.append(img.cpu().numpy())

        return np.array(imgs)

    # sample new images
    @torch.no_grad()
    def sample(
        self,
        model,
        image_shape,
        labels,
        w=2,
        clip_denoised=True,
        conditioning=True,
        timesteps=None,
    ):
        return self.p_sample_loop(
            model,
            image_shape,
            labels,
            w,
            clip_denoised,
            conditioning,
            timesteps=timesteps,
        )

    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_shape,
        labels,
        ddim_timesteps=50,
        w=2,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        conditioning=True,
    ):
        # make ddim timestep sequence
        if ddim_discr_method == "uniform":
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == "quad":
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * 0.8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{ddim_discr_method}"'
            )
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(model.parameters()).device

        labels = labels.to(device)
        batch_size = labels.shape[0]
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, *image_shape), device=device)
        seq_img = [sample_img.cpu().numpy()]

        for i in tqdm(
            reversed(range(0, ddim_timesteps)),
            desc="sampling loop time step",
            total=ddim_timesteps,
        ):
            t = torch.full(
                (batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long
            )
            prev_t = torch.full(
                (batch_size,),
                ddim_timestep_prev_seq[i],
                device=device,
                dtype=torch.long,
            )

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(
                self.alphas_cumprod, prev_t, sample_img.shape
            )

            # 2. predict noise using model

            pred_noise_none = model(
                sample_img, t, labels, torch.zeros(
                    batch_size, dtype=torch.int32).to(device)
            )
            if conditioning:
                pred_noise_c = model(
                    sample_img, t, labels, torch.ones(
                        batch_size, dtype=torch.int32).to(device)
                )
                pred_noise = (1 + w) * pred_noise_c - w * pred_noise_none
            else:
                pred_noise = pred_noise_none
            # 3. get the predicted x_0
            pred_x0 = (
                sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise
            ) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = (
                torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            )

            # 6. compute x_{t-1} of formula (12)
            x_prev = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir_xt
                + sigmas_t * torch.randn_like(sample_img)
            )

            sample_img = x_prev

        return sample_img.cpu().numpy()

    # compute train losses
    def train_losses(self, model, x_start, t, c, mask_c):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, c, mask_c)
        loss = nnF.mse_loss(noise, predicted_noise)
        return loss
