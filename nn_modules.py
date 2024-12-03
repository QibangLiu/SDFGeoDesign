# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
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
