# %%
from typing import List, Optional, Tuple, Union
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
if __package__:
    from .configs import NOTSS_configs, LoadDataSS
    from .modules.params_proj import ChannelsParamsProj
    from .modules.UNets import UNet
    from .modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from .modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from .trainer import torch_trainer
else:
    from configs import NOTSS_configs, LoadDataSS
    from modules.params_proj import ChannelsParamsProj
    from modules.UNets import UNet
    from modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from trainer import torch_trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%


class Decoder(nn.Module):
    def __init__(self, encoder, embed_dim=64, cross_attn_layers=4, num_heads=4,
                 in_channels=1, out_channels=1,
                 dropout=0.0, emd_version="nerf", padding_value=-10):
        super().__init__()
        self.padding_value = padding_value
        d = position_encoding_channels(emd_version)
        # self.Q_encoder = MLP(embed_dim, in_channels)
        self.Q_encoder = nn.Sequential(nn.Linear(d*in_channels, 3*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(3*embed_dim, 4*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(4*embed_dim, 2*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(2*embed_dim, embed_dim)
                                       )
        self.encoder = encoder
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=embed_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(cross_attn_layers)
            ]
        )
        self.output_proj = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(2*embed_dim, 4*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(4*embed_dim, 4*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(4*embed_dim, 4*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(4*embed_dim, out_channels)
                                         )

    def forward(self, xyt, sdf):
        B = len(sdf)
        # (B, latenc, embed_dim)
        latent = self.encoder(sdf)
        # (B,Nt,ndim)->(B,Nt,ndim*31)
        xyt = encode_position('nerf', position=xyt)
        # (B,N,ndim*31)->(B,Nt,embed_dim)
        x = self.Q_encoder(xyt)
        x = x[None, :, :].repeat(B, 1, 1)
        for block in self.resblocks:
            x = block(x, latent)  # (B, Nt, embed_dim)
        # (B, Nt, embed_dim)->(B, Nt, 1)
        x = self.output_proj(x)
        x = x.squeeze(-1)  # (B, Nt)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        embed_dim=64,
        num_res_blocks=1,
        norm_groups=8,
        dropout=None
    ):
        super().__init__()

        self.unet = UNet(
            img_shape,
            first_conv_channels,
            channel_mutipliers,
            has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
            dropout=dropout
        )
        in_channels = channel_mutipliers[0] * first_conv_channels
        self.conv_layers = nn.ModuleList()
        for i in range(3):
            self.conv_layers.append(
                nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1))
            self.conv_layers.append(nn.SiLU())
            in_channels = embed_dim

    def forward(self, normalized_sdf):
        x = self.unet(normalized_sdf)  # (B,1,120,120)->(B,1,120,120)
        # (B,1,120,120)->(B,embed_dim,15,15)
        for layer in self.conv_layers:
            x = layer(x)
        # (B,embed_dim,15,15)->(B,embed_dim,225)
        x = x.view(x.size(0), x.size(1), -1)
        # (B,embed_dim,225)->(B,225,embed_dim)
        x = x.permute(0, 2, 1)
        return x


def NOTModelDefinition(img_shape=(1, 120, 120),
                       channel_mutipliers=[1, 2, 4, 8],
                       has_attention=[False, False, True, True],
                       first_conv_channels=8, num_res_blocks=1,
                       norm_groups=8, dropout=None, embed_dim=128,
                       cross_attn_layers=3, num_heads=8, in_channels=1, out_channels=1,
                       emd_version="nerf",
                       padding_value=None):
    encoder = Encoder(
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        dropout=dropout,
        embed_dim=embed_dim
    )
    NOTModel = Decoder(encoder=encoder,
                       embed_dim=embed_dim,
                       cross_attn_layers=cross_attn_layers, num_heads=num_heads, in_channels=in_channels, out_channels=out_channels,
                       emd_version=emd_version,
                       padding_value=padding_value)
    tot_num_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel()
                           for p in encoder.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of encoder: {tot_num_params}, {trainable_params} of which are trainable")

    tot_num_params = sum(p.numel() for p in NOTModel.parameters())
    trainable_params = sum(p.numel()
                           for p in NOTModel.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of NOT model: {tot_num_params}, {trainable_params} of which are trainable")
    return NOTModel


# %%

def EvaluateNOTModel(trainer, train_loader, test_loader, strain):
    trainer.load_weights(device=device)

    def cal_l2_error(test_loader):
        y_pred, y_true = trainer.predict(test_loader, strain)
        error_s = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = np.linalg.norm(s_p-s_t)/np.linalg.norm(s_t)
            error_s.append(e_s)
        error_s = np.array(error_s)
        return error_s

    error_s = cal_l2_error(test_loader)
    sort_idx = np.argsort(error_s)
    idx_best = sort_idx[0]
    idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
    idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
    idx_99perc = sort_idx[int(len(sort_idx))-1]
    index_list = [idx_best, idx_32perc, idx_63perc, idx_99perc]
    labels = ["Best", "32th percentile", "63th percentile", "100th percentile"]
    for label, idx in zip(labels, index_list):
        print(f"{label} L2 error: {error_s[idx]}")

    print(
        f"Mean L2 error for test data: {np.mean(error_s)}, std: {np.std(error_s)}")

    error_s = cal_l2_error(train_loader)
    print(
        f"Mean L2 error for training data: {np.mean(error_s)}, std: {np.std(error_s)}")


def TrainGeoEncoderModel(infss, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, strain, sdf_inv_scaler, stress_inv_scaler = LoadDataSS()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    strain = strain.to(device)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            sdf = data[0].to(self.device)
            s_true = data[1].to(self.device)
            s_pred = self.models[0](strain, sdf)
            loss = nn.MSELoss()(s_true, s_pred)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data, strain):
            sd_pred = []
            sd_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data:
                    sdf = data[0].to(self.device)
                    s_true = data[1].to(self.device)
                    s_pred = self.models[0](strain, sdf)
                    s_true = stress_inv_scaler(s_true)
                    s_pred = stress_inv_scaler(s_pred)
                    sd_pred.append(s_pred.cpu().detach().numpy())
                    sd_true.append(s_true.cpu().detach().numpy())
            sd_pred = np.vstack(sd_pred)
            sd_true = np.vstack(sd_true)
            return sd_pred, sd_true

    trainer = TRAINER(infss, device, filebase)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="val_loss", save_best_only=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=20)

    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="val_loss",
    )

    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_loader, val_loader=test_loader,
                    epochs=epochs)

    trainer.load_weights(device=device)
    trainer.save_logs()
    EvaluateNOTModel(trainer, train_loader, test_loader, strain)
    return trainer


def LoadNOTModel(filebase, model_args):
    inf_model = NOTModelDefinition(**model_args)
    model_path = os.path.join(filebase, "model.ckpt")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    inf_model.load_state_dict(state_dict)
    inf_model.to(device)
    inf_model.eval()
    return inf_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start", help="start or continue, or any other string")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))
    configs = NOTSS_configs()
    filebase = configs["filebase"]
    model_args = configs["model_args"]
    print(f"\n NOTSS Filebase: {filebase}, model_args:")
    print(model_args)
    infss = NOTModelDefinition(**model_args)
    trainer = TrainGeoEncoderModel(infss, filebase, args.train_flag,
                                   epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")

# %%
