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
    from .configs import NOTSU_configs, LoadDataSU
    from .modules.UNets import UNet
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from .modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from .trainer import torch_trainer
else:
    from configs import NOTSU_configs, LoadDataSU
    from modules.UNets import UNet
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from trainer import torch_trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

class Decoder(nn.Module):
    def __init__(self, encoder, embed_dim=64, cross_attn_layers=4, num_heads=4,
                 in_channels=3, out_channels=3,
                 dropout=0.0, emd_version="nerf", padding_value=None,num_frames=26):
        super().__init__()
        self.num_frames=num_frames
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
                                         nn.Linear(4*embed_dim, out_channels*self.num_frames)
                                         )

    def forward(self, xyt, sdf):
        B,N,ndim=xyt.shape
        # (B, latenc, embed_dim)
        latent = self.encoder(sdf)
        # (B,N,ndim)->(B,N,ndim*31)
        xyt = encode_position('nerf', position=xyt)
        # (B,N,ndim*31)->(B,N,embed_dim)
        x = self.Q_encoder(xyt)
        for block in self.resblocks:
            x = block(x, latent)  # (B, N*Nt, embed_dim)
        # (B, N*Nt, embed_dim)->(B, N*Nt, 3)
        x = self.output_proj(x)
        x=x.view(B,N,self.num_frames,-1)
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
        self.conv_layers=nn.ModuleList()
        for i in range(3):
            self.conv_layers.append(nn.Conv2d(in_channels, embed_dim, kernel_size=3,stride=2, padding=1))
            self.conv_layers.append(nn.SiLU())
            in_channels=embed_dim



    def forward(self, normalized_sdf):
        x = self.unet(normalized_sdf) # (B,1,120,120)->(B,1,120,120)
        # (B,1,120,120)->(B,embed_dim,15,15)
        for layer in self.conv_layers:
            x=layer(x)
        # (B,embed_dim,15,15)->(B,embed_dim,225)
        x = x.view(x.size(0), x.size(1), -1)
        # (B,embed_dim,225)->(B,225,embed_dim)
        x = x.permute(0, 2, 1)
        return x


def NOTModelDefinition(img_shape=(1, 120, 120),
                           channel_mutipliers=[1, 2, 4, 8],
                           has_attention=[False, False, True, True],
                           first_conv_channels=8, num_res_blocks=1,
                           norm_groups=8, dropout=None,embed_dim=128,
                           cross_attn_layers=3,num_heads=8, in_channels=3, out_channels=3,
                              emd_version="nerf",
                              padding_value=None,num_frames=26):
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
                           cross_attn_layers=cross_attn_layers,num_heads=num_heads, in_channels=in_channels, out_channels=out_channels,
                              emd_version=emd_version,
                              padding_value=padding_value,num_frames=num_frames)
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

def EvaluateNOTModel(trainer,train_loader, test_loader):
    trainer.load_weights(device=device)
    def cal_l2_error(test_loader):
        y_pred, y_true = trainer.predict(test_loader)
        error_s = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:,:, 0], y_t[:,:, 0]
            ux_p, ux_t = y_p[:,:, 1], y_t[:,:, 1]
            uy_p, uy_t = y_p[:,:, 2], y_t[:,:, 2]
            e_s = np.linalg.norm(s_p-s_t)/np.linalg.norm(s_t)
            e_ux = np.linalg.norm(ux_p-ux_t)/np.linalg.norm(ux_t)
            e_uy = np.linalg.norm(uy_p-uy_t)/np.linalg.norm(uy_t)
            error_s.append((e_s+e_ux+e_uy)/3)
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

    # error_s = cal_l2_error(train_loader)
    # print(
    #     f"Mean L2 error for training data: {np.mean(error_s)}, std: {np.std(error_s)}")


def TrainNOTModel(infss, filebase, train_flag, epochs=300, lr=1e-3,window_size=None):

    train_dataloader, test_dataloader, sdf_inverse, su_inverse=LoadDataSU(bs_train=64,bs_test=128,input_T=False)
    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            sdf = data[0].to(self.device)
            xyt = data[1].to(self.device)
            y_true = data[2].to(self.device)
            y_pred = self.models[0](xyt, sdf)
            mask = (y_true != self.models[0].padding_value).float()
            loss = nn.MSELoss(reduction='none')(y_true, y_pred)
            loss = (loss*mask).sum()/(mask.sum()+1)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data:
                    sdf = data[0].to(self.device)
                    xyt = data[1].to(self.device)
                    y_true_batch = data[2].to(self.device)
                    mask = (y_true_batch != self.models[0].padding_value)
                    pred = self.models[0](xyt, sdf)
                    pred = su_inverse(pred)
                    y_true_batch = su_inverse(y_true_batch)
                    pred = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy() for x, i in zip(pred, mask)]
                    y_true_batch = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                                    for x, i in zip(y_true_batch, mask)]

                    y_pred = y_pred+pred
                    y_true = y_true+y_true_batch
            return y_pred, y_true

    trainer = TRAINER(infss, device, filebase)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="val_loss", save_best_only=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=10)

    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="val_loss",
        window_size=window_size,
    )

    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_dataloader, val_loader=test_dataloader,
                    epochs=epochs)

    trainer.load_weights(device=device)
    trainer.save_logs()
    EvaluateNOTModel(trainer,train_dataloader, test_dataloader)
    return trainer


def LoadNOTModel(file_base, model_args):
    not_model = NOTModelDefinition(**model_args)
    model_path = os.path.join(file_base, "model.ckpt")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    not_model.load_state_dict(state_dict)
    not_model.to(device)
    not_model.eval()
    return not_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start", help="start or continue, or any other string")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=None)
    args, unknown = parser.parse_known_args()
    print(vars(args))
    configs = NOTSU_configs(input_T=False)
    filebase = configs["filebase"]
    model_args = configs["model_args"]
    print(f"\n NOTSU Filebase: {filebase}, model_args:")
    print(model_args)
    not_model = NOTModelDefinition(**model_args)
    trainer = TrainNOTModel(not_model, filebase, args.train_flag,
                                   epochs=args.epochs, lr=args.learning_rate,
                                   window_size=args.window_size)
    print(filebase, " training finished")

# %%

# not_model=not_model.to(device)
# xyt=torch.randn(2,2000,3).to(device)
# sdf=torch.randn(2,1,120,120).to(device)
# out=not_model(xyt,sdf)
# out.shape
# %%
