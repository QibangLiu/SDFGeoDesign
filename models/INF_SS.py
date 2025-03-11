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
    from .configs import InfSS_configs, LoadDataInfSS
    from .modules.params_proj import ChannelsParamsProj
    from .modules.UNets import UNet
    from .modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from .trainer import torch_trainer
else:
    from configs import InfSS_configs, LoadDataInfSS
    from modules.params_proj import ChannelsParamsProj
    from modules.UNets import UNet
    from modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from trainer import torch_trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

class ImplicitNeuralFunction(nn.Module):
    def __init__(self,encoder, latent_d=120, in_c=120, ndim=1, d_hidden_sdfnn=[256, 128], emd_version="nerf"):

        super().__init__()
        self.encoder=encoder
        self.params_pre_name = 'projed_mlp_'
        self.emd_version = emd_version
        d = position_encoding_channels(emd_version)
        # the input latent vector is of shape (B, latent_d, in_c)
        # desired weight shapes [(v1,c1), (v2,c2), (v3,c3), (v4,c4),...], v is the output channel, c is the input channel
        # v1+v2+v3+...+vn = latent_d
        weight_shapes = [torch.Size([latent_d//4, d*ndim]), torch.Size(
            [latent_d//4, latent_d//4]),  torch.Size([latent_d//4, latent_d//4]), torch.Size([latent_d//4, latent_d//4])]

        self.param_shapes = {}

        for i, v in enumerate(weight_shapes):
            self.param_shapes[self.params_pre_name+str(i)+'_weight'] = v
            self.register_parameter(
                self.params_pre_name+str(i)+'_bias', nn.Parameter(torch.randn(v[0])))

        self.in_c = in_c
        # for projects the input latent vector to the weights of the MLP
        # inputs vector of shape (B, latent_d, in_c)
        # inputs: (B,v_i,in_c)
        # Linear layer weights: (v_i,c_i,in_c)
        # -> (B,v_i,c_i) weights for INF
        self.params_proj = ChannelsParamsProj(
            device=device,
            param_shapes=self.param_shapes,
            d_latent=self.in_c,
            use_ln=True,
        )
        self.nn_layers = nn.ModuleList()
        c_i = weight_shapes[-1][0]
        for ls in d_hidden_sdfnn:
            self.nn_layers.append(nn.Linear(c_i, ls))
            self.nn_layers.append(nn.SiLU())
            c_i = ls
        self.nn_layers.append(nn.Linear(c_i, 1))

    def forward(self, x, sdf):
        """
        Args:
            x (torch.Tensor): [N, 2], query points,
            latent (torch.Tensor): [B, latent_d, C], latent vectors
        """
        latent=self.encoder(sdf)
        # (N,2)--> (N,62)
        x = encode_position('nerf', position=x)
        # (1,N,62)->(B,N,62)
        x = x[None].repeat(latent.shape[0], 1, 1)
        # latent = latent.view(latent.shape[0], self.d_latent, -1)
        proj_params = self.params_proj(latent)
        for i, kw in enumerate(proj_params.keys()):
            w = proj_params[kw]
            b = getattr(self, self.params_pre_name+str(i)+'_bias')
            # (B,P,I)-> (B,P,O)
            x = torch.einsum("bpi,boi->bpo", x, w)
            x = torch.add(x, b)
            x = F.silu(x)

        for layer in self.nn_layers:
            x = layer(x)
        # (B,N,1)->(B,N)
        return x.squeeze()


class Encoder(nn.Module):
    def __init__(
        self,
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        out_c=120,
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
        self.conv=nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1)


    def forward(self, normalized_sdf):
        x = self.unet(normalized_sdf)
        x=self.conv(x)
        x=x.permute(0,2,3,1)
        x=x.mean(dim=2)
        return x


def INFModelDefinition(img_shape=(1, 120, 120),
                           channel_mutipliers=[1, 2, 4, 8],
                           has_attention=[False, False, True, True],
                           first_conv_channels=8, num_res_blocks=1,
                           norm_groups=8, dropout=None,out_c=128,
                           latent_d=120, in_c=120, ndim=1,
                              d_hidden_sdfnn=[128, 128],
                              emd_version="nerf"):
    """out_c is equal to in_c"""
    EncoderModel = Encoder(
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        dropout=dropout,
        out_c=out_c
    )
    INFSSModel = ImplicitNeuralFunction(encoder=EncoderModel,
        latent_d=latent_d, in_c=in_c, ndim=ndim, d_hidden_sdfnn=d_hidden_sdfnn, emd_version=emd_version)
    tot_num_params = sum(p.numel() for p in EncoderModel.parameters())
    trainable_params = sum(p.numel()
                           for p in EncoderModel.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of Encoder: {tot_num_params}, {trainable_params} of which are trainable")
    tot_num_params = sum(p.numel() for p in INFSSModel.parameters())
    trainable_params = sum(p.numel()
                           for p in INFSSModel.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of INF for strain-stress curve: {tot_num_params}, {trainable_params} of which are trainable")
    return INFSSModel


# %%

def EvaluateGeoEncoderModel(trainer,train_loader, test_loader, strain):
    trainer.load_weights(device=device)
    def cal_l2_error(test_loader):
        y_pred, y_true = trainer.predict(test_loader,strain)
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

    train_dataset, test_dataset,strain, sdf_inv_scaler, stress_inv_scaler=LoadDataInfSS()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    strain=strain.to(device)
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

        def predict(self, data,strain):
            sd_pred = []
            sd_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data:
                    sdf = data[0].to(self.device)
                    s_true = data[1].to(self.device)
                    s_pred = self.models[0](strain, sdf)
                    s_true=stress_inv_scaler(s_true)
                    s_pred=stress_inv_scaler(s_pred)
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
        optimizer, factor=0.7, patience=40)

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
    EvaluateGeoEncoderModel(trainer,train_loader, test_loader, strain)
    return trainer


def LoadINFModel(filebase, model_args):
    inf_model = INFModelDefinition(**model_args)
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
    configs = InfSS_configs()
    filebase = configs["filebase"]
    model_args = configs["model_args"]
    print(f"\n\INFSS Filebase: {filebase}, model_args:")
    print(model_args)
    infss = INFModelDefinition(**model_args)
    trainer = TrainGeoEncoderModel(infss, filebase, args.train_flag,
                                   epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")

# %%
