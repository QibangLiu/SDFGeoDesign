# %%
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
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
import nn_modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely.geometry import Polygon
import argparse
from typing import List, Optional, Tuple, Union
from models.modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
from models.modules.point_encoding import PointSetEmbedding, SimplePerceiver
from models.modules.transformer import Transformer
from models.modules.params_proj import ChannelsParamsProj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


def models_configs(*args, **kwargs):
    data_file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset"
    data_file = f"{data_file_base}/pc_sdf_ss_12-92_shift4_0-10000_aug.pkl"
    data_params = {"data_file": data_file,
                   "test_size": 0.2, "seed": 42}
    # GeoEncoder parameters
    out_c = 128
    latent_d = 128
    geo_encoder_file_base = f"./saved_models/geoencoder_outc{out_c}_latentdim{latent_d}_nothanh"
    geo_encoder_model_params = {
        "out_c": out_c,
        "latent_d": latent_d,
        "width": 128,
        "n_point": 128,
        "n_sample": 8,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 3,
    }
    geo_encoder_params = {
        "model_params": geo_encoder_model_params,
        "filebase": geo_encoder_file_base,
        # "data_params": data_params
    }

    img_shape = (1, out_c, latent_d)
    channel_mutipliers = [1, 2, 4, 8]
    has_attention = [False, False, True, True]
    first_conv_channels = 8
    num_res_blocks = 1
    # Forward model parameters
    fwd_model_params = {"img_shape": img_shape,
                        "channel_mutipliers": channel_mutipliers,
                        "has_attention": has_attention,
                        "first_conv_channels": first_conv_channels,
                        "num_res_blocks": num_res_blocks}
    fwd_filebase = f"./saved_models/fwdmodel_outc{out_c}_latentdim{latent_d}"
    fwd_params = {"model_params": fwd_model_params, "filebase": fwd_filebase}

    params_all = {"GeoEncoder": geo_encoder_params, "ForwardModel": fwd_params}

    return params_all


# %%
# load data
data_file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset"
data_file = f"{data_file_base}/pc_sdf_ss_12-92_shift4_0-10000_aug.pkl"


def LoadData(data_file=data_file, test_size=0.2, seed=42):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    sdf = data["sdf"].astype(np.float32)
    stress = data["stress"].astype(np.float32)
    point_cloud = data["points_cloud"]
    x_grids = data['x_grids'].astype(np.float32)
    y_grids = data['y_grids'].astype(np.float32)

    sdf = sdf.reshape(-1, 120 * 120)
    sdf_shift, sdf_scale = np.mean(sdf), np.std(sdf)
    sdf_norm = (sdf - sdf_shift) / sdf_scale
    # sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)
    stress_shift, stress_scale = np.mean(stress), np.std(stress)
    stress_norm = (stress - stress_shift) / stress_scale

    point_cloud = [torch.tensor(x[:, :2], dtype=torch.float32)
                   for x in point_cloud]  # (Nb,N,3)->(Nb, N, 2)
    point_cloud = pad_sequence(point_cloud, batch_first=True, padding_value=0)
    sdf_norm = torch.tensor(sdf_norm)
    stress_norm = torch.tensor(stress_norm)
    sdf_train, sdf_test, stress_train, stress_test, pc_train, pc_test = train_test_split(
        sdf_norm, stress_norm, point_cloud, test_size=test_size, random_state=seed
    )
    train_dataset = TensorDataset(
        pc_train, sdf_train, stress_train)
    test_dataset = TensorDataset(pc_test, sdf_test, stress_test)
    grid_coor = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
    grid_coor = torch.tensor(grid_coor)

    def y_inv_trans(y):
       return y * stress_scale + stress_shift

    def x_inv_trans(x):
        return (x * sdf_scale + sdf_shift)

    sdf_inv_scaler = x_inv_trans
    stress_inv_scaler = y_inv_trans

    return train_dataset, test_dataset, grid_coor, sdf_inv_scaler, stress_inv_scaler

# %%
# point cloud encoder


class PointCloudPerceiverChannelsEncoder(nn.Module):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(self,
                 input_channels: int = 2,
                 out_c: int = 128,
                 width: int = 128,
                 latent_d: int = 128,
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
            latent_d (int): number of context points
            n_point (int): number of points in the point set embedding
            n_sample (int): number of samples in the point set embedding
            radius (float): radius for the point set embedding
            patch_size//2 (int): padding size of dim 1 of conv in the point set embedding
            padding_mode (str): padding mode of the conv in the point set embedding
            d_hidden (list): hidden dimensions for the conv in the point set embedding
            fps_method (str): method for point sampling in the point set embedding, 'fps' or 'first', 'fps' has issue
            out_c (int): output channels
            final out shape: [B, out_c*latent_d]
        """
        self.width = width
        self.latent_d = latent_d
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
            nn.Parameter(torch.randn(self.latent_d, self.width)),
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
            points (torch.Tensor): [B, N, C]
                   C =2 or 3, or >3 if has other features
        Returns:
            torch.Tensor: [B, out_c*latent_d]
        """
        # B,N,C1]-> [B,C1,N]
        xyz = points.permute(0, 2, 1)
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
            0).repeat(batch_size, 1, 1)  # [B, latent_d, width]
        # [B, n_point+latent_d, width]
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.n_point +
                           self.latent_d, self.width)
        # [B, n_point+latent_d, width] -> [B,  n_point+latent_d, width]
        h = self.encoder(h, dataset_emb)
        h = self.processor(h)
        # [B,  n_point+latent_d, width] -> [B, latent_d, width]
        # -> [B, latent_d, out_c]
        h = self.output_proj(self.ln_post(h[:, -self.latent_d:]))
        # h = nn.Tanh()(h)  # project to [-1,1]
        # h = h.view(batch_size, -1)
        return h  # [B, latent_d, out_c]


class implicit_sdf(nn.Module):
    def __init__(self, latent_d=64, ndim=2, emd_version="nerf"):

        super().__init__()
        self.params_pre_name = 'projed_mlp_'
        self.emd_version = emd_version
        d = position_encoding_channels(emd_version)
        weight_shapes = [torch.Size([latent_d//4, d*ndim]), torch.Size(
            [latent_d//4, latent_d//4]),  torch.Size([latent_d//4, latent_d//4]), torch.Size([latent_d//4, latent_d//4])]

        self.param_shapes = {}

        for i, v in enumerate(weight_shapes):
            self.param_shapes[self.params_pre_name+str(i)+'_weight'] = v
            self.register_parameter(
                self.params_pre_name+str(i)+'_bias', nn.Parameter(torch.randn(v[0])))

        self.latent_d = latent_d
        learned_scale = 0.0625
        use_ln = True
        self.params_proj = ChannelsParamsProj(
            device=device,
            param_shapes=self.param_shapes,
            d_latent=self.latent_d,
            learned_scale=learned_scale,
            use_ln=use_ln,
        )
        l1 = nn.Linear(latent_d//4, 100)  # , bias=False
        l2 = nn.Linear(100, 100)
        l3 = nn.Linear(100, 1)
        self.nn_layers = nn.ModuleList([l1, nn.SiLU(), l2, nn.SiLU(), l3])

    def forward(self, x, latent):
        """
        Args:
            x (torch.Tensor): [N, C], query points
            latent (torch.Tensor): [B, latent_d, C], latent vectors
        """
        x = x[None].repeat(latent.shape[0], 1, 1)
        x = encode_position('nerf', position=x)
        # latent = latent.view(latent.shape[0], self.d_latent, -1)
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


def GeoEncoderModelDefinition(out_c=128, latent_d=128,
                              width=128, n_point=128,
                              n_sample=8, radius=0.2,
                              d_hidden=[128, 128],
                              num_heads=4, cross_attn_layers=1,
                              self_attn_layers=3):

    geo_encoder = PointCloudPerceiverChannelsEncoder(
        input_channels=2, out_c=out_c, width=width,
        latent_d=latent_d, n_point=n_point, n_sample=n_sample,
        radius=radius, d_hidden=d_hidden,
        num_heads=num_heads, cross_attn_layers=cross_attn_layers,
        self_attn_layers=self_attn_layers)
    sdf_NN = implicit_sdf(latent_d=latent_d)
    print("Total number of parameters of encoder: ", sum(p.numel()
                                                         for p in geo_encoder.parameters()))
    print("Total number of parameters of sdf_MLP: ", sum(p.numel()
                                                         for p in sdf_NN.parameters()))
    return geo_encoder, sdf_NN


def EvaluateGeoEncoderModel(trainer, test_loader, grid_coor, sdf_inv_scaler):
    sd_pred, sd_true = trainer.predict(test_loader, grid_coor)
    sd_pred = sdf_inv_scaler(sd_pred)
    sd_true = sdf_inv_scaler(sd_true)
    error_s = np.linalg.norm(sd_pred-sd_true, axis=1) / \
        np.linalg.norm(sd_true, axis=1)
    mean, std = np.mean(error_s), np.std(error_s)
    print(f"Mean L2 error of SDF: {mean}, std: {std}")
    sort_idx = np.argsort(error_s)
    min_index = sort_idx[0]
    max_index = sort_idx[int(len(sort_idx)*0.97)]
    median_index = sort_idx[len(sort_idx) // 2]
    print("Index for minimum geo:", min_index,
          "with error", error_s[min_index])
    print("Index for 97 percentile geo:", max_index,
          "with error", error_s[max_index])
    print("Index for median geo:", median_index,
          "with error", error_s[median_index])


def TrainGeoEncoderModel(geo_encoder, sdf_NN, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, grid_coor, sdf_inv_scaler, stress_inv_scaler = LoadData(
        seed=42)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    grid_coor = grid_coor.to(device)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            SDF = data[1].to(self.device)
            params = self.models[0](pc)
            sdf_pred = self.models[1](grid_coor, params)
            loss = nn.MSELoss()(sdf_pred, SDF)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data, grid_coor):
            sd_pred = []
            sd_true = []
            self.models[0].eval()
            self.models[1].eval()
            with torch.no_grad():
                for data in data:
                    pc = data[0].to(self.device)
                    SDF = data[1].to(self.device)
                    params = self.models[0](pc)
                    sdf_pred = self.models[1](grid_coor, params)
                    sd_pred.append(sdf_pred.cpu().detach().numpy())
                    sd_true.append(SDF.cpu().detach().numpy())
            sd_pred = np.vstack(sd_pred)
            sd_true = np.vstack(sd_true)
            return sd_pred, sd_true

    trainer = TRAINER(
        {"encoder": geo_encoder, "sdf_NN": sdf_NN}, device, filebase)

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
    EvaluateGeoEncoderModel(trainer, test_loader, grid_coor, sdf_inv_scaler)
    return trainer


def LoadGeoEncoderModel(file_base, model_params):
    geo_encoder, sdf_NN = GeoEncoderModelDefinition(**model_params)
    geo_encoder_path = os.path.join(file_base, "encoder", "model.ckpt")
    sdf_NN_path = os.path.join(file_base, "sdf_NN", "model.ckpt")
    for model, path in zip([geo_encoder, sdf_NN], [geo_encoder_path, sdf_NN_path]):
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    return geo_encoder, sdf_NN

# %%


class ForwardModel(nn.Module):
    def __init__(
        self,
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_out=51,
        num_res_blocks=1,
        norm_groups=8,
    ):
        super().__init__()
        self.unet = nn_modules.UNet(
            img_shape,
            first_conv_channels,
            channel_mutipliers,
            has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
        )
        in_channels = channel_mutipliers[0] * first_conv_channels

        self.cov = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=2, padding=1)
        sz = [50, 50, 50, 50]
        self.mlp = nn.ModuleList()
        in_sz = img_shape[1]*img_shape[2]//4
        for i in range(len(sz)):
            self.mlp.append(nn.Linear(in_sz, sz[i]))
            self.mlp.append(nn.BatchNorm1d(sz[i]))
            self.mlp.append(nn.SiLU())
            in_sz = sz[i]
        self.mlp.append(nn.Linear(in_sz, num_out))
        # self.mlp.append(nn.ReLU())

    def forward(self, x):
        x = self.unet(x)
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        for layer in self.mlp:
            x = layer(x)
        return x

    def predict(self, data, geo_encoder):
        # TODO: add predict function
        pass


def ForwardModelDefinition(img_shape=(1, 128, 128),
                           channel_mutipliers=[1, 2, 4, 8],
                           has_attention=[False, False, True, True],
                           first_conv_channels=8, num_res_blocks=1):
    num_out = 51
    fwd_model = ForwardModel(
        img_shape,
        first_conv_channels,
        channel_mutipliers,
        has_attention,
        num_out=num_out,
        num_res_blocks=num_res_blocks,
        norm_groups=None,
    )
    trainable_params = sum(p.numel()
                           for p in fwd_model.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of fwd model: {trainable_params}")
    return fwd_model


# %%


def LoadForwardModel(model_path, model_params):
    fwd_model = ForwardModelDefinition(**model_params)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    fwd_model.load_state_dict(state_dict)
    fwd_model.to(device)
    fwd_model.eval()
    return fwd_model


def TrainForwardModel(fwd_model, geo_encoder, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, _, _, stress_inv_scaler = LoadData(
        seed=42)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    geo_encoder = geo_encoder.to(device)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase, geo_encoder):
            super().__init__(models, device, filebase)
            self.geo_encoder = geo_encoder
            for param in geo_encoder.parameters():
                param.requires_grad = False

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            y_true = data[2].to(self.device)
            latent = self.geo_encoder(pc)
            latent = latent[:, None, :, :]
            y_pred = self.models[0](latent)

            loss = nn.MSELoss()(y_true, y_pred)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data_loader):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    inputs = data[0].to(self.device)  # pc,sdf,stress
                    latent = self.geo_encoder(inputs)
                    pred = self.models[0](latent)
                    pred = pred.cpu().detach().numpy()
                    y_pred.append(pred)
                    y_true.append(data[2].cpu().detach().numpy())
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            return y_pred, y_true

    trainer = TRAINER(
        fwd_model, device, filebase, geo_encoder)
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
                    epochs=epochs, print_freq=1)
    trainer.save_logs()

    EvaluateForwardModel(trainer, test_loader, stress_inv_scaler)
    return trainer


def EvaluateForwardModel(trainer, test_loader, stress_inv_scal, plot_flag=False):
    trainer.load_weights(device=device)
    s_pred, s_true = trainer.predict(test_loader)
    s_pred = stress_inv_scal(s_pred)
    s_true = stress_inv_scal(s_true)
    error_s = np.linalg.norm(s_pred-s_true, axis=1) / \
        np.linalg.norm(s_true, axis=1)

    sort_idx = np.argsort(error_s)
    idx_best = sort_idx[0]
    idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
    idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
    idx_95perc = sort_idx[int(len(sort_idx)*0.95)]
    index_list = [idx_best, idx_32perc, idx_63perc, idx_95perc]
    labels = ["Best", "32th percentile", "63th percentile", "95th percentile"]
    for label, idx in zip(labels, index_list):
        print(f"{label} L2 error: {error_s[idx]}")

    print(f"Mean L2 error: {np.mean(error_s)}, std: {np.std(error_s)}")
    if plot_flag:
        nr, nc = 1, len(index_list)
        fig = plt.figure(figsize=(nc*4.8, nr*3.6))
        strain = np.linspace(0, 0.2, 51)
        for i, index in enumerate(index_list):

            ax = plt.subplot(nr, nc, i+1)
            s_pred_i = s_pred[index]
            s_true_i = s_true[index]
            ax.plot(strain*100, s_true_i, 'r', label="True")
            ax.plot(strain*100, s_pred_i, '--b', label="Pred.")
            ax.legend()
            ax.set_xlabel(r"$\varepsilon~[\%]$")
            ax.set_ylabel(r"$\sigma~[MPa]$")
            ax.set_title(f"{labels[i]}")

            plt.tight_layout()


# %%
"""Diffusion Inverse model """
diffu_inv_filebase = "/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/saved_models/inv_sig12-92_aug10000"
diffu_model_path = f"{diffu_inv_filebase}/model.ckpt"


def DiffusionInverseModelDefinition():
    img_shape = (1, 120, 120)
    channel_multpliers = [1, 2, 4, 8]
    has_attention = [False, False, False, False]
    num_heads = 4
    num_res_blocks = 1
    norm_groups = 16
    fist_conv_channels = 32
    label_dim = 51
    inv_Unet = nn_modules.UNetTimeStep(
        img_shape=img_shape,
        label_dim=label_dim,
        one_hot=False,
        first_conv_channels=fist_conv_channels,
        channel_mutipliers=channel_multpliers,
        has_attention=has_attention,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
    )
    trainable_params = sum(p.numel()
                           for p in inv_Unet.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of inverse Unet: {trainable_params}")

    total_timesteps = 500
    gaussian_diffusion = nn_modules.GaussianDiffusion(
        timesteps=total_timesteps)

    return inv_Unet, gaussian_diffusion


def TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, filebase, train_flag, train_loader, test_loader, epochs=300, lr=5e-4):
    total_timesteps = gaussian_diffusion.timesteps

    class TorchTrainer(torch_trainer.TorchTrainer):
        def __init__(self, model, device, filebase):
            super().__init__(model, device, filebase)

        def evaluate_losses(self, data):
            '''custom loss'''
            images, labels = data[0].to(self.device), data[1].to(self.device)
            batch_size = images.shape[0]
            # random generate mask
            z_uncound = torch.rand(batch_size)
            batch_mask = (z_uncound > 0.2).int().to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, total_timesteps, (batch_size,),
                              device=device).long()

            loss = gaussian_diffusion.train_losses(
                self.models[0], images, t, labels, batch_mask
            )
            loss_tracker = {"loss": loss.item()}
            return loss, loss_tracker

    trainer = TorchTrainer(inv_Unet, device, filebase)

    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="loss", save_best_only=True)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=40)
    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="loss",
    )

    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(
        train_loader, val_loader=None, epochs=epochs, print_freq=1
    )
    trainer.save_logs()
    trainer.load_weights(device=device)

    return trainer


def LoadDiffusionInverseModel(model_path=diffu_model_path):
    inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    inv_Unet.load_state_dict(state_dict)
    inv_Unet.to(device)
    inv_Unet.eval()
    return inv_Unet, gaussian_diffusion


def EvaluateDiffusionInverseModel(fwd_model, inv_Unet, gaussian_diffusion, Ytarget, sdf_inv_scaler, stress_inv_scaler, num_sol=10, plot_flag=False):
    Ytarget = Ytarget.to(device)
    labels = Ytarget.repeat(num_sol, 1)
    Xpred = gaussian_diffusion.sample(
        inv_Unet, (1, 120, 120), labels, w=2, clip_denoised=False, conditioning=True
    )
    Xpred = torch.tensor(Xpred).to(device)
    with torch.no_grad():
        Ypred = fwd_model(Xpred)
    Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
    Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
    Xpred_inv = sdf_inv_scaler(Xpred.cpu().detach().numpy())
    L2error = np.linalg.norm(Ypred_inv - Ytarg_inv, axis=1) / \
        np.linalg.norm(Ytarg_inv, axis=1)
    sorted_idx = np.argsort(L2error)
    evl_ids = np.array([
        sorted_idx[0],
        sorted_idx[int(len(sorted_idx) * 0.33)],
        sorted_idx[int(len(sorted_idx) * 0.66)],
        sorted_idx[-1],
    ], dtype=int)
    for i, idx in enumerate(evl_ids):
        print(f"ID: {idx}, L2 error: {L2error[idx]}")

    if plot_flag:
        strain = np.linspace(0, 0.2, 51)
        legends = ["best", "33\%", "66\%", "worst"]
        fig = plt.figure(figsize=(4.8 * 5, 3.6))
        ax = plt.subplot(1, 5, 1)
        ax.plot(strain, Ytarg_inv[0], label="target")
        for i, v in enumerate(evl_ids):
            ax.plot(strain, Ypred_inv[v], label=legends[i])
        ax.legend()
        ax.set_xlabel(r"$\varepsilon~[\%]$")
        ax.set_ylabel(r"$\sigma~[MPa]$")
        for i, v in enumerate(evl_ids):
            contours = measure.find_contours(
                Xpred_inv[v], 0, positive_orientation="high")
            ax = plt.subplot(1, 5, i + 2)
            l_style = ["r-", "b--"]
            holes = []
            for j, contour in enumerate(contours):
                contour = (contour - 10) / 100
                x, y = contour[:, 1], contour[:, 0]
                if j == 0:
                    ax.fill(
                        x,
                        y,
                        alpha=1.0,
                        edgecolor="black",
                        facecolor="cyan",
                        label="Outer Boundary",
                    )
                else:
                    ax.fill(x, y, alpha=1.0, edgecolor="black",
                            facecolor="white", label="Hole")
                # ax.grid(True)
                ax.axis("off")
                ax.axis("equal")  # Keep aspect ratio square


# %%
"""UcVAE inverse model"""
ucvae_inv_filebase = "/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/saved_models/ucvae_sig12-92_aug10000"
ucvae_decoder_path = f"{ucvae_inv_filebase}/decoder/model.ckpt"
latent_dim = 10


class Encoder(nn.Module):
    def __init__(self, latent_dim, img_shape,
                 first_conv_channels,
                 channel_mutipliers,
                 has_attention,
                 num_res_blocks,
                 norm_groups,):
        super().__init__()

        self.unet = nn_modules.UNet(
            img_shape,
            first_conv_channels,
            channel_mutipliers,
            has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
        )
        in_channels = channel_mutipliers[0] * first_conv_channels
        self.cov = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=2, padding=1)
        sz = [50, 50, 50, 50]
        self.mlp = nn.ModuleList()
        in_sz = img_shape[1]*img_shape[2]//4
        for i in range(len(sz)):
            self.mlp.append(nn.Linear(in_sz, sz[i]))
            self.mlp.append(nn.BatchNorm1d(sz[i]))
            self.mlp.append(nn.SiLU())
            in_sz = sz[i]
        self.mean_layer = nn.Linear(in_sz, latent_dim)
        self.logvar_layer = nn.Linear(in_sz, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.unet(x)
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        for layer in self.mlp:
            x = layer(x)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, c_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(latent_dim+c_dim, hidden_dim))
        self.net.append(nn.BatchNorm1d(hidden_dim))
        self.net.append(nn.SiLU())
        for i in range(3):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.BatchNorm1d(hidden_dim))
            self.net.append(nn.SiLU())
        self.net.append(nn.Linear(hidden_dim, 30*30))
        self.net.append(nn.BatchNorm1d(30*30))
        self.net.append(nn.SiLU())
        self.net.append(nn.Unflatten(1, (1, 30, 30)))
        self.net.append(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.BatchNorm2d(64))
        self.net.append(nn.SiLU())
        self.net.append(nn_modules.UpSample(64))
        self.net.append(nn.BatchNorm2d(64))
        self.net.append(nn.SiLU())
        self.net.append(nn_modules.UpSample(64))
        self.net.append(nn.BatchNorm2d(64))
        self.net.append(nn.SiLU())
        self.net.append(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        for layer in self.net:
            x = layer(x)

        return x.view(x.size(0), 1, 120, 120)


def EncoderDefinition():
    img_shape = (1, 120, 120)
    first_conv_channels = 8
    channel_mutipliers = [1, 2, 4, 8]
    has_attention = [False, False, True, True]
    num_res_blocks = 1
    norm_groups = None
    encoder = Encoder(latent_dim, img_shape,
                      first_conv_channels, channel_mutipliers, has_attention, num_res_blocks, norm_groups)
    trainable_params = sum(p.numel()
                           for p in encoder.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of Encoder: {trainable_params}")
    return encoder


def DecoderDefinition():
    c_dim = 51
    decoder = Decoder(latent_dim, c_dim)
    trainable_params = sum(p.numel()
                           for p in decoder.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of Decoder: {trainable_params}")
    return decoder


def TrainUcVAEInverseModel(encoder, decoder, filebase, train_flag, train_loader, test_loader, epochs=300, lr=5e-4):
    class TorchTrainer(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            z, mu, logvar = self.models[0](images)
            recon = self.models[1](z, labels)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            recon_loss = F.mse_loss(recon, images)
            loss = kl_loss+recon_loss
            loss_dic = {"loss": loss.item(), "kl_loss": kl_loss.item(),
                        "recon_loss": recon_loss.item()}
            return loss, loss_dic

    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="loss", save_best_only=True)

    trainer = TorchTrainer(
        {"encoder": encoder, "decoder": decoder}, device, filebase)
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "params": {"factor": 0.7, "patience": 40},
        "metric_name": "loss",
    }

    trainer.compile(
        optimizer=torch.optim.Adam,
        lr=lr,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
    )

    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(
        train_loader, val_loader=None, epochs=epochs, callbacks=checkpoint, print_freq=1
    )
    trainer.save_logs()
    trainer.load_weights(device=device)

    return trainer


def LoadUcVAEDecoder(model_path=ucvae_decoder_path):
    decoder = DecoderDefinition()
    state_dict = torch.load(model_path, map_location=device)
    decoder.load_state_dict(state_dict)
    decoder.to(device)
    decoder.eval()
    return decoder


def EvaluateUcVAEInverseModel(fwd_model, decoder, Ytarget, sdf_inv_scaler, stress_inv_scaler, num_sol=10, plot_flag=False):
    Ytarget = Ytarget.to(device)
    labels = Ytarget.repeat(num_sol, 1)
    Z = torch.randn(num_sol, latent_dim).to(device)
    Xpred = decoder(Z, labels)
    with torch.no_grad():
        Ypred = fwd_model(Xpred)

    Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
    Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
    Xpred_inv = sdf_inv_scaler(Xpred.cpu().detach().numpy())
    L2error = np.linalg.norm(Ypred_inv - Ytarg_inv, axis=1) / \
        np.linalg.norm(Ytarg_inv, axis=1)
    sorted_idx = np.argsort(L2error)
    evl_ids = np.array([
        sorted_idx[0],
        sorted_idx[int(len(sorted_idx) * 0.33)],
        sorted_idx[int(len(sorted_idx) * 0.66)],
        sorted_idx[-1],
    ], dtype=int)
    for i, idx in enumerate(evl_ids):
        print(f"ID: {idx}, L2 error: {L2error[idx]}")

    if plot_flag:
        strain = np.linspace(0, 0.2, 51)
        legends = ["best", "33\%", "66\%", "worst"]
        fig = plt.figure(figsize=(4.8 * 5, 3.6))
        ax = plt.subplot(1, 5, 1)
        ax.plot(strain, Ytarg_inv[0], label="target")
        for i, v in enumerate(evl_ids):
            ax.plot(strain, Ypred_inv[v], label=legends[i])
        ax.legend()
        ax.set_xlabel(r"$\varepsilon~[\%]$")
        ax.set_ylabel(r"$\sigma~[MPa]$")
        for i, v in enumerate(evl_ids):
            contours = measure.find_contours(
                Xpred_inv[v], 0, positive_orientation="high")
            ax = plt.subplot(1, 5, i + 2)
            l_style = ["r-", "b--"]
            holes = []
            for j, contour in enumerate(contours):
                contour = (contour - 10) / 100
                x, y = contour[:, 1], contour[:, 0]
                if j == 0:
                    ax.fill(
                        x,
                        y,
                        alpha=1.0,
                        edgecolor="black",
                        facecolor="cyan",
                        label="Outer Boundary",
                    )
                else:
                    ax.fill(x, y, alpha=1.0, edgecolor="black",
                            facecolor="white", label="Hole")
                # ax.grid(True)
                ax.axis("off")
                ax.axis("equal")  # Keep aspect ratio square


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training arguments")
    parser.add_argument(
        "--model", type=str, default="forward")
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    configs = models_configs()
    if args.model == "geoencoder":
        filebase = configs["GeoEncoder"]["filebase"]
        print(filebase)
        model_params = configs["GeoEncoder"]["model_params"]
        geo_encoder, sdf_NN = GeoEncoderModelDefinition(**model_params)  # TODO
        trainer = TrainGeoEncoderModel(geo_encoder, sdf_NN, filebase, args.train_flag,
                                       epochs=args.epochs, lr=args.learning_rate)
    elif args.model == "forward":
        filebase = configs["ForwardModel"]["filebase"]
        model_params = configs["ForwardModel"]["model_params"]

        geo_encoder_filebase = configs["GeoEncoder"]["filebase"]
        geo_encoder_model_params = configs["GeoEncoder"]["model_params"]

        fwd_model = ForwardModelDefinition(**model_params)
        geo_encoder, _ = LoadGeoEncoderModel(
            geo_encoder_filebase, geo_encoder_model_params)
        trainer = TrainForwardModel(fwd_model, geo_encoder, filebase, args.train_flag,
                                    epochs=args.epochs, lr=args.learning_rate)

    # elif args.model == "forward":
    #     filebase = fwd_filebase
    #     fwd_model = ForwardModelDefinition()
    #     trainer = TrainForwardModel(fwd_model, filebase, args.train_flag, train_loader,
    #                                 test_loader, epochs=args.epochs, lr=args.learning_rate)
    #     EvaluateForwardModel(trainer, test_loader,
    #                          sdf_inv_scaler, stress_inv_scaler)
    # elif args.model == "diffusion":
    #     filebase = diffu_inv_filebase
    #     inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition()
    #     trainer = TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, filebase, args.train_flag,
    #                                          train_loader, test_loader, epochs=args.epochs, lr=args.learning_rate)
    #     fwd_model = LoadForwardModel()
    #     id = 23
    #     Ytarget = test_dataset[id][1].unsqueeze(0)
    #     EvaluateDiffusionInverseModel(
    #         fwd_model, inv_Unet, gaussian_diffusion, Ytarget, sdf_inv_scaler, stress_inv_scaler)
    # elif args.model == "ucvae":
    #     filebase = ucvae_inv_filebase
    #     encoder = EncoderDefinition()
    #     decoder = DecoderDefinition()
    #     trainer = TrainUcVAEInverseModel(encoder, decoder, filebase, args.train_flag,
    #                                      train_loader, test_loader, epochs=args.epochs, lr=args.learning_rate)
    #     fwd_model = LoadForwardModel()
    #     id = 23
    #     Ytarget = test_dataset[id][1].unsqueeze(0)
    #     EvaluateUcVAEInverseModel(
    #         fwd_model, decoder, Ytarget, sdf_inv_scaler, stress_inv_scaler)

    print(filebase, " training finished")

# %%
