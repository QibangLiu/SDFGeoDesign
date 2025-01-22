# %%
from modules.params_proj import ChannelsParamsProj
from modules.transformer import Transformer
from modules.point_encoding import PointSetEmbedding, SimplePerceiver
from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
from typing import List, Optional, Tuple, Union
import argparse
import trainer.torch_trainer as torch_trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from configs import models_configs, LoadData
# current_path = os.getcwd()
# print(current_path)
# script_directory = os.path.dirname(os.path.abspath(__file__))
# if current_path == script_directory:
#     os.chdir("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        h = nn.Tanh()(h)  # project to [-1,1]
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start", help="start or continue, or any other string")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs = models_configs()
    filebase = configs["GeoEncoder"]["filebase"]
    print(filebase)
    model_params = configs["GeoEncoder"]["model_params"]
    geo_encoder, sdf_NN = GeoEncoderModelDefinition(**model_params)  # TODO
    trainer = TrainGeoEncoderModel(geo_encoder, sdf_NN, filebase, args.train_flag,
                                   epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")
