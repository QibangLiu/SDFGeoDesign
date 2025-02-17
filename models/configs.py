
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
from sklearn.model_selection import train_test_split

# %%

script_path = os.path.dirname(os.path.abspath(__file__))

data_file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset/augmentation_split_intervel"
data_file = f"{data_file_base}/pc_sdf_ss_12-92_shift8_0-10000_aug.pkl"


POINTS_CLOUD_PADDING_VALUE = -10
NUM_POINT_POINTNET2 = 128
NX_GRID = 64


def GetGridPoints(nx):
    x_g = np.linspace(-0.1, 1.1, nx, dtype=np.float32)
    X_g, Y_g = np.meshgrid(x_g, x_g)
    grid_points = np.vstack([X_g.ravel(), Y_g.ravel()]).T
    return grid_points


def LoadData(data_file=data_file, test_size=0.2, seed=42, geoencoder=False):
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
    num_p = [x.shape[0] for x in point_cloud]
    if min(num_p) < NUM_POINT_POINTNET2:
        raise ValueError(
            f"Number of sample points {NUM_POINT_POINTNET2}\
            should be smaller than the minimum number of points in the point cloud {min(num_p)}")
    point_cloud = pad_sequence(
        point_cloud, batch_first=True, padding_value=POINTS_CLOUD_PADDING_VALUE)
    sdf_norm = torch.tensor(sdf_norm)
    stress_norm = torch.tensor(stress_norm)
    sdf_train, sdf_test, stress_train, stress_test, pc_train, pc_test = train_test_split(
        sdf_norm, stress_norm, point_cloud, test_size=test_size, random_state=seed
    )
    train_dataset = TensorDataset(
        pc_train, stress_train, sdf_train)
    test_dataset = TensorDataset(pc_test, stress_test, sdf_test)
    if geoencoder:
        grid_coor = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
    else:
        grid_coor = GetGridPoints(NX_GRID)
    # grid_coor = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
    grid_coor = torch.tensor(grid_coor)

    def y_inv_trans(y):
       return y * stress_scale + stress_shift

    def x_inv_trans(x):
        return (x * sdf_scale + sdf_shift)

    sdf_inv_scaler = x_inv_trans
    stress_inv_scaler = y_inv_trans

    return train_dataset, test_dataset, grid_coor, sdf_inv_scaler, stress_inv_scaler


# %%


def models_configs(out_c=256, latent_d=256, *args, **kwargs):
    """************GeoEncoder arguments************"""

    fps_method = "fps"
    geo_encoder_model_args = {
        "out_c": out_c,
        "latent_d": latent_d,
        "width": 128,
        "n_point": NUM_POINT_POINTNET2,
        "n_sample": 8,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 3,
        "pc_padding_val": POINTS_CLOUD_PADDING_VALUE,
        "d_hidden_sdfnn": [128, 128],
        "fps_method": fps_method,
    }
    geo_encoder_file_base = f"{script_path}/saved_weights/geoencoder_outc{out_c}_latentdim{latent_d}_fps{fps_method}"
    geo_encoder_args = {
        "model_args": geo_encoder_model_args,
        "filebase": geo_encoder_file_base,
    }
    """************Forward model arguments************"""
    img_shape = (1, out_c, latent_d)
    channel_mutipliers = [1, 2, 4, 8]
    has_attention = [False, False, True, True]
    first_conv_channels = 16
    num_res_blocks = 1
    norm_groups = None
    dropout = 0.1

    if "forward_from_pc" in kwargs and kwargs["forward_from_pc"] == True:
        fwd_filebase = f"{script_path}/saved_weights/fwd_fromPC_outc{out_c}_latentdim{latent_d}_noatt"
        fwd_img_shape = img_shape
    elif "forward_from_latent" in kwargs and kwargs["forward_from_latent"] == True:
        fwd_filebase = f"{script_path}/saved_weights/fwd_fromLatent_outc{out_c}_latentdim{latent_d}_noatt"
        fwd_img_shape = img_shape
    else:
        fwd_filebase = f"{script_path}/saved_weights/fwd_outc{out_c}_latentdim{latent_d}_noatt_normgroups-{norm_groups}_dropout-{dropout}_nx{NX_GRID}"
        fwd_img_shape = (1, NX_GRID, NX_GRID)

    fwd_model_args = {"img_shape": fwd_img_shape,
                        "channel_mutipliers": channel_mutipliers,
                        "has_attention": has_attention,
                        "first_conv_channels": first_conv_channels,
                        "num_res_blocks": num_res_blocks,
                        "norm_groups": norm_groups,
                        "dropout": dropout}
    fwd_args = {"model_args": fwd_model_args, "filebase": fwd_filebase}

    """************Inverse diffusion model arguments************"""
    channel_multpliers = [1, 2, 4, 8]
    fist_conv_channels = 16
    num_heads = 4
    norm_groups = 8
    num_res_blocks = 1
    total_timesteps = 500
    dropout = None
    if "diffusion_from_lattent" in kwargs and kwargs["diffusion_from_lattent"] == True:
        inv_img_shape = img_shape
        inv_diffusion_filebase = f"{script_path}/saved_weights/inv_diffusion_from_lattent_outc{out_c}_latentdim{latent_d}"
        has_attention = [False, False, False, True]

    else:
        inv_img_shape = (1, NX_GRID, NX_GRID)
        has_attention = [False, False, True, True]
        inv_diffusion_filebase = f"{script_path}/saved_weights/inv_diffusion_outc{out_c}_latentdim{latent_d}_dropout{dropout}_nx{NX_GRID}"
    inv_diffusion_model_args = {"img_shape": inv_img_shape, "channel_multpliers": channel_multpliers,
                                  "has_attention": has_attention, "fist_conv_channels": fist_conv_channels,
                                  "num_heads": num_heads, "norm_groups": norm_groups, "num_res_blocks": num_res_blocks,
                                  "total_timesteps": total_timesteps, "dropout": dropout}
    inv_diffusion_args = {
        "model_args": inv_diffusion_model_args, "filebase": inv_diffusion_filebase}

    args_all = {"GeoEncoder": geo_encoder_args,
                "ForwardModel": fwd_args, "InvDiffusion": inv_diffusion_args}
    return args_all
