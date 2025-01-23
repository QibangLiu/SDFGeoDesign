
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


def models_configs(out_c=128, latent_d=128, *args, **kwargs):
    data_file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset"
    data_file = f"{data_file_base}/pc_sdf_ss_12-92_shift4_0-10000_aug.pkl"
    data_params = {"data_file": data_file,
                   "test_size": 0.2, "seed": 42}
    """************GeoEncoder parameters************"""


    geo_encoder_file_base = f"{script_path}/saved_weights/geoencoder_outc{out_c}_latentdim{latent_d}"
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
    """************Forward model parameters************"""
    img_shape = (1, out_c, latent_d)
    channel_mutipliers = [1, 2, 4, 8]
    has_attention = [False, False, False, False]
    first_conv_channels = 8
    num_res_blocks = 1
    # Forward model parameters

    if "forward_from_pc" in kwargs and kwargs["forward_from_pc"] == True:
        fwd_filebase = f"{script_path}/saved_weights/fwd_fromPC_outc{out_c}_latentdim{latent_d}_noatt"
    else:
        fwd_filebase = f"{script_path}/saved_weights/fwd_outc{out_c}_latentdim{latent_d}_noatt"

    fwd_model_params = {"img_shape": img_shape,
                        "channel_mutipliers": channel_mutipliers,
                        "has_attention": has_attention,
                        "first_conv_channels": first_conv_channels,
                        "num_res_blocks": num_res_blocks}
    fwd_params = {"model_params": fwd_model_params, "filebase": fwd_filebase}

    """************Inverse diffusion model parameters************"""
    channel_multpliers = [1, 2, 4, 8]
    has_attention = [False, False, True, True]
    fist_conv_channels = 32
    num_heads = 4
    norm_groups = 16
    num_res_blocks = 1
    total_timesteps = 500
    inv_diffusion_filebase = f"{script_path}/saved_weights/inv_diffusion_outc{out_c}_latentdim{latent_d}"
    inv_diffusion_model_params = {"img_shape": img_shape, "channel_multpliers": channel_multpliers,
                                  "has_attention": has_attention, "fist_conv_channels": fist_conv_channels,
                                  "num_heads": num_heads, "norm_groups": norm_groups, "num_res_blocks": num_res_blocks,
                                  "total_timesteps": total_timesteps}
    inv_diffusion_params = {
        "model_params": inv_diffusion_model_params, "filebase": inv_diffusion_filebase}

    params_all = {"GeoEncoder": geo_encoder_params,
                  "ForwardModel": fwd_params, "InvDiffusion": inv_diffusion_params}
    return params_all
