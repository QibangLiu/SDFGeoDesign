
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Union
import os
import pickle
from sklearn.model_selection import train_test_split
import time
# %%

script_path = os.path.dirname(os.path.abspath(__file__))

data_filebase = f"{script_path}/../data/PeriodUnitCell"

PADDING_VALUE = -1000
# %%


class ListDataset(Dataset):
    """for list of tensors"""

    def __init__(self, data: Union[list, tuple]):
        """
        args:
            data: list of data, each element is a list of tensors
            e.g. [(pc1, xyt1, S1), (pc2, xyt2, S2), ...]

        """
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        one_data = [d[idx] for d in self.data]
        return one_data
# %%


def LoadDataSS(test_size=0.2, seed=42):
    SS_curve_file = f"{data_filebase}/SS_curve.npy"
    stress = np.load(SS_curve_file)
    sdf_file = f"{data_filebase}/sdf.npz"
    sdf = np.load(sdf_file)["sdf"]
    sdf_shift, sdf_scale = np.mean(sdf), np.std(sdf)
    sdf_norm = (sdf - sdf_shift) / sdf_scale
    sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)
    stress_shift, stress_scale = np.mean(stress), np.std(stress)
    stress_norm = (stress - stress_shift) / stress_scale

    def s_inv_trans(y):
       return y * stress_scale + stress_shift

    def sdf_inv_trans(x):
        return (x * sdf_scale + sdf_shift)

    sdf_norm = torch.tensor(sdf_norm)
    stress_norm = torch.tensor(stress_norm)
    sdf_train, sdf_test, stress_train, stress_test = train_test_split(
        sdf_norm, stress_norm, test_size=test_size, random_state=seed
    )
    train_dataset = TensorDataset(
        sdf_train, stress_train)
    test_dataset = TensorDataset(sdf_test, stress_test)
    sdf_inv_scaler = sdf_inv_trans
    stress_inv_scaler = s_inv_trans

    strain = torch.tensor(np.linspace(0, 1, 51), dtype=torch.float32)

    return train_dataset, test_dataset, strain[:, None], sdf_inv_scaler, stress_inv_scaler


def NOTSS_configs():
    file_base = f"{script_path}/saved_weights/NOTSS"
    notss_args = {"img_shape": (1, 120, 120),
                  "channel_mutipliers": [1, 2, 4, 8],
                  "has_attention": [False, False, False, False],
                  "first_conv_channels": 8, "num_res_blocks": 1,
                  "norm_groups": 8, "dropout": None, "embed_dim": 64,
                  "cross_attn_layers": 2, "num_heads": 8, "in_channels": 1, "out_channels": 1,
                  "emd_version": "nerf"}

    args_all = {"model_args": notss_args, "filebase": file_base}
    return args_all


# %%
NUM_FRAMES = 26


def LoadSUScaler():
    scaler_file = f"{data_filebase}/SU_scalers_{NUM_FRAMES}fram.npz"

    scalers = np.load(scaler_file)
    su_shift, su_scaler = scalers["global_shift"], scalers["global_scale"]
    su_shift = torch.tensor(su_shift)[None, :]  # (1, 1,1,3)
    su_scaler = torch.tensor(su_scaler)[None, :]  # (1, 1, 1, 3)

    def SUInverse(x):
        su_sig = su_scaler.to(x.device)
        su_mu = su_shift.to(x.device)
        return x*su_sig+su_mu
    return SUInverse


def LoadDataSU(bs_train=32, bs_test=128, test_size=0.2, seed=42, padding_value=PADDING_VALUE, input_T=True):
    start = time.time()

    scaler_file = f"{data_filebase}/SU_scalers_{NUM_FRAMES}fram.npz"
    su_scalers = np.load(scaler_file)
    su_shift, su_scaler = su_scalers["global_shift"], su_scalers["global_scale"]

    SU_file = f"{data_filebase}/mises_disp{NUM_FRAMES}fram.pkl"
    with open(SU_file, "rb") as f:
        SU = pickle.load(f)
    nodes_file = f"{data_filebase}/mesh_coords.pkl"
    with open(nodes_file, "rb") as f:
        nodes = pickle.load(f)

    sdf_file = f"{data_filebase}/sdf.npz"
    sdf = np.load(sdf_file)["sdf"]

    Nt = SU[0].shape[1]
    t = torch.tensor(np.linspace(0, 1, Nt, dtype=np.float32))  # t is strain
    t = t[None, :, None]  # (1, Nt, 1)

    SU = [torch.tensor((su-su_shift)/su_scaler) for su in SU]  # (Nb, N,Nt, 3)

    if input_T:
        xyt = [torch.cat((torch.tensor(x[:, None, :2]).repeat(
            1, Nt, 1), t.repeat(x.shape[0], 1, 1)), dim=-1) for x in nodes]  # (Nb, N, Nt, 3)
    else:
        xyt = [torch.tensor(x[:, :2]) for x in nodes]
    sdf = sdf.reshape(-1, 120 * 120)
    sdf_shift, sdf_scale = np.mean(sdf), np.std(sdf)
    sdf = (sdf - sdf_shift) / sdf_scale
    sdf = sdf.reshape(-1, 1, 120, 120)
    sdf = torch.tensor(sdf)

    train_ids, test_ids = train_test_split(
        np.arange(len(sdf)), test_size=test_size, random_state=seed)

    train_xyt = [xyt[i] for i in train_ids]
    test_xyt = [xyt[i] for i in test_ids]
    train_S = [SU[i] for i in train_ids]
    test_S = [SU[i] for i in test_ids]
    train_sdf = sdf[train_ids]
    test_sdf = sdf[test_ids]

    train_dataset = ListDataset(
        (train_sdf, train_xyt, train_S, torch.tensor(train_ids)))
    test_dataset = ListDataset(
        (test_sdf, test_xyt, test_S, torch.tensor(test_ids)))

    def pad_collate_fn(batch):
        # Extract sdf (same-length)
        sdf_batch = torch.stack([item[0] for item in batch])
        xyt_batch = [item[1]
                     for item in batch]  # Extract xyt (variable-length)
        SU = [item[2] for item in batch]  # Extract S (variable-length)
        sample_ids = torch.stack([item[3] for item in batch])
        # y_batch = torch.stack([item[1] for item in batch])  # Extract and stack y (fixed-length)
        # Pad sequences
        xyt_padded = pad_sequence(
            xyt_batch, batch_first=True, padding_value=padding_value)
        SU_padded = pad_sequence(SU, batch_first=True,
                                 padding_value=padding_value)
        return sdf_batch, xyt_padded, SU_padded, sample_ids

    train_dataloader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True,
                                  collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False,
                                 collate_fn=pad_collate_fn)
    su_shift = torch.tensor(su_shift)[None, :]  # (1, 1,1,3)
    su_scaler = torch.tensor(su_scaler)[None, :]  # (1, 1, 1, 3)

    def SUInverse(x):
        su_sig = su_scaler.to(x.device)
        su_mu = su_shift.to(x.device)
        return x*su_sig+su_mu

    def sdf_inv_trans(x):
        return (x * sdf_scale + sdf_shift)
    sdf_inv_scaler = sdf_inv_trans
    su_inv_scaler = SUInverse

    print(f"Data loading time: {time.time()-start:.2f} s")

    return train_dataloader, test_dataloader, sdf_inv_scaler, su_inv_scaler


def LoadCells():
    mesh_file = f"{data_filebase}/mesh_cells10K.pkl"
    with open(os.path.join(mesh_file), "rb") as f:
        cells = pickle.load(f)
    return cells


def NOTSU_configs(input_T=True):
    if input_T:
        file_base = f"{script_path}/saved_weights/NOTSU_inpT_test"
        notsu_args = {"img_shape": (1, 120, 120),
                      "channel_mutipliers": [1, 2, 4, 8],
                      "has_attention": [False, False, False, False],
                      "first_conv_channels": 8, "num_res_blocks": 1,
                      "norm_groups": 8, "dropout": None, "embed_dim": 64,
                      "cross_attn_layers": 2, "num_heads": 8, "in_channels": 3, "out_channels": 3,
                      "emd_version": "nerf",
                      "padding_value": PADDING_VALUE}
    else:
        # file_base = f"{script_path}/saved_weights/NOTSU_noinpT_frame_scaler_test"
        file_base = f"{script_path}/saved_weights/NOTSU"
        notsu_args = {"img_shape": (1, 120, 120),
                      "channel_mutipliers": [1, 2, 4, 8],
                      "has_attention": [False, False, False, False],
                      "first_conv_channels": 8, "num_res_blocks": 1,
                      "norm_groups": 8, "dropout": None, "embed_dim": 64,
                      "cross_attn_layers": 2, "num_heads": 8, "in_channels": 2, "out_channels": 3,
                      "emd_version": "nerf",
                      "padding_value": PADDING_VALUE, "num_frames": NUM_FRAMES}
    args_all = {"model_args": notsu_args, "filebase": file_base}
    return args_all


# %%


def LoadDataInv(test_size=0.2, seed=420):
    SS_curve_file = f"{data_filebase}/SS_curve.npy"
    stress = np.load(SS_curve_file)
    sdf_file = f"{data_filebase}/sdf.npz"
    sdf = np.load(sdf_file)["sdf"]
    sdf = sdf.reshape(-1, 120 * 120)
    sdf_shift, sdf_scale = np.mean(sdf), np.std(sdf)
    sdf_norm = (sdf - sdf_shift) / sdf_scale
    sdf_norm = sdf_norm.reshape(-1, 1, 120, 120)
    stress_shift, stress_scale = np.mean(stress), np.std(stress)
    stress_norm = (stress - stress_shift) / stress_scale

    def s_inv_trans(y):
       return y * stress_scale + stress_shift

    def sdf_inv_trans(x):
        return (x * sdf_scale + sdf_shift)

    sdf_norm = torch.tensor(sdf_norm)
    stress_norm = torch.tensor(stress_norm)
    sdf_train, sdf_test, stress_train, stress_test = train_test_split(
        sdf_norm, stress_norm, test_size=test_size, random_state=seed
    )
    train_dataset = TensorDataset(
        sdf_train, stress_train)
    test_dataset = TensorDataset(sdf_test, stress_test)
    sdf_inv_scaler = sdf_inv_trans
    stress_inv_scaler = s_inv_trans

    return train_dataset, test_dataset, sdf_inv_scaler, stress_inv_scaler


def INV_configs():
    """************Inverse diffusion model arguments************"""
    channel_multpliers = [1, 2, 4, 8]
    fist_conv_channels = 16
    num_heads = 4
    norm_groups = 8
    num_res_blocks = 1
    total_timesteps = 500
    dropout = None

    inv_img_shape = (1, 120, 120)
    has_attention = [False, False, True, True]
    inv_diffusion_filebase = f"{script_path}/saved_weights/inv_diffusion"
    inv_diffusion_model_args = {"img_shape": inv_img_shape, "channel_multpliers": channel_multpliers,
                                  "has_attention": has_attention, "fist_conv_channels": fist_conv_channels,
                                  "num_heads": num_heads, "norm_groups": norm_groups, "num_res_blocks": num_res_blocks,
                                  "total_timesteps": total_timesteps, "dropout": dropout}
    inv_diffusion_args = {
        "model_args": inv_diffusion_model_args, "filebase": inv_diffusion_filebase}

    return inv_diffusion_args
