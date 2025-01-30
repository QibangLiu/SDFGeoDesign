# %%
import os
from models.forward import LoadForwardModel
from models.configs import models_configs, LoadData
from models.inverse_diffusion_from_sdf import LoadDiffusionInverseModel
from utils.sdf2geo import classify_contours
from utils.run_abaqus import run_abaqus_sim
import torch
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%


def filter_out_unit_cell(sdfs_all, Ypred_all, Ytarg_all):
    geo_contours = []
    periodic_ids = []
    for i, sdf in enumerate(sdfs_all):
        shell_contours, holes_contours, is_periodic = classify_contours(sdf)
        if is_periodic:
            periodic_ids.append(i)
            geo_contours.append((shell_contours, holes_contours))

    sdfs_all = sdfs_all[periodic_ids]
    Ypred_all = Ypred_all[periodic_ids]
    Ytarg_all = Ytarg_all[periodic_ids]
    return geo_contours, Ypred_all, Ytarg_all, sdfs_all


# %%
# configs = models_configs()
# fwd_filebase = configs["ForwardModel"]["filebase"]
# fwd_args = configs["ForwardModel"]["model_args"]
# geoencoder_args = configs["GeoEncoder"]["model_args"]
# inv_filebase = configs["InvDiffusion"]["filebase"]
# inv_args = configs["InvDiffusion"]["model_args"]

# inv_Unet, gaussian_diffusion = LoadDiffusionInverseModel(
#     inv_filebase, inv_args)
# fwd = LoadForwardModel(fwd_filebase, fwd_args, geoencoder_args)

# # %%
# train_dataset, test_dataset, grid_coor, sdf_inv_scaler, stress_inv_scaler = LoadData()

# # %%
# id = 23
# num_sol = 100
# Ytarget = test_dataset[id][1].unsqueeze(0)
# Ytarget = Ytarget.to(device)
# labels = Ytarget.repeat(num_sol, 1)
# sdf = gaussian_diffusion.sample(
#     inv_Unet, labels, w=2, clip_denoised=False, conditioning=True
# )
# sdf = torch.tensor(sdf).to(device)
# with torch.no_grad():
#     Ypred = fwd.forward_from_sdf(sdf)
# Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
# Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
# Xpred_inv = sdf_inv_scaler(sdf.cpu().detach().numpy())
# Xpred_inv = Xpred_inv.squeeze()

# # np.savez(f"test_data.npz", Xpred_inv=Xpred_inv,
# #          Ypred_inv=Ypred_inv, Ytarg_inv=Ytarg_inv)

# %%
data = np.load("test_data.npz")
Xpred_inv = data["Xpred_inv"]
Ypred_inv = data["Ypred_inv"]
Ytarg_inv = data["Ytarg_inv"]
# %%
# filter out the periodic unit cells
geo_contours, Ypred_inv, Ytarg_inv, sdfs_inv = filter_out_unit_cell(
    Xpred_inv, Ypred_inv, Ytarg_inv)
# %%
L2error = np.linalg.norm(Ypred_inv - Ytarg_inv, axis=1) / \
    np.linalg.norm(Ytarg_inv, axis=1)
fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(L2error, bins=20, color="skyblue", edgecolor="black")
ax.set_xlabel("L2 relative error")
ax.set_ylabel("Frequency")

sorted_idx = np.argsort(L2error)
mean, std = np.mean(L2error), np.std(L2error)
print(f"Mean L2 error of the diffusion design results: {mean}, std: {std}")
evl_ids = np.array([
    sorted_idx[0],
    sorted_idx[int(len(sorted_idx) * 0.33)],
    sorted_idx[int(len(sorted_idx) * 0.66)],
    sorted_idx[-1],
], dtype=int)
for i, idx in enumerate(evl_ids):
    print(f"ID: {idx}, L2 error: {L2error[idx]}")

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

    shell_contours, holes_contours = geo_contours[v]

    ax = plt.subplot(1, 5, i + 2)
    l_style = ["r-", "b--"]
    holes = []
    for contour in shell_contours:
        x, y = contour[:, 1], contour[:, 0]
        ax.fill(x, y, alpha=1.0, edgecolor="black",
                facecolor="cyan", label="Outer Boundary")

    for contour in holes_contours:
        x, y = contour[:, 1], contour[:, 0]
        ax.fill(x, y, alpha=1.0, edgecolor="black",
                facecolor="white", label="Hole")
    # ax.grid(True)
    # ax.axis("off")
    ax.axis("equal")  # Keep aspect ratio square

# %%


id = 23
working_dir = f"./abaqus_sims/case{id}"
# Please change to your abaqus executable path
abaqus_exe = "/projects/bbkg/Abaqus/2024/Commands/abaqus"
# Note: if the stress_strain.csv file exists, the abaqus sim will not be executed
i = 1
femdata = run_abaqus_sim(geo_contours[evl_ids[i]], working_dir, abaqus_exe)
fem_stress = femdata[:, 1]
fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
ax.plot(strain, Ytarg_inv[0], label="target")
ax.plot(strain, fem_stress, label="fem")
ax.plot(strain, Ypred_inv[evl_ids[1]], label="predicted")
ax.legend()
# %%
