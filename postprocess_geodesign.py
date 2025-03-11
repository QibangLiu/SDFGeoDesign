# %%
from scipy.optimize import fsolve
import os
from models import NOT_SS
from models import configs
from models.inverse_diffusion_from_sdf import LoadDiffusionInverseModel
from utils.sdf2geo import filter_out_unit_cell
from utils.run_abaqus import run_abaqus_sim
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# TODO: change to your abaqus path, set None if you don't have abaqus
ABAQUS_EXE = "/projects/bbkg/Abaqus/2024/Commands/abaqus"
# %%
inv_config = configs.INV_configs()
inv_filebase = inv_config["filebase"]
inv_args = inv_config["model_args"]
fwd_config = configs.NOTSS_configs()
fwd_filebase = fwd_config["filebase"]
fwd_args = fwd_config["model_args"]

inv_Unet, gaussian_diffusion = LoadDiffusionInverseModel(
    inv_filebase, inv_args)
fwd = NOT_SS.LoadNOTModel(fwd_filebase, fwd_args)

# %%
train_dataset, test_dataset, sdf_inv_scaler, stress_inv_scaler = configs.LoadDataInv()
strain = torch.tensor(np.linspace(0, 1, 51), dtype=torch.float32)
strain = strain[:, None].to(device)
# %%


def inv_diffusion(Ytarget, fwd=fwd, num_sol=100, seed=None, w=2):
    # Ytarget = test_dataset[testID][1].unsqueeze(0)
    start = time.time()
    Ytarget = Ytarget.to(device)
    labels = Ytarget.repeat(num_sol, 1)
    sdf = gaussian_diffusion.sample(
        inv_Unet, labels, w=w, clip_denoised=False, seed=seed
    )
    sdf = torch.tensor(sdf).to(device)
    with torch.no_grad():
        Ypred = fwd(strain, sdf)
    Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
    Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
    Xpred = sdf.cpu().detach().numpy()
    end = time.time()
    print(f"Time taken for {num_sol} solutions: {end-start}")
    return Xpred, Ypred_inv, Ytarg_inv

# %%


def design_perform(design_results, filebase, seed=None, threshold=0.05, abaqus_exe=ABAQUS_EXE, overwrite=False):
    Xpred, Ypred_inv, Ytarg_inv = design_results
    Xpred_inv = sdf_inv_scaler(Xpred)
    Xpred_inv = Xpred_inv.squeeze()
    """evaluate the  accuracy design results by forward model"""

    L2error = np.linalg.norm(Ypred_inv - Ytarg_inv, axis=1) / \
        np.linalg.norm(Ytarg_inv, axis=1)
    fig = plt.figure(figsize=(4.8, 3.6))
    ax = plt.subplot(1, 1, 1)
    _ = ax.hist(L2error, bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("L2 relative error")
    ax.set_ylabel("Frequency")
    mean, std = np.mean(L2error), np.std(L2error)
    print(f"Mean L2 error of the diffusion design results: {mean}, std: {std}")
    """random select the 4 designs"""
    # filter out the periodic unit cells
    geo_contours, periodic_ids = filter_out_unit_cell(
        Xpred_inv)
    Ypred_inv = Ypred_inv[periodic_ids]
    X_pred = Xpred[periodic_ids]
    Ytarg_inv = Ytarg_inv[periodic_ids]
    L2error = L2error[periodic_ids]
    if threshold is not None:
        np.random.seed(seed)
        evl_ids = np.random.choice(
            np.where(L2error < threshold)[0], 4, replace=False)
    else:
        evl_ids = np.argsort(L2error)[:4]
    for i, idx in enumerate(evl_ids):
        print(f"ID: {idx}, L2 error: {L2error[idx]}")
    strain = np.linspace(0, 0.2, 51)
    cases = ["c", "d", "e", "f"]
    """run abaqus simulation for selected designs"""
    fem_stress = []
    for i, idx in enumerate(evl_ids):
        print(f"Running Abaqus simulation for case {cases[i]}")
        working_dir = os.path.join(filebase, cases[i])
        femdata = run_abaqus_sim(
            geo_contours[evl_ids[i]], working_dir, abaqus_exe, sdf_norm=X_pred[evl_ids[i]], overwrite=overwrite)
        fem_stress.append(femdata[:, 1])
    fem_stress = np.array(fem_stress)
    """plot the results"""
    legends = [
        f"{cases[i]} ({L2error[evl_ids[i]]*100:.1f}%)" for i in range(4)]
    nr, nc = 2, 3
    fig, axes = plt.subplots(nr, nc, figsize=(
        nc*4.8, nr*3.6), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(strain*100, Ytarg_inv[0], '-bo', label="Target", markersize=4)
    for i, v in enumerate(evl_ids):
        ax.plot(strain*100, Ypred_inv[v], '--', label=legends[i])
    ax.legend()
    ax.set_xlabel(r"$\varepsilon~[\%]$")
    ax.set_ylabel(r"$\sigma~[MPa]$")
    for i, v in enumerate(evl_ids):
        shell_contours, holes_contours = geo_contours[v]
        ax = axes[0, i+1] if i < 2 else axes[1, i-1]
        for contour in shell_contours:
            x, y = contour[:, 1], contour[:, 0]
            ax.fill(x, y, alpha=1.0, edgecolor="black",
                    facecolor="cyan", label="Outer Boundary")
        for contour in holes_contours:
            x, y = contour[:, 1], contour[:, 0]
            ax.fill(x, y, alpha=1.0, edgecolor="black",
                    facecolor="white", label="Hole")
        ax.axis("equal")  # Keep aspect ratio square
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.1, 1.1])
    L2error_fem = np.linalg.norm(fem_stress - Ytarg_inv[:1], axis=1) / \
        np.linalg.norm(Ytarg_inv[:1], axis=1)
    legends = [f"{cases[i]} ({L2error_fem[i]*100:.1f}%)" for i in range(4)]
    ax = axes[1, 0]
    ax.plot(strain*100, Ytarg_inv[0], '-bo', label="Target", markersize=4)
    for i in range(4):
        if fem_stress[i][0] is np.nan:
            continue
        ax.plot(strain*100, fem_stress[i], '--', label=legends[i])
    ax.legend()
    ax.set_xlabel(r"$\varepsilon~[\%]$")
    ax.set_ylabel(r"$\sigma~[MPa]$")
    return geo_contours, Ypred_inv, Ytarg_inv


# %%
# good: 37388 \/ 55161,71114
# not bad: 3653,16681,79017,18765
seed = np.random.randint(0, 100000)  # 54010
seed = 37388
print(f"Random seed: {seed}")
np.random.seed(seed)
testID = np.random.randint(0, len(test_dataset))
print(f"Test ID: {testID}")
seed_ = np.random.randint(0, 100000)
Ytarget = test_dataset[testID][1].unsqueeze(0)
filebase = f"./abaqus_sims/testID{testID}"
design_results = inv_diffusion(Ytarget, num_sol=200, seed=seed_, w=10)
geo_contours, Ypred, Ytarg = design_perform(
    design_results, filebase, threshold=None,  seed=seed_, overwrite=True)

# %%
# Ondemand desgin


def ramberg_osgood_eq(sigma, epsilon, E, sigma_0, alpha, n):
    return epsilon - (sigma / E + alpha * (sigma / sigma_0) ** n)


def ondemand_target(E, sigma_0, alpha, n, idx_e=None):
    epsilon_v = np.linspace(0, 0.2, 51)
    if idx_e is None:
        sigma = np.array([fsolve(ramberg_osgood_eq, x0=epsilon * E, args=(epsilon, E, sigma_0, alpha, n))[0]
                          for epsilon in epsilon_v])
    else:
        epsilon_1st = epsilon_v[:idx_e]
        sigma_1st = np.array([fsolve(ramberg_osgood_eq, x0=epsilon * E, args=(epsilon, E, sigma_0, alpha, n))[0]
                              for epsilon in epsilon_1st])
        epsilon_2nd = epsilon_v[idx_e-1:]
        sigma_2nd = (sigma_1st[-1] + E*0.6 * (epsilon_2nd-epsilon_1st[-1]))
        sigma = np.concatenate((sigma_1st, sigma_2nd[1:]))
        fig = plt.figure(figsize=(4.8, 3.6))
    plt.figure(figsize=(6, 4))
    plt.plot(epsilon_v, sigma, color='b')
    plt.xlabel(r"$\varepsilon~[\%]$")
    plt.ylabel(r"$\sigma~[MPa]$")
    plt.title("Target Stress-Strain Curve (Ramberg-Osgood)")
    return sigma.astype(np.float32)


s_shift = stress_inv_scaler(0)
s_scale = stress_inv_scaler(1)-s_shift
# %%
# Material properties
E = 800     # Young's modulus (MPa)
sigma_0 = 30  # Reference yield stress (Pa)
alpha = 0.002   # Coefficient
n = 10           # Hardening exponent
y_demand1 = ondemand_target(E, sigma_0, alpha, n)
y_demand_target1 = (y_demand1-s_shift)/s_scale
y_demand_target1 = torch.tensor(y_demand_target1)[None, :].to(device)
filebase = f"./abaqus_sims/ondemand_case1"
# seed = np.random.randint(0, 100000)  # 54010
seed = 420  # 42  # 39157
np.random.seed(seed)
print(f"Random seed: {seed}")
seed_ = np.random.randint(0, 100000)
design_results = inv_diffusion(y_demand_target1, num_sol=500, seed=seed_, w=10)
geo_contours, Ypred, Ytarg = design_perform(
    design_results, filebase, threshold=None,  seed=seed_, overwrite=True)
# %%
# Material properties
E = 1000     # Young's modulus (MPa)
sigma_0 = 30  # Reference yield stress (Pa)
alpha = 0.002   # Coefficient
n = 10           # Hardening exponent
y_demand1 = ondemand_target(E, sigma_0, alpha, n, idx_e=45)
y_demand_target1 = (y_demand1-s_shift)/s_scale
y_demand_target1 = torch.tensor(y_demand_target1)[None, :].to(device)
filebase = f"./abaqus_sims/ondemand_case2"
seed = np.random.randint(0, 100000)  # 54010
seed = 43
np.random.seed(seed)
print(f"Random seed: {seed}")
seed_ = np.random.randint(0, 100000)
geo_contours, Ypred, Ytarg = design_test(
    y_demand_target1, filebase, num_sol=500, threshold=None, w=2, seed=seed_, overwrite=True)

# %%
