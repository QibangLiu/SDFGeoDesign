# %%
from skimage import measure
import models
from torch.utils.data import TensorDataset, DataLoader
import torch_trainer
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

_, _, sdf_test, stress_test, sdf_inv_scaler, stress_inv_scaler, strain = models.LoadData(
    seed=52)
test_dataset = TensorDataset(sdf_test, stress_test)

fwd_model = models.LoadForwardModel()


inv_Unet, gaussian_diffusion = models.LoadDiffusionInverseModel()
# %%
id = 22
num_sol = 10
img_shape = sdf_test.shape[1:]
Ytarget = test_dataset[id][1].unsqueeze(0)
Ytarget = Ytarget.to(device)
labels = Ytarget.repeat(num_sol, 1)
Xpred = gaussian_diffusion.sample(
    inv_Unet, img_shape, labels, w=2, clip_denoised=False, conditioning=True
)

Xpred = torch.tensor(Xpred[-1]).to(device)
# %%

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
