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
fwd_model = models.LoadForwardModel()
_, _, sdf_test, stress_test, sdf_inv_scaler, stress_inv_scaler, strain = models.LoadData(
    seed=42)
test_dataset = TensorDataset(sdf_test, stress_test)
test_loader = DataLoader(
    test_dataset, batch_size=1024, shuffle=False)
trainer = torch_trainer.TorchTrainer(
    fwd_model, device, filebase=models.fwd_filebase)

# %%
h = trainer.load_logs()
if h is not None:
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(h["loss"], label="loss")
    ax.plot(h["val_loss"], label="val_loss")
    ax.legend()
    ax.set_yscale("log")


# %%
s_pred, s_true = trainer.predict(test_loader)
s_pred = stress_inv_scaler(s_pred)
s_true = stress_inv_scaler(s_true)
error_s = np.linalg.norm(s_pred-s_true, axis=1) / \
    np.linalg.norm(s_true, axis=1)

fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(error_s, bins=20)
ax.set_xlabel("L2 relative error")
ax.set_ylabel("Frequency")
# %%
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
nr, nc = 1, len(index_list)
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
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
sdf_test_inv = sdf_inv_scaler(sdf_test.numpy())

nr, nc = 1, len(index_list)
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
for i, idx in enumerate(index_list):
    contours = measure.find_contours(
        sdf_test_inv[idx], 0, positive_orientation='high')
    ax = plt.subplot(nr, nc, i+1)
    l_style = ['r-', 'b--']
    holes = []
    for j, contour in enumerate(contours):
        contour = (contour-10)/100
        x, y = contour[:, 1], contour[:, 0]
        if j == 0:
            ax.fill(x, y, alpha=1.0, edgecolor="black",
                    facecolor="cyan", label="Outer Boundary")
        else:
            ax.fill(x, y, alpha=1.0, edgecolor="black",
                    facecolor="white", label="Hole")
        # ax.grid(True)
        ax.axis("off")
        ax.axis("equal")  # Keep aspect ratio square
