# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.geoencoder import LoadGeoEncoderModel
from models.configs import models_configs, LoadData
from torch.utils.data import DataLoader
from skimage import measure
# import os
# os.chdir("./models")

device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
configs = models_configs(out_c=256, latent_d=256)
filebase = configs["GeoEncoder"]["filebase"]
model_params = configs["GeoEncoder"]["model_params"]
print(f"\n\nGeoEncoder Filebase: {filebase}, model_params:")
print(model_params)
geo_encoder, sdf_NN = LoadGeoEncoderModel(
    filebase, model_params)

# %%
train_dataset, test_dataset, grid_coor, sdf_inv_scaler, _ = LoadData(
    seed=42)


x_grids = grid_coor[:, 0].reshape(120, -1).cpu().numpy()
y_grids = grid_coor[:, 1].reshape(120, -1).cpu().numpy()
grid_coor = grid_coor.to(device)
# %%


def predict(data, geo_encoder, sdf_NN, grid_coor):
    sd_pred = []
    sd_true = []
    geo_encoder = geo_encoder.to(device)
    sdf_NN = sdf_NN.to(device)
    geo_encoder.eval()
    sdf_NN.eval()
    with torch.no_grad():
        if isinstance(data, DataLoader):
            for data in data:
                pc = data[0].to(device)
                SDF = data[1].to(device)
                params = geo_encoder(pc)
                sdf_pred = sdf_NN(grid_coor, params)
                sd_pred.append(sdf_pred.detach().cpu().numpy())
                sd_true.append(SDF.detach().cpu().numpy())
            sd_pred = np.vstack(sd_pred)
            sd_true = np.vstack(sd_true)
        else:
            pc = data[0].to(device)
            SDF = data[1].to(device)
            params = geo_encoder(pc)
            sdf_pred = sdf_NN(grid_coor, params)
            sd_pred.append(sdf_pred.detach().cpu().numpy())
            sd_true.append(SDF.detach().cpu().numpy())
            sd_pred = np.vstack(sd_pred)
            sd_true = np.vstack(sd_true)
    return sd_pred, sd_true


# %%
titles = ["best", "50% percentile", "97% percentile"]


def plot_geo(sd_pred_test, sd_ture_test):
    error_s = np.linalg.norm(sd_pred-sd_true, axis=1) / \
        np.linalg.norm(sd_true, axis=1)
    mean, std = np.mean(error_s), np.std(error_s)
    fig = plt.figure(figsize=(4.8, 3.6))
    ax = plt.subplot(1, 1, 1)

    _ = ax.hist(error_s, bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("L2 relative error")
    ax.set_ylabel("Frequency")

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
    indices = np.array([min_index, median_index, max_index])
    nr, nc = 1, 3
    fig = plt.figure(figsize=(nc*4.8, nr*3.6))
    for i, index in enumerate(indices):

        ax = plt.subplot(nr, nc, i+1)
        sd_pred_i = sd_pred_test[index].reshape(x_grids.shape)
        sd_true_i = sd_ture_test[index].reshape(x_grids.shape)
        pred_geo = measure.find_contours(
            sd_pred_i, 0, positive_orientation='high')
        true_geo = measure.find_contours(
            sd_true_i, 0, positive_orientation='high')
        for c, contour in enumerate(true_geo):
            if c == 0:
                ax.plot(contour[:, 1], contour[:, 0],
                        'r', linewidth=2, label="Truth")
            else:
                ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
        for c, contour in enumerate(pred_geo):
            if c == 0:
                ax.plot(contour[:, 1], contour[:, 0], '--b',
                        linewidth=2, label="Predicted")
            else:
                ax.plot(contour[:, 1], contour[:, 0], '--b', linewidth=2)

        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_title(f"{titles[i]}")

        plt.tight_layout()


# %%
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
sd_pred, sd_true = predict(test_loader, geo_encoder, sdf_NN, grid_coor)
sd_pred = sdf_inv_scaler(sd_pred)
sd_true = sdf_inv_scaler(sd_true)
plot_geo(sd_pred, sd_true)

# %%
# Extract data from test_dataset
test_data = []
numps = []
for data in test_dataset:
  test_data.append((data[0].cpu().numpy()[:], data[1].cpu().numpy()))


test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
sd_pred, sd_true = predict(test_loader, geo_encoder, sdf_NN, grid_coor)
sd_pred = sdf_inv_scaler(sd_pred)
sd_true = sdf_inv_scaler(sd_true)
plot_geo(sd_pred, sd_true)
# %%
data1 = test_dataset[:1]

# mask = (data1[0][:, 0] == 0).cpu().numpy().astype(np.int32)
# idx = np.where(mask == 1)[0][3]
data1_reduced = (data1[0][0][:320][None], data1[1][0][None])
sd_pred_reduced, sd_true = predict(
    data1_reduced, geo_encoder, sdf_NN, grid_coor)
error_s = np.linalg.norm(sd_pred_reduced-sd_true, axis=1) / \
    np.linalg.norm(sd_pred_reduced, axis=1)
print("Error for reduced data:", error_s)
# %%
