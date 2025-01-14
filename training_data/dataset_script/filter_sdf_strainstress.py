
# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import pickle
from skimage import measure
# save data as pickle
# with open('/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data.pkl', 'wb') as f:
#   pickle.dump(data_valid, f)
# %%
# Load npz data
loaded_data = np.load(
    '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/dataset/sdf_stress_strain_data.npz')


# %%

# Access data
loaded_sdf = loaded_data['sdf']
loaded_stress = loaded_data['stress']
loaded_strain = loaded_data['strain']
loaded_x_grids = loaded_data['x_grids']
loaded_y_grids = loaded_data['y_grids']
loaded_sample_ids = loaded_data['valid_sample_ids']

print("Loaded SDF shape:", loaded_sdf.shape)
print("Loaded stress shape:", loaded_stress.shape)
print("Loaded strain shape:", loaded_strain.shape)
print("Loaded x_grids shape:", loaded_x_grids.shape)
print("Loaded y_grids shape:", loaded_y_grids.shape)
print("Loaded sample_ids shape:", loaded_sample_ids.shape)
# %%
s_max = np.max(loaded_stress, axis=1)
_ = plt.hist(s_max, bins=100)
# %%


def visualize_sample(mins=10, case=500):
    id_m = np.where(s_max > mins)[0]
    clip_s = loaded_stress[id_m]
    sort_id = np.argsort(s_max[id_m])
    clip_sdf = loaded_sdf[id_m]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    print("Max stress:", np.max(clip_s[sort_id][case]))
    ax.plot(clip_s[sort_id][case])
    ax = fig.add_subplot(122)
    SDF = clip_sdf[sort_id][case]
    measure.find_contours(SDF, 0, positive_orientation='high')
    contours = measure.find_contours(SDF, 0, positive_orientation='high')

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)


# %%
s_max = np.max(loaded_stress, axis=1)
id_m = np.where((s_max > 12) & (s_max < 92))[0]

data_valid = {
    'sdf': loaded_sdf[id_m],
    'stress': loaded_stress[id_m],
    'strain': loaded_strain,
    'x_grids': loaded_x_grids,
    'y_grids': loaded_y_grids,
    'sample_ids': loaded_sample_ids[id_m]
}
np.savez('/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/dataset/sdf_stress_strain_data_12-92.npz', **data_valid)

# %%
