# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import pickle
from skimage import measure
# %%
# fem data
filebase_ss = "/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/abaqus/femDataR1"


secs = [f.path for f in os.scandir(filebase_ss) if f.is_dir()]
secs = natsort.natsorted(secs)

valid_sample_ids = []
stresses = np.empty((0, 51))
for sec in secs:
    print("Processing sec: ", sec)
    samples = [f.name for f in os.scandir(sec) if f.is_dir()]
    samples = natsort.natsorted(samples)
    for sample in samples:
        print("Processing sample: ", sample)
        _, id = sample.split('_')
        ss_file = os.path.join(sec, sample, "stress_strain.csv")
        if os.path.exists(ss_file):
            data = np.loadtxt(ss_file, delimiter=',', skiprows=1)
            if len(data) == 51:
                valid_sample_ids.append(id)
                stresses = np.vstack((stresses, data[:, 1]))

stresses = np.array(stresses)
strain = data[:, 0]
valid_sample_ids = np.array(valid_sample_ids, dtype=int)
# %%
# geo data
geos_file = '/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_pcn_all.pkl'
with open(geos_file, "rb") as f:
    geo_data = pickle.load(f)
vertices_all = geo_data['vertices']
inner_loops_all = geo_data['inner_loops']
out_loop_all = geo_data['out_loop']
points_cloud_all = geo_data['points_cloud']
sdf_all = geo_data['sdf']
x_grids = geo_data['x_grids']
y_grids = geo_data['y_grids']
sdf_all = np.array(sdf_all).reshape(-1, *x_grids.shape)
sdf_valid = sdf_all[valid_sample_ids]


# %%
# save data
data_valid = {
    'sdf': sdf_valid,
    'stress': stresses,
    'strain': strain,
    'x_grids': x_grids,
    'y_grids': y_grids
}
np.savez('/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data.npz', **data_valid)

# %%
# save data as pickle
# with open('/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data.pkl', 'wb') as f:
#   pickle.dump(data_valid, f)
# %%
# Load npz data
loaded_data = np.load(
    '/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data.npz')

# Access data
loaded_sdf = loaded_data['sdf']
loaded_stress = loaded_data['stress']
loaded_strain = loaded_data['strain']
loaded_x_grids = loaded_data['x_grids']
loaded_y_grids = loaded_data['y_grids']

print("Loaded SDF shape:", loaded_sdf.shape)
print("Loaded stress shape:", loaded_stress.shape)
print("Loaded strain shape:", loaded_strain.shape)
print("Loaded x_grids shape:", loaded_x_grids.shape)
print("Loaded y_grids shape:", loaded_y_grids.shape)
# %%
s_max = np.max(loaded_stress, axis=1)
_ = plt.hist(s_max, bins=100)
# %%
id_m = np.where(s_max > 10)[0]
clip_s = loaded_stress[id_m]
sort_id = np.argsort(s_max[id_m])
clip_sdf = loaded_sdf[id_m]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(121)
case = 500
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
id_m = np.where((s_max > 1) & (s_max < 120))[0]

data_valid = {
    'sdf': loaded_sdf[id_m],
    'stress': loaded_stress[id_m],
    'strain': loaded_strain,
    'x_grids': loaded_x_grids,
    'y_grids': loaded_y_grids
}
np.savez('/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_1-120.npz', **data_valid)

# %%
