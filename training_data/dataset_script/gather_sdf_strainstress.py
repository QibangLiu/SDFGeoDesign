# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import pickle
from skimage import measure
# %%
# fem data
filebase_ss = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/femDataR1"


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
geos_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_pcn_all.pkl'
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
    'y_grids': y_grids,
    'valid_sample_ids': valid_sample_ids
}
np.savez('/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/dataset/sdf_stress_strain_data.npz', **data_valid)
