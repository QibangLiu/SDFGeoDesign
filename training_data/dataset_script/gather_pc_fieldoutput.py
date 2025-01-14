# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import pickle


# %%
loaded_data = np.load(
    "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92.npz")
loaded_sample_ids = loaded_data['sample_ids'][10000]


# %%
filebase_ss = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/femDataR1"
secs = [f.path for f in os.scandir(filebase_ss) if f.is_dir()]
secs = natsort.natsorted(secs)
valid_sample_ids = []
mises_all = []
disp_all = []
mesh_coords_all = []
mesh_connect_all = []
for sec in secs[:1]:
    print("Processing sec: ", sec)
    samples = [f.name for f in os.scandir(sec) if f.is_dir()]
    samples = natsort.natsorted(samples)
    for sample in samples:
        print("Processing sample: ", sample)
        _, id = sample.split('_')
        mises_fn = os.path.join(sec, sample, "mises_stress.npy")
        disp_fn = os.path.join(sec, sample, "displacement.npy")
        mesh_fn = os.path.join(sec, sample, "mesh_data.npz")
        ss_file = os.path.join(sec, sample, "stress_strain.csv")
        if os.path.exists(ss_file) and os.path.exists(mises_fn) and os.path.exists(disp_fn) and os.path.exists(mesh_fn):
            mises = np.load(mises_fn)

            disp = np.load(disp_fn)
            disp = np.concatenate([np.zeros_like(disp[0])[None], disp], axis=0)
            mesh = np.load(mesh_fn)
            if len(mises) == 50 or len(disp) == 50:
                mises = np.concatenate(
                    [np.zeros_like(mises[0])[None], mises], axis=0)[::2]
                disp = np.concatenate(
                    [np.zeros_like(disp[0])[None], disp], axis=0)[::2]
                mesh = np.load(mesh_fn)
                mises_all.append(mises)
                disp_all.append(disp)
                mesh_coords_all.append(mesh['nodes_coords'])
                mesh_connect_all.append(mesh['elements_connectivity'])
                valid_sample_ids.append(id)

valid_sample_ids = np.array(valid_sample_ids, dtype=np.int32)
# %%
geos_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_pcn_all.pkl'
with open(geos_file, "rb") as f:
    geo_data = pickle.load(f)
# vertices_all = geo_data['vertices']
# inner_loops_all = geo_data['inner_loops']
# out_loop_all = geo_data['out_loop']
points_cloud_all = geo_data['points_cloud']
# sdf_all = geo_data['sdf']
# x_grids = geo_data['x_grids']
# y_grids = geo_data['y_grids']
# sdf_all = np.array(sdf_all).reshape(-1, *x_grids.shape)
point_cloud_valid = points_cloud_all[valid_sample_ids]
# %%

data_mises = {"mises": mises_all, "valid_sample_ids": valid_sample_ids}
data_disp = {"disp": disp_all, "valid_sample_ids": valid_sample_ids}
data_mesh = {"mesh_coords": mesh_coords_all, "mesh_connect": mesh_connect_all, "point_cloud": point_cloud_valid,
             "valid_sample_ids": valid_sample_ids}


file_path = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/pc_fieldoutput"
os.makedirs(file_path, exist_ok=True)

with open(os.path.join(file_path, "mises.pkl"), "wb") as f:
  pickle.dump(data_mises, f)

with open(os.path.join(file_path, "disp.pkl"), "wb") as f:
  pickle.dump(data_disp, f)

with open(os.path.join(file_path, "mesh_pc.pkl"), "wb") as f:
  pickle.dump(data_mesh, f)
