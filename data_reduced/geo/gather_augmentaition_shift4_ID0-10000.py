# %%
import numpy as np
import pickle
# %%
sdf_ss_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset/sdf_stress_strain_data_12-92.npz'
sdf_ss_data = np.load(sdf_ss_file)
sdf_ss = sdf_ss_data['sdf'][:10000]
stress = sdf_ss_data['stress'][:10000]
strain = sdf_ss_data['strain']
sample_ids = sdf_ss_data['sample_ids'][:10000]
x_grids = sdf_ss_data['x_grids']
y_grids = sdf_ss_data['y_grids']
# %%
geos_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_all.pkl'
with open(geos_file, "rb") as f:
    geo_data = pickle.load(f)
points_cloud_all = geo_data['points_cloud']
points_cloud_all = [points_cloud_all[i] for i in sample_ids]
# %%
aug_sdf_ss_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset/augmentation/pc_sdf_stress_strain_12-92_shift4_0-10000.pkl'
with open(aug_sdf_ss_file, "rb") as f:
    aug_sdf_ss_data = pickle.load(f)

aug_sdf_ss = aug_sdf_ss_data['sdf']
aug_stress = aug_sdf_ss_data['stress']
aug_pc = aug_sdf_ss_data['points_cloud']


# %%
all_sdf = np.concatenate((sdf_ss[:], aug_sdf_ss), axis=0)
all_stress = np.concatenate((stress[:], aug_stress), axis=0)
points_cloud_all = points_cloud_all + aug_pc
aug_data = {'sdf': all_sdf, 'stress': all_stress,
            'strain': strain, 'points_cloud': points_cloud_all, "x_grids": x_grids, "y_grids": y_grids}

# %%
file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset"
aug_file = f"{file_base}/pc_sdf_ss_12-92_shift4_0-10000_aug.pkl"
with open(aug_file, "wb") as f:
    pickle.dump(aug_data, f)
print(f"Saved to {aug_file}, {len(all_sdf)} samples")
# %%
