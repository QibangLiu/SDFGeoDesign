# %%
import numpy as np

# %%
sdf_ss_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92.npz'
sdf_ss_data = np.load(sdf_ss_file)
sdf_ss = sdf_ss_data['sdf']
stress = sdf_ss_data['stress']
strain = sdf_ss_data['strain']
sample_ids = sdf_ss_data['sample_ids']

# %%
aug_sdf_ss_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift3_0-28781.npz'
aug_sdf_ss_data = np.load(aug_sdf_ss_file)
aug_sdf_ss = aug_sdf_ss_data['sdf']
aug_stress = aug_sdf_ss_data['stress']
aug_strain = aug_sdf_ss_data['strain']
aug_sample_ids = aug_sdf_ss_data['sample_ids']


# %%
all_sdf = np.concatenate((sdf_ss[:], aug_sdf_ss), axis=0)
all_stress = np.concatenate((stress[:], aug_stress), axis=0)
all_sample_ids = np.concatenate((sample_ids[:], aug_sample_ids), axis=0)

aug_data = {'sdf': all_sdf, 'stress': all_stress,
            'strain': strain, 'sample_ids': all_sample_ids}
np.savez('/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift3_0-28781_aug.npz', **aug_data)
# %%
