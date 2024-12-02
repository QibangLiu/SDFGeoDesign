# %%
import os
import pickle
import numpy as np
import json


# %%
file_base='/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/femDataR1'
geos_file = '/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF/training_data/geo_sdf_randv_pcn_all.pkl'
with open(geos_file, "rb") as f:
    geo_data = pickle.load(f)
vertices_all = geo_data['vertices']
inner_loops_all = geo_data['inner_loops']
out_loop_all = geo_data['out_loop']
points_cloud_all = geo_data['points_cloud']
sdf_all = geo_data['sdf']
x_grid = geo_data['x_grids']
y_grid = geo_data['y_grids']

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Calculate the number of samples per section
num_sections = 200
total_samples = len(vertices_all)
samples_per_section = total_samples // num_sections
remainder = total_samples % num_sections

# Distribute the data into sections
start_idx = 0
for sec_id in range(num_sections):
  end_idx = start_idx + samples_per_section + (1 if sec_id < remainder else 0)
  for idx in range(start_idx, end_idx):
    working_dir=os.path.join(file_base, f'sec_{sec_id}/sample_{idx}')
    os.makedirs(working_dir, exist_ok=True)
    sample_data = {
      'vertices': vertices_all[idx],
      'inner_loops': inner_loops_all[idx],
      'out_loop': out_loop_all[idx],
    }
    sample_file = os.path.join(working_dir, 'sample.json')
    with open(sample_file, 'w') as f:
      json.dump(sample_data, f, default=ndarray_to_list)
  print(f'Section {sec_id} done')
  start_idx = end_idx
