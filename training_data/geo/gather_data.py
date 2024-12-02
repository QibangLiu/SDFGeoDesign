# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from shapely.geometry import Polygon, Point
import pickle
import shapely.geometry as sg
import timeit
import json
# %%
filenamses=['geo_sdf_randv_'+str(i)+'.pkl' for i in range(1,2)]
vertices_all=[]
inner_loops_all=[]
out_loop_all=[]
points_cloud_all=[]
sdf_all=[]

for filename in filenamses:
    with open(filename, "rb") as f:
        geo_data = pickle.load(f)
    vertices_all+=geo_data['vertices']
    inner_loops_all+=geo_data['inner_loops']
    out_loop_all+=geo_data['out_loop']
    points_cloud_all+=geo_data['points_cloud']
    sdf_all+=geo_data['sdf']
x_grid=geo_data['x_grids']
y_grid=geo_data['y_grids']
geo_data_all={'vertices':vertices_all,'inner_loops':inner_loops_all,
              'out_loop':out_loop_all,'points_cloud':points_cloud_all,
              'sdf':sdf_all,'x_grids':x_grid,'y_grids':y_grid}
# with open('geo_sdf_randv_all.pkl', "wb") as f:
#   pickle.dump(geo_data_all, f)





# %%
def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

with open('geo_sdf_randv_all.json', 'w') as file:
    json.dump(geo_data_all, file, default=ndarray_to_list)

# %%
with open('data.json', 'r') as file:
    data = json.load(file)
