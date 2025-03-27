# %%
import pickle

import numpy as np
import matplotlib.pyplot as plt
# %%
filename = 'geo_sdf_randv_all.pkl'
with open(filename, "rb") as f:
    geo_data = pickle.load(f)
vertices_all = geo_data['vertices']
inner_loops_all = geo_data['inner_loops']
out_loop_all = geo_data['out_loop']
points_cloud_all = geo_data['points_cloud']
sdf_all = geo_data['sdf']
x_grid = geo_data['x_grids']
y_grid = geo_data['y_grids']
# %%
id_s = len(vertices_all[0])

plt.plot(points_cloud_all[0][id_s:, 0], points_cloud_all[0][id_s:, 1], 'bo')
plt.plot(points_cloud_all[0][:id_s, 0], points_cloud_all[0][:id_s, 1], 'rs')
# %%

# %%

n_cp = 484


def sort_points_counter_clockwise(points):
    # Compute the centroid of the points

    # Compute angles of each point w.r.t. the centroid
    angles = np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5)

    # Sort points by angle in counter-clockwise direction
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


points_cloud_all_new = []
for vert, pc in zip(vertices_all, points_cloud_all):
    l_vert = len(vert)
    bp = pc[l_vert:]  # boundary points
    n_bp = n_cp-l_vert
    sort_bp = bp
    while len(sort_bp) < n_bp:
        sort_bp = sort_points_counter_clockwise(sort_bp)
        new_points = []
        for i in range(len(sort_bp)-1):
            p1 = sort_bp[i]
            p2 = sort_bp[i+1]
            if (np.isclose(p1[0], 0) and np.isclose(p2[0], 0) and abs(p1[1]-p2[1]) < 0.035) \
                    or (np.isclose(p1[0], 1) and np.isclose(p2[0], 1) and abs(p1[1]-p2[1]) < 0.035) \
                    or (np.isclose(p1[1], 0) and np.isclose(p2[1], 0) and abs(p1[0]-p2[0]) < 0.035) \
                    or (np.isclose(p1[1], 1) and np.isclose(p2[1], 1) and abs(p1[0]-p2[0]) < 0.035):
                mid_point = (p1 + p2) / 2
                new_points.append(mid_point)
        new_points = np.array(new_points)
        n_add = n_bp-len(sort_bp)
        sort_bp = np.vstack((sort_bp, new_points[:n_add]))
    pc_new = np.vstack((vert, sort_bp))
    points_cloud_all_new.append(pc_new)


# %%
points_cloud_all_new = np.array(points_cloud_all_new)
geo_data_all_new = {'vertices': vertices_all, 'inner_loops': inner_loops_all,
                    'out_loop': out_loop_all, 'points_cloud': points_cloud_all_new,
                    'sdf': sdf_all, 'x_grids': x_grid, 'y_grids': y_grid}
with open('geo_sdf_randv_pcn_all.pkl', "wb") as f:
    pickle.dump(geo_data_all_new, f)

# %%
