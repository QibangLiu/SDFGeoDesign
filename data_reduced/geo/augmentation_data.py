# %%
from skimage import measure
import pandas as pd
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.geometry import LineString, Polygon, box, MultiPolygon, Point
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from shapely.ops import unary_union, polygonize, split
from shapely.affinity import translate
import os
import timeit
import json
# %%
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
grid_points = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
# %%
sdf_ss_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92.npz'
sdf_ss_data = np.load(sdf_ss_file)
sdf_ss = sdf_ss_data['sdf']
stress = sdf_ss_data['stress']
strain = sdf_ss_data['strain']
sample_ids = sdf_ss_data['sample_ids']
# %%
idx = 0
geo_id = sample_ids[idx]
xmin, xmax = 0, 1
vertices = vertices_all[geo_id]
inner_loops = inner_loops_all[geo_id]
out_loop = out_loop_all[geo_id]


def get_polygon(vertices, out_loop, inner_loops, x0, y0=0.0):

    # Create the outer loop polygon
    out_loop_coords = [vertices[i] for i in out_loop]
    # Create the inner loops polygons
    inner_holes = [[vertices[i] for i in inner] for inner in inner_loops]

    # Combine the outer polygon with the inner polygons to create a single polygon with holes
    uc = Polygon(out_loop_coords, inner_holes)

    right_uc = translate(uc, xoff=1, yoff=0)
    top_uc = translate(uc, xoff=0, yoff=1)
    top_righ_uc = translate(uc, xoff=1, yoff=1)
    merged_polygon = unary_union([uc, right_uc, top_uc, top_righ_uc])

    # rectangle = Polygon([(x0, y0), (x0+1, y0), (x0+1, y0+1), (x0, y0+1)])
    rectangle = box(x0, y0, x0+1, y0+1)
    cut_polygon = merged_polygon.intersection(rectangle)
    return cut_polygon


def get_periodic_geo(cut_polygon, x0, y0=0.0, ensure_perodic_tb=True):
    if isinstance(cut_polygon, MultiPolygon):
        return None

    boundary_coords = np.array(cut_polygon.exterior.coords)
    interiors_coords = [np.array(interior.coords)
                        for interior in cut_polygon.interiors]
    boundary_coords = boundary_coords - np.array([[x0, y0, 0]])
    interiors_coords = [interior - np.array([[x0, y0, 0]])
                        for interior in interiors_coords]

    vertices_new = boundary_coords

    if ensure_perodic_tb:
        id_bot = np.where(vertices_new[:, 1] == 0)[0][::-1]
        id_top = np.where(vertices_new[:, 1] == 1)[0]
        rm_ids = []
        for b in id_bot:
            got_it = False
            for t in id_top:
                if np.isclose(vertices_new[b, 0], vertices_new[t, 0]):
                    got_it = True
                    break
            if not got_it:
                rm_ids.append(b)

        for t in id_top:
            got_it = False
            for b in id_bot:
                if np.isclose(vertices_new[b, 0], vertices_new[t, 0]):
                    got_it = True
                    break
            if not got_it:
                rm_ids.append(t)
        vertices_new = np.delete(vertices_new, rm_ids, axis=0)

    out_loop_new = np.array(range(len(vertices_new)), dtype=int)

    inner_loops_new = []
    for interior in interiors_coords:
        start = len(vertices_new)
        vertices_new = np.concatenate((vertices_new, interior))
        end = len(vertices_new)
        inner_loops_new.append(np.array(range(start, end), dtype=int))
    return vertices_new, out_loop_new, inner_loops_new


def min_distance_to_boundary(point, boundaries):
    """Calculate the minimum distance from a point to multiple boundaries."""
    return min(boundary.distance(Point(point)) for boundary in boundaries)


def SDF_from_GEO(vertices_z, inner_loops, out_loop, grid_points):
    vertices = vertices_z[:, :2]
    # Define the boundary as a polygon (e.g., a square)
    outer_boundary = vertices[out_loop]
    holes = [vertices[loop] for loop in inner_loops]
    full_polygon = Polygon(outer_boundary, holes)
    # List of all boundaries (outer boundary + holes)
    boundaries = [Polygon(outer_boundary).exterior] + \
        [Polygon(hole).exterior for hole in holes]
    distances = np.array([min_distance_to_boundary(p, boundaries)
                         for p in grid_points])

    # Determine if each point is inside the polygon (outer boundary but outside holes)
    inside_outer_boundary = np.array(
        [full_polygon.contains(Point(p)) for p in grid_points])

    # Check if the point is within any of the holes
    inside_holes = np.array([
        any(Polygon(hole).contains(Point(p)) for hole in holes)
        for p in grid_points
    ])

    # Combine conditions for the signed distance function
    # Points inside the outer boundary but not in holes are considered "inside"
    signed_distances = distances * \
        np.where(inside_outer_boundary & ~inside_holes, -1, 1)
    return signed_distances


def shift_geo(filter_sample_id, num_shifts=4):
    geo_id = sample_ids[filter_sample_id]
    vertices = vertices_all[geo_id]
    inner_loops = inner_loops_all[geo_id]
    out_loop = out_loop_all[geo_id]

    got_num = 0
    try_num = 0
    SDFs_new = []
    stress_new = []
    geo_ids = []
    filter_sample_ids = []
    while got_num < num_shifts and try_num < 100:
        x0 = np.random.uniform(0.1, 0.9)
        cut_polygon = get_polygon(vertices, out_loop, inner_loops, x0)
        if isinstance(cut_polygon, Polygon):
            got_num += 1
            vertices_new, out_loop_new, inner_loops_new = get_periodic_geo(
                cut_polygon, x0)
            SDF = SDF_from_GEO(vertices_new, inner_loops_new,
                               out_loop_new, grid_points)
            SDFs_new.append(SDF)
            stress_new.append(stress[idx])
            geo_ids.append(geo_id)
            filter_sample_ids.append(filter_sample_id)
        try_num += 1
    return SDFs_new, stress_new, geo_ids, filter_sample_ids


# %%
geo_ids = []
SDFs_new = []
stress_new = []
filter_sample_ids = []
start, end = 0, len(sample_ids)
start_time = timeit.default_timer()
num_shifts = 3
for idx in range(start, end):
    try:
        shift_SDFs, shift_stress, shift_geo_id, shift_filter_sample_id = shift_geo(
            idx, num_shifts)
        SDFs_new.extend(shift_SDFs)
        stress_new.extend(shift_stress)
        geo_ids.extend(shift_geo_id)
        filter_sample_ids.extend(shift_filter_sample_id)
    except Exception as e:
        print(f'Error: {e}, idx={idx}')

    print(f'{idx} done, cost time: {timeit.default_timer()-start_time:.2f}s')

SDFs_new = np.array(SDFs_new).reshape(-1, *x_grids.shape)
stress_new = np.array(stress_new)
geo_ids = np.array(geo_ids)
filter_sample_ids = np.array(filter_sample_ids)
data_new = {'sdf': SDFs_new, 'stress': stress_new, 'strain': strain,
            'sample_ids': geo_ids, 'filter_sample_ids': filter_sample_ids, 'x_grids': x_grids, 'y_grids': y_grids}

file_new = f'/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/sdf_stress_strain_data_12-92_shift{num_shifts}_{start}-{end}.npz'

np.savez(file_new, **data_new)

# cut_polygon = get_polygon(vertices, out_loop, inner_loops, 0.5)
# vertices_new, out_loop_new, inner_loops_new = get_periodic_geo(
#     cut_polygon, 0.5)
# SDF = SDF_from_GEO(vertices_new, inner_loops_new, out_loop_new, grid_points)

# %%
# plt.plot(vertices_new[out_loop_new, 0], vertices_new[out_loop_new, 1], 'o-')
# for loop in inner_loops_new:
#     plt.plot(vertices_new[loop, 0], vertices_new[loop, 1], '*-')

# contours=measure.find_contours(SDF.reshape(*x_grids.shape),
#                       0, positive_orientation='high')

# contour = contours[0]
# scale_x, shift_x = np.max(contour[:, 1]) - \
#     np.min(contour[:, 1]), np.min(contour[:, 1])
# scale_y, shift_y = np.max(contour[:, 0]) - \
#     np.min(contour[:, 0]), np.min(contour[:, 0])
# plt.plot((contour[:, 1]-shift_x)/scale_x,
#          (contour[:, 0]-shift_y)/scale_y, 'r--', linewidth=2)
# for contour in contours[1:]:
#     plt.plot((contour[:, 1]-shift_x)/scale_x,
#              (contour[:, 0]-shift_y)/scale_y, 'b--', linewidth=2)

# %%


# def ndarray_to_list(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation"
# working_dir = os.path.join(
#     file_base, f'sample_{geo_id}_x0_{x0:.2f}_y0_{y0:.2f}')
# os.makedirs(working_dir, exist_ok=True)
# sample_data = {
#     'vertices': vertices_new,
#     'inner_loops': inner_loops_new,
#     'out_loop': out_loop_new,
# }
# sample_file = os.path.join(working_dir, 'sample.json')
# with open(sample_file, 'w') as f:
#     json.dump(sample_data, f, default=ndarray_to_list)

# # %%
# working_dir = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.70_y0_0.00"
# stress_strain_file = os.path.join(working_dir, 'stress_strain.csv')
# stress_strain_data = pd.read_csv(stress_strain_file)

# working_dir = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.50_y0_0.00"
# stress_strain_file = os.path.join(working_dir, 'stress_strain.csv')
# stress_strain_data1 = pd.read_csv(stress_strain_file)

# working_dir = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.05_y0_0.00"
# stress_strain_file = os.path.join(working_dir, 'stress_strain.csv')
# stress_strain_data2 = pd.read_csv(stress_strain_file)

# working_dir = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.00_y0_0.00"
# stress_strain_file = os.path.join(working_dir, 'stress_strain.csv')
# stress_strain_data3 = pd.read_csv(stress_strain_file)

# working_dir = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.55_y0_0.35"
# stress_strain_file = os.path.join(working_dir, 'stress_strain.csv')
# stress_strain_data4 = pd.read_csv(stress_strain_file)


# # plt.plot(strain, stress[idx])
# plt.plot(stress_strain_data['strain'],
#          stress_strain_data['stress'], '-', label='x0=0.7,y0=0.0')
# plt.plot(stress_strain_data1['strain'],
#          stress_strain_data1['stress'], '--', label='x0=0.5,y0=0.0')
# plt.plot(stress_strain_data2['strain'],
#          stress_strain_data2['stress'], '--', label='x0=0.05,y0=0.0')
# plt.plot(stress_strain_data3['strain'],
#          stress_strain_data3['stress'], '--', label='x0=0.00,y0=0.0')
# plt.plot(stress_strain_data4['strain'],
#          stress_strain_data4['stress'], '--', label='x0=0.55,y0=0.35')
# plt.legend()
# # %%
