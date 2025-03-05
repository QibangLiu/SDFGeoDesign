
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
import shapely.geometry as sg
import pickle
import time

# %%


def min_distance_to_boundary(point, boundaries):
    """Calculate the minimum distance from a point to multiple boundaries."""
    return min(boundary.distance(Point(point)) for boundary in boundaries)


# %%

def SDF_from_GEO(vertices_z, inner_loops, out_loop, grid_points):
    vertices = vertices_z[:, :2]
    # Define the boundary as a polygon (e.g., a square)
    outer_boundary = vertices[out_loop]
    holes = [vertices[loop] for loop in inner_loops]
    full_polygon = sg.Polygon(outer_boundary, holes)
    # List of all boundaries (outer boundary + holes)
    boundaries = [sg.Polygon(outer_boundary).exterior] + \
        [sg.Polygon(hole).exterior for hole in holes]
    distances = np.array([min_distance_to_boundary(p, boundaries)
                         for p in grid_points])

    # Determine if each point is inside the polygon (outer boundary but outside holes)
    inside_outer_boundary = np.array(
        [full_polygon.contains(sg.Point(p)) for p in grid_points])

    # Check if the point is within any of the holes
    inside_holes = np.array([
        any(sg.Polygon(hole).contains(sg.Point(p)) for hole in holes)
        for p in grid_points
    ])

    # Combine conditions for the signed distance function
    # Points inside the outer boundary but not in holes are considered "inside"
    signed_distances = distances * \
        np.where(inside_outer_boundary & ~inside_holes, -1, 1)
    return signed_distances


# %%
x_g = np.linspace(-0.1, 1.1, 120)
y_g = np.linspace(-0.1, 1.1, 120)
X_g, Y_g = np.meshgrid(x_g, y_g)
grid_points = np.vstack([X_g.ravel(), Y_g.ravel()]).T

# %%
geos_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_pcn_all.pkl'
with open(geos_file, "rb") as f:
    geo_data = pickle.load(f)
vertices_all = geo_data['vertices']
inner_loops_all = geo_data['inner_loops']
out_loop_all = geo_data['out_loop']

# %%
sdf = SDF_from_GEO(
    vertices_all[0], inner_loops_all[0], out_loop_all[0], grid_points)
# %%
nsamp = 200
start_time = time.time()
for i in range(nsamp):
    if i % 10 == 0:
        print(i)
    sdf = SDF_from_GEO(
        vertices_all[i], inner_loops_all[i], out_loop_all[i], grid_points)

print(f"time for one sample: {(time.time()-start_time)/nsamp}")
# %%
