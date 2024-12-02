# %%
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from shapely.geometry import Polygon, Point
import scipy.io as scio
import pickle
import shapely.geometry as sg
# %%
# Define the boundary as a polygon (e.g., a square)
# boundary_points = np.array([
#     [0, 0],
#     [1, 0],
#     [1, 1],
#     [0, 1]
# ])

with open("./training_data/geo_randv_all.pkl", "rb") as f:
    geo_data = pickle.load(f)
    vertices_all = geo_data['vertices']
    inner_loops_all = geo_data['inner_loops']
    out_loop_all = geo_data['out_loop']
    points_cloud_all = geo_data['points_cloud']
# %%


def min_distance_to_boundary(point, boundaries):
    """Calculate the minimum distance from a point to multiple boundaries."""
    return min(boundary.distance(Point(point)) for boundary in boundaries)


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


#
# %%

x1 = np.arange(-0.05, 0.05, 0.004)
x2 = np.linspace(0.05, 0.95, 69, endpoint=False)
x3 = np.arange(0.95, 1.05, 0.004)
x_g = np.concatenate((x1, x2, x3))
y_g = x_g
# x_g = np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 120)
# y_g = np.linspace(ymin-(ymax-ymin)*0.1, ymax+(ymax-ymin)*0.1, 120)
X_g, Y_g = np.meshgrid(x_g, y_g)
grid_points = np.vstack([X_g.ravel(), Y_g.ravel()]).T


# Reshape the signed distances to match the grid
i = 5
vert = vertices_all[i]
inner_loops = inner_loops_all[i]
out_loop = out_loop_all[i]
SDF = SDF_from_GEO(vert, inner_loops, out_loop, grid_points)
SDF = SDF.reshape(X_g.shape)
# %%
# Plot the signed distance function
plt.contourf(X_g, Y_g, SDF, levels=50, cmap='coolwarm')
plt.colorbar(label='Signed Distance')

# Plot the boundary and holes
plt.plot(vert[out_loop, 0], vert[out_loop, 1],
         color='black', linewidth=2, label='Outer Boundary')
for hole in inner_loops:
    plt.plot(vert[hole, 0], vert[hole, 1], color='black',
             linestyle='--', linewidth=2, label='Hole')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Signed Distance Function (SDF) with Holes')
plt.legend()
plt.show()
# %%
measure.find_contours(SDF, 0, positive_orientation='high')
contours = measure.find_contours(SDF, 0, positive_orientation='high')

# Plot the contours
# plt.figure()
# plt.contour(X, Y, SDF, levels=[0], colors='blue')
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contours of the Signed Distance Function (SDF)')
plt.show()
# %%
