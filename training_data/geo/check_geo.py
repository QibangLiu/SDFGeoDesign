# %%
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

with open("./training_data/geo_sdf_randv_all.pkl", "rb") as f:
    geo_data = pickle.load(f)
    vertices_all=geo_data['vertices']
    inner_loops_all=geo_data['inner_loops']
    out_loop_all=geo_data['out_loop']
    points_cloud_all=geo_data['points_cloud']
    sdf_all=geo_data['sdf']
    x_grid=geo_data['x_grids']
    y_grid=geo_data['y_grids']
# %%
i=np.random.randint(0,len(vertices_all))
vertices=vertices_all[i]
inner_loops=inner_loops_all[i]
out_loop=out_loop_all[i]
points_cloud=points_cloud_all[i]
sdf=sdf_all[i]

X, Y = x_grid, y_grid
grid_points = np.vstack([X.ravel(), Y.ravel()]).T

# Reshape the signed distances to match the grid
SDF = sdf.reshape(X.shape)

# Plot the signed distance function
plt.contourf(X, Y, SDF, levels=50, cmap='coolwarm')
plt.colorbar(label='Signed Distance')

# Plot the boundary and holes
plt.plot(vertices[out_loop,0],vertices[out_loop,1], color='black', linewidth=2, label='Outer Boundary')
for hole in inner_loops:
    plt.plot(vertices[hole,0],vertices[hole,1], color='black', linestyle='--', linewidth=2, label='Hole')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.title('Signed Distance Function (SDF) with Holes')
plt.legend()

# %%
from skimage import measure
measure.find_contours(SDF,0,positive_orientation='high')
contours = measure.find_contours(SDF, 0, positive_orientation='high')

# Plot the contours
# plt.figure()
# plt.contour(X, Y, SDF, levels=[0], colors='blue')

contour=contours[0]
scale_x,shift_x = np.max(contour[:, 1]) - np.min(contour[:, 1]),np.min(contour[:, 1])
scale_y,shift_y = np.max(contour[:, 0]) - np.min(contour[:, 0]),np.min(contour[:, 0])
plt.plot((contour[:, 1]-shift_x)/scale_x, (contour[:, 0]-shift_y)/scale_y, 'r', linewidth=2)
for contour in contours[1:]:
    plt.plot((contour[:, 1]-shift_x)/scale_x, (contour[:, 0]-shift_y)/scale_y, 'b', linewidth=2)
plt.scatter(points_cloud[:,0],points_cloud[:,1], color='g', s=10)

plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.title('Contours of the Signed Distance Function (SDF)')


# %%
