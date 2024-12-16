# %%
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.geometry import LineString, Polygon
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from shapely.ops import unary_union, polygonize, split
# %%
# geos_file = '/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/geo/geo_sdf_randv_pcn_all.pkl'
# with open(geos_file, "rb") as f:
#     geo_data = pickle.load(f)
# vertices_all = geo_data['vertices']
# inner_loops_all = geo_data['inner_loops']
# out_loop_all = geo_data['out_loop']
# points_cloud_all = geo_data['points_cloud']
# sdf_all = geo_data['sdf']
# x_grids = geo_data['x_grids']
# y_grids = geo_data['y_grids']

# %%
idx = 24
xmin, xmax = 0, 1
vertices = vertices_all[idx]
inner_loops = inner_loops_all[idx]
out_loop = out_loop_all[idx]

# %%
unit_cell_out_loops = []
unit_cell_out_loop = []
boundary_edges = []
for i in range(len(out_loop)-1):
  p1, p2 = vertices[out_loop[i]], vertices[out_loop[i+1]]
  unit_cell_out_loop.append(out_loop[i])
  if (p1[0] == xmin and p2[0] == xmin) or (p1[0] == xmax and p2[0] == xmax) or (p1[1] == xmin and p2[1] == xmin) or (p1[1] == xmax and p2[1] == xmax):
     boundary_edges.append((i, i+1))
     print(f"point {i}: {p1}, point {i+1}: {p2}")

# %%
start = 0
for be in boundary_edges:
  if be[0] == start:
    start = be[1]
    continue
  end = be[0]
  unit_cell_out_loops.append(out_loop[start:end+1])
  start = be[1]

if start < len(out_loop)-1:
  unit_cell_out_loops.append(out_loop[start:])
if len(unit_cell_out_loops[0]) < 2:
  unit_cell_out_loops = unit_cell_out_loops[1:]

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
for loop in inner_loops:
  ax.plot(vertices[loop, 0], vertices[loop, 1], 'r')
for i, loop in enumerate(unit_cell_out_loops):
  ax.plot(vertices[loop, 0], vertices[loop, 1], label=i)
# ax.legend()

# %%

vertices_full_domain = vertices.copy()
bot_domain_out_loops_noconn = unit_cell_out_loops.copy()
# left to right
for id, loop in enumerate(unit_cell_out_loops):
  add_loop = []
  num_v = len(loop)
  for i in range(num_v):
    p1 = vertices_full_domain[loop[i]]
    vert1 = np.array([p1[0]+1, p1[1], 0])
    if (i == 0 or i == num_v-1) and np.isclose(p1[0], xmin):
      dis = np.linalg.norm(vertices_full_domain - vert1, axis=1)
      close_idx = np.argmin(dis)
      if not np.isclose(dis[close_idx], 0):
        raise ValueError('left to right: Not Periodic Geo ')
      else:
        add_loop.append(close_idx)
    else:
      vertices_full_domain = np.vstack([vertices_full_domain, vert1])
      add_loop.append(len(vertices_full_domain)-1)
  bot_domain_out_loops_noconn.append(np.array(add_loop, dtype=int))

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
for loop in inner_loops:
  ax.plot(vertices[loop, 0], vertices[loop, 1], 'r')
for i, loop in enumerate(bot_domain_out_loops_noconn):
  ax.plot(vertices_full_domain[loop, 0],
          vertices_full_domain[loop, 1], label=i)
ax.legend(loc='center right')
# %%
visited = set()
bot_domain_out_loops = []
for i in range(len(bot_domain_out_loops_noconn)):
  if i in visited:
    continue
  connect_loop = (bot_domain_out_loops_noconn[i])
  # visited.add(i)
  for j in range(0, len(bot_domain_out_loops_noconn)):
    if j in visited:
      continue
    loop = bot_domain_out_loops_noconn[j]
    if connect_loop[-1] == loop[0]:
      connect_loop = np.concatenate([connect_loop, loop[1:]])
      visited.add(j)
  bot_domain_out_loops.append(connect_loop)

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
for loop in inner_loops:
  ax.plot(vertices[loop, 0], vertices[loop, 1], 'r')
for i, loop in enumerate(bot_domain_out_loops):
  ax.plot(vertices_full_domain[loop, 0],
          vertices_full_domain[loop, 1], label=i)
ax.legend(loc='center right')
# %%
bot_domain_inner_loops = inner_loops.copy()
for loop in inner_loops:
  add_loop = []
  num_v = len(loop)
  for i in range(num_v):
    p1 = vertices_full_domain[loop[i]]
    vert1 = np.array([p1[0]+1, p1[1], 0])
    vertices_full_domain = np.vstack([vertices_full_domain, vert1])
    add_loop.append(len(vertices_full_domain)-1)
  bot_domain_inner_loops.append(add_loop)

# %%
# top to bottom
full_domain_out_loops = bot_domain_out_loops.copy()
full_domain_inner_loops = bot_domain_inner_loops.copy()
for loop in bot_domain_out_loops:
  add_loop = []
  num_v = len(loop)
  for i in range(num_v):
    p1 = vertices_full_domain[loop[i]]
    vert1 = np.array([p1[0], p1[1]+1, 0])
    if (i == 0 or i == num_v-1) and np.isclose(p1[1], xmin):
      dis = np.linalg.norm(vertices_full_domain - vert1, axis=1)
      close_idx = np.argmin(dis)
      if not np.isclose(dis[close_idx], 0):
        raise ValueError('Not Periodic Geo')
      else:
        add_loop.append(close_idx)
    else:
      vertices_full_domain = np.vstack([vertices_full_domain, vert1])
      add_loop.append(len(vertices_full_domain)-1)
  full_domain_out_loops.append(np.array(add_loop, dtype=int))

# %%
visited = set()
full_domain_out_loops_connected = []
for i in range(len(full_domain_out_loops)):
  if i in visited:
    continue
  connect_loop = list(full_domain_out_loops[i])
  for j in range(0, len(full_domain_out_loops)):
    if j in visited:
      continue
    loop = full_domain_out_loops[j]
    if connect_loop[-1] == loop[0]:
      connect_loop = np.concatenate([connect_loop, loop[1:]])
      visited.add(j)
  full_domain_out_loops_connected.append(connect_loop)
# %%

for loop in bot_domain_inner_loops:
  add_loop = []
  num_v = len(loop)
  for i in range(num_v):
    p1 = vertices_full_domain[loop[i]]
    vert1 = np.array([p1[0], p1[1]+1, 0])
    vertices_full_domain = np.vstack([vertices_full_domain, vert1])
    add_loop.append(len(vertices_full_domain)-1)
  full_domain_inner_loops.append(add_loop)
# %%

vertices_full_domain = np.array(vertices_full_domain)

fig = plt.figure()
ax = plt.subplot(1, 2, 1)
for loop in inner_loops:
  ax.plot(vertices[loop, 0], vertices[loop, 1], 'r')
ax.plot(vertices[out_loop, 0], vertices[out_loop, 1], 'b')

ax = plt.subplot(1, 2, 2)
for loop in full_domain_inner_loops:
  ax.plot(vertices_full_domain[loop, 0], vertices_full_domain[loop, 1], 'r')
for i, loop in enumerate(full_domain_out_loops_connected):
  ax.plot(vertices_full_domain[loop, 0],
          vertices_full_domain[loop, 1], label=i)
ax.legend(loc='center right')
# %%
contours = []
for loop in full_domain_inner_loops:
  contours.append(LineString(vertices_full_domain[loop]))

for loop in full_domain_out_loops_connected:
  contours.append(LineString(vertices_full_domain[loop]))

# %%
x0, y0 = 0.52, 0.5
rectangle = Polygon([(x0, y0), (x0+1, y0), (x0+1, y0+1), (x0, y0+1)])
inside_all = []
outside_all = []
for contour in contours:
    inside = contour.intersection(rectangle)  # Part inside the rectangle
    outside = contour.difference(rectangle)  # Part outside the rectangle
    inside_all.append(inside)
    outside_all.append(outside)


# %%

# %%


def combine_multilines_to_linestring(geometry):
    if geometry.geom_type == 'MultiLineString':
        return linemerge(geometry.geoms)

    else:
        return geometry  # Return the original LineString if it's not MultiLineString
# Function to remove horizontal segments


# Plotting
fig, ax = plt.subplots()

# Plot the rectangle (Polygon)
ax.add_patch(plt.Polygon(list(rectangle.exterior.coords),
             fill=True, alpha=0.3, label="Rectangle", color="gray"))

# Plot each contour and its inside and outside parts
inside_lines = []
for i, contour in enumerate(contours):
    ax.plot(*combine_multilines_to_linestring(contour).xy,
            label=f"Contour {i + 1}", color="blue")
    # Plot the inside parts (if any)
    if not inside_all[i].is_empty:
        line = (
            combine_multilines_to_linestring(inside_all[i]))
        inside_lines.append(line)
        # ax.plot(
        #     *line.xy, label=f"Inside {i + 1}", color="green")

    # Plot the outside parts (if any)
    # if not outside_all[i].is_empty:
    #     # line = remove_horizontal_segments(
    #     #     combine_multilines_to_linestring(outside_all[i]))

    #     ax.plot(
    #         *.xy, label=f"Inside {i + 1}", color="red")


# Add labels, legend, and show the plot
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.legend()
plt.title("Contours and their Inside/Outside Parts Relative to Rectangle")
plt.grid(True)
plt.show()
# %%
tl = linemerge(inside_lines)
# %%

# %%
cutting_lines = tl
# Merge lines with the rectangle's boundary to create cutting edges
merged = unary_union([rectangle.boundary, cutting_lines])
resulting_polygons = list(polygonize(merged))
resulting_polygons[0]
# %%
