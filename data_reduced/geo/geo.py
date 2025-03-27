"""For generate random 2D geometries with signed distance function (SDF)
    the code is messy, but it works
"""

# %%
import gstools as gs
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import points_in_poly
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
import shapely.geometry as sg
import os
import sys
import pickle
import timeit

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
# %%


def calculate_area(curve, vertices):
    x = vertices[curve, 0]
    y = vertices[curve, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


xmin, xmax = 0, 1
ymin, ymax = 0, 1
x = np.linspace(xmin, xmax, 64, endpoint=True)
y = np.linspace(ymin, ymax, 64, endpoint=True)
dx = x[1]-x[0]
dy = y[1]-y[0]
model = gs.Gaussian(dim=2, var=10, len_scale=0.15)


x_g = np.linspace(xmin-(xmax-xmin)*0.1, xmax+(xmax-xmin)*0.1, 120)
y_g = np.linspace(ymin-(ymax-ymin)*0.1, ymax+(ymax-ymin)*0.1, 120)
X_g, Y_g = np.meshgrid(x_g, y_g)
grid_points = np.vstack([X_g.ravel(), Y_g.ravel()]).T
# %%


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def has_narrow_regions(out_curve, holes, vertices, threshold):
    all_curves = [out_curve]+holes
    for curve1 in all_curves:
        for curve2 in all_curves:
            if curve1 is curve2:
                for i in range(len(curve1)-1):
                    for j in range(i+5, len(curve1)-6):
                        if euclidean_distance(vertices[curve1[i]], vertices[curve1[j]]) < threshold:
                            return True
            else:
                for i in range(len(curve1)-1):
                    for j in range(len(curve2)-1):
                        if euclidean_distance(vertices[curve1[i]], vertices[curve2[j]]) < threshold:
                            return True
    return False


def get_2D_geo(seed=None):

    srf = gs.SRF(
        model,
        generator="Fourier",
        period=(xmax-xmin, ymax-ymin),  # periodic in x and y
        mode_no=32,
        seed=seed
    )
    srf((x, y), mesh_type="structured")

    # check periodicity
    srf.field[-1, :] = srf.field[0, :]
    srf.field[:, -1] = srf.field[:, 0]

    # select a contour value
    scale = (np.max(srf.field)-np.min(srf.field))*0.4
    ave = (np.max(srf.field)+np.min(srf.field))/2
    v = np.random.uniform(ave, ave+scale)
    # v=-1.9218069268471787
    # print("v=", v)
    # v=ave
    # positive_orientation='high' clockwise contour
    contours = measure.find_contours(srf.field, v, positive_orientation='high')
    # z=np.where(srf.field>v,1,0)
    # bin_data={'x':x,'y':y,'z':z}
    # Convert contours to vertices and faces
    vertices = []
    curves = []
    left, right, bottom, top = [], [], [], []
    cbl, cbr, ctl, ctr = None, None, None, None
    print_tol = 0.08*(xmax-xmin)
    fac = 0.3
    for contour in contours:
        if np.isclose(contour[0], contour[-1]).all():
            closed = True
            ctour = contour[:-1]
        else:
            closed = False
            ctour = contour

        if not closed:
            # cross the boundary
            p1 = ctour[0]
            p2 = ctour[-1]
            xp1, yp1 = p1[1]*dx, p1[0]*dy
            xp2, yp2 = p2[1]*dx, p2[0]*dy
            if np.isclose(xp1, xmin) and np.isclose(xp2, xmax):
                return None
            elif np.isclose(xp1, xmax) and np.isclose(xp2, xmin):
                return None
            elif np.isclose(yp1, ymin) and np.isclose(yp2, ymax):
                return None
            elif np.isclose(yp1, ymax) and np.isclose(yp2, ymin):
                return None

        for i in range(len(ctour)):
            point = ctour[i]
            xp, yp = point[1]*dx, point[0]*dy
            vertices.append([xp, yp, 0])  # z-coordinate is 0 for 2D contours
            if closed:
                fac = 0.5
                if abs(xp-xmin) < fac*print_tol\
                        or abs(xp-xmax) < fac * print_tol \
                        or abs(yp-ymin) < fac * print_tol \
                        or abs(yp-ymax) < fac * print_tol:
                    return None
            elif (i == 0 or i == len(ctour)-1):
                if np.isclose(xp, xmin) and np.isclose(yp, ymin):
                    cbl = len(vertices) - 1
                elif np.isclose(xp, xmax) and np.isclose(yp, ymin):
                    cbr = len(vertices) - 1
                elif np.isclose(xp, xmin) and np.isclose(yp, ymax):
                    ctl = len(vertices) - 1
                elif np.isclose(xp, xmax) and np.isclose(yp, ymax):
                    ctr = len(vertices) - 1
                elif np.isclose(xp, xmin):
                    left.append(len(vertices) - 1)
                elif np.isclose(xp, xmax):
                    right.append(len(vertices) - 1)
                elif np.isclose(yp, ymin):
                    bottom.append(len(vertices) - 1)
                elif np.isclose(yp, ymax):
                    top.append(len(vertices) - 1)
            elif (i > 5 and i < len(ctour)-5):
                # point far from the ends, not close to the boundary

                if abs(xp-xmin) < fac*print_tol\
                        or abs(xp-xmax) < fac*print_tol \
                        or abs(yp-ymin) < fac*print_tol \
                        or abs(yp-ymax) < fac*print_tol:
                    # print("Error: close points on the boundary!!!", xp, yp)
                    return None

        if not closed:
            curves.append(
                list(range(len(vertices) - len(ctour), len(vertices))))
        else:
            curves.append(list(range(len(vertices) - len(ctour), len(vertices)))
                          + [len(vertices) - len(ctour)])

    if cbl is None:
        vertices.append([xmin, ymin, 0])
        cbl = len(vertices) - 1
    if cbr is None:
        vertices.append([xmax, ymin, 0])
        cbr = len(vertices) - 1
    if ctl is None:
        vertices.append([xmin, ymax, 0])
        ctl = len(vertices) - 1
    if ctr is None:
        vertices.append([xmax, ymax, 0])
        ctr = len(vertices) - 1
    vertices = np.array(vertices)
    bottom = bottom+[cbl, cbr]
    top = top+[ctl, ctr]
    left = left+[cbl, ctl]
    right = right+[cbr, ctr]
    left = np.array(left, dtype=int)
    right = np.array(right, dtype=int)
    bottom = np.array(bottom, dtype=int)
    top = np.array(top, dtype=int)

    left_points = np.array([vertices[i] for i in left])
    idx = np.argsort(left_points[:, 1])[::-1]
    left = left[idx]

    right_points = np.array([vertices[i] for i in right])
    idx = np.argsort(right_points[:, 1])
    right = right[idx]

    bottom_points = vertices[bottom]
    idx = np.argsort(bottom_points[:, 0])
    bottom = bottom[idx]

    top_points = vertices[top]
    idx = np.argsort(top_points[:, 0])[::-1]
    top = top[idx]
    if len(left) != len(right) or len(bottom) != len(top):
        return None
    vertices[right, 1] = vertices[left[::-1], 1]
    vertices[top, 0] = vertices[bottom[::-1], 0]

    if srf.field[0, 0] > v:  # rm low value part
        eb = 0
    else:
        eb = 1

    if srf.field[0, -1] > v:
        er = 0
    else:
        er = 1

    if srf.field[-1, -1] > v:
        et = 0
    else:
        et = 1

    if srf.field[-1, 0] > v:
        el = 0
    else:
        el = 1

    bottom_edges, top_edges, left_edges, right_edges = [], [], [], []
    for i in range(len(bottom)-1):
        if i % 2 == eb:
            bottom_edges.append([bottom[i], bottom[i+1]])

    for i in range(len(right)-1):
        if i % 2 == er:
            right_edges.append([right[i], right[i+1]])
    for i in range(len(top)-1):
        if i % 2 == et:
            top_edges.append([top[i], top[i+1]])
    for i in range(len(left)-1):
        if i % 2 == el:
            left_edges.append([left[i], left[i+1]])
    # find out the closed edges
    closed_edges, open_edges = [], []
    for curve in curves:
        if curve[0] == curve[-1]:
            closed_edges.append(curve)
        else:
            open_edges.append(curve)
    open_edges = bottom_edges+right_edges+top_edges+left_edges+open_edges
    if len(open_edges) == 0:
        return None
    envolved_edges = open_edges[0]
    open_edges.remove(open_edges[0])

    while True:
        got_it = False
        for idx, edge in enumerate(open_edges):
            if envolved_edges[-1] == edge[0]:
                envolved_edges = envolved_edges+edge[1:]
                open_edges.remove(edge)
                got_it = True
                break
            elif envolved_edges[-1] == edge[-1]:
                envolved_edges = envolved_edges+edge[::-1][1:]
                open_edges.remove(edge)
                got_it = True
                break
        if (idx == len(open_edges)-1 and got_it == False) or len(open_edges) == 0:
            break
    if len(open_edges) > 0:
        return None
    # rememove the small closed edges
    closed_e = []
    for curve in closed_edges:
        area = calculate_area(curve, vertices)
        r = np.sqrt(area/np.pi)
        if r > 0.5*print_tol:
            p = np.mean(vertices[curve], axis=0)
            inside = points_in_poly(p[None,], vertices[envolved_edges])
            if inside[0]:
                closed_e.append(curve)

    # closed_e.append(envolved_edges)

    interior_edges = []
    boundary_edges = []
    verts = []
    for edge in closed_e:
        ctour = edge[:-1]
        i = 0
        edge_start = len(verts)
        while i < (len(ctour)):
            p1 = vertices[ctour[i]]
            if i == len(ctour)-1:
                verts.append(p1)
                break
            p1 = vertices[ctour[i]]
            p2 = vertices[ctour[i+1]]
            distance = np.linalg.norm(p1 - p2)
            # merge these close points
            if distance < 0.15*(dx+dy):
                verts.append((p1+p2)*0.5)
                i += 1
            else:
                verts.append(p1)
            i += 1

        interior_edges.append(np.array(list(range(edge_start, len(verts)))
                                       + [edge_start], dtype=int))

    ctour = envolved_edges[:-1]
    edge_start = len(verts)
    i = 0
    while i < (len(ctour)):
        p1 = vertices[ctour[i]]
        if i == len(ctour)-1:
            verts.append(p1)
            break
        p1 = vertices[ctour[i]]
        p2 = vertices[ctour[i+1]]
        distance = np.linalg.norm(p1 - p2)
        # merge these close points
        if distance < print_tol:
            if (np.isclose(p1[0], xmin) and np.isclose(p2[0], xmin)) \
                    or (np.isclose(p1[0], xmax) and np.isclose(p2[0], xmax)) \
                    or (np.isclose(p1[1], ymin) and np.isclose(p2[1], ymin)) \
                    or (np.isclose(p1[1], ymax) and np.isclose(p2[1], ymax)):
                return None

        if distance < 0.15*(dx+dy):

            p1_on_boundary = (np.isclose(p1[0], xmin) or np.isclose(
                p1[0], xmax) or np.isclose(p1[1], ymin) or np.isclose(p1[1], ymax))
            p2_on_boundary = (np.isclose(p2[0], xmin) or np.isclose(
                p2[0], xmax) or np.isclose(p2[1], ymin) or np.isclose(p2[1], ymax))
            if p1_on_boundary:
                verts.append(p1)
            elif p2_on_boundary:
                verts.append(p2)
            else:
                verts.append((p1+p2)*0.5)
            i += 1
        else:
            verts.append(p1)
        i += 1

    boundary_edges = np.array(list(range(edge_start, len(verts)))
                              + [edge_start], dtype=int)

    verts = np.array(verts)

    if has_narrow_regions(boundary_edges, interior_edges, verts, fac*print_tol):
        # print("Error: narrow regions!!!")
        return None

    X, Y = np.meshgrid(np.linspace(xmin, xmax, 32),
                       np.linspace(xmin, xmax, 32))
    # XY=np.array([X.flatten(),Y.flatten()]).T
    # XY=np.concatenate([XY,np.zeros((XY.shape[0],1))],axis=1)
    top_row = np.column_stack((X[0, :], Y[0, :]))
    bottom_row = np.column_stack((X[-1, :], Y[-1, :]))
    left_column = np.column_stack((X[:, 0], Y[:, 0]))
    right_column = np.column_stack((X[:, -1], Y[:, -1]))

    # Combine boundary points
    XY = np.vstack((top_row, bottom_row, left_column, right_column))

    XY = np.concatenate([XY, np.zeros((XY.shape[0], 1))], axis=1)
    inside_boundary_outside_interior = points_in_poly(
        XY, verts[boundary_edges])

    for edges in interior_edges:
        inside_interior = points_in_poly(XY, verts[edges])
        inside_boundary_outside_interior = np.logical_and(
            inside_boundary_outside_interior, ~inside_interior)
    XY = XY[inside_boundary_outside_interior]
    tree = cKDTree(verts)
    distances, _ = tree.query(XY)
    not_close_points = XY[distances > 0.4*(dx+dy)]
    points_cloud = np.concatenate([verts, not_close_points], axis=0)

    sdf = SDF_from_GEO(verts, interior_edges, boundary_edges, grid_points)
    return verts, interior_edges, boundary_edges, points_cloud, sdf


# %%
# seed=np.random.randint(1000)
vertices_all, inner_loops_all, out_loop_all, points_cloud_all, sdf_all = [], [], [], [], []
file_base = './geodata'
os.makedirs(file_base, exist_ok=True)
file_name = os.path.join(file_base, 'geo_sdf_randv'+'.pkl')
start_time = timeit.default_timer()
while len(vertices_all) < 5000:
    geo = get_2D_geo(seed=None)
    if geo is not None:
        vertices, inner_loops, out_loop, points_cloud, sdf = geo
        vertices_all.append(vertices)
        inner_loops_all.append(inner_loops)
        out_loop_all.append(out_loop)
        points_cloud_all.append(points_cloud)
        sdf_all.append(sdf)
        print(f"Get {len(vertices_all)} geometries!!!")
    if len(vertices_all) % 100 == 1:
        geo_data = {'vertices': vertices_all, 'inner_loops': inner_loops_all,
                    'out_loop': out_loop_all, 'points_cloud': points_cloud_all,
                    'sdf': sdf_all, 'x_grids': X_g, 'y_grids': Y_g}
        with open(file_name, "wb") as f:
            pickle.dump(geo_data, f)
        time_end = timeit.default_timer()
        print('time cost', time_end-start_time, 's')

time_end = timeit.default_timer()
print('time cost', time_end-start_time, 's')

# %%
# fig, ax = plt.subplots(figsize=(4.8, 4.8))
# for curve in inner_loops:
#     ax.plot(vertices[curve, 0], vertices[curve, 1], 'r', linewidth=2)
# ax.plot(vertices[out_loop, 0], vertices[out_loop, 1], 'b', linewidth=2)
# ax.plot(points_cloud[:, 0], points_cloud[:, 1], 'o', color='r', markersize=1)
# ax.grid(True)
# # %%
# SDF = sdf.reshape(X_g.shape)
# plt.contourf(X_g, Y_g, SDF, levels=50, cmap='coolwarm')
# plt.colorbar(label='Signed Distance')
# plt.axis('equal')
# # Plot the boundary and holes
# plt.plot(vertices[out_loop, 0], vertices[out_loop, 1],
#          color='black', linewidth=2, label='Outer Boundary')
# for curve in inner_loops:
#     plt.plot(vertices[curve, 0], vertices[curve, 1], color='black',
#              linestyle='--', linewidth=2, label='Hole')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Signed Distance Function (SDF) with Holes')
# plt.legend()
# plt.show()
# # %%

# measure.find_contours(SDF, 0, positive_orientation='high')
# contours = measure.find_contours(SDF, 0, positive_orientation='high')

# for contour in contours:
#     plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Contours of the Signed Distance Function (SDF)')
# plt.axis('equal')
