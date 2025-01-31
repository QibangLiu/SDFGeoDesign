
# %%
from scipy.spatial.distance import pdist, squareform
import os
import json
import numpy as np
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ABAQUS_SCRIPT = os.path.join(SCRIPT_PATH, 'abaqus_script_sim.py')
# %%


def check_boundary(points, min_x=0, min_y=0, max_x=1, max_y=1, tol=1e-6):

    # Check if each point is close to the min/max x or y values
    mask = np.any([
        np.isclose(points[:, 0], min_x, atol=tol),  # Close to min_x
        np.isclose(points[:, 0], max_x, atol=tol),  # Close to max_x
        np.isclose(points[:, 1], min_y, atol=tol),  # Close to min_y
        np.isclose(points[:, 1], max_y, atol=tol)   # Close to max_y
    ], axis=0)

    return np.where(mask)[0]


def merge_points(points, threshold=0.0048):
    """ need merge the points that are too close,
    before run abaqus simulation, otherwise, abaqus will report error
    """
    dist_matrix = squareform(pdist(points))
    merged = []
    visited = set()
    for i, p in enumerate(points):
        if i in visited:
            continue
        close_indices = np.where(dist_matrix[i] < threshold)[0]
        idx_bc = check_boundary(points[close_indices])
        if len(idx_bc) > 0:
            merged.append(points[close_indices[idx_bc[0]]])
        else:
            merged.append(points[close_indices].mean(axis=0))
        visited.update(close_indices)
    merged.append(merged[0])  # close the loop
    return np.array(merged)


# %%
def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_geo_file(geo_contours, working_dir):
    shell_c, hole_c = geo_contours
    shell_contours = [merge_points(contour) for contour in shell_c]
    holes_contours = [merge_points(contour) for contour in hole_c]
    verts = []
    interior_edges = []
    for contour in holes_contours:
        ctour = contour[::-1, ::-1]
        edge_start = len(verts)
        verts.extend(ctour.tolist())
        interior_edges.append(np.array(list(range(edge_start, len(verts)))
                                       + [edge_start], dtype=int))

    ctour = shell_contours[0][:-1, ::-1]
    edge_start = len(verts)
    verts.extend(ctour.tolist())
    shell_edges = np.array(
        list(range(edge_start, len(verts))) + [edge_start], dtype=int)
    verts = np.array(verts)
    verts = np.hstack([verts, np.zeros_like(verts[:, :1])])
    sample_data = {'vertices': verts,
                   'inner_loops': interior_edges, 'out_loop': shell_edges}
    sample_file = os.path.join(working_dir, 'sample.json')
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, default=ndarray_to_list)


# %%
def run_abaqus_sim(geo_contours, working_dir, abaqus_exe=None, run_abaqus=False):
    """
    Args:
        geo_contours: tuple, (shell_contours, hole_contours)
        working_dir: str, the working directory
        abaqus_exe: str, the path to the abaqus executable
        run_abaqus: bool, whether to run the abaqus simulation,if True, run the simulation,
                even if the simulation result already exists
    Returns:
        femdata: np.ndarray, the stress-strain data
    """
    if len(geo_contours[0]) > 1:
        raise ValueError("Only one shell contour is supported")

    os.makedirs(working_dir, exist_ok=True)
    prev_dir = os.getcwd()
    ss_file = os.path.join(working_dir, 'stress_strain.csv')
    if not os.path.exists(ss_file) or run_abaqus:
        write_geo_file(geo_contours, working_dir)
        # Define the Abaqus command
        abaqus_command = f"{abaqus_exe} cae -noGUI {ABAQUS_SCRIPT}"
        # Execute the Abaqus command
        os.chdir(working_dir)
        os.system(abaqus_command)
        os.chdir(prev_dir)
    if not os.path.exists(ss_file):
        raise ValueError("Failed to run Abaqus simulation!!!")
    else:
        femdata = np.loadtxt(ss_file, delimiter=',', skiprows=1)
    return femdata
