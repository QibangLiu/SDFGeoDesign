# %%
from shapely.geometry import Polygon
from skimage import measure
import numpy as np
# %%


def remove_consecutive_duplicates(contour, tol=1e-3):
    """
    Removes consecutive duplicate points in a contour based on a given tolerance.

    Parameters:
        contour (np.ndarray): An (N, 2) array of (x, y) points.
        tol (float): The tolerance for considering points as duplicates.

    Returns:
        np.ndarray: Contour with consecutive duplicates removed.
    """
    if len(contour) == 0:
        return contour  # Return empty if input is empty

    # Compute distance between consecutive points
    diffs = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    # Keep first point and remove consecutive close points
    mask = np.hstack([[True], diffs > tol])
    return contour[mask]


def simplify_shell_contour(contour, min_x=0, max_x=1, min_y=0, max_y=1, tol=5e-3):
    contour = remove_consecutive_duplicates(contour)
    x_coords = contour[:, 1]
    y_coords = contour[:, 0]
    to_be_removed = []
    for i in range(1, len(contour) - 1):
        if np.isclose(x_coords[i-1:i+2], min_x, atol=tol).all() or \
           np.isclose(x_coords[i-1:i+2], max_x, atol=tol).all() or \
           np.isclose(y_coords[i-1:i+2], min_y, atol=tol).all() or \
           np.isclose(y_coords[i-1:i+2], max_y, atol=tol).all():
            to_be_removed.append(i)

    contour = np.delete(contour, to_be_removed, axis=0)
    return contour


def change_start_contour(contour, min_x=0, max_x=1, min_y=0, max_y=1, tol=2e-2, tol_mm=1.0e-2):
    x_coords = contour[:, 1]
    y_coords = contour[:, 0]
    index_not_boundary = np.where(~np.isclose(x_coords, min_x, atol=tol)
                                  & ~np.isclose(x_coords, max_x, atol=tol)
                                  & ~np.isclose(y_coords, min_y, atol=tol)
                                  & ~np.isclose(y_coords, max_y, atol=tol))[0]
    if len(index_not_boundary) > 0:
        index = index_not_boundary[0]
        contour = np.concatenate([contour[index:], contour[1:index+1]], axis=0)

    x_coords = contour[:, 1]
    y_coords = contour[:, 0]
    left_indices = np.where(np.isclose(x_coords, min_x, atol=tol_mm))[0]
    right_indices = np.where(np.isclose(x_coords, max_x, atol=tol_mm))[0]
    bottom_indices = np.where(np.isclose(y_coords, min_y, atol=tol_mm))[0]
    top_indices = np.where(np.isclose(y_coords, max_y, atol=tol_mm))[0]
    contour[left_indices, 1] = min_x
    contour[right_indices, 1] = max_x
    contour[bottom_indices, 0] = min_y
    contour[top_indices, 0] = max_y

    return contour


def check_periodic(contours,  min_x=0, max_x=1, min_y=0, max_y=1, tol_mm=5e-3, tol_p=0.05):
    """check periodic boundary, and also average the points coordiates on the boundary"""
    left_y = np.empty(0, dtype=float)
    right_y = np.empty(0, dtype=float)
    top_x = np.empty(0, dtype=float)
    bottom_x = np.empty(0, dtype=float)
    left_locs, right_locs, top_locs, bottom_locs = [], [], [], []
    for i, contour in enumerate(contours):
        x_coords = contour[:, 1]
        y_coords = contour[:, 0]

        # Extract boundary points
        left_indices = np.where(np.isclose(x_coords, min_x, atol=tol_mm))[0]
        right_indices = np.where(np.isclose(x_coords, max_x, atol=tol_mm))[0]
        bottom_indices = np.where(np.isclose(y_coords, min_y, atol=tol_mm))[0]
        top_indices = np.where(np.isclose(y_coords, max_y, atol=tol_mm))[0]

        left_y = np.append(left_y, y_coords[left_indices])
        left_locs.extend([(i, idx) for idx in left_indices])
        right_y = np.append(right_y, y_coords[right_indices])
        right_locs.extend([(i, idx) for idx in right_indices])
        top_x = np.append(top_x, x_coords[top_indices])
        top_locs.extend([(i, idx) for idx in top_indices])
        bottom_x = np.append(bottom_x, x_coords[bottom_indices])
        bottom_locs.extend([(i, idx) for idx in bottom_indices])

    left_id = np.argsort(left_y)
    left_y = left_y[left_id]
    left_locs = np.array(left_locs)[left_id]

    right_id = np.argsort(right_y)
    right_y = right_y[right_id]
    right_locs = np.array(right_locs)[right_id]

    top_id = np.argsort(top_x)
    top_x = top_x[top_id]
    top_locs = np.array(top_locs)[top_id]

    bottom_id = np.argsort(bottom_x)
    bottom_x = bottom_x[bottom_id]
    bottom_locs = np.array(bottom_locs)[bottom_id]

    is_periodic_x, is_periodic_y = False, False
    if len(left_y) == len(right_y) and np.all(np.isclose(left_y, right_y, atol=tol_p)):
        is_periodic_x = True
        # print("Left and right boundary  match")
    if len(top_x) == len(bottom_x) and np.all(np.isclose(top_x, bottom_x, atol=tol_p)):
        is_periodic_y = True
        # print("Top and bottom boundary match")

    if is_periodic_x and is_periodic_y:
        for i, (l, r) in enumerate(zip(left_locs, right_locs)):
            if contours[l[0]][l[1], 0] == min_y or contours[l[0]][l[1], 0] == max_y:
                contours[r[0]][r[1], 0] = contours[l[0]][l[1], 0]
            elif contours[r[0]][r[1], 0] == min_y or contours[r[0]][r[1], 0] == max_y:
                contours[l[0]][l[1], 0] = contours[r[0]][r[1], 0]
            else:
                y_ave = (left_y[i]+right_y[i])*0.5
                contours[l[0]][l[1], 0] = y_ave
                contours[r[0]][r[1], 0] = y_ave
        for i, (t, b) in enumerate(zip(top_locs, bottom_locs)):
            if contours[t[0]][t[1], 1] == min_x or contours[t[0]][t[1], 1] == max_x:
                contours[b[0]][b[1], 1] = contours[t[0]][t[1], 1]
            elif contours[b[0]][b[1], 1] == min_x or contours[b[0]][b[1], 1] == max_x:
                contours[t[0]][t[1], 1] = contours[b[0]][b[1], 1]
            else:
                x_ave = (top_x[i]+bottom_x[i])*0.5
                contours[t[0]][t[1], 1] = x_ave
                contours[b[0]][b[1], 1] = x_ave
        return True
    else:
        return False


def classify_contours(sdf):
    contours = measure.find_contours(
        sdf, 0, positive_orientation="high")
    polygons = [Polygon(contour) for contour in contours]
    holes_ids = []
    for i in range(len(polygons)):
        for j in range(len(polygons)):
            if i != j and polygons[j].contains(polygons[i]):
                holes_ids.append(i)
                break
    shell_ids = [i for i in range(len(polygons)) if i not in holes_ids]
    # because the sdf is evaluate on 120x120 grid， [-0.1,1.1]x[-0.1,1.1]
    shell_contours = [(contours[i]*1.2/119-0.1) for i in shell_ids]
    holes_contours = [(contours[i]*1.2/119-0.1) for i in holes_ids]
    shell_contours = [change_start_contour(
        contour) for contour in shell_contours]
    shell_contours = [simplify_shell_contour(
        contour) for contour in shell_contours]

    is_periodic = check_periodic(shell_contours)

    return shell_contours, holes_contours, is_periodic


# %%

def filter_out_unit_cell(sdfs_all, no_isolated=True):
    geo_contours = []
    periodic_ids = []
    for i, sdf in enumerate(sdfs_all):
        shell_contours, holes_contours, is_periodic = classify_contours(sdf)
        if is_periodic:
            if no_isolated:
                if len(shell_contours) == 1:
                    periodic_ids.append(i)
                    geo_contours.append((shell_contours, holes_contours))
            else:
                periodic_ids.append(i)
                geo_contours.append((shell_contours, holes_contours))

    return geo_contours, periodic_ids
