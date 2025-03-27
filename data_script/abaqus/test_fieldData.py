# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load the data
file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/abaqus/femDataR1/sec_0/sample_0/"
mises = np.load(file_base + "mises_stress.npy")
U = np.load(file_base + "displacement.npy")
mesh = np.load(file_base + "mesh_data.npz")


# %%
coord = mesh['nodes_coords']
ele_conec = mesh['elements_connectivity']
ele_conec = ele_conec-1
# Plot the mises stress with coordinates

plt.figure(figsize=(10, 8))
plt.scatter(coord[:, 0], coord[:, 1], c=mises[20], cmap='viridis', marker='o')
plt.colorbar(label='Mises Stress')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Mises Stress Distribution')
plt.show()
# %%
