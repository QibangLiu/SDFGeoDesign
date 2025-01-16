# %%
from odbAccess import *
import os
import numpy as np
import sys
from collections import defaultdict
import timeit
# %%


def extract_nodal_mises_stress(step):
    num_steps = len(step.frames)
    averaged_mises_allframes = []
    for ith in range(1, num_steps-1):
        frame = step.frames[ith]
        stress_field = frame.fieldOutputs['S']
        mises_field = stress_field.getScalarField(
            invariant=MISES)  # Von Mises stress
        element_nodal_mises = mises_field.getSubset(position=ELEMENT_NODAL)
        # Use defaultdict for efficient accumulation
        mises_sum = defaultdict(float)  # Sum of von Mises stress for each node
        count = defaultdict(int)        # Count of contributions for each node
        # Accumulate von Mises stress and counts
        for value in element_nodal_mises.values:
            node_label = value.nodeLabel
            mises_sum[node_label] += value.data  # Add the von Mises stress
            # Increment the contribution count
            count[node_label] += 1
        # Compute averaged von Mises stresses
        sorted_node_labels = sorted(mises_sum.keys())
        averaged_mises_frame = np.array(
            [mises_sum[node_label] / count[node_label] for node_label in sorted_node_labels])
        averaged_mises_allframes.append(averaged_mises_frame)
    averaged_mises_allframes = np.array(averaged_mises_allframes)
    np.save('mises_stress.npy', averaged_mises_allframes)


def extract_nodal_disp(step):
    num_steps = len(step.frames)
    Uxy_allframes = []
    for ith in range(1, num_steps-1):
        frame = step.frames[ith]
        disp_field = frame.fieldOutputs['U']
        Uxy = []
        for value in disp_field.values:
            Uxy.append(value.data)
        Uxy = np.array(Uxy)
        Uxy_allframes.append(Uxy)
    Uxy_allframes = np.array(Uxy_allframes)
    np.save('displacement.npy', Uxy_allframes)


def extract_mesh_data(assembly):
    instance = assembly.instances["UC_ASSEM"]
    nodes_coords = []
    for node in instance.nodes:
        nodes_coords.append(node.coordinates)
    nodes_coords = np.array(nodes_coords)
    elements_connectivity = []
    for element in instance.elements:
        conne = list(element.connectivity)
        last_node = conne[-1]
        conne.append(last_node)
        elements_connectivity.append(conne[:4])
    elements_connectivity = np.array(elements_connectivity, dtype=np.int32)
    mesh_data = {
        'nodes_coords': nodes_coords,
        'elements_connectivity': elements_connectivity
    }
    np.savez('mesh_data.npz', **mesh_data)


# working_dir = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/femDataR1/sec_0/sample_0"


def get_fieldData(working_dir):
    os.chdir(working_dir)
    stress_file = os.path.join(working_dir, "mises_stress.npy")
    disp_file = os.path.join(working_dir, "displacement.npy")
    mesh_file = os.path.join(working_dir, "mesh_data.npz")
    job_name = 'MyJob.odb'
    # access .odb
    odb = openOdb(job_name)
    step = odb.steps.values()[0]
    assembly = odb.rootAssembly
    if not os.path.exists(stress_file):
        extract_nodal_mises_stress(step)
    if not os.path.exists(disp_file):
        extract_nodal_disp(step)
    if not os.path.exists(mesh_file):
        extract_mesh_data(assembly)
    odb.close()


def rerun_abaqus(working_dir):
    os.chdir(working_dir)
    file_extensions = [
        ".lck", ".dat", ".msg", ".prt", ".sim", ".sta",
        ".stt", ".com", ".log", ".odb", ".cid", ".mdl", ".odb_f", ".env"
    ]
    for ext in file_extensions:
        file_path = f"MyJob{ext}"
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists("MyJob.inp"):
        os.system(
            "/projects/bbkg/Abaqus/2024/Commands/abaqus job=MyJob input=MyJob.inp")


# %%
file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/abaqus/femDataR1"
sec_id_start, sec_id_end = int(sys.argv[1]), int(sys.argv[2])
count = 0
rerun_fem = False
time_start = timeit.default_timer()
for sec_id in range(sec_id_start, sec_id_end):
    file_pre = os.path.join(file_base, f"sec_{sec_id}")
    sample_idxs = [int(f.name.split('_')[1])
                   for f in os.scandir(file_pre) if f.is_dir()]
    sample_idxs.sort()
    working_dirs = [os.path.join(
        file_pre, f"sample_{idx}") for idx in sample_idxs]
    for working_dir in working_dirs:
        odb_file = os.path.join(working_dir, "MyJob.odb")
        lck_file = os.path.join(working_dir, "MyJob.lck")
        stress_file = os.path.join(working_dir, "mises_stress.npy")
        disp_file = os.path.join(working_dir, "displacement.npy")
        mesh_file = os.path.join(working_dir, "mesh_data.npy")
        if os.path.exists(lck_file):
            os.remove(lck_file)
        if os.path.exists(odb_file) and (not os.path.exists(stress_file)
                                         or not os.path.exists(disp_file) or not os.path.exists(mesh_file)):
            if os.path.exists(lck_file) and rerun_fem:
                rerun_abaqus(working_dir)
            try:
                get_fieldData(working_dir)
                count += 1
                print(f"Got fieldData: {working_dir}")

            except Exception as e:
                print(f"Error in getting fieldData: {working_dir}")

    print("finished extract field data of section: ", sec_id)
    print(f"Time elapsed: {timeit.default_timer() - time_start}")

print(f"Got {count} strain stress curves")
