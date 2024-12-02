# %%
from odbAccess import *
import os

import os
import numpy as np
import sys


# %%
def get_strain_stress(working_dir):
    os.chdir(working_dir)
    job_name = 'MyJob'
    odb = job_name + '.odb'
    # access .odb
    odb = openOdb(odb)
    step=odb.steps.keys()[0]  # step_name
    num_steps=len(odb.steps[step].frames)
    his_strain=np.zeros(num_steps)
    his_stress=np.zeros(num_steps)
    # pull values
    num_nodes=0
    for iter, key in enumerate(odb.steps[step].historyRegions.keys()):
        if key != 'Assembly ASSEMBLY':
            RF2_data = np.array(odb.steps[step].historyRegions[key].historyOutputs['RF2'].data)
            U2_data=np.array(odb.steps[step].historyRegions[key].historyOutputs['U2'].data)
            if len(RF2_data)==num_steps:
                his_stress += RF2_data[:, 1]
                his_strain += U2_data[:, 1]
                num_nodes += 1
    his_strain /= num_nodes
    stress_strain_curve = np.stack((-his_strain[:-1], -his_stress[:-1]), axis=1)
    np.savetxt(os.path.join('./', 'stress_strain.csv'), stress_strain_curve, delimiter = ',', comments = '', header = 'strain,stress')
    odb.close()


def rerun_abaqus(working_dir):
    os.chdir(working_dir)
    file_extensions = [
    ".lck", ".dat", ".msg", ".prt", ".sim", ".sta",
    ".stt", ".com", ".log", ".odb", ".cid", ".mdl",".odb_f",".env"
    ]
    for ext in file_extensions:
        file_path = f"MyJob{ext}"
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists("MyJob.inp"):
        os.system("/projects/bbkg/Abaqus/2024/Commands/abaqus job=MyJob input=MyJob.inp")

# %%
file_base="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/femDataR1"
sec_id_start, sec_id_end = int(sys.argv[1]), int(sys.argv[2])
count=0
rerun_fem=False
for sec_id in range(sec_id_start, sec_id_end):
    file_pre=os.path.join(file_base, f"sec_{sec_id}")
    sample_idxs=[int(f.name.split('_')[1]) for f in os.scandir(file_pre) if f.is_dir()]
    sample_idxs.sort()
    working_dirs = [os.path.join(file_pre, f"sample_{idx}") for idx in sample_idxs]
    for working_dir in working_dirs:
        odb_file=os.path.join(working_dir, "MyJob.odb")
        lck_file=os.path.join(working_dir, "MyJob.lck")
        if os.path.exists(odb_file) and not os.path.exists(os.path.join(working_dir, "stress_strain.csv")):
            if os.path.exists(lck_file) and rerun_fem:
                rerun_abaqus(working_dir)
            try:
                get_strain_stress(working_dir)
                count+=1
                print(f"Got strain stress curve: {working_dir}")
            except Exception as e:
                print(f"Error in getting strain stress curve:: {working_dir}")


print(f"Got {count} strain stress curves")
