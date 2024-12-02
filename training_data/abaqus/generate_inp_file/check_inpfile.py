# %%
import os
from natsort import natsorted
# %%
file_base="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/femDataR1"

subfolders = natsorted([f.path for f in os.scandir(file_base) if f.is_dir()])
no_inp_files=[]
for sec in subfolders:
    samples=natsorted([f.path for f in os.scandir(sec) if f.is_dir()])
    for sample in samples:
      inp_file=os.path.join(sample,"MyJob.inp")
      if not os.path.exists(inp_file):
          no_inp_files.append(inp_file)



# %%
print(no_inp_files)
print('number of missing inp files:',len(no_inp_files))
