#!/bin/bash


python_script="/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/data_reduced/abaqus/abaqus_script_sim.py"

# working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/visual_Simu"
# cd "$working_dir"
# /projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script

working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/abaqus/femDataR1/sec_0/sample_0"
working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/abaqus/femDataR1/sec_0/sample_702"

cd "$working_dir"
/projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script



