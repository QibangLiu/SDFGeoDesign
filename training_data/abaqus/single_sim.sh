#!/bin/bash


python_script="/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/training_data/abaqus/abaqus_script_sim.py"

# working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.00_y0_0.00"
# cd "$working_dir"
# /projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script


# working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.05_y0_0.00"
# cd "$working_dir"
# /projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script

# working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.50_y0_0.00"
# cd "$working_dir"
# /projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script


# working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/test_augmentation/sample_0_x0_0.70_y0_0.00"
# cd "$working_dir"
# /projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script

working_dir="/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/abaqus/visual_Simu"
cd "$working_dir"
/projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script
