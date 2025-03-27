#!/bin/bash


python_script="abaqus_script_sim.py"

# cd "$working_dir"
# /projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script

working_dir=" "./femdata/abaqus/femDataR1/sec_0/sample_0"

cd "$working_dir"
/projects/bbkg/Abaqus/2024/Commands/abaqus cae -noGUI $python_script



