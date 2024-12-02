#!/bin/bash
#SBATCH -J abaqus_job_inp
#SBATCH --output=./slurm_output/abaqus_inp_job%j.log
#SBATCH --account=bdsy-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH --mem=32g #199?


#script_path="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/generate_inp_file/abaqus_script_inp.py"
#script_path="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/generate_inp_file/abaqus_script_inp_50_100.py"
script_path="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/generate_inp_file/abaqus_script_inp_100_200.py"

/projects/bbkg/Abaqus/2024/Commands/abaqus cae noGUI="$script_path"
