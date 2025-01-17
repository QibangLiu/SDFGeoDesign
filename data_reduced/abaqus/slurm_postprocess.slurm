#!/bin/bash
#SBATCH -J abaqus_postprocess
#SBATCH --output=./slurm_output/abaqus_postprocess%j.log
#SBATCH --account=bdsy-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH --mem=32g #199?

sec_id_s="$1"
sec_id_e="$2"

#/projects/bbkg/Abaqus/2024/Commands/abaqus python -u extract_strain_stress.py $sec_id_s $sec_id_e
/projects/bbkg/Abaqus/2024/Commands/abaqus python -u extract_field_results.py $sec_id_s $sec_id_e
