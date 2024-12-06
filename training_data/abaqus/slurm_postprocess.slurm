#!/bin/bash
#SBATCH -J abaqus_postprocess
#SBATCH --output=./slurm_output/abaqus_postprocess%j.log
#SBATCH --account=bdsy-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=00:10:00      # hh:mm:ss for the job
#SBATCH --mem=2g #199?


#/projects/bbkg/Abaqus/2024/Commands/abaqus python extract_strain_stress.py 190 200
srun python -u test_print.py
