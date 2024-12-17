#!/bin/bash
#SBATCH -J AI
#SBATCH --output=./slurm_output/fwd_models%j.log
#SBATCH --account=bdsy-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8-interactive
#SBATCH --time=05:10:00      # hh:mm:ss for the jo
#SBATCH --mem=31g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load anaconda3_gpu/23.3.1
export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
which python

srun python3 -u fwd.py
