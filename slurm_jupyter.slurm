#!/bin/bash
#SBATCH -J Jupyter
#SBATCH --output=./slurm_output/jupyterServer%j.log
#SBATCH --account=bdsy-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4 ##-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=02:00:00      # hh:mm:ss for the jo
#SBATCH --mem=64g #<62.5?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load anaconda3_gpu/23.3.1
#MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT --port=8891
export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
srun jupyter-lab --no-browser  --ip=0.0.0.0 #srun
