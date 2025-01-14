#!/bin/bash
#SBATCH -J gather_data
#SBATCH --output=./slurm_output/gather_data%j.log
#SBATCH --account=bdsy-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=01:00:00      # hh:mm:ss for the job
#SBATCH --mem=128g #199?
module load anaconda3_gpu/23.3.1
export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
which python


#srun python3 -u gather_sdf_strainstress.py
srun python3 -u gather_pc_fieldoutput.py --setID=3
