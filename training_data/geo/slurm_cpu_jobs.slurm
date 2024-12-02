#!/bin/bash
#SBATCH -J geo_gen
#SBATCH --output=./slurm_output/cpujobs%j.log
#SBATCH --account=bcxr-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH --mem=64g #199?
###SBATCH --nodelist=cn120
# # ### GPU options ###
##SBATCH --gpus-per-node=1
##SBATCH --gpus-per-task=1

# . "/sw/external/python/anaconda3-2023.03_cuda/etc/profile.d/conda.sh"
# conda activate fenicsx
module load anaconda3_gpu/23.3.1
ID=$1
#ID=9
echo "ID: $ID"
which python
python geo.py $ID
#python gather_data.py
