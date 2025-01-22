#!/bin/bash
#SBATCH -J geo_gen
#SBATCH --output=./slurm_output/augmentation_data%j.log
#SBATCH --account=bcxr-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --mem=96g #199?
###SBATCH --nodelist=cn120
# # ### GPU options ###
##SBATCH --gpus-per-node=1
##SBATCH --gpus-per-task=1

# . "/sw/external/python/anaconda3-2023.03_cuda/etc/profile.d/conda.sh"
# conda activate fenicsx
module load anaconda3_gpu/23.3.1
export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
# ID=$1
# #ID=9
# echo "ID: $ID"
which python
# python geo.py $ID
#python gather_data.py
#srun python -u ../gather_sdf_strainstress.py

srun python -u augmentation_data.py
#srun python -u gather_augmentaition_shift4_ID0-10000.py
