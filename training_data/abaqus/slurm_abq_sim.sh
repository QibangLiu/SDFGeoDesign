#!/bin/bash
#SBATCH -J abaqus_job
#SBATCH --output=./slurm_output/abaqus_job%j.log
#SBATCH --account=bdsy-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --mem=2g #199?
# Check if a section ID argument is provided
# if [ -z "$1" ]; then
#     echo "Usage: $0 <section_id>"
#     exit 1
# fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <arg1> <arg2>"
    exit 1
fi

# Set variables
sec_id_s="$1"
sec_id_e="$2"

# Loop through each section ID
for sec_id in $(seq $sec_id_s $sec_id_e); do
    # Set working directory
    file_base="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/femDataR1/"
    working_pre="${file_base}/sec_${sec_id}"
    # List subdirectories in sorted natural order
    mapfile -t working_dirs < <(find "$working_pre" -mindepth 1 -maxdepth 1 -type d | sort -V)

    # Loop through each working directory
    for working_dir in "${working_dirs[@]}"; do
        odb_file="${working_dir}/MyJob.odb"
        # clean files
        if [ -f "${working_dir}/MyJob.lck" ]; then
            echo "Cleaning dir: $working_dir"
            cd $working_dir
            rm MyJob.lck
            rm MyJob.dat
            rm MyJob.msg
            rm MyJob.prt
            rm MyJob.sim
            rm MyJob.sta
            rm MyJob.stt
            rm MyJob.com
            rm MyJob.log
            rm MyJob.odb
            rm -r MyJob.simdir*
            rm MyJob.*.SMABulk
            rm MyJob.cid
            rm MyJob.dat
            rm MyJob.mdl
            rm MyJob.odb_f
            rm MyJob.env
        fi
        if [ ! -f "$odb_file" ]; then
            echo "Processing: $working_dir"
            cd "$working_dir"
            # Run Abaqus simulation
            inp_file="${working_dir}/MyJob.inp"
            if [ -f "$inp_file" ]; then
                time_s=$(date +%s)  # Start time in seconds
                /projects/bbkg/Abaqus/2024/Commands/abaqus job=MyJob input=MyJob.inp
                # Calculate and display elapsed time
                time_e=$(date +%s)
                elapsed_time=$((time_e - time_s))
                echo "Abaqus simulation finished, time cost: ${elapsed_time} seconds"
            else
                echo "No input file found in $working_dir"
            fi
        fi
    done
done

# file_base="/work/hdd/bbpq/qibang/repository_Wbbpq/TRAINING_DATA/GeoSDF/abaqus/femDataR1/"
# working_pre="${file_base}sec_${sec_id}"
# # List subdirectories in sorted natural order
# mapfile -t working_dirs < <(find "$working_pre" -mindepth 1 -maxdepth 1 -type d | sort -V)

# # Loop through each working directory
# for working_dir in "${working_dirs[@]}"; do
#     odb_file="${working_dir}/MyJob.odb"
#     # clean files
#     if [ -f "${working_dir}/MyJob.lck" ]; then
#         echo "Cleaning dir: $working_dir"
#         cd $working_dir
#         rm MyJob.lck
#         rm MyJob.dat
#         rm MyJob.msg
#         rm MyJob.prt
#         rm MyJob.sim
#         rm MyJob.sta
#         rm MyJob.stt
#         rm MyJob.com
#         rm MyJob.log
#         rm MyJob.odb
#         rm -r MyJob.simdir*
#         rm MyJob.*.SMABulk
#         rm MyJob.cid
#         rm MyJob.dat
#         rm MyJob.mdl
#     fi
#     if [ ! -f "$odb_file" ]; then
#         echo "Processing: $working_dir"
#         cd "$working_dir"
#         # Run Abaqus simulation
#         inp_file="${working_dir}/MyJob.inp"
#         if [ -f "$inp_file" ]; then
#             time_s=$(date +%s)  # Start time in seconds
#             /projects/bbkg/Abaqus/2024/Commands/abaqus job=MyJob input=MyJob.inp
#             # Calculate and display elapsed time
#             time_e=$(date +%s)
#             elapsed_time=$((time_e - time_s))
#             echo "Abaqus simulation finished, time cost: ${elapsed_time} seconds"
#         else
#             echo "No input file found in $working_dir"
#         fi
#     fi
# done
