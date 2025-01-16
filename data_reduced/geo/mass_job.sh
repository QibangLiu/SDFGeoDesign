#!/bin/bash


for i in {1..10}  # Adjust the range as needed
do
    sbatch slurm_cpu_jobs.slurm $i  # Submits the job script
done
