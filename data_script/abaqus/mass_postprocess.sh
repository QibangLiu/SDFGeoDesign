#!/bin/bash

# for sec_id in {1..25}  # Adjust the range as needed
# do
#     echo "Submitting job for section $sec_id"
#     sbatch slurm_abq_sim.slurm $sec_id  # Submits the job script
# done

start=0 ## TODO: change to 0
end_sec=24
num_groups=25 # number of simultaneous jobs to run
sections=$(seq $start $end_sec)
total_sections=$(($end_sec - $start + 1))
group_size=$((($total_sections) / $num_groups))
remainder=$(($total_sections % $num_groups))


for group in $(seq 1 $num_groups)
do
  end=$((start + group_size - 1))
  if [ $group -le $remainder ]; then
    end=$((end + 1))
  fi
  end_python=$((end + 1))  # need +1 because python is 0-indexed, # TODO
  echo "Submitting job for sections: $start to $end_python"
  sbatch slurm_postprocess.slurm $start $end_python
  start=$((end + 1))
done
