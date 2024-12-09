#!/bin/bash
#SBATCH --job-name=batches_ONERUN              
#SBATCH --output=0_outputs_onerun/onerun.out    # Make sure this directory is made before submitting job
#SBATCH --time=4-00:00:00              # Time limit for each task (D-HH:MM:SS)
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4           
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=16000

# Append start time info to output file
start_time=$(date +%s)
echo "Starting job ONERUN at $(date)" >> 0_outputs_onerun/timer.out

source activate /bgfs/jkitzes/ter38/.conda/envs/mapping/

python 0_make_batches_onerun.py

# Append timing info to output file
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ONERUN completed at $(date) and took $duration seconds" >> 0_outputs_onerun/timer.out
