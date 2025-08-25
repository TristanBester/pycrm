#!/bin/bash
# specify a partition
#SBATCH -p bigbatch 
# specify number of nodes
#SBATCH -N 1
# specify the wall clock time limit for the job hh:mm:ss
#SBATCH -t 72:00:00
# specify the job name
#SBATCH -J ctd3
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/tbester/warehouse/slurm_logs/out/out_file.%N.%j.%a.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/tbester/warehouse/slurm_logs/error/error_file.%N.%j.%a.err
# array of 100 tasks, one per seed
#SBATCH --array=0-99%10   # seeds 0..99, max 10 run concurrently

# Get the hostname of the current machine
HOSTNAME=$(hostname)

# Check if the hostname contains the substring "login"
if [[ $HOSTNAME == *"login"* ]]; then
    echo "Error: This script cannot be run on a machine with 'login' in its hostname."
    exit 1
fi

cd ~/warehouse
uv sync --group examples

cd ~/warehouse/examples/crm/continuous
SEED=$SLURM_ARRAY_TASK_ID
echo "Running C-TD3 with seed $SEED"
uv run ctd3.py --seed=$SEED