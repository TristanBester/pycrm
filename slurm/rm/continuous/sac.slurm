#!/bin/bash
# specify a partition
#SBATCH -p bigbatch
# specify number of nodes
#SBATCH -N 1
# specify the wall clock time limit for the job hh:mm:ss
#SBATCH -t 72:00:00
# specify the job name
#SBATCH -J sac
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/tbester/warehouse/slurm_logs/out/out_file.%N.%j.%a.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/tbester/warehouse/slurm_logs/error/error_file.%N.%j.%a.err
# array of 100 tasks, one per seed
#SBATCH --array=0-99%10   # seeds 0..99, max 10 run concurrently

# Prevent running on login nodes
HOSTNAME=$(hostname)
if [[ $HOSTNAME == *"login"* ]]; then
    echo "Error: This script cannot be run on a machine with 'login' in its hostname."
    exit 1
fi

# Move into workspace and set up environment
cd ~/warehouse
uv sync --group examples

cd ~/warehouse/examples/rm/continuous

# Use SLURM_ARRAY_TASK_ID as the seed
SEED=$SLURM_ARRAY_TASK_ID
echo "Running SAC with seed $SEED"

uv run sac.py --seed=$SEED
