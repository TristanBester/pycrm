#!/bin/bash
# specify a partition
#SBATCH -p bigbatch 
# specify number of nodes
#SBATCH -N 1
# specify the wall clock time limit for the job hh:mm:ss
#SBATCH -t 72:00:00
# specify the job name
#SBATCH -J csac
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/tbester/warehouse/slurm_logs/out/out_file.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/tbester/warehouse/slurm_logs/error/error_file.%N.%j.err

# Get the hostname of the current machine
HOSTNAME=$(hostname)

# Check if the hostname contains the substring "login"
if [[ $HOSTNAME == *"login"* ]]; then
    echo "Error: This script cannot be run on a machine with 'login' in its hostname."
    exit 1
fi

cd ~/warehouse
uv sync --extra experiments

cd ~/warehouse/examples/crm/continuous
uv run csac.py --seed 0 & 
uv run csac.py --seed 1 & 
uv run csac.py --seed 2 & 
uv run csac.py --seed 3 & 
uv run csac.py --seed 4 & 
uv run csac.py --seed 5 & 
uv run csac.py --seed 6 & 
uv run csac.py --seed 7 & 
uv run csac.py --seed 8 & 
uv run csac.py --seed 9 & 
wait;
