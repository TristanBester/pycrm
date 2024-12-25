#!/bin/bash
# specify a partition
#SBATCH -p bigbatch 
# specify number of nodes
#SBATCH -N 1
# specify the wall clock time limit for the job hh:mm:ss
#SBATCH -t 72:00:00
# specify the job name
#SBATCH -J sac-cf-joints-p
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

cd ~/warehouse/experiments/warehouse/exp/cf/train
uv run sac.py train.seed=$(( $1 + 0 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 1 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 2 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 3 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 4 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 5 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 6 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 7 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
uv run sac.py train.seed=$(( $1 + 8 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote &   
uv run sac.py train.seed=$(( $1 + 9 )) train.n_procs=3 exp.control_type=joints exp.use_wandb=true exp.wandb_project=regular-v1.1-joints environment=remote & 
wait;

