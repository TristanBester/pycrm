#!/bin/bash
mkdir -p ~/warehouse/slurm_logs/error
mkdir -p ~/warehouse/slurm_logs/out

# Setup environment 
cd ~/warehouse
uv sync --group examples

# Start training
cd ~/warehouse/slurm/examples/crm/continuous

echo "TRAINING..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] csac.slurm



