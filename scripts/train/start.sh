#!/bin/bash
mkdir -p ~/warehouse/slurm_logs/error
mkdir -p ~/warehouse/slurm_logs/out

# Setup environment 
cd ~/warehouse
uv sync --extra experiments

# Start training
cd ~/warehouse/scripts/train


echo "Training EE-Control..."
echo "SERIAL..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/ee/serial/sac.sbatch
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/ee/serial/csac.sbatch
sleep 30s

echo "PARALLEL..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/ee/parallel/sac.sbatch
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/ee/parallel/csac.sbatch
sleep 30s


echo "Training Joint-Control..."
echo "SERIAL..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/joints/serial/sac.sbatch
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/joints/serial/csac.sbatch
sleep 30s

echo "PARALLEL..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/joints/parallel/sac.sbatch
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] cs/joints/parallel/csac.sbatch
