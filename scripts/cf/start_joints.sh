#!/bin/bash
mkdir -p ~/warehouse/slurm_logs/error
mkdir -p ~/warehouse/slurm_logs/out

# Setup environment 
cd ~/warehouse
uv sync --extra experiments

# Start training
cd ~/warehouse/scripts/cf

echo "TRAINING JOINT-CONTROL..."
echo "Training 0-9..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/sac.sbatch 0
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/csac.sbatch 0
sleep 30s

echo "Training 10-19..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/sac.sbatch 10
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/csac.sbatch 10
sleep 30s

echo "Training 20-29..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/sac.sbatch 20
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/csac.sbatch 20
sleep 30s

echo "Training 30-39..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/sac.sbatch 30
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/csac.sbatch 30
sleep 30s

echo "Training 40-49..."
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/sac.sbatch 40
sleep 30s
sbatch --exclude=mscluster[8,9,35,42,44,46,47,54,57,59,61,62,65,67,68,75,76] train/joints/csac.sbatch 40
sleep 30s
