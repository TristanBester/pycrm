#!/bin/bash
mkdir -p ~/warehouse/slurm_logs/error
mkdir -p ~/warehouse/slurm_logs/out

# Start training
cd ~/warehouse/scripts/train

echo "Training EE single-process..."
sbatch ee/serial/sac.sbatch
sbatch ee/serial/csac.sbatch

echo "Training EE multi-process..."
sbatch ee/vec/sac.sbatch
sbatch ee/vec/csac.sbatch

echo "Training joints single-process..."
sbatch joints/serial/sac.sbatch
sbatch joints/serial/csac.sbatch

echo "Training joints multi-process..."
sbatch joints/vec/sac.sbatch
sbatch joints/vec/csac.sbatch
