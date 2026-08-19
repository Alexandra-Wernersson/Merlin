#!/bin/bash
# Tuned for Snellius specifically — on a different cluster you'll likely need
# to adapt the #SBATCH preamble below too 
#SBATCH --job-name=merlin-train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/train_%j.out

# Train the network against an already-filled Zarr store (merlin-train) — 1 GPU.
#
# Usage:
#   sbatch jobs/submit_train.sh [path/to/config.yaml]
# Defaults to input/config_example.yaml. --cpus-per-task above should be a
# few more than that config's TRAINING.num_workers (DataLoader workers).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

module load 2024

export LD_PRELOAD=/sw/arch/RHEL9/EB_production/2024/software/GCCcore/13.3.0/lib64/libstdc++.so.6
# cloelib's TensorFlow-based emulators otherwise preallocate ~the entire GPU
# on first use and starve PyTorch's training allocations (CUDA OOM).
export TF_FORCE_GPU_ALLOW_GROWTH=true

CONFIG="${1:-input/config_example.yaml}"
/home/abellan/.local/bin/merlin-train "$CONFIG"
