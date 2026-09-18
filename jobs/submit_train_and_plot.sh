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

# Train (merlin-train), then generate the corner plot for whichever
# train_<N> slot it was just assigned -- chained in one job so the plot is
# ready without a separate manual step.
#
# Usage:
#   sbatch jobs/submit_train_and_plot.sh path/to/config.yaml

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

module load 2024

export LD_PRELOAD=/sw/arch/RHEL9/EB_production/2024/software/GCCcore/13.3.0/lib64/libstdc++.so.6
# cloelib's TensorFlow-based emulators otherwise preallocate ~the entire GPU
# on first use and starve PyTorch's training/inference allocations (CUDA OOM).
export TF_FORCE_GPU_ALLOW_GROWTH=true

CONFIG="${1:-input/config_example.yaml}"

OUTPUT=$(/home/abellan/.local/bin/merlin-train "$CONFIG" | tee /dev/stderr)
TRAIN_ID=$(echo "$OUTPUT" | grep -oP 'Training run: train_\K[0-9]+')

/home/abellan/.local/bin/merlin-plot "$CONFIG" --mode corner --train-id "$TRAIN_ID" --smooth 2
