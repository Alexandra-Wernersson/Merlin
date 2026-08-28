#!/bin/bash
# Tuned for Snellius specifically — on a different cluster you'll likely need
# to adapt the #SBATCH preamble below too
#SBATCH --job-name=merlin-optuna
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=6:00:00
#SBATCH --output=jobs/logs/optuna_%j.out

# Hyperparameter search over an already-filled Zarr store (optuna_search.py)
# — 1 GPU, many sequential trials in a single job. 6h wall-clock vs. the
# script's own --timeout-hours default of 4.75h leaves 75min of buffer for
# the last trial in flight (the timeout is only checked between trials, and
# TRAINING.max_epochs=-1 means a single trial's length is otherwise only
# bounded by early-stopping patience) plus final summary/cleanup, before
# SLURM's hard kill.
#
# Usage:
#   sbatch jobs/submit_optuna.sh [path/to/config.yaml] [extra optuna_search.py args...]
# Defaults to input/config_example.yaml. --cpus-per-task above should be a
# few more than optuna_search.py's --num-workers (default 12).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

module load 2024

export LD_PRELOAD=/sw/arch/RHEL9/EB_production/2024/software/GCCcore/13.3.0/lib64/libstdc++.so.6
# cloelib's TensorFlow-based emulators otherwise preallocate ~the entire GPU
# on first use and starve PyTorch's training allocations (CUDA OOM).
export TF_FORCE_GPU_ALLOW_GROWTH=true

CONFIG="${1:-input/config_example.yaml}"
shift || true
# Explicit interpreter, not bare "python" — in this SLURM batch context,
# module-loaded PATH doesn't take effect the way it does interactively, and
# bare "python" silently resolves to an unrelated Python 3.9 with a
# different (older) site-packages, missing optuna_integration entirely. The
# merlin-* console scripts (submit_train.sh/submit_simulate.sh) don't hit
# this because their shebang line pins the same interpreter directly.
/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/bin/python3.12 optuna_search.py "$CONFIG" "$@"
