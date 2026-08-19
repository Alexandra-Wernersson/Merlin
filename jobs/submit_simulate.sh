#!/bin/bash
# Tuned for Snellius specifically — on a different cluster you'll likely need
# to adapt the #SBATCH preamble below too 
#SBATCH --job-name=merlin-simulate
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/simulate_%j.out

# Fill a Zarr simulation store (merlin-simulate) — CPU-only, no GPU needed.
# Runs the Fisher analysis first too, if PRIORS.use_Fisher_priors is set.
#
# Usage:
#   sbatch jobs/submit_simulate.sh [path/to/config.yaml]
# Defaults to input/config_example.yaml. --cpus-per-task above should match
# (or exceed) that config's SIMULATION.n_workers.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

module load 2024

export LD_PRELOAD=/sw/arch/RHEL9/EB_production/2024/software/GCCcore/13.3.0/lib64/libstdc++.so.6
# Cosmetic: suppresses the harmless per-worker "CUDA_ERROR_NO_DEVICE" noise from TensorFlow/XLA on this GPU-less partition.
export TF_CPP_MIN_LOG_LEVEL=3

CONFIG="${1:-input/config_example.yaml}"
/home/abellan/.local/bin/merlin-simulate "$CONFIG"
