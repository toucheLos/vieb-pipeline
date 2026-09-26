#!/bin/bash
# STATES §1: per-animal feature files (pose stream + mask stream + head coverage).
#SBATCH --job-name=vieb_feat
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 scripts/features.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12
