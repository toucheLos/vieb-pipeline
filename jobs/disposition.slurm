#!/bin/bash
# Phase D: correct, abstain, or down-weight -- and the leave-one-bone-out gate.
#
#     python3 scripts/disposition.py --write-grid
#     sbatch --array=0-29%15 jobs/disposition.slurm
#     python3 scripts/disposition.py --combine --split report
#
# MEASURED: ~1 s per animal, peak RSS 0.10 GB. At that cost the whole corpus
# runs locally in five minutes and this file exists for reproducibility rather
# than necessity.
#
# The configuration is fixed in results/DISPOSITION_PREREGISTRATION.md, which was
# committed BEFORE this ran on anything. Q1 is not re-scored here.
#SBATCH --job-name=vieb_disp
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/disposition
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/disposition.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
