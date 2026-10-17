#!/bin/bash
# STEP 4 C2-C4 (tune split): one phase of scripts/hsmm_tune.py.
#     sbatch jobs/hsmm_tune.slurm prep
#     sbatch --array=0-14%15 jobs/hsmm_tune.slurm fit          # index -> fold*3 + family
#     sbatch jobs/hsmm_tune.slurm select | final | controls | report
#     sbatch --array=0-50%25 jobs/hsmm_tune.slurm planted_refit  # boot 0 + 50 replicates
#SBATCH --job-name=vieb_hsmm
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS=2
PHASE="$1"; shift || true
if [ "$PHASE" = "fit" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python3 scripts/hsmm_tune.py fit --index "$SLURM_ARRAY_TASK_ID"
elif [ "$PHASE" = "planted_refit" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python3 scripts/hsmm_tune.py planted_refit --boot "$SLURM_ARRAY_TASK_ID"
else
    python3 scripts/hsmm_tune.py "$PHASE" "$@"
fi
