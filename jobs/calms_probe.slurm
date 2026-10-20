#!/bin/bash
# STEP 3A: one probe cell, or combine.
#     sbatch --array=0-9%10 jobs/calms_probe.slurm cell
#     sbatch jobs/calms_probe.slurm combine
#SBATCH --job-name=vieb_probe
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS=4
if [ "$1" = "cell" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python3 scripts/calms_probe.py cell --index "$SLURM_ARRAY_TASK_ID"
else
    python3 scripts/calms_probe.py "$@"
fi
