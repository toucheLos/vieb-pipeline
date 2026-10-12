#!/bin/bash
# STEP 1 §1-2: prep (noise.json, partitions.json), or build one arm's inputs.
#     sbatch jobs/calms_prep.slurm prep
#     sbatch --array=0-44%10 jobs/calms_prep.slurm build      # index -> arms()
#SBATCH --job-name=vieb_calmsprep
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS=2
if [ "$1" = "build" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python3 scripts/calms_prep.py build --index "$SLURM_ARRAY_TASK_ID"
else
    python3 scripts/calms_prep.py "$@"
fi
