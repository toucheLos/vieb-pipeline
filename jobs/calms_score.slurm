#!/bin/bash
# STEP 1 §4-7: one scoring cell (instrument x condition), or combine.
#     sbatch --array=0-17%9 jobs/calms_score.slurm cell      # D 0-8, C 9-17
#     sbatch --array=18-26%9 jobs/calms_score.slurm cell     # M 18-26
#     sbatch jobs/calms_score.slurm combine
#SBATCH --job-name=vieb_calmsscore
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS=1
if [ "$1" = "cell" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python3 scripts/calms_score.py cell --index "$SLURM_ARRAY_TASK_ID"
else
    python3 scripts/calms_score.py "$@"
fi
