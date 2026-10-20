#!/bin/bash
# STEP 3A Amendment 1: bout-aware probe cells (scripts/calms_bout_probe.py), then combine.
#     sbatch --array=0-17%9 jobs/calms_bout_probe.slurm cell
#     sbatch --dependency=afterok:<array> jobs/calms_bout_probe.slurm combine
#SBATCH --job-name=vieb_boutprobe
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=10:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [ "$1" = "cell" ]; then
    python3 scripts/calms_bout_probe.py cell --index "$SLURM_ARRAY_TASK_ID"
else
    python3 scripts/calms_bout_probe.py combine
fi
