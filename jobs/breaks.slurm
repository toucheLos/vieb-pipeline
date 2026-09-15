#!/bin/bash
# Step 1: the acceleration-mismatch statistic on the carried arm, swept.
#
#     python3 scripts/breaks.py --write-grid
#     sbatch --array=0-29%12 jobs/breaks.slurm --shard
#     python3 scripts/breaks.py --combine
#
# Sharded by ANIMAL, with every grid cell computed inside the task: the
# statistic is two convolutions, so the sweep is cheap and the expensive part is
# reading the ego shard once.
#
# This stage emits a STATISTIC, not a finding. The reference threshold is
# ExBias's k_mad = 3.0 and carries no stated false-positive rate; Step 2
# calibrates against the surrogate families, because a changepoint detector
# fires on noise and a boundary rate without a null beside it is not evidence.
#
# CPU: two sliding-window dot products and a least-squares fit per segment.
#SBATCH --job-name=vieb_breaks
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/tok/breaks
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/breaks.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
