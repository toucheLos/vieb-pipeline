#!/bin/bash
# Step 2: the gate. Four structureless nulls through the identical pipeline.
#
#     sbatch --array=0-29%12 jobs/seg_validate.slurm --shard
#     sbatch            jobs/seg_validate.slurm --separability
#     sbatch            jobs/seg_validate.slurm --diagnose
#     python3 scripts/seg_validate.py --combine
#
# READ results/SEGMENTATION_PREREGISTRATION.md FIRST. It was committed before
# any surrogate for this stage was generated and it fixes the gate: if the
# corpus boundary rate sits inside the surrogate range anywhere the comparison
# is powered, THE SEGMENTATION ROUTE CLOSES and that is the finding.
#
# --diagnose is POST-HOC, added after the probe failed on all four nulls. It
# attaches no verdict and does not reopen the gate; it asks only what the probe
# separates on -- a roughness LEVEL, which the scale-adaptive threshold divides
# out, or roughness VARIABILITY, which is the non-stationarity under test.
#
# --separability runs once, not per animal, because it needs both classes at
# the same time. It probes WINDOWS rather than frames: a phase surrogate matches
# every first- and second-order moment by construction, so a linear probe on
# single frames is at chance by construction and its PASS would mean nothing.
#
# CPU: two convolutions per recording per arm, a least-squares fit per segment,
# and one logistic regression per surrogate.
#SBATCH --job-name=vieb_segval
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/tok/seg_validate
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/seg_validate.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
