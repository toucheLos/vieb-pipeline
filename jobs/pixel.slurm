#!/bin/bash
# Step 2: the arena's own noise floor, and an ezTrack-family freeze score.
#
#     python3 scripts/pixel_pilot.py --phase sample
#     sbatch jobs/pixel.slurm
#     python3 scripts/pixel_pilot.py --phase combine
#
# READ results/PIXEL_PREREGISTRATION.md FIRST, INCLUDING AMENDMENT 1. Section 8's
# original draw gave 48 of 54 animals a single recording, which made Q1 -- a
# WITHIN-ANIMAL correlation -- uncomputable. The sampling unit is the ANIMAL:
# 30 animals, 10 per box, all ten of each animal's (context, day) cells.
#
# CPU, and deliberately so. The work is sequential video decoding at ~254 fps on
# one core; a GPU would idle behind the decoder. Videos are READ-ONLY and are
# never seeked: a random seek costs 5-36 ms against 0.5-0.65 ms for a sequential
# frame, measured.
#
# Submitted to the `gpu` partition without --gres only because `normal` runs
# fully allocated here. Nothing in this job touches CUDA.
#SBATCH --job-name=vieb_pixel
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/pixel/rec
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) shard=${SLURM_ARRAY_TASK_ID}"
test -f work/pixel/manifest.json || { echo "run --phase sample first"; exit 1; }
python3 scripts/pixel_pilot.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12
