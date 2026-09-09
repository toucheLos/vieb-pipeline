#!/bin/bash
# Phase B: the cleaning bakeoff -- every arm on three axes that can disagree.
#
#     python3 scripts/cleaning.py --write-grid
#     sbatch --array=0-29%15 jobs/effect.slurm
#     python3 scripts/cleaning.py --combine
#
# Sharded by ANIMAL: ell_a is a per-animal constant and is the only unit in
# which two differently sized mice are comparable.
#
# MEASURED: ~170 s per animal with the Viterbi arm (12.6 s per recording; every
# other arm is ~0.05 s), peak RSS 0.27 GB. No raw_pose here -- every arm
# starts from shapeflow's hold-for-filtering array, which is rebuilt per
# recording from work/clean, so the 1.88 GB corpus array is never touched.
#SBATCH --job-name=vieb_cleaning
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/cleaning
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/cleaning.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
