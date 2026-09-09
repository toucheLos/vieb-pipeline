#!/bin/bash
# Phase A: what the cleaning actually changes, layer by layer.
#
#     python3 scripts/effect.py --write-grid
#     sbatch --array=0-29%15 jobs/effect.slurm
#     python3 scripts/effect.py --combine
#
# Sharded by ANIMAL: ell_a is a per-animal constant and is the only unit in
# which two differently sized mice are comparable.
#
# MEASURED: ~33 s per animal, peak RSS 1.81 GB. The memory is dominated by
# raw_pose.npz, which is 1.88 GB and is cached whole -- slicing one recording
# out of it per call would re-read 1.88 GB 3,846 times. 16 GB leaves headroom
# for the four pose arrays each recording holds at once.
#SBATCH --job-name=vieb_effect
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/effect
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/effect.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
