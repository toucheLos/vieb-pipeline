#!/bin/bash
# Step 4: rungs 0-2 on each alphabet, scored by held-out description length.
#
#     python3 scripts/ladder.py --write-grid
#     sbatch --array=0-7%4 jobs/ladder.slurm --cell
#     python3 scripts/ladder.py --combine
#
# One array task per (arm, n_states) cell, read from work/grids/ladder.txt by
# array index. Each task fits on `fit`, scores on `report`, and takes the
# duration grid from `tune` -- so all three splits are read and none is mixed.
#
# READ results/TOK_PREREGISTRATION.md BEFORE READING ANY NUMBER THIS PRODUCES.
# The run-length stop condition retired all eight alphabets and the ladder runs
# anyway, under an override recorded in SS1 of that document. Every shard carries
# `retired_by_runlength: true`.
#
# CPU. Fitting is counting and the scoring is a sparse lookup plus twelve vector
# operations per run; there is no matmul here for a GPU to hold. The k=2 context
# vocabulary is the memory driver -- up to one key per run, np.unique over ~7.5M
# int64 -- which is why --mem is 64G rather than the 48G the quantize stage uses.
#SBATCH --job-name=vieb_ladder
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/tok/ladder
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-8}"
python3 scripts/ladder.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-8}" "$@"
