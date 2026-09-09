#!/bin/bash
# Step 1: the bone-length violation sweep. BLOCKING -- nothing downstream is
# interpretable until this and its R^2 join land.
#
#     python3 scripts/bones.py --write-grid
#     sbatch --array=0-29%12 jobs/bones.slurm
#     python3 scripts/bones.py --combine
#
# Sharded by ANIMAL because the reference length is fitted per animal per bone
# on that animal's own frames. A corpus-wide reference would flag small mice
# everywhere, which is a confound wearing a quality gate's clothes.
#
# 30 tasks, not 298. An animal takes ~3 s measured (13 recordings, 75k frames,
# peak RSS 0.13 GB), so one array task per animal would be mostly scheduler
# overhead. Each task takes a STRIDED slice of the grid, so no task's wall clock
# depends on whether its block held the long recordings.
#
# CPU: this is distance arithmetic and percentiles, and there is nothing here a
# GPU would help with.
#SBATCH --job-name=recur_bones
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/bones
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/bones.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
