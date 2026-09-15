#!/bin/bash
# Stage 1: the egocentric SE(2) transform. The only mandatory preprocessing.
#
#     python3 scripts/ego.py --write-grid
#     sbatch --array=0-9%10 jobs/ego.slurm --pose-arm raw --scale-arm bodylen
#     python3 scripts/ego.py --combine --pose-arm raw --scale-arm bodylen --regressor
#
# Sharded by ANIMAL because ell_a is the animal's own median body length pooled
# over all of its recordings -- not per recording, which is what disarmed a
# control on this project once already.
#
# TWO ARM AXES, and they are different questions. --scale-arm is about ell_a
# (bodylen / unitlen); --pose-arm is about which coordinates the transform
# consumes (raw / wiener). Arms are submitted separately rather than in one
# task: they write separate shards and are compared only at combine time, so a
# failure in one leaves the other readable.
#
# 10 tasks, not 30 and not 298. MEASURED: one animal is 0.7 s wall and 0.08 GB
# peak, so the whole corpus is ~3.5 minutes of arithmetic and any larger array
# is scheduler overhead. The earlier 30-task shape predates that measurement.
#
# CPU. The transform is trigonometry and one 2x2 solve per frame; there is
# nothing here a GPU would help with.
#SBATCH --job-name=recur_ego
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/ego
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/ego.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
