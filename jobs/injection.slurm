#!/bin/bash
# Phase F: does the arm get closer to the truth, and what does it break?
#
#     python3 scripts/injection.py --write-grid
#     sbatch --array=0-29%15 jobs/injection.slurm
#     python3 scripts/injection.py --combine --split report
#
# MEASURED: ~2 min 40 s per animal, peak RSS 0.20 GB. Five arms, and four of them
# have to be cleaned before they can be scored -- the median and Viterbi arms are
# recomputed here rather than stored. At 30 tasks the corpus is about half an
# hour.
#
# The configuration is fixed in results/CONTINUITY_PREREGISTRATION.md, committed
# BEFORE this ran on the corpus. This does not restate the bakeoff and does not
# re-score Q1.
#SBATCH --job-name=vieb_inject
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/injection
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/injection.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" \
    ${SPIKE_BL:+--spike-bl "$SPIKE_BL"} ${MEASURED_PARKS:+--measured-parks} "$@"
