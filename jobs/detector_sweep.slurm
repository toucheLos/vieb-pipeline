#!/bin/bash
# Step B: can the detector find what is demonstrably there?
#
#     python3 scripts/detector_sweep.py --diagnose --group shape
#     sbatch --array=0-11%6 jobs/detector_sweep.slurm --group shape --n-tasks 12
#     python3 scripts/detector_sweep.py --combine
#
# READ results/DETECTOR_PREREGISTRATION.md FIRST. The gate is named there:
# planted-instance isolation >= 20% at the primary planted duration, and NO
# parameter is tuned until it passes.
#
# The diagnosis ran first and it already narrows this: at the detector of record
# 73.8% of planted instances produce NO boundary at either edge, while merged and
# split are both 0.0%. So the refractory period is swallowing nothing and the
# threshold is not the problem -- which is why this sweep is over the derivative
# half-width and the polynomial degree, the two parameters that set what the
# criterion can resolve at all, and not over k_mad.
#
# CPU: two convolutions and a least-squares fit per recording per cell.
#SBATCH --job-name=vieb_detsweep
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/tok/detector
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
python3 scripts/detector_sweep.py --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
