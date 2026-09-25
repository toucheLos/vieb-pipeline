#!/bin/bash
# STABILISE 7 calibration: gates 3 and 4 on the animal's own pixels,
# from the incumbent alone.
#
#     sbatch --array=0 --export=ALL,LIMIT=1 jobs/stabilise7_calibrate.slurm   # smoke first
#     sbatch jobs/stabilise7_calibrate.slurm
#     python3 scripts/stabilise7_calibrate.py --phase combine
#
# Incumbent only: no candidate arm is computed. See the script docstring.
#
# GPU: SAM on every third frame (~145 ms each on an A100) plus the planted
# sequences. The checkpoint is outside the repository and its SHA-256 is
# checked against the registered value before anything runs.
#SBATCH --job-name=vieb_s7cal
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/stabilise7_cal/rec
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) shard=${SLURM_ARRAY_TASK_ID} limit=${LIMIT:-0}"
test -f work/pixel/manifest.json || { echo "run pixel_pilot --phase sample"; exit 1; }
python3 scripts/stabilise7_calibrate.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12 \
    --limit "${LIMIT:-0}"
