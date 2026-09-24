#!/bin/bash
# STABILISE: keypoint-free crop registration, validated as an instrument.
#
#     sbatch --array=0 --export=ALL,LIMIT=1 jobs/stabilise.slurm   # smoke first
#     sbatch jobs/stabilise.slurm
#     python3 scripts/stabilise.py --phase combine
#
# READ results/STABILISE_PREREGISTRATION.md FIRST, with Amendment 1 (D22).
#
# CPU: two sequential decodes per recording (mask pass, difference pass) plus
# 22 planted 61-frame sequences. ECC and the warps are CPU work, and a GPU would
# idle behind the decoder. Arm S (SAM) is not run here.
#SBATCH --job-name=vieb_stab
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/stabilise/rec
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) shard=${SLURM_ARRAY_TASK_ID} limit=${LIMIT:-0}"
test -f work/pixel/manifest.json || { echo "run pixel_pilot --phase sample"; exit 1; }
python3 scripts/stabilise.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12 \
    --limit "${LIMIT:-0}"
