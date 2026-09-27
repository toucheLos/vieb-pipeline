#!/bin/bash
# MOSEQ §7: pixel motion + freezing (FreezeThresh 800) on report days 3-7 A/B.
# CPU work (sequential video decode); `gpu` partition without --gres, as pixel.slurm.
#     python3 scripts/moseq_freeze.py --phase manifest   # first
#SBATCH --job-name=vieb_mqfrz
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 scripts/moseq_freeze.py --phase shard --shard "${SLURM_ARRAY_TASK_ID}" --of 12
