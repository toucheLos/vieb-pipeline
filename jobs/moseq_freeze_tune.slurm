#!/bin/bash
# MOSEQ §7: pixel motion + freezing (FreezeThresh 800) on tune days 3-7 A/B (STEP 4 known-answer check).
# CPU work (sequential video decode); `gpu` partition without --gres, as pixel.slurm.
#     python3 scripts/moseq_freeze.py --phase manifest --split tune   # first
#SBATCH --job-name=vieb_mqfrz_tune
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 scripts/moseq_freeze.py --phase shard --shard "${SLURM_ARRAY_TASK_ID}" --of 12 --split tune
