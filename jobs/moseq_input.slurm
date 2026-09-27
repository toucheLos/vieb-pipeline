#!/bin/bash
# MOSEQ §1: unsmoothed repaired pose + white/phase surrogates as kpms input.
#     python3 scripts/moseq_input.py --phase select   # first
#SBATCH --job-name=vieb_mqin
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/moseq_input.py --phase build --shard "${SLURM_ARRAY_TASK_ID}" --of 12
