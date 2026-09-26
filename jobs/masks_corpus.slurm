#!/bin/bash
# Plan Stage A1: guarded SAM masks, every 6th frame of all 3,846 recordings.
#     sbatch --array=0 --export=ALL,LIMIT=1 jobs/masks_corpus.slurm   # smoke
#     sbatch jobs/masks_corpus.slurm
# Resumable: finished recordings are skipped.
#SBATCH --job-name=vieb_masks
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/masks
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/masks_corpus.py --shard "${SLURM_ARRAY_TASK_ID}" --of 16 --limit "${LIMIT:-0}"
