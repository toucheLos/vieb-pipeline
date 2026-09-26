#!/bin/bash
# SAM with keypoints as POINT prompts vs the box alone: does the head stay in?
# Descriptive; every 30th frame of the 300 STABILISE 7 recordings.
#     sbatch jobs/point_prompts.slurm
#     python3 scripts/point_prompts.py --phase combine
#SBATCH --job-name=vieb_points
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/point_prompts.py --mode "${MODE:-plain}" --shard "${SLURM_ARRAY_TASK_ID}" --of 12
