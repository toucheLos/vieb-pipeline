#!/bin/bash
# SAM mask vs DLC keypoints: per-keyframe agreement, and who is wrong when not.
# Descriptive; CPU; reads STABILISE 7's saved masks.
#     sbatch jobs/agreement.slurm
#SBATCH --job-name=vieb_agree
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/agreement.py --shard 0 --of 1
python3 scripts/agreement.py --phase combine
