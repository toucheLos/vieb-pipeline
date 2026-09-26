#!/bin/bash
# Full-length overlay videos: SAM's mask and the DLC keypoints on every frame,
# reconstructed from STABILISE 7's saved records (no new SAM call).
#     sbatch jobs/sam_videos.slurm
#SBATCH --job-name=vieb_samvid
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/sam_videos.py
