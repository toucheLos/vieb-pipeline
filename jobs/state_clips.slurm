#!/bin/bash
#SBATCH --job-name=vieb_stclips
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
source "$REPO/env.sh"
python3 scripts/state_clips.py --model moseq
python3 scripts/state_clips.py --model states
