#!/bin/bash
# Evidence images for STABILISE 1-4 (descriptive; nothing enters a gate).
#     sbatch jobs/stabilise_figures.slurm
#SBATCH --job-name=vieb_stabfig
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/stabilise_figures.py
