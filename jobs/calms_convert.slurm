#!/bin/bash
# STEP 1 §1.0: CalMS21 JSON -> per-sequence npz (pose and labels apart) + facts.
#SBATCH --job-name=vieb_calmsconv
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
umask 077
python3 scripts/calms_convert.py "$@"
