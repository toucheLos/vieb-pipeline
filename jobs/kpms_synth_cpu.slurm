#!/bin/bash
# Paper 1 optional analysis, CPU phases: build the LDS synthetic recordings; score dwell.
#SBATCH --job-name=vieb_kpsynthc
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/kpms_synth.py "$1"
