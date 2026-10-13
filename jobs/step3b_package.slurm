#!/bin/bash
# STEP 3B.1-3B.2: pilot set + anatomy-only labelling package.
#SBATCH --job-name=vieb_3bpkg
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
umask 077
python3 scripts/step3b_package.py
