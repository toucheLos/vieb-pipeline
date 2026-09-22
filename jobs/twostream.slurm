#!/bin/bash
#   sbatch --array=0-11 jobs/twostream.slurm
#   python3 scripts/twostream.py --combine
#SBATCH --job-name=vieb_twostream
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/twostream
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 "${VIEB_SCRIPT:-scripts/twostream.py}" "$@"
