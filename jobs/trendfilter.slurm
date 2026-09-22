#!/bin/bash
#   sbatch --array=0-11 jobs/trendfilter.slurm --phase sweep
#SBATCH --job-name=vieb_tf
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/trendfilter_sweep
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/trendfilter.py --shard "${SLURM_ARRAY_TASK_ID}" --of 60 "$@"
