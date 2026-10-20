#!/bin/bash
# STEP 4: Amendment 4 planted bout recall, or the known-answer check (scripts/hsmm_known.py).
#     sbatch jobs/hsmm_known.slurm planted_bouts
#     sbatch jobs/hsmm_known.slurm known        # after jobs/moseq_freeze_tune.slurm
#SBATCH --job-name=vieb_hsmm_known
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS=2
python3 scripts/hsmm_known.py "$1"
