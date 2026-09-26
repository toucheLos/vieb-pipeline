#!/bin/bash
# STATES §4 gate 1: each surrogate through the identical pipeline.
#SBATCH --job-name=vieb_stsurr
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
KINDS=(phase var5 microstate microstate0 white)
python3 scripts/states.py --phase surrogate --kind "${KINDS[$SLURM_ARRAY_TASK_ID]}"
