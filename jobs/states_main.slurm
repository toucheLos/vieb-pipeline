#!/bin/bash
# STATES §2-§3: leak gates, K* and tau* on tune, the corpus fit (two seeds).
#SBATCH --job-name=vieb_states
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 scripts/states.py --phase leak
python3 scripts/states.py --phase select
python3 scripts/states.py --phase fit
