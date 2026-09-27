#!/bin/bash
# MOSEQ §3: forward-filter scoring of one model per task (array 0-10).
#SBATCH --job-name=vieb_mqscore
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-10
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
MODELS=(real_s0 real_s1 real_s2 real_s3 real_s4 white_s0 white_s1 white_s2 phase_s0 phase_s1 phase_s2)
python3 scripts/moseq_gates.py score --model "${MODELS[$SLURM_ARRAY_TASK_ID]}"
