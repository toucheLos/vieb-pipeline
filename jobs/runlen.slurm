#!/bin/bash
# G1: how long is a violating run, and where does the violating mass live?
#
#     python3 scripts/runlen.py --write-grid
#     sbatch --array=0-29%15 jobs/runlen.slurm
#     python3 scripts/runlen.py --combine --split report
#
# MEASURED: ~2 min 35 s per animal, peak RSS 0.10 GB. Three arms; two of them
# must be cleaned before their violations can be counted. At 30 tasks the corpus
# is about half an hour.
#
# Nothing ever measured this. Every run-length figure in this repo's documents
# was a hardcoded literal in a Markdown generator, backed by no artifact.
#SBATCH --job-name=vieb_runlen
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/runlen
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/runlen.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
