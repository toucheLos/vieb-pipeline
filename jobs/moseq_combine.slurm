#!/bin/bash
# MOSEQ §3-7: gates, reported statistics, context test -> results/moseq.json.
#SBATCH --job-name=vieb_mqcomb
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 scripts/moseq_gates.py combine
