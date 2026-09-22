#!/bin/bash
# Stage 0: noise colour, descriptor floors, dynamics probe.
#
#   sbatch jobs/dynamics_floor.slurm
#
# One task. Measured after batching the AR fit across channels: ~0.1 s per
# 240-frame contrast, so the 13-cell probe ladder at 240 instances each is
# ~5 minutes and the descriptor floor ~1 minute.
#
# Registered in results/DYNAMICS_PREREGISTRATION.md.
#SBATCH --job-name=vieb_dynfloor
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
python3 "${VIEB_STAGE0_SCRIPT:-scripts/dynamics_floor.py}" "$@"
