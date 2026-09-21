#!/bin/bash
# Do the detector's boundaries survive a change of cleaning arm?
#
#   sbatch --array=0-11 jobs/arm_concordance.slurm
#   python3 scripts/arm_concordance.py --combine
#
# Sharded by ANIMAL, 12 tasks over 60 tune animals. The binding cost is the
# viterbi arm at ~12.6 s per recording (`jobs/cleaning.slurm` measures the
# same thing), so a single task would run ~2.6 hours and twelve run ~13
# minutes each.
#
# Registered in results/NOISEFLOOR_PREREGISTRATION.md §6.
#SBATCH --job-name=vieb_arm_conc
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
mkdir -p work/logs work/arm_concordance
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were
# frozen.
python3 scripts/freeze.py
python3 scripts/arm_concordance.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12
