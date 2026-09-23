#!/bin/bash
# Step 1: does the island replicate on the 149 held-out `fit` animals?
#
#     python3 scripts/fit_replicate.py --prep      # builds both target banks
#     sbatch jobs/fit_replicate.slurm
#     python3 scripts/fit_replicate.py --combine
#
# READ results/FIT_REPLICATION_PREREGISTRATION.md FIRST. It was committed alone,
# before this script existed, and it fixes theta, the scale, the cohort size and
# the incumbent's values.
#
# THE GATE IS THE TRANSFER RULE, NOT THE CONTRAST. A fit segment landing within
# theta of a report island member could be reporting a shared low-speed regime
# rather than the same state. So the rate is read against a second target bank of
# 361 non-island report segments matched to the island on joint (log duration,
# log mean speed). If the two rates overlap, the rule captures slowness and
# THE CONTRAST IS NOT READ whichever way it would have landed.
#
# Submitted to the `gpu` partition and asked for NO --gres: the work is CPU-only
# but `normal` was fully allocated and this finishes in minutes.
# CPU, one task per shard. The work is a 560-dimensional exact nearest neighbour
# against 722 targets -- small enough that the detector, not the distance, is
# the cost, and the detector is single-threaded per animal.
#SBATCH --job-name=vieb_fitrep
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/fit_replicate
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) shard=${SLURM_ARRAY_TASK_ID}"
test -f work/fit_replicate/targets.npz || { echo "run --prep first"; exit 1; }
python3 scripts/fit_replicate.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12
