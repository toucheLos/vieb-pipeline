#!/bin/bash
# Step 2: do segments recur across animals? Q1's statistic, segments as unit.
#
#     sbatch jobs/seg_recur.slurm --group both  --planted
#     sbatch jobs/seg_recur.slurm --group shape
#     sbatch jobs/seg_recur.slurm --group twist
#     python3 scripts/seg_recur.py --combine
#
# READ results/SEGRECUR_PREREGISTRATION.md FIRST. It was committed before any
# bank or search existed. THE GATE: if the segment excess sits BELOW a
# length-matched windowed control in the same space, the boundaries cut through
# behaviours rather than between them, Step 1's criterion is wrong, and the arm
# stops there with an answer.
#
# One job per CHANNEL GROUP, not per animal: a cross-animal nearest neighbour
# needs every animal's units in one bank, so the unit of work is the group.
# The three groups are never pooled -- shape|twist boundary sets give Jaccard
# 0.226, so they are largely different events, and disagreement is the finding.
#
# GPU, asserted and never auto-detected: the search is an exact leave-one-animal
# -out nearest neighbour and there is no CPU fallback that finishes.
#SBATCH --job-name=vieb_segrecur
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/tok/seg_recur
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; \
print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python3 scripts/seg_recur.py --device cuda "$@"
