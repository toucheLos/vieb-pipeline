#!/bin/bash
# Step 3: the 3-8 Hz grooming gate. Head-region motion energy, one pass per
# recording per registered radius.
#
#     sbatch jobs/grooming.slurm
#     python3 scripts/grooming_gate.py --phase combine
#
# READ results/GROOMING_PREREGISTRATION.md FIRST, and DEVIATIONS.md D19.
#
# THE GATE IS NOT THE SPECTRUM ALONE. Candidates are selected on high
# head-region motion, so a statistic on head-region amplitude would guarantee
# its own answer. Section 3 (30 clips confirmed BY EYE) and section 4
# (peak_excess, exactly invariant to rescaling) are both required, and section
# 9.4 forbids reading the spectrum as an answer about GROOMING until the clips
# are confirmed.
#
# CPU: three sequential decodes per recording, one per registered radius. The
# decoder is the cost and a GPU would idle behind it.
#SBATCH --job-name=vieb_groom
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/grooming/rec
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) shard=${SLURM_ARRAY_TASK_ID}"
test -f work/pixel/manifest.json || { echo "run pixel_pilot --phase sample"; exit 1; }
python3 scripts/grooming_gate.py --shard "${SLURM_ARRAY_TASK_ID}" --of 12
