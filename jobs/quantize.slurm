#!/bin/bash
# Step 2: the alphabet sweep. Four modes, and the order between them matters.
#
#     python3 scripts/quantize.py --write-grid
#     sbatch --array=0-29%15 jobs/quantize.slurm --abstain
#     sbatch jobs/quantize.slurm --basis
#     sbatch --array=0-7%4   jobs/quantize.slurm --cell
#     python3 scripts/quantize.py --combine
#
# --abstain is sharded by ANIMAL and is cell-independent: the mask comes from
# four sources in three repositories and costs one np.load per recording, so
# computing it inside each of the eight cells would do that work eight times for
# an identical answer.
#
# --basis runs ONCE and must land before any cell. It freezes the standardising
# SD and the speed quantile cuts, fitted on tune. If each cell fitted its own SD
# the two arms would be standardised differently and the MDL comparison between
# them would be measuring the standardisation rather than the tokenization.
#
# --cell is one (arm, n_states) point, read from work/grids/quantize.txt by
# array index. Eight cells: 2 arms x 4 alphabet sizes.
#
# CPU, and that is not a fallback. MiniBatchKMeans is sklearn, which has no GPU
# path; a GPU k-means would mean a new implementation to validate against this
# one, which is a build rather than a speed-up. The assignment is one tiled
# matmul and would suit a GPU, but it is minutes on CPU and is not the
# bottleneck. jobs/q1.slurm stays the GPU template if that ever changes.
#
# MEMORY: --basis and --cell hold the 60 tune animals' X (~0.3 GB float32, ~1.2
# GB once float64 copies exist) plus one (chunk, N) distance block, which
# scripts/quantize.py caps at 2 GB by deriving the chunk from N. --abstain is
# per animal and needs almost nothing.
#SBATCH --job-name=vieb_quantize
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/tok
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-30}"
python3 scripts/quantize.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-30}" "$@"
