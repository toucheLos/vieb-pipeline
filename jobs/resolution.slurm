#!/bin/bash
# Distortion, frailty and symbol homogeneity. One shared pass, three readings.
#
#     python3 scripts/resolution.py --write-grid
#     sbatch --array=0-7%4 jobs/resolution.slurm --covariates
#     sbatch --array=0-7%4 jobs/resolution.slurm --distortion
#     sbatch            jobs/resolution.slurm --frailty
#     python3 scripts/resolution.py --combine
#
# --covariates is the expensive one and must land first: it joins the egocentric
# array against the label shard for all 298 animals and writes a run table with
# the two run-level covariates (mean speed, mean distance to centroid) that
# everything after reads. ~1.2 GB of ego arrays read per cell.
#
# --frailty runs on ONE cell (plain/N=256, the MDL-selected one) rather than the
# whole grid: it is a decomposition of a reported finding, not a sweep.
#
# CPU. Bincounts, bin-overlap arithmetic and a residual against an already
# chosen centroid -- no argmin is recomputed here, so there is no matmul for a
# GPU to hold.
#SBATCH --job-name=vieb_resolution
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
mkdir -p work/logs work/tok/runs work/tok/resolution
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were frozen.
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-8}"
python3 scripts/resolution.py --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT:-8}" "$@"
