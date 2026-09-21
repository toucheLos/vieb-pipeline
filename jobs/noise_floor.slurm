#!/bin/bash
# The tracking-noise floor under the boundary detector.
#
#   sbatch jobs/noise_floor.slurm
#
# One task: 60 tune animals, ~770 recordings. Measured on a 2-animal smoke
# test at ~1 s per animal for the detector phases and ~10 s per animal for
# the skull-residual calibration, so the whole run is minutes rather than
# hours and does not need the array pattern the corpus stages use.
#
# Registered in results/NOISEFLOOR_PREREGISTRATION.md.
#SBATCH --job-name=vieb_noise_floor
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
# Provenance gate: refuse if the inherited artifacts moved since they were
# frozen.
python3 scripts/freeze.py
python3 scripts/noise_floor.py "$@"
