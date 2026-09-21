#!/bin/bash
# Order-controlled discontinuity plant: the first accuracy number.
#
#   sbatch jobs/plant.slurm
#
# 60 tune animals x 2 arms x 3 orders x 6 amplitudes. Measured on a 2-animal
# smoke test at ~7 s per animal for the whole sweep.
#
# Registered in results/PLANT_PREREGISTRATION.md.
#SBATCH --job-name=vieb_plant
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
python3 scripts/plant.py "$@"
