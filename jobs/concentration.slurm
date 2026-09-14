#!/bin/bash
# H1: are tracking errors spread evenly, or concentrated on particular footage?
#
#     sbatch jobs/concentration.slurm
#     python3 scripts/concentration.py --combine --split report
#
# MEASURED: ~0.4 s per animal, peak RSS 0.04 GB -- it reads Step 1's bit-packed
# masks rather than recomputing violations, so there is nothing to compute. A
# single job, not an array; 298 animals is about two minutes.
#
# The cheap half of the ensemble decision. It bounds how much an ensemble could
# help -- averaging removes errors members make INDEPENDENTLY -- and it cannot
# decide the question, which every read it emits says in its own reason string.
#SBATCH --job-name=vieb_conc
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/concentration
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"

python3 scripts/concentration.py --write-grid
while read -r tag; do
    python3 scripts/concentration.py --animal "$tag" --force
done < work/grids/concentration.txt

python3 scripts/concentration.py --combine --split report
