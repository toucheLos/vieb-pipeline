#!/bin/bash
# REPAIR's gates, then the ego transform on the arm they adopt (registered rule).
#SBATCH --job-name=vieb_repcomb
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 scripts/repair.py --phase combine
ARM=$(python3 -c "import json; r=json.load(open('results/repair.json'))['results']; print('repaired' if r['adopted_arm']=='repaired' else 'repaired_noveto')")
echo "adopted arm: $ARM"
python3 scripts/ego.py --write-grid
python3 scripts/ego.py --task 0 --n-tasks 1 --pose-arm "$ARM" --scale-arm bodylen
