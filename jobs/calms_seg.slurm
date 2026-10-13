#!/bin/bash
# STEP 1 §3 D and C on one arm (GPU: the cross-video kNN is exact on CUDA).
#     sbatch --array=0-44%4 jobs/calms_seg.slurm       # index -> calms_prep arms()
#     sbatch jobs/calms_seg.slurm m0_s0 --limit 6        # smoke
#     LIST="--list seg --new" sbatch --array=0-164%6 jobs/calms_seg.slurm
#SBATCH --job-name=vieb_calmsseg
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARM=$(python3 scripts/calms_prep.py arms ${LIST:-} | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
else
    ARM="$1"; shift
fi
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'"
python3 scripts/calms_seg.py --arm "$ARM" --device cuda "$@"
