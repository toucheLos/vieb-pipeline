#!/bin/bash
# MOSEQ §2: one kpms phase. Arguments are forwarded to scripts/moseq_kpms.py:
#     sbatch jobs/moseq_kpms.slurm setup --arm real
#     sbatch jobs/moseq_kpms.slurm fit --arm real --seed 0
# As an array over the registered 11 (arm, seed) pairs, set MOSEQ_PHASE and use
# --array=0-10: the task index picks from PAIRS below.
#SBATCH --job-name=vieb_kpms
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
module purge
module load python/3.11.4
export LD_LIBRARY_PATH=$(find /home/tul26194/moseq/venv-moseq/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
source /home/tul26194/moseq/venv-moseq/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
python -c "import jax; print('JAX devices:', jax.devices())"
PAIRS=(real:0 real:1 real:2 real:3 real:4 white:0 white:1 white:2 phase:0 phase:1 phase:2)
if [ -n "${MOSEQ_PHASE:-}" ]; then
    IFS=: read -r ARM SEED <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
    python scripts/moseq_kpms.py "$MOSEQ_PHASE" --arm "$ARM" --seed "$SEED" "$@"
else
    python scripts/moseq_kpms.py "$@"
fi
