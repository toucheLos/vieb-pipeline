#!/bin/bash
# STEP 1 §3 M: Keypoint-MoSeq on CalMS21, the MOSEQ protocol unchanged
# (scripts/moseq_kpms.py: latent_dim 10, nlags 3, kappa 1e4 x50 then 1e6 to 550).
# One array task = one arm: PCA (setup), fit at the arm's seed, apply to the
# scored videos.
#     sbatch --array=0-44%4 jobs/calms_kpms.slurm            # index -> calms_prep arms()
#     sbatch jobs/calms_kpms.slurm m0_s0 --limit 2 --iters-arhmm 3 --iters-full 3   # smoke
#SBATCH --job-name=vieb_calmskpms
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARM=$(source "$REPO/env.sh" >/dev/null 2>&1; python3 scripts/calms_prep.py arms | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
    EXTRA=("$@")
else
    ARM="$1"; shift; EXTRA=("$@")
fi
SEED="${ARM##*_s}"
module purge
module load python/3.11.4
export LD_LIBRARY_PATH=$(find /home/tul26194/moseq/venv-moseq/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
source /home/tul26194/moseq/venv-moseq/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export MOSEQ_WORK="$REPO/work/calms/moseq"
python -c "import jax; print('JAX devices:', jax.devices())"
echo "arm=$ARM seed=$SEED"
python scripts/moseq_kpms.py setup --arm "$ARM" "${EXTRA[@]}"
python scripts/moseq_kpms.py fit   --arm "$ARM" --seed "$SEED" "${EXTRA[@]}"
python scripts/moseq_kpms.py apply --arm "$ARM" --seed "$SEED" "${EXTRA[@]}"
