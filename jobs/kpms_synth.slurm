#!/bin/bash
# Paper 1 optional analysis: KPMS on its own synthetic (LDS) null, the MOSEQ protocol unchanged.
#     sbatch jobs/kpms_synth_cpu.slurm build      # CPU, first
#     sbatch jobs/kpms_synth.slurm setup
#     sbatch --array=0-2 jobs/kpms_synth.slurm fit   # seed = array index; fit + apply
#SBATCH --job-name=vieb_kpsynth
#SBATCH --output=work/logs/%x_%A_%a.out
#SBATCH --error=work/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs
module purge
module load python/3.11.4
export LD_LIBRARY_PATH=$(find /home/tul26194/moseq/venv-moseq/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
source /home/tul26194/moseq/venv-moseq/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export MOSEQ_WORK="$REPO/work/kpms_synth"
if [ "$1" = "setup" ]; then
    python scripts/moseq_kpms.py setup --arm lds
else
    S="${SLURM_ARRAY_TASK_ID:-0}"
    python scripts/moseq_kpms.py fit   --arm lds --seed "$S"
    python scripts/moseq_kpms.py apply --arm lds --seed "$S"
fi
