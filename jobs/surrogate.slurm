#!/bin/bash
# The falsifier. One job per surrogate arm, running the WHOLE chain serially.
#
#     sbatch jobs/surrogate.slurm --fit-microstate     # once, before the two
#     sbatch jobs/surrogate.slurm --kind phase          #   visit-stitching arms
#     sbatch jobs/surrogate.slurm --kind var5
#     sbatch jobs/surrogate.slurm --kind microstate
#     sbatch jobs/surrogate.slurm --kind microstate0
#     python3 scripts/surrogate.py --combine
#
# --fit-microstate fits the null's own partition and pooled chain on the
# CORPUS's tune split and writes work/surrogate/microstate_fit.npz. It is a
# prerequisite for `microstate` and `microstate0` only. Two partitions exist in
# this arm: that one is the null's internal granularity, and the tokenizer fits
# its own on the surrogate afterwards.
#
# READ results/FALSIFIER_PREREGISTRATION.md FIRST. It was committed before any
# surrogate was generated and it fixes the stopping rule: if the animal-level
# interval on (corpus - surrogate) contains zero, the stack is fitting
# quantization structure and the coarse sweep does not run.
#
# The chain is quantize and ladder UNMODIFIED, with VIEB_EGO_DIR and
# VIEB_TOK_DIR redirected at the surrogate tree. There is deliberately no
# surrogate-specific copy of any stage: a control that runs through different
# code from the thing it controls is not a control.
#
# Serial rather than an array because the stages depend on each other -- basis
# before cells, cells before the ladder -- and the whole chain is under an hour.
#
# CPU: FFTs, a least-squares VAR fit, MiniBatchKMeans and counting.
#SBATCH --job-name=vieb_surrogate
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
REPO="/home/tul26194/vieb-pipeline"; cd "$REPO"
mkdir -p work/logs work/surrogate
source "$REPO/env.sh"
python3 scripts/freeze.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --generate-only stops after writing the surrogate ego shards. Step 2 of the
# segmentation arm needs those and nothing downstream of them, and the
# tokenizer chain is ~20 minutes per arm that would go unread.
KIND=""; FIT=0; GEN_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --kind) KIND="$2"; shift 2 ;;
        --fit-microstate) FIT=1; shift ;;
        --generate-only) GEN_ONLY=1; shift ;;
        *) shift ;;
    esac
done
if [ "$FIT" = "1" ]; then
    echo "node=$(hostname) fit-microstate"
    python3 scripts/surrogate.py --fit-microstate
    echo "done: fit-microstate"
    exit 0
fi
[ -n "$KIND" ] || { echo "pass --kind K or --fit-microstate" >&2; exit 2; }
echo "node=$(hostname) kind=$KIND"

ROOT="${VIEB_SURROGATE_ROOT:-$REPO/work/surrogate}/$KIND"
python3 scripts/surrogate.py --generate --kind "$KIND"

if [ "$GEN_ONLY" = "1" ]; then
    echo "done: $KIND (generate only)"
    exit 0
fi

export VIEB_EGO_DIR="$ROOT/ego"
export VIEB_TOK_DIR="$ROOT/tok"
python3 scripts/quantize.py --basis
for N in 256 512; do
    python3 scripts/quantize.py --cell --arm plain --n-states "$N"
    python3 scripts/ladder.py  --cell --arm plain --n-states "$N"
done
echo "done: $KIND"
