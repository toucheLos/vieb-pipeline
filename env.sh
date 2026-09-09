#!/bin/bash
# Source this before anything else.
#
#     source /home/tul26194/vieb-pipeline/env.sh
#
# Inherited verbatim from ~/recur/env.sh, including its two warnings, because
# this repo consumes float32 arrays that shapeflow produced under 3.11.4 and
# asserts numeric agreement with them.
#
# 1. NEVER INVOKE venv/bin/python3 DIRECTLY. It is linked against
#    libpython3.11.so.1.0, which only exists on the module path, and without
#    `module load python/3.11.4` it dies with a shared-library error that looks
#    nothing like the missing module it actually is.
#
# 2. `module load python/3.11.4` DOES NOT WIN ON A LOGIN SHELL. ~/.bashrc runs a
#    conda init block whose PATH entry sits ahead of the module's. The check
#    below asserts BOTH sys.prefix and sys.version -- an earlier version checked
#    only the prefix, which was right while the version was wrong.
#
# WHAT IS DIFFERENT HERE: PYTHONPATH carries recur as well as this repo.
#
# This repo imports recur's utility layer (`Read`, `boot`, `splits`, `anchors`,
# `util`, `labels`, `kendall`, `geom/*`, `qc/swap`) and content-hashes recur's
# and shapeflow's ARTIFACTS through vieb/io/spine.py. Hash the artifacts, import
# the utilities: the provenance rule is about data that can change underneath a
# result, not about reusing a verdict type. Forking `Read` would give this
# project two verdict vocabularies that drift.

# THE VENV IS A SYMLINK TO RECUR'S, and that is a deliberate, recorded choice.
# The pins in requirements.txt are identical to recur's by design -- this repo
# consumes float32 arrays recur and shapeflow produced -- so two venvs would be
# ~5 GB of duplicated torch to hold the same versions. The cost is real and is
# stated here rather than discovered: `pip install` from either repo lands in
# both. Anything installed for this repo must be added to BOTH requirements.txt
# files, or recur's environment will carry a package its pins do not name.
# Break the symlink and build a real venv the moment the two need to diverge.

REPO="/home/tul26194/vieb-pipeline"
RECUR="${VIEB_RECUR:-/home/tul26194/recur}"
WANT_PY="3.11"

module load python/3.11.4
# shellcheck disable=SC1091
source "$REPO/venv/bin/activate"

export PYTHONPATH="$REPO:$RECUR${PYTHONPATH:+:$PYTHONPATH}"
export VIEB_RECUR="$RECUR"

# Compared by REALPATH, because venv's activate resolves the symlink above and
# puts recur's real bin dir on PATH. Resolving both sides keeps the check doing
# the job it exists for -- catching conda base shadowing the module -- without
# failing on the shared venv.
_want="$(readlink -f "$REPO/venv/bin/python3")"
_got="$(readlink -f "$(which python3)")"
if [ "$_got" != "$_want" ]; then
    echo "WRONG INTERPRETER: which python3 resolves to $_got, expected $_want" >&2
    return 1 2>/dev/null || exit 1
fi
python3 - <<PYCHECK || { echo "interpreter check failed" >&2; return 1 2>/dev/null || exit 1; }
import os, sys
assert os.path.realpath(sys.prefix) == os.path.realpath("$REPO/venv"), sys.prefix
got = "%d.%d" % sys.version_info[:2]
assert got == "$WANT_PY", (
    f"python {got}, expected $WANT_PY -- the venv was built with the wrong "
    f"interpreter (almost certainly conda base shadowing the module)")
import recur, vieb   # both must import, or PYTHONPATH is wrong
assert os.path.dirname(os.path.dirname(recur.__file__)) == "$RECUR", recur.__file__
PYCHECK
unset _want _got WANT_PY
