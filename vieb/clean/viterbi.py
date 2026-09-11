r"""Anipose's Viterbi filter, loaded from its own source without its import chain.

## What it is, and why it is not a smoother

`viterbi_path` picks the most likely *path* of keypoint positions across frames
under a motion prior, from a candidate set per frame. Its effect is to decline
implausible jumps and bridge dropouts, rather than to shave amplitude off every
fast movement the way a low-pass or a constant-velocity Kalman smoother does.
That is a materially better fit for this corpus, where fast rare movement is the
signal and the whole worry about filtering is that it deletes exactly that.

It still carries a transition prior and can still suppress real ballistic
motion. The failure mode is narrower and it is **boundable**: `reassigned` in the
returned detail is the fraction of frames whose position the filter actually
changed, and an arm that reassigns 40% of frames is not de-glitching.

## The correction this module records

An earlier version of this repo asserted that the Viterbi filter could not run
here, on the grounds that all 3,080 `_full.pickle` files carry exactly one
detection per bodypart-frame. The detection count is right; the conclusion was
wrong, and it came from the paper's phrase "a set of top detections per frame"
rather than from the code.

The candidate set at frame `i` is **not** just frame `i`'s detections. Reading
the implementation below: for `j` in `0..n_back-1` it gathers the detections from
frame `i-j`, weighting their scores by `2**-j`, and falls back to an explicit
missing particle `[-1, -1, 0.001]` when a frame has none. So the state space is
`num_max * n_back + 1` wide, and at one detection per frame that is still
`n_back` historical candidates plus a missing state. The decision it makes is
whether to accept this frame's jump or carry an older position forward -- which
is the de-glitching, and it needs no multi-detection input at all. Anipose's own
`wrap_points()` carries the comment ``# n_possible = 1`` for exactly this case.

## Why the functions are loaded rather than imported

`anipose.filter_pose` imports `.common`, which imports `aniposelib.boards`, which
imports `numba`. That is the calibration and CLI half of Anipose -- three
packages and an LLVM toolchain -- and the filter touches none of it. Installing
it would also drag `opencv-contrib-python` on top of recur's pinned
`opencv-python-headless` **in a venv the two repos share**, which puts recur's
render path at risk for nothing.

So the two functions the filter actually needs are extracted from the installed
package's own source file by AST and executed with a numpy/scipy namespace. This
is Anipose's implementation, verbatim and version-recorded -- not a
reimplementation, and not a copy pasted into this repo where it would rot.
`anipose` is a real pinned dependency; only its import chain is bypassed.
"""

from __future__ import annotations

import ast
import os
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]

#: The functions lifted from `anipose/filter_pose.py`. `viterbi_path` calls
#: `remove_dups`; nothing else in that file is reachable from either.
NEEDED = ("remove_dups", "viterbi_path")

#: Anipose's defaults. `thres_dist` is in PIXELS and is the scale of the motion
#: prior -- the animal is ~110 px nose-to-tail here, so 30 px is about a quarter
#: of a body length per frame before a jump starts costing likelihood.
N_BACK = 3
THRES_DIST_PX = 30.0


def _source_path() -> str:
    import anipose
    return os.path.join(os.path.dirname(anipose.__file__), "filter_pose.py")


def _version() -> str:
    try:
        from importlib.metadata import version
        return str(version("anipose"))
    except Exception:                                  # noqa: BLE001
        return "unknown"


def _load() -> dict[str, Any]:
    """Extract `NEEDED` from anipose's source and exec them in a clean namespace.

    By AST rather than by regex, so a function that moves or is reformatted
    upstream either loads correctly or fails loudly instead of being silently
    half-matched.
    """
    from scipy import stats
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from scipy.special import logsumexp

    src = open(_source_path()).read()
    tree = ast.parse(src)
    picked: list[ast.stmt] = [n for n in tree.body
                              if isinstance(n, ast.FunctionDef)
                              and n.name in NEEDED]
    missing = set(NEEDED) - {n.name for n in picked
                             if isinstance(n, ast.FunctionDef)}
    if missing:
        raise SystemExit(
            f"anipose {_version()} does not define {sorted(missing)} in "
            f"{_source_path()}. The upstream file has changed shape; re-read it "
            f"rather than guessing which function replaced them.")
    ns: dict[str, Any] = {"np": np, "arr": np.array, "stats": stats,
                          "cdist": cdist, "cKDTree": cKDTree,
                          "logsumexp": logsumexp}
    exec(compile(ast.Module(body=picked, type_ignores=[]), _source_path(),
                 "exec"), ns)
    return ns


_NS: dict[str, Any] = _load()
viterbi_path = _NS["viterbi_path"]
SOURCE = _source_path()
VERSION = _version()


def viterbi(pose: npt.ArrayLike, conf: npt.ArrayLike | None, fps: float, *,
            n_back: int = N_BACK, thres_dist: float = THRES_DIST_PX,
            **_: Any) -> F64:
    """Anipose's Viterbi filter, one keypoint at a time, for ONE recording.

    Frames where the chosen particle is the **missing** state come back as
    `(-1, -1)`. Those are interpolated over, which is what Anipose's own pipeline
    does next; leaving a sentinel in a pose array would be read downstream as the
    animal teleporting to the origin.
    """
    p = np.asarray(pose, dtype=np.float64)
    t, k = p.shape[0], p.shape[1]
    c = (np.ones((t, k)) if conf is None
         else np.asarray(conf, dtype=np.float64))
    out = p.copy()
    if t < n_back + 2:
        return out
    idx = np.arange(t)
    for j in range(k):
        pts = p[:, j, :][:, None, :]                  # (T, n_possible=1, 2)
        scores = np.clip(c[:, j][:, None], 1e-6, 1.0)
        new, _ = viterbi_path(pts, scores, n_back=n_back, thres_dist=thres_dist)
        new = np.asarray(new, dtype=np.float64)
        gone = (new[:, 0] == -1) & (new[:, 1] == -1)
        if gone.all():
            continue
        if gone.any():
            for coord in range(2):
                new[gone, coord] = np.interp(idx[gone], idx[~gone],
                                             new[~gone, coord])
        out[:, j, :] = new
    return out


def reassignment(before: npt.ArrayLike, after: npt.ArrayLike,
                 *, tol: float = 1e-9) -> Detail:
    """How much of the recording the filter actually changed.

    The bound the user of a path-selection filter needs: it is a de-glitcher only
    if it declines a few frames. One that reassigns a large fraction is choosing
    a different trajectory, and whatever that is, it is not de-glitching.
    """
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    d = np.linalg.norm(b - a, axis=-1)
    moved = d > tol
    return {
        "reassigned": float(moved.mean()),
        "reassigned_per_keypoint": moved.mean(axis=0).tolist(),
        "median_move_px": float(np.median(d[moved])) if moved.any() else 0.0,
        "max_move_px": float(d.max()),
        "anipose_version": VERSION, "source": SOURCE,
        "n_back": N_BACK, "thres_dist_px": THRES_DIST_PX,
    }
