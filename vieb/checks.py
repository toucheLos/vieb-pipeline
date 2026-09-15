r"""Assertions about quantities that claim to be probabilities.

## Why this module exists

`ladder.duration_pmf` spread each bin's mass over one whole duration too many.
`f(d)` summed to **1.115**, so every duration charge in the MDL objective was
wrong by roughly 0.11 nats. It was caught by a test written for a different
reason, not by review, and it would have been invisible in every downstream
number because a constant offset on every arm cancels in the comparisons and
survives in the absolute figures.

The rule this module exists to enforce:

> **A pmf that is never asserted to sum to 1 is not a pmf.**

It is repo-level rather than under `vieb/tok` because `vieb/qc` builds shares
too, and a second copy of this file would be a second thing to keep right.

## What these do and do not catch

They catch a normalisation that is wrong **now**, at the moment the array is
built, on real data. They do not catch a distribution that is correctly
normalised and wrong in some other way -- a pmf over the wrong support, or one
whose bins do not mean what the caller thinks. Those need a test.

`assert_pmf` deliberately refuses an all-zero vector rather than treating it as
a degenerate pass. A caller with no observations has a **missing** distribution,
not a uniform one, and the `Read` vocabulary already has `NOT_A_RESULT` for
saying so. Silently returning zeros is how `bones.py:332`'s degenerate branch
came to return a "share" that sums to nothing.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

__all__ = ["TOL", "assert_pmf", "assert_share", "assert_log_pmf",
           "assert_unit"]

#: Absolute tolerance on a sum. Float64 over a few thousand bins accumulates
#: well below this; anything above it is an arithmetic error, not rounding.
TOL = 1e-9


def assert_pmf(p: npt.ArrayLike, *, name: str, axis: int | None = None,
               tol: float = TOL, allow_empty: bool = False) -> F64:
    """Refuse `p` unless it is non-negative and sums to 1. Returns `p`.

    Returns the array so the call can wrap the expression that builds it rather
    than sitting on a line below, where it is easy to delete and easy to forget
    when a branch is added.

    `axis` checks each slice along that axis independently -- rows of a
    transition matrix, say.
    """
    a = np.asarray(p, dtype=np.float64)
    if a.size == 0:
        if allow_empty:
            return a
        raise AssertionError(f"{name}: empty, and an empty distribution is "
                             f"missing rather than uniform")
    if not np.isfinite(a).all():
        bad = int((~np.isfinite(a)).sum())
        raise AssertionError(f"{name}: {bad} non-finite entries")
    if (a < -tol).any():
        raise AssertionError(f"{name}: {int((a < -tol).sum())} negative "
                             f"entries, smallest {float(a.min()):.6g}")
    total = a.sum() if axis is None else a.sum(axis=axis)
    off = np.abs(np.asarray(total) - 1.0)
    if (off > tol).any():
        worst = float(np.max(off))
        got = float(np.asarray(total).ravel()[int(np.argmax(off))])
        where = "" if axis is None else f" (worst slice along axis {axis})"
        raise AssertionError(
            f"{name}: sums to {got:.12g}, off by {worst:.3g}{where}. A pmf "
            f"that does not sum to 1 is not a pmf -- see vieb/checks.py")
    return a


def assert_log_pmf(logp: npt.ArrayLike, *, name: str, axis: int | None = None,
                   tol: float = TOL) -> F64:
    """`assert_pmf` on `exp(logp)`, with `-inf` entries allowed.

    `-inf` is how a forbidden outcome is written -- the zero diagonal a
    run-length-encoded stream requires, for one -- so it is a legitimate value
    here and `exp` maps it to the zero it means.
    """
    a = np.asarray(logp, dtype=np.float64)
    if np.isposinf(a).any():
        raise AssertionError(f"{name}: +inf in a log-probability")
    if np.isnan(a).any():
        raise AssertionError(f"{name}: {int(np.isnan(a).sum())} NaN entries")
    assert_pmf(np.exp(a), name=name, axis=axis, tol=tol)
    return a


def assert_unit(total: npt.ArrayLike, *, name: str, tol: float = TOL) -> F64:
    """Refuse unless EVERY element is 1. For a per-cell mass already summed.

    Separate from `assert_pmf` for memory rather than taste. A table of
    `(context, elapsed bin)` cells, each of which is its own simplex, would have
    to be stacked along a new axis to use `assert_pmf(axis=-1)` -- and at the
    `k = 2` history depth that is 7.5M contexts x 13 bins x 3 outcomes, about
    2.3 GB of temporary, to check an identity that holds cellwise. The caller
    adds the masses itself and this checks the one array that results.
    """
    a = np.asarray(total, dtype=np.float64)
    if a.size == 0:
        raise AssertionError(f"{name}: empty")
    if not np.isfinite(a).all():
        raise AssertionError(f"{name}: {int((~np.isfinite(a)).sum())} "
                             f"non-finite cell totals")
    off = float(np.abs(a - 1.0).max())
    if off > tol:
        i = int(np.argmax(np.abs(a - 1.0)))
        raise AssertionError(
            f"{name}: worst cell sums to {float(a.ravel()[i]):.12g}, off by "
            f"{off:.3g} over {a.size} cells. A per-cell distribution that does "
            f"not sum to 1 is not a distribution -- see vieb/checks.py")
    return a


def assert_share(x: npt.ArrayLike, *, name: str, tol: float = TOL) -> F64:
    """Refuse `x` unless every entry is a finite fraction in `[0, 1]`.

    For quantities that are shares but not a partition of anything -- a
    violation rate, a cumulative Lorenz point, the mass at or below a bucket.
    They have no sum to check, and the bound is the only thing that can be.
    """
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise AssertionError(f"{name}: {int((~np.isfinite(a)).sum())} "
                             f"non-finite entries")
    lo, hi = float(a.min()) if a.size else 0.0, float(a.max()) if a.size else 0.0
    if lo < -tol or hi > 1.0 + tol:
        raise AssertionError(f"{name}: outside [0, 1] -- range "
                             f"[{lo:.6g}, {hi:.6g}]")
    return a
