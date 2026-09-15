r"""Surrogates in the egocentric space, for the falsifier.

`TOK_PREREGISTRATION.md` §7: *if surrogates achieve comparable MDL, the stack is
fitting quantization structure rather than behaviour.* The claim at risk is rung
1 beating rung 0 by +11.3 nats/s on 89 of 89 animals. This module builds the
signals that test it.

## Why in ego space and not in pose

The pipeline under test consumes the 17-dimensional egocentric array. A
surrogate built on raw pose would then pass through the SE(2) transform, which
is a bijection and would faithfully carry whatever the surrogate has -- but it
would also be one more stage between the null and the thing being nulled. Making
the surrogate where the pipeline starts keeps the comparison to one substitution.

## The dead directions, and why they are copied rather than generated

Removing SE(2) equivariance from 14 coordinates costs three degrees of freedom.
Two of them are whole columns -- `s` at the origin keypoint is identically
`(0, 0)` -- and the third is a linear combination, the aligned axis's
y-component. A surrogate carrying variance in a direction the corpus never moves
in would be trivially separable from it, and would hand the quantizer structure
the corpus does not have.

**Phase randomisation preserves all three for free.** One phase per frequency,
shared across channels, multiplies every column by the same `exp(i*phi)`, so any
linear relationship between columns survives exactly and a zero column stays
zero. That is a substantive reason to prefer it as the primary arm, not just a
convenience.

**A VAR does not.** Its residual noise is full-rank in whatever subspace it was
fitted on, so it is fitted on the live channels only and the dead ones are
copied.

## What each destroys

`phase` keeps every channel's power spectrum and the cross-spectrum exactly --
so autocorrelation, so the run-length structure that quantizing a smooth signal
produces -- and destroys everything above second order. `var5` keeps what five
lags can express and destroys the rest. Neither contains a stereotyped sequence,
a repeated motif, or a behaviour.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
from recur.read import Read
from recur.recurrence import phase as ph
from recur.recurrence import surrogate as sg

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["KINDS", "VAR_ORDER", "falsifier_read", "generate", "live_columns"]

#: The two arms. `phase` is primary.
KINDS: tuple[str, ...] = ("phase", "var5")
#: Lag order for the VAR arm. Five, because VAR(5) is the registered null behind
#: Q1's +1.639% and the two results should sit on one footing.
VAR_ORDER = 5

#: recur predates annotation discipline; the boundary is declared once.
_randomize_block = cast("Callable[..., F64]", ph.randomize_block)
_fit_var = cast("Callable[..., Any]", sg.fit_var)
_simulate_var = cast("Callable[..., F64]", sg.simulate_var)


def live_columns(x: npt.ArrayLike, *, tol: float = 0.0) -> BOOL:
    """Columns that actually vary. The rest are the SE(2) rank deficit."""
    a = np.asarray(x, dtype=np.float64)
    return np.asarray(a.std(axis=0) > tol, dtype=bool)


def _phase_block(block: F64, rng: np.random.Generator) -> F64:
    return np.asarray(_randomize_block(block, rng), dtype=np.float64)


def _var_block(block: F64, rng: np.random.Generator, *, order: int) -> F64:
    """VAR(`order`) on the live columns; dead columns copied through.

    Falls back to phase randomisation when the fit is degenerate -- too few
    frames for the lag order, or a singular residual covariance -- which is
    `recur.recurrence.surrogate.ar_surrogate`'s own behaviour and is recorded
    per recording rather than hidden.
    """
    live = live_columns(block)
    if block.shape[0] <= order * (int(live.sum()) + 1) or not live.any():
        return _phase_block(block, rng)
    sub = block[:, live]
    mu = sub.mean(axis=0)
    fit = _fit_var(sub - mu, order)
    if fit is None:
        return _phase_block(block, rng)
    sim = _simulate_var(fit, sub.shape[0], rng, seed_state=(sub[:order] - mu))
    out = block.copy()
    out[:, live] = np.asarray(sim, dtype=np.float64) + mu
    return out


def generate(x: npt.ArrayLike, bounds: npt.ArrayLike, kind: str,
             rng: np.random.Generator, *, order: int = VAR_ORDER) -> Detail:
    """A surrogate of `x`, one recording at a time. Never across a seam.

    A surrogate built across a boundary would splice two recordings into one
    spectrum and then be compared against data where crossing a seam is
    forbidden everywhere else in this repo.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.int64)
    if kind not in KINDS:
        raise ValueError(f"unknown surrogate {kind!r}; one of {KINDS}")
    out = np.empty_like(a)
    n_fallback = 0
    for r in range(b.shape[0] - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        blk = a[lo:hi]
        if blk.shape[0] < 4:
            out[lo:hi] = blk
            continue
        if kind == "phase":
            out[lo:hi] = _phase_block(blk, rng)
        else:
            before = _var_block(blk, rng, order=order)
            out[lo:hi] = before
            live = live_columns(blk)
            if blk.shape[0] <= order * (int(live.sum()) + 1) or not live.any():
                n_fallback += 1
    # A zero column must stay a zero column, whatever the generator did: the
    # quantizer would otherwise find structure in a direction the corpus has
    # none in. Free under phase randomisation, asserted here for both arms.
    dead = ~live_columns(a)
    out[:, dead] = a[:, dead]
    return {"x": out.astype(np.float32), "kind": kind,
            "n_recordings": int(b.shape[0] - 1), "n_fallback": int(n_fallback),
            "n_dead_columns": int(dead.sum())}


def falsifier_read(delta: Mapping[str, Any], corpus: Mapping[str, Any],
                   surrogate: Mapping[str, Any], *, kind: str,
                   scored_object: Detail, n_effective: int) -> Read:
    """Does the corpus beat its surrogate on `rung 1 - rung 0`?

    The stopping rule is `FALSIFIER_PREREGISTRATION.md` §7 and is read exactly
    as registered: the verdict turns on whether the animal-level interval on
    `d_corpus - d_surrogate` excludes zero, and on **no threshold for the size
    of the gap**. A threshold set now would be arbitrary; one set after seeing
    the number would be selected against the outcome.
    """
    ci = dict(delta.get("animal_interval", {}))
    lo, hi = float(ci.get("lo", np.nan)), float(ci.get("hi", np.nan))
    point = float(ci.get("point", np.nan))
    detail: Detail = {
        "kind": kind,
        "corpus_advantage": corpus.get("mean_delta"),
        "surrogate_advantage": surrogate.get("mean_delta"),
        "gap": point, "animal_interval": ci,
        "frame_interval": dict(delta.get("frame_interval", {})),
        "frame_interval_note": ("printed to show how much narrower the wrong "
                                "method looks; the verdict does not read it"),
        "n_animals": int(n_effective),
    }
    if not np.isfinite(point):
        return Read("NOT_A_RESULT", scored_object,
                    f"the {kind} comparison produced no finite per-animal "
                    f"difference, so there is nothing to test",
                    n_effective=n_effective, degenerate=True, detail=detail)
    tail = (f"corpus {float(corpus.get('mean_delta', np.nan)):+.3f} against "
            f"{kind} {float(surrogate.get('mean_delta', np.nan)):+.3f} nats/s, "
            f"a gap of {point:+.3f} [{lo:+.3f}, {hi:+.3f}] animal-level over "
            f"{n_effective} animals")

    if lo > 0:
        return Read("PASS", scored_object,
                    f"the corpus beats its {kind} surrogate: {tail}, interval "
                    f"excluding zero. Rung 1's advantage over rung 0 is not "
                    f"explained by quantizing a smooth, second-order-matched "
                    f"signal. This does NOT license 'the model learned "
                    f"behaviour' -- two linear-Gaussian surrogates are two "
                    f"alternatives, not all of them",
                    n_effective=n_effective, detail=detail)
    if hi < 0:
        return Read("FAIL", scored_object,
                    f"the {kind} surrogate BEATS the corpus: {tail}, interval "
                    f"excluding zero on the wrong side. Every ladder number is "
                    f"withdrawn",
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"the corpus does not beat its {kind} surrogate: {tail}, and "
                f"the interval contains zero. The stack is fitting quantization "
                f"structure rather than behaviour, the coarse sweep does not "
                f"run, and that is the finding",
                n_effective=n_effective, detail=detail)
