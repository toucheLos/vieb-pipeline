r"""Cause-specific hazard, piecewise-constant on a log-spaced duration grid.

Nonparametric, no shape assumption, and it lets you **see** whether the hazard of
leaving a state is flat, rising or falling with how long the animal has already
been in it. That shape is the honest replacement for the retracted dwell test:
`audit/dwell.py` fitted a model with `kappa = 1e6`, which manufactures
non-geometric dwell on white noise, and its PASS is still sitting in two result
files. Nothing here carries a persistence prior.

## Discrete time, because the data is discrete

Durations are whole frames. A run of duration :math:`d` means the animal occupied
the state for frames :math:`0 \ldots d-1`, produced :math:`d-1` observed
*stay* events, and then one final event which is either an **exit** to some state
:math:`j` or **censoring** by the end of the recording.

At each at-risk frame the outcome is one of :math:`N` things: stay, or exit to one
of the :math:`N-1` other states. So a cell of the table is a multinomial and is
smoothed as one:

.. math::

    p(\text{exit to } j \mid c, b) =
        \frac{n_{cbj} + \alpha}{R_{cb} + \alpha N}, \qquad
    p(\text{stay} \mid c, b) =
        \frac{R_{cb} - \sum_j n_{cbj} + \alpha}{R_{cb} + \alpha N}

with :math:`R_{cb}` the frames at risk in context :math:`c` and elapsed-bin
:math:`b`. Parametrising the *stay* outcome alongside the exits is what keeps the
probabilities on the simplex: smoothing the exit hazards alone lets their sum
exceed one in a thin cell, and a thin cell is exactly where a deep history lands.

## The diagonal is zero and that is not a detail

Run-length encoding makes `AA` impossible, so :math:`j = u` is not a cause. A
model able to emit it would spend mass on a pair the data cannot contain and
would inflate every real 2-gram, which is the reason
`recur.seq.motif.transition_matrix` zeroes its diagonal too.

## Censoring

A terminal run contributes its stays and **nothing else** -- never a
:math:`S \lambda_j` term for an exit that was not observed. Getting this wrong
biases every duration estimate downward, because every truncated long run would
be recorded as having ended when the camera stopped.

The same applies at an abstain boundary in the arm that conditions abstain away:
a run that ends because tracking failed did not end because the animal did
something, and scoring it as a genuine exit would let the model earn likelihood
for predicting tracking dropout.

## No expansion over frames

At-risk counts are accumulated by **bin overlap** rather than by materialising one
row per elapsed frame. The corpus has 22.4M frames; expanding would be 176 MB of
int64 for a quantity that is twelve vector operations.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["ALPHA", "N_BINS", "TAIL_Q", "bin_of", "duration_grid", "fit",
           "log_likelihood", "shape_by_state"]

#: Laplace prior, inherited from `recur.seq.motif.transition_matrix`.
ALPHA = 1.0
#: Elapsed-time bins. Twelve is enough to show a monotone trend across three
#: orders of magnitude of duration without leaving cells empty at the tail.
N_BINS = 12
#: The grid's upper edge is this quantile of TUNE run length. Not the maximum:
#: one 40-second run would put eleven of the twelve bins below one frame.
TAIL_Q = 0.999


def duration_grid(durations: npt.ArrayLike, *, n_bins: int = N_BINS,
                  q: float = TAIL_Q) -> I64:
    """Log-spaced elapsed-time bin edges, in frames. Fitted on **tune** only.

    Returns `n_bins + 1` edges starting at 0, with the last one open at
    infinity. The grid is a hyperparameter and is chosen where every other
    hyperparameter on this branch is chosen.
    """
    d = np.asarray(durations, dtype=np.float64)
    d = d[np.isfinite(d) & (d >= 1)]
    if d.size == 0:
        raise ValueError("no durations to build a grid on")
    top = max(float(np.quantile(d, q)), 2.0)
    inner = np.unique(np.round(np.geomspace(1.0, top, int(n_bins))).astype(np.int64))
    edges = np.concatenate([[0], inner, [np.iinfo(np.int64).max]])
    return np.unique(edges).astype(np.int64)


def bin_of(elapsed: npt.ArrayLike, edges: npt.ArrayLike) -> I64:
    """Which elapsed-time bin a frame index falls in. Half-open `[lo, hi)`."""
    e = np.asarray(edges, dtype=np.int64)
    return (np.searchsorted(e, np.asarray(elapsed, dtype=np.int64),
                            side="right") - 1).astype(np.int64)


def _overlap(duration: I64, edges: I64, b: int) -> F64:
    """Frames of `[0, duration)` that fall in bin `b`. Vectorised over runs."""
    lo, hi = int(edges[b]), int(edges[b + 1])
    d = np.asarray(duration, dtype=np.float64)
    return np.clip(np.minimum(d, float(hi)) - float(lo), 0.0, None)


def fit(context: npt.ArrayLike, duration: npt.ArrayLike,
        next_state: npt.ArrayLike, censored: npt.ArrayLike, *,
        edges: npt.ArrayLike, n_context: int, n_states: int,
        alpha: float = ALPHA) -> Detail:
    """Count frames at risk and exits, per `(context, elapsed bin, cause)`.

    `next_state` is ignored wherever `censored` is true, and no exit is recorded
    there. Exits are held sparsely: the dense table is
    `n_context x n_bins x n_states`, which at 2,048 symbols and `k = 2` would be
    larger than the number of atoms this cluster has memory for.
    """
    c = np.asarray(context, dtype=np.int64)
    d = np.asarray(duration, dtype=np.int64)
    j = np.asarray(next_state, dtype=np.int64)
    cen = np.asarray(censored, dtype=bool)
    e = np.asarray(edges, dtype=np.int64)
    n_bins = int(e.shape[0] - 1)

    # Stays happen over [0, d-1) in BOTH cases: a completed run's last frame is
    # the exit, and a censored run's last frame is unobserved.
    stay_span = np.maximum(d - 1, 0)
    at_risk = np.zeros((int(n_context), n_bins), dtype=np.float64)
    for b in range(n_bins):
        w = _overlap(stay_span, e, b)
        nz = w > 0
        if nz.any():
            at_risk[:, b] = np.bincount(c[nz], weights=w[nz],
                                        minlength=int(n_context))

    done = ~cen
    exit_bin = bin_of(np.maximum(d - 1, 0), e)
    # One at-risk frame for the terminal event of every completed run: the frame
    # on which the exit was actually observed.
    if done.any():
        np.add.at(at_risk, (c[done], exit_bin[done]), 1.0)

    keys = ((c[done] * n_bins + exit_bin[done]).astype(np.int64)
            * int(n_states) + j[done])
    uniq, counts = np.unique(keys, return_counts=True)
    exits_per_cell = np.zeros((int(n_context), n_bins), dtype=np.float64)
    if done.any():
        np.add.at(exits_per_cell, (c[done], exit_bin[done]), 1.0)

    denom = at_risk + alpha * float(n_states)
    log_stay = np.log(np.maximum(at_risk - exits_per_cell, 0.0) + alpha) \
        - np.log(denom)
    return {
        "edges": e, "n_bins": n_bins, "n_context": int(n_context),
        "n_states": int(n_states), "alpha": float(alpha),
        "at_risk": at_risk, "exits_per_cell": exits_per_cell,
        "log_stay": log_stay, "log_denom": np.log(denom),
        "exit_keys": uniq, "exit_counts": counts.astype(np.float64),
        "n_runs": int(d.shape[0]), "n_censored": int(cen.sum()),
        "n_cells_observed": int((at_risk > 0).sum()),
    }


def _log_exit(model: Mapping[str, Any], c: I64, b: I64, j: I64) -> F64:
    """`log p(exit to j | c, b)`, looked up sparsely and backed off by Laplace."""
    n_bins, n_states = int(model["n_bins"]), int(model["n_states"])
    keys = (c * n_bins + b).astype(np.int64) * n_states + j
    idx = np.searchsorted(model["exit_keys"], keys)
    idx = np.clip(idx, 0, max(model["exit_keys"].shape[0] - 1, 0))
    hit = (model["exit_keys"].shape[0] > 0) & (model["exit_keys"][idx] == keys)
    n = np.where(hit, model["exit_counts"][idx], 0.0)
    out = np.log(n + float(model["alpha"])) - model["log_denom"][c, b]
    return np.asarray(out, dtype=np.float64)


def log_likelihood(model: Mapping[str, Any], context: npt.ArrayLike,
                   duration: npt.ArrayLike, next_state: npt.ArrayLike,
                   censored: npt.ArrayLike) -> F64:
    """Per-run log-probability of `(duration, next state)` jointly.

    A completed run pays for `d - 1` stays and one exit; a censored run pays for
    its stays alone. Both are accumulated by bin overlap, never by expanding to
    one row per frame.
    """
    c = np.asarray(context, dtype=np.int64)
    d = np.asarray(duration, dtype=np.int64)
    j = np.asarray(next_state, dtype=np.int64)
    cen = np.asarray(censored, dtype=bool)
    e = np.asarray(model["edges"], dtype=np.int64)
    stay_span = np.maximum(d - 1, 0)

    out = np.zeros(d.shape[0], dtype=np.float64)
    for b in range(int(model["n_bins"])):
        w = _overlap(stay_span, e, b)
        nz = w > 0
        if nz.any():
            out[nz] += w[nz] * model["log_stay"][c[nz], b]
    done = ~cen
    if done.any():
        eb = bin_of(stay_span[done], e)
        out[done] += _log_exit(model, c[done], eb, j[done])
    return out


def shape_by_state(model: Mapping[str, Any], states: npt.ArrayLike,
                   *, top: int = 12) -> list[Detail]:
    """Total exit hazard against elapsed-time bin, for the busiest states.

    The readable output of this whole module. A **flat** profile is memoryless
    given the state; **falling** means the longer the animal has been doing
    something the less likely it is to stop; **rising** is the opposite. The
    retracted dwell instrument claimed to adjudicate this and could not, so the
    profile is reported as a shape rather than reduced to a test statistic.
    """
    at_risk = np.asarray(model["at_risk"], dtype=np.float64)
    exits = np.asarray(model["exits_per_cell"], dtype=np.float64)
    order = np.argsort(at_risk.sum(axis=1))[::-1]
    s = np.asarray(states, dtype=np.int64)
    rows: list[Detail] = []
    for c in order[:int(top)]:
        r, x = at_risk[c], exits[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            haz = np.where(r > 0, x / np.maximum(r, 1e-12), np.nan)
        rows.append({"context": int(c),
                     "state": int(s[c]) if c < s.shape[0] else int(c),
                     "frames_at_risk": float(r.sum()),
                     "hazard_by_bin": [None if not np.isfinite(v) else float(v)
                                       for v in haz]})
    return rows
