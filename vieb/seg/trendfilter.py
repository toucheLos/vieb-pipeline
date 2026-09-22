"""Group-sparse l1 trend filtering: the axiom as an estimator, not a difference.

READ results/TRENDFILTER_PREREGISTRATION.md FIRST.

## Why this and not a second derivative

The frozen detector takes a numerical second derivative of DLC position and
thresholds its mismatch. `NOISEFLOOR.md` measured the jitter that step
amplifies, and `PLANT.md` measured the cost: **16 sigma-bar, about 0.99 body
lengths, to recover an acceleration discontinuity at 50%**, against 4 for a
position jump.

Penalising the **third difference of position** estimates the same quantity by
regularised fitting instead. The fit is piecewise-quadratic -- piecewise-linear
velocity, piecewise-constant acceleration -- so its knots are **exactly
acceleration jumps**, which is the axiom written down rather than approximated.
**It never differentiates the data.**

    minimise over X:   1/2 * sum_t w_t ||Y_t - X_t||^2  +  lam * sum_t ||(D3 X)_t||_2

* the group norm across channels makes a knot **shared**: a behavioural
  boundary is a boundary in the animal, not in one keypoint;
* `w_t = 1/sigma^2(c_t)` from the skull-bone calibration is where the error
  model enters -- a low-confidence frame pulls the fit less hard.

## How it is solved

ADMM with `Z = D3 X`. The X-step is a symmetric banded system
`(W + rho D3' D3) X = W Y + rho D3'(Z - U)` with bandwidth 3, solved by
`scipy.linalg.solveh_banded` in O(T). The Z-step is a group soft-threshold.
Convex, one parameter, deterministic, and no `cvxpy` -- which is not installed
and could not be, since the venv is shared with `~/recur` (`vieb/clean/arms.py`
:11-22 records the same refusal for `movement`).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.linalg import solveh_banded

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.read import Read                                         # noqa: E402

__all__ = ["ALPHAS", "KNOT_OFFSET", "RHO", "MAX_ITERS", "ABS_TOL",
           "JITTER_SHARE", "operator", "lambda_max", "scale_of", "fit",
           "knots",
           "peaks_of", "lambda_read", "compare_read"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: The registered grid, as a fraction of each recording's own `lambda_max`.
#: Scale-adaptive per recording, exactly as `k_mad` is.
#: Multipliers on `scale_of`. The registered grid was a fraction of
#: `lambda_max` and is superseded -- see `scale_of` and `DEVIATIONS.md` D13.
ALPHAS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
#: A knot at row `r` of `D3` is the boundary at frame `r + 2` -- the first
#: sample after the break. Fixed in the registration so it cannot be chosen to
#: flatter a recovery number.
KNOT_OFFSET = 2
RHO = 1.0
MAX_ITERS = 4000
ABS_TOL = 1e-6
#: Boyd residual balancing. `rho` is an ADMM penalty and NOT a model parameter:
#: it changes how fast the solver converges and not what it converges to, so
#: adapting it is a numerical decision, which is why
#: `TRENDFILTER_PREREGISTRATION.md` §6 forbids tuning it against a recovery
#: number rather than forbidding it outright.
#:
#: It must be adapted. With `rho` fixed the solver stalls on `D3`'s
#: conditioning; with `rho = lam` -- the other obvious choice -- the soft
#: threshold `lam/rho` becomes identically 1 and the shrinkage stops depending
#: on `lam` at all, which silently turns the whole sweep into one operating
#: point.
RHO_MU = 10.0
RHO_TAU = 2.0
RHO_EVERY = 10
#: At most one boundary in twenty at the finest retained scale may be
#: attributable to jitter. Invented in the registration and named there.
JITTER_SHARE = 0.05

_CACHE: dict[int, tuple[Any, F64, F64]] = {}


def operator(t: int) -> tuple[Any, F64, F64]:
    """`(D3, banded(D3' D3), banded(D3 D3'))` for length `t`, cached.

    The banded form is extracted numerically rather than from the interior
    stencil `[20, -15, 6, -1]`, so the edge rows are exact instead of
    hand-derived -- an off-by-one at the edge would move boundaries at the
    start of every recording, where nothing would notice.
    """
    if t not in _CACHE:
        if t < 8:
            raise ValueError("need at least 8 frames")
        e = np.ones(t - 3, dtype=np.float64)
        bands: Any = [-e, 3.0 * e, -3.0 * e, e]
        d3 = sp.diags(bands, [0, 1, 2, 3], shape=(t - 3, t)).tocsr()
        g = (d3.T @ d3).tocsr()
        ab = np.zeros((4, t), dtype=np.float64)
        for d in range(4):
            ab[3 - d, d:] = g.diagonal(d)
        h = (d3 @ d3.T).tocsr()
        m = t - 3
        gb = np.zeros((4, m), dtype=np.float64)
        for d in range(4):
            gb[3 - d, d:] = h.diagonal(d)
        _CACHE[t] = (d3, ab, gb)
        if len(_CACHE) > 24:
            _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[t]


def lambda_max(y: npt.ArrayLike) -> float:
    """The smallest `lam` at which the fit has NO knots: the DUAL norm

        max_t || ((D3 D3')^-1 D3 Y)_t ||_2

    **Not** `max_t ||(D3 Y)_t||_2`, which is what
    `TRENDFILTER_PREREGISTRATION.md` §3 wrote down and which is wrong for the
    job §3 gives it. That quantity has no relationship to where knots vanish:
    measured on real-scale data it is **2.3e10 times smaller** than the value
    at which the solution empties, so the entire registered alpha grid would
    have lived in the fully dense regime and `lambda_read` would have returned
    `GRID_LIMITED` by construction rather than by evidence. Recorded as
    `DEVIATIONS.md` D13.

    With this definition `alpha = 1` gives no knots and `alpha -> 0` gives the
    interpolating fit, so the registered grid spans the path it was meant to.
    """
    a = np.asarray(y, dtype=np.float64)
    d3, _, gb = operator(a.shape[0])
    v = np.asarray(d3 @ a, dtype=np.float64)
    if v.size == 0:
        return 0.0
    dual = solveh_banded(gb, v)
    return float(np.max(np.linalg.norm(dual, axis=1)))


def fit(y: npt.ArrayLike, w: npt.ArrayLike, lam: float, *, rho: float = RHO,
        max_iters: int = MAX_ITERS, abs_tol: float = ABS_TOL,
        warm: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Solve the trend filter. Returns the fit, the knot rows, and residuals.

    Iterations stop on the primal/dual residual rather than at a fixed count,
    so the setting is numerical and not tuned against an outcome.

    `warm` carries `z`, `u` and `rho` from a neighbouring `lam` on the same
    recording. Sweeping the path warm cuts the sparse end from thousands of
    iterations to hundreds and changes nothing about the fixed point.
    """
    a = np.asarray(y, dtype=np.float64)
    t, c = a.shape
    d3, ab, _ = operator(t)
    wt = np.asarray(w, dtype=np.float64).reshape(-1)
    mat = rho * ab.copy()
    mat[3, :] += wt
    wy = wt[:, None] * a
    x = a.copy()
    if warm is not None:
        z = np.array(warm["z"], dtype=np.float64, copy=True)
        u = np.array(warm["u"], dtype=np.float64, copy=True)
        rho = float(warm.get("rho_final", rho))
        mat = rho * ab.copy()
        mat[3, :] += wt
    else:
        z = np.asarray(d3 @ x, dtype=np.float64)
        u = np.zeros_like(z)
    it = 0
    pri = dua = float("nan")
    for it in range(1, max_iters + 1):
        if it > 1 and it % RHO_EVERY == 1:
            if pri > RHO_MU * dua:
                rho, u = rho * RHO_TAU, u / RHO_TAU
                mat = rho * ab.copy()
                mat[3, :] += wt
            elif dua > RHO_MU * pri:
                rho, u = rho / RHO_TAU, u * RHO_TAU
                mat = rho * ab.copy()
                mat[3, :] += wt
        x = solveh_banded(mat, wy + rho * (d3.T @ (z - u)))
        v = np.asarray(d3 @ x, dtype=np.float64) + u
        n = np.linalg.norm(v, axis=1, keepdims=True)
        z_new = v * np.maximum(0.0, 1.0 - (lam / rho) / np.maximum(n, 1e-300))
        dx = np.asarray(d3 @ x, dtype=np.float64)
        u = u + dx - z_new
        pri = float(np.linalg.norm(dx - z_new))
        dua = float(rho * np.linalg.norm(z_new - z))
        z = z_new
        if pri < abs_tol * max(1.0, float(np.linalg.norm(dx))) and \
                dua < abs_tol * max(1.0, float(np.linalg.norm(z))):
            break
    return {"x": x, "z": z, "u": u, "n_iter": it, "primal": pri, "dual": dua,
            "rho_final": float(rho), "converged": bool(it < max_iters)}


def knots(z: npt.ArrayLike, *, offset: int = KNOT_OFFSET) -> I64:
    """Boundary frames: rows of `Z` the soft-threshold did not zero."""
    v = np.asarray(z, dtype=np.float64)
    if v.size == 0:
        return np.zeros(0, dtype=np.int64)
    out = np.flatnonzero(np.linalg.norm(v, axis=1) > 0.0) + int(offset)
    return np.asarray(out, dtype=np.int64)


def scale_of(y: npt.ArrayLike) -> float:
    """`median_t ||(D3 Y)_t||_2` -- the robust scale `alpha` multiplies.

    Deliberately the analogue of `mad_threshold`: that sets a cut at
    `median(D) + k * MAD(D)` of the detector's own scalar, and this sets a
    penalty at `alpha` times a robust scale of the same quantity this
    estimator penalises. Both are per recording, so both mean "this far above
    this recording's own typical roughness".
    """
    a = np.asarray(y, dtype=np.float64)
    d3, _, _ = operator(a.shape[0])
    v = np.asarray(d3 @ a, dtype=np.float64)
    return float(np.median(np.linalg.norm(v, axis=1))) if v.size else 0.0


def peaks_of(y: npt.ArrayLike, w: npt.ArrayLike, alpha: float, *,
             blocked: npt.ArrayLike | None = None,
             rho: float = RHO,
             warm: Mapping[str, Any] | None = None) -> I64:
    """Boundaries for ONE recording, slice-local -- the detector interface.

    The same shape as `floor.peaks_of` so the plant harness can take either
    detector without knowing which it has. The abstain veto is re-applied here
    rather than inside the solver: an external boundary set that skipped it
    would be measuring a different exclusion from every other stage.
    """
    a = np.asarray(y, dtype=np.float64)
    if a.shape[0] < 8:
        return np.zeros(0, dtype=np.int64)
    lm = scale_of(a)
    if not np.isfinite(lm) or lm <= 0.0:
        return np.zeros(0, dtype=np.int64)
    got = fit(a, w, float(alpha) * lm, rho=rho, warm=warm)
    k = knots(got["z"])
    k = k[(k >= 0) & (k < a.shape[0])]
    if blocked is not None:
        b = np.asarray(blocked, dtype=bool)
        k = k[~b[k]]
    return np.asarray(k, dtype=np.int64)


def lambda_read(rows: Mapping[float, Mapping[str, Any]], *,
                scored_object: dict[str, Any], n_effective: int) -> Read:
    """Pick the smallest alpha whose jitter share clears the registered bar.

    `GRID_LIMITED` rather than an extended grid if none does: extending after
    seeing where the answer falls is selecting a hyperparameter against an
    outcome, which is what `LADDER.md` refused when MDL pushed below N = 256.
    """
    ok: list[float] = []
    table: dict[str, Any] = {}
    for a in ALPHAS:
        r = rows.get(a)
        if r is None:
            continue
        share = (float(r["jitter_hi"]) / float(r["corpus"])
                 if float(r["corpus"]) > 0 else float("inf"))
        table[f"{a:g}"] = {"corpus_rate": float(r["corpus"]),
                           "jitter_rate": float(r["jitter"]),
                           "jitter_hi": float(r["jitter_hi"]),
                           "share": share}
        if share < JITTER_SHARE:
            ok.append(a)
    detail = {"grid": list(ALPHAS), "share_bar": JITTER_SHARE,
              "by_alpha": table, "n_passing": len(ok)}
    if not ok:
        return Read("GRID_LIMITED", scored_object,
                    (f"no alpha on the registered grid brings the jitter-only "
                     f"boundary rate below {JITTER_SHARE:.0%} of the corpus "
                     f"rate. The grid is NOT extended after the fact; the "
                     f"direction is reported and the selection is refused"),
                    n_effective=n_effective, detail=detail)
    star = min(ok)
    detail["lambda_star_alpha"] = star
    got = table[f"{star:g}"]
    if star == ALPHAS[0]:
        return Read("GRID_LIMITED", scored_object,
                    (f"the smallest alpha on the grid ({star:g}) already "
                     f"clears the bar at share {got['share']:.4f}, so the "
                     f"selection is limited in the direction of finer scales "
                     f"and the grid is not extended to chase it"),
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                (f"lambda* is alpha = {star:g}: jitter-only rate "
                 f"{got['jitter_rate']:.4f}/s against a corpus rate of "
                 f"{got['corpus_rate']:.4f}/s, a share of {got['share']:.4f} "
                 f"below the registered {JITTER_SHARE:.2f}. Chosen against a "
                 f"measured floor, not by eye"),
                n_effective=n_effective, detail=detail)


def compare_read(new: float, old: float, *, order: int, amp: float,
                 scored_object: dict[str, Any], n_effective: int) -> Read:
    """Prediction 2: does the estimator beat the frozen detector at its own
    axiom's break?"""
    detail = {"order": int(order), "amp_sigma": float(amp),
              "trendfilter_recall": float(new), "frozen_recall": float(old),
              "delta": float(new - old)}
    if new > old:
        return Read("PASS", scored_object,
                    (f"at order {order}, {amp:g} sigma the trend filter "
                     f"recovers {new:.4f} against the frozen detector's "
                     f"{old:.4f}, a gain of {new - old:+.4f}. Estimating the "
                     f"break beats differencing toward it"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"at order {order}, {amp:g} sigma the trend filter recovers "
                 f"{new:.4f} against the frozen detector's {old:.4f} "
                 f"({new - old:+.4f}). An estimator built directly on the "
                 f"axiom is no better at the axiom's own break, which points "
                 f"at the data rather than the criterion"),
                n_effective=n_effective, detail=detail)
