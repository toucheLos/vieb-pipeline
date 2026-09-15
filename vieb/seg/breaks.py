r"""Where the trajectory stops being smooth. Boundaries, not states.

The tokenizer is closed by measurement: median run 1 frame in all eight cells,
every homogeneity interval excluding zero, and the falsifier firing. Two
architectures have now produced the same signature, and the diagnosis is that
this corpus has no reliable frame-level atoms. **Imposing a vocabulary before
finding boundaries is where every failure on this project has lived.**

This module inverts that. It never asks what a frame is. It asks where the
signal breaks.

## The criterion, and whose it is

Ported from `~/exbias/exbias.py:69-150`, whose axiom is stated there and is worth
restating because it is an assumption and not a fact:

    Behaviour is piecewise-smooth. An ACTION is a maximal interval on which the
    pose trajectory is smooth. A TRANSITION is a discontinuity in acceleration.

Its physical warrant: position cannot jump, velocity cannot jump, but
acceleration can — a muscle activation switches in milliseconds. So a motor
program change is a C² discontinuity. Its known failure modes are equally
explicit: gradual transitions produce no kink, continuous tremor produces no
boundaries at all, and context-defined behaviours are invisible to kinematics.

**Ported, not imported.** `vieb/io/spine.py` hashes exbias as *data*
(`CONSUMED_DIRS["exbias_segments"]`); importing its code would break this repo's
own "hash the artifacts, import the utilities" rule and pull in a second venv.
And its segments are unusable here for a separate reason — `extract.slurm` feeds
it raw DeepLabCut `.h5`, upstream of the gap policy, the swap correction and
every QC mask, in alphabetical bodypart order against shapeflow's file order.

## Three things that are load-bearing and look cosmetic

**The windows are strictly one-sided.** `left[t]` uses `x[t-h+1 : t+1]`,
`right[t]` uses `x[t : t+h]`. A centred convolution lets both windows straddle
`t`, so they share the very frames whose disagreement is being measured and `D`
collapses to ~0. That is a recorded past bug, not a hypothetical.

**Non-maximum suppression runs strongest-peak-first.** The obvious
left-to-right rule is order-dependent: an earlier, weaker peak permanently
suppresses a later, stronger one inside the refractory window, and the boundary
lands on noise instead of on the kink.

**The guard band is what makes segmentation testable.** A boundary frame is the
frame of maximal acceleration mismatch — the kink itself — and half-open bounds
put it inside the segment, so every segment would be a smooth piece with a
discontinuity glued to its front. Measured on real data: with no guard, detected
segments score R² 0.583 against 0.621 for length-matched random intervals, a
**50.6% win rate — the detector buys nothing**. With one to three frames excluded
at each end it is 0.62 against 0.59, 65–69%.

## What is NOT here, deliberately

No minimum-duration rule, no stickiness prior, no smoothing of the boundary
series. That imports persistence, which is what retracted the dwell result —
κ = 10⁶ manufactures heavy-tailed dwell on white noise. The only length floor is
`identifiability_floor`, which is a property of the **estimator**: a degree-`d`
polynomial has `d+1` free parameters per channel, and fitting them to fewer
samples is interpolation whose R² is ~1 by construction.

Non-maximum suppression is the detector's **resolution limit** rather than a
duration prior, and `D` is written out unsmoothed beside the boundaries so the
distinction is auditable rather than asserted.

**No threshold is set here.** This module emits the continuous statistic and a
peak-picker that takes a threshold. Step 2 calibrates that threshold against
surrogates so the boundary rate carries a stated false-positive rate; ExBias's
`k_mad = 3.0` is adaptive to the recording but says nothing about how often it
fires on a signal with no boundaries in it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from recur.read import Read

from vieb.tok import ego

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["CHANNEL_GROUPS", "DEGREE", "DERIV_SEC", "K_MAD",
           "MIN_SAMPLES_PER_PARAM", "boundaries", "breaks_read",
           "discontinuity", "fit_quality", "guard_frames",
           "identifiability_floor", "mad_threshold", "min_segment_frames",
           "second_deriv_weights", "segment_table"]

#: Samples per free parameter before a polynomial fit means anything. Under a
#: white-noise null `E[R²] = (p-1)/(n-1)`, so at `n = 2p` the null already sits
#: near 0.43; three puts it near 0.27 and keeps the adjusted statistic stable.
MIN_SAMPLES_PER_PARAM = 3
#: Polynomial degree, and the derivative half-window in SECONDS. Both are
#: ExBias's run values, and both are swept rather than trusted.
DEGREE = 3
DERIV_SEC = 0.133
#: ExBias's threshold, carried as a REFERENCE POINT only. Step 2 calibrates.
K_MAD = 3.0

#: Which channels a break is looked for in. They may disagree, and where they
#: disagree is reported rather than resolved by fiat.
CHANNEL_GROUPS: dict[str, tuple[int, ...]] = {
    "shape": tuple(range(ego.N_POSE)),
    "twist": tuple(range(ego.N_POSE, ego.N_DIMS)),
    "both": tuple(range(ego.N_DIMS)),
}


def second_deriv_weights(h: int, side: str) -> F64:
    """Weights `w` with `w · y` the second-derivative coefficient of the
    least-squares quadratic through `h` equally spaced points.

    Fitting `y = c0 + c1 s + c2 s²`, the second derivative is `2 c2` and
    `c2 = pinv(V)[2] · y`. Precomputing `w` turns the detector into two
    convolutions.
    """
    s = np.arange(int(h), dtype=np.float64)
    if side == "left":
        s = -s[::-1]                      # the window ENDS at t
    elif side != "right":
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    v = np.stack([np.ones(int(h)), s, s ** 2], axis=1)
    return np.asarray(2.0 * np.linalg.pinv(v)[2], dtype=np.float64)


def discontinuity(x: npt.ArrayLike, h: int) -> F64:
    """`D(t) = ‖ẍ⁺(t) − ẍ⁻(t)‖` — the acceleration mismatch at every frame.

    `x` must already be standardised; the channels carry different units and an
    unstandardised norm would be a statement about which channel has the largest
    variance. `D` is zeroed in the first and last `h` frames, where one of the
    two windows would be padded rather than observed.
    """
    a = np.asarray(x, dtype=np.float64)
    hh = int(h)
    if a.ndim != 2:
        raise ValueError(f"x must be (T, C), got {a.shape}")
    t = a.shape[0]
    if t < 2 * hh + 1:
        return np.zeros(t, dtype=np.float64)
    wl = second_deriv_weights(hh, "left")
    wr = second_deriv_weights(hh, "right")
    left = sliding_window_view(
        np.pad(a, ((hh - 1, 0), (0, 0)), mode="edge"), hh, axis=0) @ wl
    right = sliding_window_view(
        np.pad(a, ((0, hh - 1), (0, 0)), mode="edge"), hh, axis=0) @ wr
    d = np.sqrt(((right - left) ** 2).sum(axis=1))
    d[:hh] = 0.0
    d[-hh:] = 0.0
    return np.asarray(d, dtype=np.float64)


def identifiability_floor(degree: int = DEGREE,
                          k: int = MIN_SAMPLES_PER_PARAM) -> int:
    """Minimum frames for a degree-`d` fit to carry information.

    A property of the estimator, **not** a prior about behaviour: `d+1` free
    parameters fitted to `n ≤ d+1` samples is interpolation, and its R² is ~1 by
    construction.
    """
    return int(k) * (int(degree) + 1)


def guard_frames(fps: float, *, guard_sec: float | None = None,
                 deriv_sec: float = DERIV_SEC) -> int:
    """Frames excluded at each end of a segment before fitting.

    Defaults to half the derivative window: `h` is the detector's localisation
    uncertainty, so `h/2` is the natural exclusion radius. In seconds, so it
    means the same thing at 30 fps and at 250.
    """
    g = (deriv_sec / 2.0) if guard_sec is None else guard_sec
    return max(1, int(round(float(g) * float(fps))))


def min_segment_frames(degree: int = DEGREE, guard: int = 0) -> int:
    """`identifiability_floor(degree) + 2·guard`, the shortest fittable segment."""
    return identifiability_floor(degree) + 2 * int(guard)


def mad_threshold(d: npt.ArrayLike, k_mad: float = K_MAD) -> float:
    """`median + k · 1.4826 · MAD`. Robust, and the bulk of `D` is the null.

    Within an action `D ≈ 0` by the axiom, so the body of the distribution is
    what "no boundary here" looks like. This is a reference threshold; it
    carries **no stated false-positive rate**, which is what Step 2 supplies.
    """
    a = np.asarray(d, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) + 1e-12
    return med + float(k_mad) * 1.4826 * mad


def boundaries(d: npt.ArrayLike, threshold: float, *, min_gap: int,
               blocked: npt.ArrayLike | None = None) -> I64:
    """Peaks of `D` above `threshold`, refractory by non-maximum suppression.

    **Strongest first.** A left-to-right rule is order-dependent: an earlier,
    weaker peak permanently suppresses a later, stronger one inside the
    refractory window, and the boundary lands on noise rather than on the kink.

    `blocked` marks frames where a boundary may not be declared — abstained
    frames, whose pose is partly an interpolant and whose acceleration is
    therefore partly the gap policy's.
    """
    a = np.asarray(d, dtype=np.float64)
    if a.shape[0] < 3:
        return np.zeros(0, dtype=np.int64)
    body = a[1:-1]
    is_peak = (body > float(threshold)) & (body >= a[:-2]) & (body > a[2:])
    cand = np.flatnonzero(is_peak) + 1
    if blocked is not None:
        cand = cand[~np.asarray(blocked, dtype=bool)[cand]]
    if cand.size == 0:
        return np.zeros(0, dtype=np.int64)
    gap = max(2, int(min_gap))
    taken = np.zeros(a.shape[0], dtype=bool)
    out: list[int] = []
    for t in cand[np.argsort(-a[cand], kind="stable")]:
        if not taken[t]:
            out.append(int(t))
            taken[max(0, int(t) - gap + 1):int(t) + gap] = True
    return np.asarray(sorted(out), dtype=np.int64)


def fit_quality(block: npt.ArrayLike, degree: int = DEGREE) -> tuple[float, float]:
    """`(R², adjusted R²)` of a degree-`d` polynomial fit, per segment.

    **On the original samples.** Resampling `n` frames to a fixed length before
    fitting manufactures smoothness — the fit then explains an interpolant. And
    the reportable statistic is the **adjusted** R², whose expectation is 0 under
    a white-noise null at *every* length; thresholding raw R² makes the check a
    length filter that keeps under-determined short segments and discards the
    long ones where the axiom is best supported.
    """
    y = np.asarray(block, dtype=np.float64)
    n, p = y.shape[0], int(degree) + 1
    if n <= p:
        raise ValueError(f"a degree-{degree} fit needs more than {p} samples, "
                         f"got {n}: that is interpolation, not a fit")
    s = np.linspace(-1.0, 1.0, n)
    v = np.vander(s, p, increasing=True)
    coef, *_ = np.linalg.lstsq(v, y, rcond=None)
    resid = y - v @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if np.isfinite(r2) else float("nan")
    return float(r2), float(adj)


def segment_table(x: npt.ArrayLike, peaks: npt.ArrayLike, *, lo: int, hi: int,
                  abstain: npt.ArrayLike, guard: int, degree: int = DEGREE,
                  fps: float = 30.0) -> list[Detail]:
    """Segments between seams, abstain blocks and detected boundaries.

    **Abstain splits and is never dropped.** Dropping an abstained stretch would
    fuse the segments either side of it into one and manufacture a duration the
    animal never held — the same error `motif.sequences(..., gap=ABSTAIN)` exists
    to prevent in the symbol stream.

    A segment shorter than `min_segment_frames` after the guard band is recorded
    with `fittable: false` rather than discarded: how much of the stream is
    unfittable is part of the result.
    """
    a = np.asarray(x, dtype=np.float64)
    ab = np.asarray(abstain, dtype=bool)
    cuts = {int(lo), int(hi)}
    cuts.update(int(p) for p in np.asarray(peaks, dtype=np.int64))
    # Every abstain run is its own boundary pair.
    run = np.diff(np.concatenate([[0], ab[lo:hi].astype(np.int8), [0]]))
    for ab_s, ab_e in zip(np.flatnonzero(run == 1), np.flatnonzero(run == -1)):
        cuts.add(int(lo) + int(ab_s))
        cuts.add(int(lo) + int(ab_e))
    edges = sorted(cuts)

    out: list[Detail] = []
    floor = min_segment_frames(degree, guard)
    for _a, _b in zip(edges[:-1], edges[1:]):
        s_, e_ = int(_a), int(_b)
        n = e_ - s_
        if n <= 0:
            continue
        frac_ab = float(ab[s_:e_].mean())
        fs, fe = s_ + int(guard), e_ - int(guard)
        row: Detail = {
            "start": s_, "stop": e_, "n_frames": n,
            "duration_s": n / float(fps),
            "fit_start": fs, "fit_stop": fe,
            "abstain_frac": frac_ab,
            "fittable": bool(n >= floor and frac_ab < 1.0),
            "fit_r2": float("nan"), "fit_r2_adj": float("nan"),
        }
        if row["fittable"] and fe - fs > int(degree) + 1:
            r2, adj = fit_quality(a[fs:fe], degree)
            row["fit_r2"], row["fit_r2_adj"] = r2, adj
        else:
            row["fittable"] = False
        out.append(row)
    return out


def breaks_read(summary: Mapping[str, Any], *, scored_object: Detail,
                n_effective: int) -> Read:
    """Reports the boundary statistic. **Never a verdict on the boundaries.**

    There is deliberately no threshold here and no PASS condition on the
    boundary rate. Whether these boundaries are real is Step 2's question, and it
    needs the surrogate families to answer it — a detector fires on noise, and a
    rate without a null beside it is not evidence of anything. What this read
    can fail on is the detector being degenerate: no peaks anywhere, or a
    statistic that is identically zero.
    """
    rate = float(summary.get("boundary_rate_per_s", float("nan")))
    n_seg = int(summary.get("n_segments", 0))
    detail: Detail = dict(summary)
    if not np.isfinite(rate) or n_seg < 2:
        return Read("NOT_A_RESULT", scored_object,
                    f"the detector produced {n_seg} segments, so there is no "
                    f"boundary statistic to report",
                    n_effective=n_effective, degenerate=True, detail=detail)
    return Read("GRID_LIMITED", scored_object,
                f"{n_seg:,} segments at {rate:.3f} boundaries/s, median duration "
                f"{float(summary.get('median_duration_s', float('nan'))):.3f} s, "
                f"{float(summary.get('frac_fittable', float('nan'))):.1%} "
                f"fittable, mean adjusted R^2 "
                f"{float(summary.get('mean_fit_r2_adj', float('nan'))):.4f}. "
                f"This is a statistic at one threshold and NOT a finding: "
                f"whether these boundaries are real needs the surrogate "
                f"families, and a detector fires on noise",
                n_effective=n_effective, detail=detail)
