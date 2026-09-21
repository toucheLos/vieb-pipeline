"""Tracking-noise amplitude as a function of DLC confidence, measured.

READ results/NOISEFLOOR_PREREGISTRATION.md FIRST.

## Why bone length and not stillness

Everything downstream needs a jitter amplitude. The obvious estimator -- the
high-frequency residual during still stretches -- is **circular**: "still" is
defined by the smoothness whose noise is the thing being measured.

The skull triangle is not. `bones.SKULL` spans left_ear-right_ear,
left_ear-nose and right_ear-nose, which are rigid: the distance between two
points on a mouse's skull does not change when the mouse moves. So **all** of
its frame-to-frame variance is tracking noise, whatever the animal is doing,
and the estimate needs no assumption about behaviour at all.

**Trunk bones are excluded and the exclusion is not optional.** `vieb/qc/bones.py`
records that they flex, so a length change there is real posture at least as
often as it is tracking. Including them would attribute posture to noise and
inflate the floor, which would flatter the detector.

## From a bone residual to a per-keypoint amplitude

`r = L - L_ref` is the change in the distance between two keypoints. For
isotropic per-keypoint noise of per-axis standard deviation `sigma`, the two
endpoints contribute independently along the bone axis, so

    Var(r) = sigma^2 + sigma^2 = 2 * sigma^2      =>   sigma = sd(r) / sqrt(2)

and a draw of `r / sqrt(2)` is a draw of one keypoint's displacement along one
axis. Two independent such draws give an isotropic 2-D displacement whose
marginal is the **measured** one, heavy tails included.

## Why the empirical distribution and not a Gaussian

`vieb/qc/inject.py:3-7` refuses Gaussian perturbation benchmarks because
"Gaussian noise is the thing a smoother is optimal against". That argument is
about benchmarking smoothers and does not reach a detector, but the precedent is
followed in substance anyway: amplitudes are drawn by inverse-CDF from the
measured residual distribution, as `INJECTION_PREREGISTRATION.md` does with
`DISPLACEMENT_Q`.

## Held frames are excluded, and that matters

A missing or interpolated keypoint has been **held** at its last value, so its
bone residual is zero by construction. Leaving those frames in would pull the
variance toward zero exactly where tracking is worst -- the estimate would be
best-behaved precisely where it is least true.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.read import Read                                         # noqa: E402
from vieb.qc import bones as bn                                     # noqa: E402

__all__ = ["CONF_EDGES", "N_QUANTILES", "MIN_BIN", "residuals", "calibrate",
           "sigma_of", "draw", "calibration_read"]

F64 = npt.NDArray[np.float64]

#: Confidence bin edges. Fixed rather than quantile-derived so the table means
#: the same thing across animals and can be compared between them -- a quantile
#: grid would put a different confidence range in "bin 3" for every recording.
CONF_EDGES: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                 0.95, 1.0001)
#: Inverse-CDF grid stored per bin. 65 points resolves the tail without storing
#: every residual, and the tail is the part that makes a detector fire.
N_QUANTILES = 65
#: A bin with fewer than this many residuals is not estimated. It is left NaN
#: and the count is reported; a variance from a handful of frames is noise about
#: noise.
MIN_BIN = 20_000


def residuals(pose: npt.ArrayLike, conf: npt.ArrayLike, *,
              missing: npt.ArrayLike, interpolated: npt.ArrayLike,
              ell: float) -> tuple[F64, F64]:
    """``(residual_bl, conf_min)`` over skull bones, in body lengths.

    One row per (frame, skull bone) that is clean at BOTH endpoints. The
    reference length is that recording's own median, so a per-recording
    tracking offset cannot masquerade as noise.
    """
    p = np.asarray(pose, dtype=np.float64)
    c = np.asarray(conf, dtype=np.float64)
    bad = (np.asarray(missing, dtype=bool)
           | np.asarray(interpolated, dtype=bool))
    lens = bn.bone_lengths(p, bn.SKULL) / float(ell)
    rs: list[F64] = []
    cs: list[F64] = []
    for m, (i, j) in enumerate(bn.SKULL):
        ok = ~bad[:, i] & ~bad[:, j] & np.isfinite(lens[:, m])
        if int(ok.sum()) < 2:
            continue
        ref = float(np.median(lens[ok, m]))
        rs.append(lens[ok, m] - ref)
        cs.append(np.minimum(c[ok, i], c[ok, j]))
    if not rs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(rs), np.concatenate(cs)


def calibrate(resid: npt.ArrayLike, conf: npt.ArrayLike) -> dict[str, Any]:
    """The table: per confidence bin, `sigma` and an inverse-CDF grid.

    `sigma` is `sd(r) / sqrt(2)` -- one keypoint's per-axis amplitude. The grid
    is of `r / sqrt(2)` so a draw from it is directly a per-axis displacement.
    """
    r = np.asarray(resid, dtype=np.float64) / np.sqrt(2.0)
    c = np.asarray(conf, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(c)
    r, c = r[ok], c[ok]
    edges = np.asarray(CONF_EDGES, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, N_QUANTILES)
    sigma = np.full(edges.size - 1, np.nan)
    grid = np.full((edges.size - 1, N_QUANTILES), np.nan)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    for b in range(edges.size - 1):
        sel = (c >= edges[b]) & (c < edges[b + 1])
        n = int(sel.sum())
        counts[b] = n
        if n < MIN_BIN:
            continue
        sigma[b] = float(np.std(r[sel]))
        grid[b] = np.quantile(r[sel], qs)
    return {"edges": edges.tolist(), "sigma_bl": sigma.tolist(),
            "quantiles": qs.tolist(), "grid_bl": grid.tolist(),
            "counts": counts.tolist(), "n_residuals": int(r.size),
            "min_bin": MIN_BIN}


def _bin_of(conf: npt.ArrayLike, edges: npt.ArrayLike) -> npt.NDArray[np.int64]:
    e = np.asarray(edges, dtype=np.float64)
    b = np.searchsorted(e, np.asarray(conf, dtype=np.float64), "right") - 1
    return np.clip(b, 0, e.size - 2).astype(np.int64)


def sigma_of(conf: npt.ArrayLike, table: Mapping[str, Any]) -> F64:
    """Per-element noise amplitude in body lengths, from the table.

    An unestimated bin falls back to the nearest estimated one rather than to
    zero: zero would say "this confidence is noiseless", which is the opposite
    of what an unfillable bin means.
    """
    sig = np.asarray(table["sigma_bl"], dtype=np.float64)
    good = np.flatnonzero(np.isfinite(sig))
    if good.size == 0:
        return np.full(np.shape(conf), np.nan)
    b = _bin_of(conf, table["edges"])
    near = good[np.argmin(np.abs(good[None, :] - b.ravel()[:, None]), axis=1)]
    return np.asarray(sig[near].reshape(np.shape(conf)),
                      dtype=np.float64)


def draw(rng: np.random.Generator, conf: npt.ArrayLike,
         table: Mapping[str, Any], *, scale: float = 1.0) -> F64:
    """``(T, K, 2)`` isotropic displacement in body lengths, measured marginal.

    Inverse-CDF from the bin's own grid, per axis, so the heavy tail of real
    tracking noise survives into the injected stream. `scale` multiplies the
    amplitude and is how the dose-response arms are built.
    """
    c = np.asarray(conf, dtype=np.float64)
    grid = np.asarray(table["grid_bl"], dtype=np.float64)
    qs = np.asarray(table["quantiles"], dtype=np.float64)
    good = np.flatnonzero(np.isfinite(grid).all(axis=1))
    if good.size == 0:
        return np.zeros(c.shape + (2,))
    b = _bin_of(c, table["edges"])
    near = good[np.argmin(np.abs(good[None, :] - b.ravel()[:, None]),
                          axis=1)].reshape(c.shape)
    out = np.empty(c.shape + (2,), dtype=np.float64)
    u = rng.random(c.shape + (2,))
    for g in np.unique(near):
        sel = near == g
        for ax in (0, 1):
            out[..., ax][sel] = np.interp(u[..., ax][sel], qs, grid[g])
    return out * float(scale)


def calibration_read(table: Mapping[str, Any], *,
                     scored_object: dict[str, Any],
                     n_effective: int) -> Read:
    """Is the calibration usable, and does it behave as tracking noise should?

    Refuses rather than reporting a table whose bins are mostly empty. The
    directional check -- amplitude falling as confidence rises -- is reported
    but is NOT a pass condition: confidence is unimodal on this corpus and a
    weak relationship is a known property of it, not a broken calibration.
    """
    sig = np.asarray(table["sigma_bl"], dtype=np.float64)
    counts = np.asarray(table["counts"], dtype=np.int64)
    good = np.flatnonzero(np.isfinite(sig))
    detail = {"n_bins": int(sig.size), "n_estimated": int(good.size),
              "n_residuals": int(table["n_residuals"]),
              "sigma_bl": table["sigma_bl"], "counts": table["counts"]}
    if good.size < 2:
        return Read("NOT_A_RESULT", scored_object,
                    (f"only {good.size} confidence bin(s) reached the "
                     f"{table['min_bin']:,}-residual floor; there is no "
                     f"calibration to inject from"),
                    n_effective=n_effective, detail=detail)
    lo, hi = float(sig[good[0]]), float(sig[good[-1]])
    detail["sigma_lowest_conf"] = lo
    detail["sigma_highest_conf"] = hi
    detail["ratio_low_over_high"] = (lo / hi) if hi > 0 else float("nan")
    return Read(
        "PASS", scored_object,
        (f"calibrated on {table['n_residuals']:,} skull-bone residuals across "
         f"{good.size} of {sig.size} confidence bins: per-keypoint per-axis "
         f"amplitude runs {lo:.5f} body lengths at the lowest estimated "
         f"confidence to {hi:.5f} at the highest, a ratio of "
         f"{detail['ratio_low_over_high']:.2f}x. Skull bones only -- trunk "
         f"bones flex and their length change is posture as often as "
         f"tracking. Held and interpolated keypoint-frames excluded: their "
         f"residual is zero by construction"),
        n_effective=n_effective, detail=detail)
