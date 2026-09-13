r"""The closest thing this corpus has to ground truth, and why it is not.

A cleaning arm can only be judged against something. Every axis used so far is a
proxy: violation rate asks whether a length is plausible, distortion asks how far
the arm moved things, retention asks how much fast content survived. **None of
them knows where the keypoint actually was**, so none can tell a repair from a
confident error, and the one thing they all reward is smoothness.

This module builds the missing reference. Frames that pass *every* instrument in
this repo at once -- no bone violation, no continuity spike, no missing keypoint,
and DLC confidence above a floor -- are treated as correct, corruption is injected
into them with known magnitude and direction, and the arm is then asked to put
them back.

## The selection bias, stated first because it is the main threat

A pool selected for cleanliness is biased towards **slow and still** behaviour: a
continuity spike is exactly what a fast keypoint produces, so fast frames are
preferentially excluded. The consequence is specific and it runs against the
conclusion this phase is likely to reach:

    damage measured on this pool UNDERSTATES what a smoother does to fast
    movement, which is the quantity in dispute.

So `segment_speed` exists and every result built on this module must be
**stratified by it**. A pooled scalar over this pool is the misleading statistic;
the damage-versus-speed curve is the informative one. Measured on 12 tune
recordings: 23.4% of frames are clean, 14.4% sit in runs of a second or more,
95 segments with a median of 69 frames.

It is a *pseudo*-ground truth and nothing here should be written as though a
human labelled it.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from recur.qc.swap import runs_of

from . import bones, continuity as ct

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

#: A frame joins the pool only if every keypoint's continuity residual is below
#: this. 0.02 body lengths is roughly the p95 of the corpus-wide spike
#: distribution (median 0.0046), so it admits ordinary motion and excludes
#: excursions. Pre-registered.
CLEAN_SPIKE_BL = 0.02

#: DLC confidence floor. Confidence predicts violations at AUC 0.60-0.67 here --
#: weak, but this is a filter for a reference pool rather than a detector, and
#: a weak signal is worth using when the cost of a wrong frame is a corrupted
#: ground truth.
MIN_CONF = 0.60

#: Shorter than this and a temporal arm has no context to work with, so the
#: comparison would measure edge effects. One second at 30 fps.
MIN_SEGMENT_FRAMES = 30

#: Speed strata. Five bins, because the damage-versus-speed curve is the point.
N_SPEED_BINS = 5


def clean_mask(pose: npt.ArrayLike, conf: npt.ArrayLike,
               missing: npt.ArrayLike, ell: float, *,
               pairs: Sequence[tuple[int, int]], eps: float = 0.10,
               spike_bl: float = CLEAN_SPIKE_BL,
               min_conf: float = MIN_CONF) -> tuple[BOOL, Detail]:
    """``(mask, why)`` -- frames that every instrument in this repo calls clean.

    Takes ONE recording. Each criterion is recorded separately in `why` so a pool
    that comes out too small or too biased can be diagnosed rather than guessed
    at.
    """
    p = np.asarray(pose, dtype=np.float64)
    c = np.asarray(conf, dtype=np.float64)
    miss = np.asarray(missing, dtype=bool)

    keep = bones.rigid_pairs(bones.log_lengths(p, pairs), pairs)
    lengths = bones.metric_lengths(p, bones.SKULL, "raw", pairs=pairs, keep=keep)
    l_hat = np.array([bones.reference_length(lengths[:, m], eps)["l_hat"]
                      for m in range(len(bones.SKULL))])
    bone_ok = ~bones.frame_mask(bones.violations(lengths, l_hat, eps))

    r_minus, r_plus = ct.residuals(p, ell)
    spike, _ = ct.decompose(r_minus, r_plus)
    with np.errstate(invalid="ignore"):
        worst = np.where(np.isfinite(spike).any(axis=1),
                         np.nanmax(np.where(np.isfinite(spike), spike, -np.inf),
                                   axis=1), np.nan)
    cont_ok = np.isfinite(worst) & (worst <= float(spike_bl))

    present = ~miss.any(axis=1)
    conf_ok = c.min(axis=1) >= float(min_conf)
    mask = bone_ok & cont_ok & present & conf_ok
    return mask, {
        "frac_bone_ok": float(bone_ok.mean()),
        "frac_continuity_ok": float(cont_ok.mean()),
        "frac_present": float(present.mean()),
        "frac_confident": float(conf_ok.mean()),
        "frac_all": float(mask.mean()),
        "spike_bl": float(spike_bl), "min_conf": float(min_conf), "eps": float(eps),
    }


def segments(mask: npt.ArrayLike, *,
             min_frames: int = MIN_SEGMENT_FRAMES) -> I64:
    """``(n, 2)`` half-open contiguous runs of `mask` at least `min_frames` long.

    Seam-safe by contract: `runs_of` is given one recording's mask, so no segment
    can span two recordings.
    """
    runs = runs_of(np.asarray(mask, dtype=bool))
    if runs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    long = runs[(runs[:, 1] - runs[:, 0]) >= int(min_frames)]
    return np.asarray(long, dtype=np.int64)


def segment_speed(pose: npt.ArrayLike, seg: Sequence[int], ell: float,
                  fps: float, *, keypoint: int = 3) -> float:
    """Median speed of one segment, body lengths per second.

    The centre by default: it is the landmark every instrument in this repo
    trusts most -- Viterbi reassigns it 0.0224% of the time against the nose's
    0.5315%, Wiener smooths it least, and it has the lowest continuity residual
    of the seven. Its speed is the animal's locomotion rather than a landmark's
    wobble.
    """
    p = np.asarray(pose, dtype=np.float64)[int(seg[0]):int(seg[1]), int(keypoint)]
    if p.shape[0] < 2 or not np.isfinite(ell) or ell <= 0:
        return float("nan")
    step = np.linalg.norm(np.diff(p, axis=0), axis=-1) * float(fps) / float(ell)
    step = step[np.isfinite(step)]
    return float(np.median(step)) if step.size else float("nan")


def speed_bins(speeds: npt.ArrayLike, *, n_bins: int = N_SPEED_BINS) -> F64:
    """Quantile edges over segment speeds, at most ``n_bins + 1`` of them.

    Quantiles rather than equal width, because segment speed is heavily skewed
    and equal-width bins would put almost everything in the first one. Duplicate
    edges are dropped, so the caller gets the number of strata the data actually
    supports rather than the number requested.
    """
    v = np.asarray(speeds, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.full(int(n_bins) + 1, np.nan)
    qs = np.linspace(0.0, 100.0, int(n_bins) + 1)
    edges = np.percentile(v, qs)
    # Deduplicate. A speed distribution flat enough to tie its quantiles has
    # fewer real strata than asked for, and tied edges make `assign_bin`
    # ambiguous -- every tied value lands in the highest tied bin, which silently
    # files the slowest segments as the fastest. Returning fewer edges says so.
    return np.asarray(np.unique(edges), dtype=np.float64)


def assign_bin(speed: float, edges: npt.ArrayLike) -> int:
    """Which speed stratum a segment falls in, or -1 if it cannot be placed."""
    e = np.asarray(edges, dtype=np.float64)
    if not np.isfinite(speed) or e.size < 2 or not np.isfinite(e).all():
        return -1
    idx = int(np.searchsorted(e[1:-1], float(speed), side="right"))
    return max(0, min(idx, e.size - 2))
