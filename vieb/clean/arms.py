r"""Cleaning arms, behind one signature, so the scoring harness cannot tell them apart.

Every arm is ``clean(pose, conf, fps, **params) -> pose`` and every one of them
receives the **identical** input: shapeflow's hold-for-filtering array, which is
what its Wiener filter actually consumed. Comparing a new filter against a
different input than the incumbent got would be a comparison between two
pipelines rather than between two filters.

## Why these are implemented rather than imported

`movement` (`filter_by_confidence`, `interpolate_over_time`, `rolling_filter`,
`savgol_filter`) is the right reference and the semantics below are matched to
it deliberately -- reflect padding, centred window, `min_periods` defaulting to
the full window so a NaN anywhere in the window propagates. It is not installed
because it resolves to **~70 transitive packages** (napari-video, Cartopy, numba,
zarr, pynwb, skia-python, ...) and this repo's venv is **shared with recur**, so
installing it would put packages in recur's environment that recur's pins do not
name. For two filters that are `scipy.signal.savgol_filter` and a rolling median
underneath, that trade is not worth making. If the arms ever need more of
`movement` than this, build a separate venv and install it properly.

Sources, and what each contributes:

* `movement` -- https://movement.neuroinformatics.dev/ -- `savgol`, `median`
* Anipose -- https://github.com/lambdaloop/anipose -- `viterbi`, loaded from the
  installed package's own source; see `vieb/clean/viterbi.py` for why the import
  chain is bypassed
* refineDLC -- https://github.com/wer-kle/refineDLC -- `position_outlier`, whose
  rule is a threshold on consecutive-frame displacement by absolute value, a
  percentile, MAD or IQR, followed by interpolation
* DeepLabCut's own `post_processing/filtering.py` -- the median arm again, at a
  different default window

## One arm cannot run here, and that is a result

**Confidence filtering is inert.** shapeflow's threshold estimator *refuses* on
this corpus for all seven keypoints: there is no low-confidence mode to separate,
and its own gate masks 0.000% of keypoint-frames. An arm that thresholds
likelihood would be a no-op, and reporting it as one arm among several would
imply it had been given a chance.

## CORRECTED: the Viterbi arm CAN run here

An earlier version of this docstring said Anipose's Viterbi filter was
unavailable, because all 3,080 `_full.pickle` files carry exactly one detection
per bodypart-frame. The detection count is right and **the conclusion was wrong**
-- it came from the paper's phrase "a set of top detections per frame" rather
than from the implementation, which builds its candidate set from the previous
`n_back` frames as well as the current one. See `vieb/clean/viterbi.py`.

Measured on one recording: it reassigns **0.31%** of keypoint-frames, and when it
does move one it moves it by a median 46.65 px. The Wiener filter moves **86%**
of keypoint-frames by a median 1.42 px. Those are opposite signatures -- a
de-glitcher against a shrinkage -- and having both in the bakeoff is the point.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from recur.util import frames, odd_frames

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: Outlier rules for the refineDLC arm.
RULES: tuple[str, ...] = ("mad", "iqr", "percentile", "absolute")


def held_array(pose: npt.ArrayLike, missing: npt.ArrayLike) -> F64:
    r"""shapeflow's hold-for-filtering array, reproduced.

    Missing frames must not drive a filter, so shapeflow interpolates each
    keypoint across every non-missing index, filters *that*, and **never stores
    it** -- the flags are kept instead, so nothing downstream mistakes a held
    value for a measurement.

    Every arm here gets this same array, because it is what the incumbent got.
    Ported rather than imported: this repo reads shapeflow's artifacts and never
    its code, and `held` is not among the artifacts it wrote.
    """
    p = np.asarray(pose, dtype=np.float64).copy()
    miss = np.asarray(missing, dtype=bool)
    t = p.shape[0]
    idx = np.arange(t)
    for k in range(p.shape[1]):
        ok = ~miss[:, k]
        if ok.all() or not ok.any():
            continue
        for c in range(2):
            p[:, k, c] = np.interp(idx, idx[ok], p[ok, k, c])
    return p


def _window(window_s: float, fps: float, *, odd: bool = True) -> int:
    """Seconds to frames, converted exactly once, never below one.

    Windows are specified in seconds throughout. A config that held a frame
    count would silently mean a quarter of its intended duration on the 250 fps
    rat corpus while looking correct.
    """
    return odd_frames(window_s, fps) if odd else frames(window_s, fps)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def _viterbi_arm(pose: npt.ArrayLike, conf: npt.ArrayLike | None,
                 fps: float, **kw: Any) -> F64:
    """Anipose's Viterbi path filter. Imported lazily: it reads anipose's source
    at module load, and an arm nobody selected should not make that a hard
    dependency of importing this module."""
    from .viterbi import viterbi
    return viterbi(pose, conf, fps, **kw)


def identity(pose: npt.ArrayLike, conf: npt.ArrayLike | None,
             fps: float, **_: Any) -> F64:
    """No cleaning. The floor every other arm has to beat."""
    return np.asarray(pose, dtype=np.float64).copy()


def savgol(pose: npt.ArrayLike, conf: npt.ArrayLike | None, fps: float, *,
           window_s: float = 0.25, polyorder: int = 2, **_: Any) -> F64:
    """Savitzky-Golay along time, per keypoint per coordinate.

    Matches `movement.filtering.savgol_filter`: `scipy.signal.savgol_filter`
    over the time axis at `polyorder=2`. A polynomial fit preserves peak height
    where a moving average flattens it, which is the reason it is the usual
    choice for kinematics.
    """
    from scipy.signal import savgol_filter as _savgol

    p = np.asarray(pose, dtype=np.float64)
    w = _window(window_s, fps)
    if p.shape[0] <= w or w <= polyorder:
        return p.copy()
    return np.asarray(_savgol(p, w, polyorder, axis=0), dtype=np.float64)


def rolling_median(pose: npt.ArrayLike, conf: npt.ArrayLike | None, fps: float, *,
                   window_s: float = 0.25, **_: Any) -> F64:
    """Centred rolling median with reflect padding.

    Matches `movement.filtering.rolling_filter(statistic="median")`: pad by half
    the window in `reflect` mode, take a centred rolling median, trim the pad.
    Reflect rather than edge-hold because holding the first value flat makes the
    start of every recording look like freezing.
    """
    from scipy.ndimage import median_filter

    p = np.asarray(pose, dtype=np.float64)
    w = _window(window_s, fps)
    if p.shape[0] <= w:
        return p.copy()
    half = w // 2
    padded = np.pad(p, ((half, half), (0, 0), (0, 0)), mode="reflect")
    out = median_filter(padded, size=(w, 1, 1), mode="nearest")
    return np.asarray(out[half:half + p.shape[0]], dtype=np.float64)


def position_outlier(pose: npt.ArrayLike, conf: npt.ArrayLike | None,
                     fps: float, *, rule: str = "mad", k: float = 5.0,
                     max_gap_s: float = 0.5, **_: Any) -> F64:
    r"""refineDLC's position filter: gate implausible jumps, then interpolate.

    Frame-to-frame displacement per keypoint is thresholded, flagged frames are
    dropped, and the gaps are linearly interpolated. Unlike a smoother this
    leaves ordinary frames **exactly** as measured, which is the property that
    makes it interesting here: it is the only arm that can reduce the violation
    rate without moving anything else.

    `rule` picks how the threshold is set, all four from refineDLC:

    ``mad``         median + k * 1.4826 * MAD of the displacement
    ``iqr``         Q3 + k * IQR
    ``percentile``  the k-th percentile (k in 0-100)
    ``absolute``    k pixels flat

    A run longer than `max_gap_s` is left alone rather than interpolated across
    -- the same reasoning as shapeflow's gap policy, that filling a long gap
    fabricates smooth motion and smooth motion reads as behaviour.

    **An isolated spike costs its neighbour.** A one-frame excursion breaks two
    consecutive displacements -- the step out and the step back -- so the rule
    flags both frames and replaces one that was fine. That is a property of
    thresholding displacement rather than position, it is what refineDLC
    specifies, and it is kept: changing the rule to recover the neighbour would
    benchmark something other than the published method. The cost is one extra
    interpolated frame per isolated spike and it is asserted in the tests so
    nobody rediscovers it as a bug.
    """
    if rule not in RULES:
        raise ValueError(f"{rule!r} is not one of {RULES}")
    p = np.asarray(pose, dtype=np.float64).copy()
    t = p.shape[0]
    if t < 3:
        return p
    limit = frames(max_gap_s, fps)
    idx = np.arange(t)
    for kp in range(p.shape[1]):
        step = np.linalg.norm(np.diff(p[:, kp], axis=0), axis=-1)
        finite = step[np.isfinite(step)]
        if finite.size < 10:
            continue
        if rule == "mad":
            med = float(np.median(finite))
            mad = float(np.median(np.abs(finite - med)) * 1.4826)
            thr = med + k * mad
        elif rule == "iqr":
            q1, q3 = np.percentile(finite, [25, 75])
            thr = float(q3 + k * (q3 - q1))
        elif rule == "percentile":
            thr = float(np.percentile(finite, k))
        else:
            thr = float(k)
        bad = np.zeros(t, dtype=bool)
        # A jump lands ON the later frame of the pair that produced it.
        bad[1:] = step > thr
        if not bad.any():
            continue
        # Refuse to interpolate across a long run, and never across an endpoint.
        edges = np.diff(np.concatenate([[0], bad.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a > limit or a == 0 or b == t:
                bad[a:b] = False
        if not bad.any():
            continue
        ok = ~bad
        for c in range(2):
            p[bad, kp, c] = np.interp(idx[bad], idx[ok], p[ok, kp, c])
    return p


#: Arm name -> (function, params). Window lengths in SECONDS.
#:
#: `wiener` and `butterworth` are not here: shapeflow already computed them and
#: they are read off disk, so recomputing would risk a second implementation
#: disagreeing with the arrays every existing result was built on.
ARMS: dict[str, tuple[Callable[..., F64], Detail]] = {
    "raw": (identity, {}),
    # No 0.10 s savgol arm: at 30 fps that is 3 frames, and a quadratic through
    # 3 points is exact, so `savgol(window=3, polyorder=2)` is the identity. It
    # would occupy a slot in the bakeoff while doing nothing.
    "savgol_0.17": (savgol, {"window_s": 0.17}),
    "savgol_0.33": (savgol, {"window_s": 0.33}),
    "savgol_0.50": (savgol, {"window_s": 0.50}),
    "median_0.10": (rolling_median, {"window_s": 0.10}),
    "median_0.25": (rolling_median, {"window_s": 0.25}),
    "median_0.50": (rolling_median, {"window_s": 0.50}),
    "outlier_mad5": (position_outlier, {"rule": "mad", "k": 5.0}),
    "outlier_mad3": (position_outlier, {"rule": "mad", "k": 3.0}),
    "outlier_iqr3": (position_outlier, {"rule": "iqr", "k": 3.0}),
    "outlier_p99": (position_outlier, {"rule": "percentile", "k": 99.0}),
    # Anipose's defaults. `thres_dist` is swept on `tune` if this arm places;
    # one setting on the corpus run, because it costs ~12.6 s per recording
    # against ~0.05 s for every other arm.
    "viterbi": (_viterbi_arm, {}),
}

#: Arms read from shapeflow's stored arrays rather than recomputed.
STORED_ARMS: dict[str, str] = {"wiener": "pose", "butterworth": "pose_butterworth"}


def apply(name: str, pose: npt.ArrayLike, conf: npt.ArrayLike | None,
          fps: float) -> F64:
    """Run one arm by name."""
    fn, params = ARMS[name]
    return fn(pose, conf, fps, **params)
