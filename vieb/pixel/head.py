"""Head-region motion energy, and the spectral peak the grooming gate tests.

`results/GROOMING_PREREGISTRATION.md` §1, §4.

## The signal

Per frame, the **mean |delta|** inside a disc on the skull triangle
(`LEFT_EAR`, `RIGHT_EAR`, `NOSE`) of radius `0.6 x` the recording's median body
length, after the same `GaussianBlur(SIGMA=1)` `motion.scan` uses, **minus the
same frame's arena mean |delta|**. Mean rather than a thresholded count: the
statistic downstream is spectral, and a count above a cutoff is a coarse
quantisation of exactly the thing being transformed.

The radius is 0.6 rather than 0.5 body lengths because the forepaws during a
grooming stroke travel **in front of** the nose; a disc that stops at the nose
excludes the motion under test.

## The statistic, and why it is not an amplitude

Candidates are selected on **high** head-region motion, so any amplitude
statistic would be testing the selection rule. `peak_excess` is
**log-power in the band minus a background fitted to `log P` against `log f`
over the analysis range EXCLUDING the band**. Multiplying a spectrum by any
constant shifts the observed band power and the fitted background by the same
amount, so `peak_excess` is **invariant to the rescaling amplitude selection
performs**. What it is not invariant to is the spectrum's SHAPE at those
frequencies, which is the thing under test.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from vieb.pixel import motion as mo
from vieb.seg import descriptors as ds
from vieb.tok import ego as tego

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

#: §1: the disc is centred on the skull triangle -- the same three points
#: `bones.SKULL` uses -- and its radius is in body lengths, so it is scale-free
#: across boxes and camera distances.
SKULL = (tego.LEFT_EAR, tego.RIGHT_EAR, tego.NOSE)
#: Amendment 1: the control region. No grooming stroke happens at the hips, so a
#: stabilised difference there carries REGISTRATION NOISE and nothing else. A
#: 3-8 Hz excess that appears equally here is not behaviour.
HIPS = (tego.LEFT_HIP, tego.RIGHT_HIP, tego.TAIL_BASE)
RADIUS_BL = 0.6
#: §1: below this many locatable skull points the disc is not trusted.
MIN_SKULL_POINTS = 2
#: §4: the band under test, and the range the background is fitted over.
BAND = (3.0, 8.0)
SUB_BANDS = (("low", 3.0, 6.0), ("high", 6.0, 8.0))
FIT_RANGE = (0.5, 15.0)
#: §2: window length in seconds, and the two percentiles that define a candidate.
WIN_S = 2.0
STILL_PCT = 25.0
HEAD_PCT = 75.0


def skull_disc(pose_frame: npt.ArrayLike, radius_px: float,
               shape: tuple[int, int], *,
               points: tuple[int, ...] = SKULL) -> B1 | None:
    """``(h, w)`` True inside the disc on `points`, or None when not trusted."""
    p = np.asarray(pose_frame, dtype=np.float64)[list(points)]
    ok = np.isfinite(p).all(axis=1)
    if int(ok.sum()) < MIN_SKULL_POINTS:
        return None
    cy, cx = float(p[ok][:, 1].mean()), float(p[ok][:, 0].mean())
    h, w = int(shape[0]), int(shape[1])
    y0, y1 = max(0, int(cy - radius_px)), min(h, int(cy + radius_px) + 1)
    x0, x1 = max(0, int(cx - radius_px)), min(w, int(cx + radius_px) + 1)
    m = np.zeros((h, w), dtype=bool)
    if y1 <= y0 or x1 <= x0:
        return None
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
    m[y0:y1, x0:x1] = (yy ** 2 + xx ** 2) <= radius_px ** 2
    return m if m.any() else None


def scan_head(video: str, pose: npt.ArrayLike, *,
              radii_px: Sequence[float], dilate_px: float) -> dict[str, Any]:
    """One sequential pass: head-region and arena mean |delta| per frame.

    **Every registered radius is computed in ONE pass**, because the expensive
    parts -- the decode, the blur and the egocentric warp -- are shared and only
    the disc changes. Three radii in three passes would triple the cost and
    produce identical numbers.

    Separate from `motion.scan` rather than folded into it: that function's
    output is frozen into `PIXEL_NOISEFLOOR.md` and `PIXEL_PILOT.md`, and adding
    fields to it would mean either rescanning everything or carrying two
    incompatible versions of the same artefact.
    """
    import cv2                                   # headless; frame READING only

    p = np.asarray(pose, dtype=np.float64)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    try:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"no frames in {video}")
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape
        blur_old = cv2.GaussianBlur(grey.astype(np.float64), (0, 0), mo.SIGMA)
        mask_old = mo.exclusion_mask(p[0], dilate_px, (h, w))
        n = p.shape[0]
        rad = [float(r) for r in radii_px]
        arena = np.full(n, np.nan)
        head = {r: np.full(n, np.nan) for r in rad}
        ego_head = {r: np.full(n, np.nan) for r in rad}
        ego_hip = {r: np.full(n, np.nan) for r in rad}
        warp_old = _ego_warp(blur_old, p[0], (h, w))
        t = 1
        while t < n:
            ok, frame = cap.read()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey.astype(np.float64), (0, 0), mo.SIGMA)
            dif = np.abs(blur - blur_old)
            mask = mo.exclusion_mask(p[t], dilate_px, (h, w))
            arena_px = ~(mask | mask_old)
            if arena_px.any():
                arena[t] = float(dif[arena_px].mean())
            # Amendment 1: the same difference, taken AFTER registering both
            # frames on the animal, plus its hindquarter control.
            warp = _ego_warp(blur, p[t], (h, w))
            edif = (np.abs(warp - warp_old)
                    if warp is not None and warp_old is not None else None)
            for r in rad:
                disc = skull_disc(p[t], r, (h, w))
                if disc is not None:
                    head[r][t] = float(dif[disc].mean())
                if edif is None:
                    continue
                for pts, dest in ((SKULL, ego_head), (HIPS, ego_hip)):
                    dm = _fixed_disc(p[t], pts, r, (h, w))
                    if dm is not None:
                        dest[r][t] = float(edif[dm].mean())
            blur_old, mask_old, warp_old = blur, mask, warp
            t += 1
    finally:
        cap.release()
    # §1: head motion ABOVE this recording's own floor. The floor spans a factor
    # of 20 across recordings, so a global correction would be meaningless.
    a0 = arena[:t]
    out: dict[str, Any] = {"arena": a0, "n_frames": int(t),
                           "video": os.path.basename(video)}
    for r in rad:
        out[f"energy|{r}"] = head[r][:t] - a0
        out[f"energy_ego|{r}"] = ego_head[r][:t] - a0
        out[f"energy_ego_hip|{r}"] = ego_hip[r][:t] - a0
    return out


def _ego_warp(grey: npt.ArrayLike, pose_frame: npt.ArrayLike,
              shape: tuple[int, int]) -> F64 | None:
    """Rotate and translate a frame into the ANIMAL's frame. Amendment 1.

    Removes body translation and body rotation by construction, so a difference
    of two warped frames is motion RELATIVE TO THE ANIMAL -- which is what
    "still body, moving head" means, and what a disc in image coordinates cannot
    express (D20: it correlates +0.93 with keypoint speed).

    Returns None when the body frame is undefined. `ego.heading` and
    `ego.ORIGIN` are reused rather than re-derived, so the pixel frame and the
    keypoint frame are the same frame by construction.
    """
    import cv2

    p = np.asarray(pose_frame, dtype=np.float64)
    if not (np.isfinite(p[tego.ORIGIN]).all()
            and np.isfinite(p[list(tego.AXIS)]).all()):
        return None
    ang = float(np.degrees(tego.heading(p[None, :, :])[0]))
    h, w = int(shape[0]), int(shape[1])
    cx, cy = float(p[tego.ORIGIN, 0]), float(p[tego.ORIGIN, 1])
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    # Put the body origin at the image centre, so the disc is at a FIXED place
    # in every warped frame and the difference is taken over the same pixels.
    m[0, 2] += w / 2.0 - cx
    m[1, 2] += h / 2.0 - cy
    g = np.asarray(grey, dtype=np.float64)
    return np.asarray(cv2.warpAffine(g, m, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0.0), dtype=np.float64)


def _fixed_disc(pose_frame: npt.ArrayLike, points: tuple[int, ...],
                radius_px: float, shape: tuple[int, int]) -> B1 | None:
    """The disc for a warped frame: `points` carried into the animal's frame."""
    p = np.asarray(pose_frame, dtype=np.float64)
    if not (np.isfinite(p[tego.ORIGIN]).all()
            and np.isfinite(p[list(tego.AXIS)]).all()):
        return None
    q = p[list(points)]
    ok = np.isfinite(q).all(axis=1)
    if int(ok.sum()) < MIN_SKULL_POINTS:
        return None
    ang = float(tego.heading(p[None, :, :])[0])
    c, s_ = np.cos(-ang), np.sin(-ang)
    d = q[ok] - p[tego.ORIGIN]
    rx = c * d[:, 0] - s_ * d[:, 1]
    ry = s_ * d[:, 0] + c * d[:, 1]
    h, w = int(shape[0]), int(shape[1])
    cx, cy = w / 2.0 + float(rx.mean()), h / 2.0 + float(ry.mean())
    y0, y1 = max(0, int(cy - radius_px)), min(h, int(cy + radius_px) + 1)
    x0, x1 = max(0, int(cx - radius_px)), min(w, int(cx + radius_px) + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    m = np.zeros((h, w), dtype=bool)
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
    m[y0:y1, x0:x1] = (yy ** 2 + xx ** 2) <= radius_px ** 2
    return m if m.any() else None


def spectrum_stats(x: npt.ArrayLike, *, fps: float,
                   band: tuple[float, float] = BAND,
                   fit_range: tuple[float, float] = FIT_RANGE
                   ) -> dict[str, float]:
    """§4's statistics for one window, and the background SLOPE they rest on.

    Returns `peak_excess`, `band_share`, the fitted background `slope`, and the
    two sub-band excesses §0 requires reported separately. All NaN when the
    window cannot support the estimate -- too short, any NaN, constant, or fewer
    than 4 background bins to fit a line through. **NaN is a refusal**: callers
    drop the window rather than substituting a zero.

    ## What was calibrated before this touched a video, and what it corrects

    * **Exactly invariant to multiplicative rescaling.** `peak_excess` is
      identical to six decimals across a 1000x rescaling, because band power and
      the fitted background shift by the same constant in logs. This is the
      property §4 needs and it holds exactly. `band_share` shares it.
    * **Its null is small and nearly colour-independent**: on backgrounds with no
      peak, `peak_excess` is +0.053 (white), +0.059 (1/f) and +0.012 (1/f^2),
      sd ~0.09. **Not zero**, so §6's "0 by construction" is wrong and the
      contrast against a matched control -- where the offset cancels -- is the
      only fair reading. Recorded as D19.
    * **It is NOT more shape-robust than `band_share`, which §4 implied.**
      Measured: raising the broadband floor at a fixed 5 Hz peak drops
      `peak_excess` from +2.56 to +0.15 while `band_share` drops 0.93 to 0.31;
      steepening the background from beta 0 to 2.5 RAISES `peak_excess` from
      +0.77 to +2.11 while `band_share` falls 0.56 to 0.40. Neither is
      shape-free, and `peak_excess` is the more slope-sensitive of the two.

    So the protection against §2's circularity is **not the statistic alone**. It
    is exact rescaling-invariance, plus §5's amplitude-decile check, plus the
    speed-matched control -- and, because of the slope finding, plus reporting
    `slope` for both arms so a background difference cannot masquerade as a peak.
    """
    nan = {"peak_excess": float("nan"), "band_share": float("nan"),
           "slope": float("nan"),
           **{f"excess_{nm}": float("nan") for nm, _, _ in SUB_BANDS}}
    a = np.asarray(x, dtype=np.float64)
    if a.size < 8 or not np.isfinite(a).all() or float(np.ptp(a)) == 0.0:
        return nan
    freq, psd = ds.multitaper_psd(a[:, None], fps=fps)
    if freq.size == 0:
        return nan
    keep = (freq >= fit_range[0]) & (freq <= fit_range[1]) & (psd > 0)
    inb = keep & (freq >= band[0]) & (freq < band[1])
    out = keep & ~inb
    if int(out.sum()) < 4 or not inb.any():
        return nan
    slope, intercept = np.polyfit(np.log(freq[out]), np.log(psd[out]), 1)
    lp = np.log(psd)

    def excess(lo: float, hi: float) -> float:
        m = keep & (freq >= lo) & (freq < hi)
        if not m.any():
            return float("nan")
        return float(np.mean(lp[m] - (intercept + slope * np.log(freq[m]))))

    tot = float(psd[keep].sum())
    return {"peak_excess": excess(band[0], band[1]),
            "band_share": (float(psd[inb].sum() / tot) if tot > 0
                           else float("nan")),
            "slope": float(slope),
            **{f"excess_{nm}": excess(lo, hi) for nm, lo, hi in SUB_BANDS}}


def peak_excess(x: npt.ArrayLike, *, fps: float,
                band: tuple[float, float] = BAND,
                fit_range: tuple[float, float] = FIT_RANGE) -> float:
    """§4's gate statistic. See `spectrum_stats` for what it is and is not."""
    return spectrum_stats(x, fps=fps, band=band,
                          fit_range=fit_range)["peak_excess"]


def band_share(x: npt.ArrayLike, *, fps: float,
               band: tuple[float, float] = BAND,
               fit_range: tuple[float, float] = FIT_RANGE) -> float:
    """§4's secondary. Also exactly scale-invariant; also not shape-free."""
    return spectrum_stats(x, fps=fps, band=band,
                          fit_range=fit_range)["band_share"]
