"""Frame differencing, the animal mask, and the arena's own noise floor.

`results/PIXEL_PREREGISTRATION.md` §1-§3. The per-frame pipeline is ezTrack's,
read from source and not remembered:

    grey -> GaussianBlur(sigma=1) -> |new - old| -> count of pixels > mt_cutoff

## One pass, and the whole sweep comes out of it

`Measure_Motion` stores a *count above a threshold*, so a sweep over `mt_cutoff`
would normally cost one video decode per cutoff. Storing the per-frame
**histogram** of |delta| instead makes the count at any cutoff a suffix sum, so
the registered 27-point grid costs **one sequential pass**. The histogram is
exact to `1/BIN_SCALE` of a grey level, which is finer than any cutoff the grid
reaches, and every value at or above `BIN_MAX` lands in one overflow bin --
harmless, because a cutoff is never set up there.

## Two histograms, because the arena is the control

`hist_full` counts every pixel; `hist_arena` counts only pixels that are outside
the animal in **both** frames of the pair. The arena histogram is what §3's
cutoff is derived from -- we have no animal-free calibration video, so the
animal is masked out of the video we do have -- and the per-frame arena count is
what the trichotomy in §6 compares the animal region against.

## The mask is the keypoint box dilated by one body length

Dilating by a **fraction of the animal's own body length** rather than a fixed
number of pixels keeps the margin scale-free across boxes and camera distances.
A frame with fewer than `MIN_KEYPOINTS` locatable points contributes **no**
arena pixels: a mask that cannot be trusted to contain the animal would leak the
animal into its own control.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B1 = npt.NDArray[np.bool_]

#: ezTrack's `SIGMA`, verified in `Measure_Motion` and in all three notebooks.
#: A blur radius in pixels, so it carries across resolutions unchanged.
SIGMA = 1.0
#: Histogram resolution: bins of 1/4 of a grey level, up to `BIN_MAX`.
BIN_SCALE = 4
BIN_MAX = 64.0
N_BINS = int(BIN_MAX * BIN_SCALE) + 1          # last bin is the overflow
#: §3: the mask is the keypoint bounding box dilated by this many body lengths.
DILATE_BODY_LENGTHS = 1.0
#: §3: below this many locatable keypoints the mask is not trusted at all.
MIN_KEYPOINTS = 3
#: §3: ezTrack's `Calibrate` rule, and the percentile it uses.
FLOOR_PERCENTILE = 99.99
CUTOFF_MULTIPLIER = 2.0
#: §3 refusal: fewer arena pixel-pairs than this and the percentile is not one.
MIN_ARENA_PAIRS = 200_000


def quantise(dif: F64) -> npt.NDArray[np.int64]:
    """|delta| -> histogram bin index, everything at/above `BIN_MAX` clamped."""
    q = np.floor(np.asarray(dif, dtype=np.float64) * BIN_SCALE)
    return np.asarray(np.clip(q, 0, N_BINS - 1), dtype=np.int64)


def bin_edge(index: int) -> float:
    """The |delta| value the given bin index starts at."""
    return float(index) / BIN_SCALE


def exclusion_mask(pose_frame: npt.ArrayLike, dilate_px: float,
                   shape: tuple[int, int]) -> B1:
    """``(h, w)`` True where the animal is presumed to be.

    The bounding box of the frame's locatable keypoints, dilated by `dilate_px`.
    Returns an **all-True** mask when fewer than `MIN_KEYPOINTS` are locatable:
    an undefined mask excludes everything rather than excluding nothing, so a
    frame we cannot mask contributes no arena pixels instead of contributing the
    animal as if it were arena.
    """
    h, w = int(shape[0]), int(shape[1])
    p = np.asarray(pose_frame, dtype=np.float64)
    ok = np.isfinite(p).all(axis=1)
    m = np.zeros((h, w), dtype=bool)
    if int(ok.sum()) < MIN_KEYPOINTS:
        m[:] = True
        return m
    q = p[ok]
    x0 = int(np.floor(q[:, 0].min() - dilate_px))
    x1 = int(np.ceil(q[:, 0].max() + dilate_px))
    y0 = int(np.floor(q[:, 1].min() - dilate_px))
    y1 = int(np.ceil(q[:, 1].max() + dilate_px))
    m[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
    return m


def motion_from_hist(hist: npt.ArrayLike, cutoff: float) -> I64:
    """``(T,)`` ezTrack's `Motion`: pixels whose |delta| EXCEEDS `cutoff`.

    `Measure_Motion` uses a strict `>`, so a bin is counted only if every value
    it can hold is above the cutoff. The bin containing the cutoff is dropped --
    conservative by at most one bin width, `1/BIN_SCALE` of a grey level.
    """
    h = np.asarray(hist, dtype=np.int64)
    first = int(np.floor(float(cutoff) * BIN_SCALE)) + 1
    if first >= h.shape[1]:
        return np.zeros(h.shape[0], dtype=np.int64)
    return np.asarray(h[:, max(first, 0):].sum(axis=1), dtype=np.int64)


def floor_percentile(hist_arena_total: npt.ArrayLike,
                     percentile: float = FLOOR_PERCENTILE) -> float:
    """The `percentile`-th |delta| of the pooled arena sample, from its histogram.

    ezTrack's `Calibrate` takes `np.percentile(cal_dif, 99.99)` over a random
    pixel sample of an animal-free video. This is the same statistic over every
    arena pixel of the video we actually have, which is a far larger sample of
    the same quantity.

    Returns the containing bin's **upper** edge, so the answer is at most one
    bin width -- 1/BIN_SCALE of a grey level -- ABOVE the true percentile and
    never below it. The registered sweep moves the cutoff by factors of 2, which
    dwarfs a quarter of a grey level, so the convention cannot decide a verdict;
    it is fixed here only so that it is not decided silently.
    """
    h = np.asarray(hist_arena_total, dtype=np.float64)
    tot = float(h.sum())
    if tot <= 0:
        return float("nan")
    c = np.cumsum(h) / tot
    idx = int(np.searchsorted(c, percentile / 100.0, side="left"))
    return bin_edge(min(idx, N_BINS - 1) + 1)


def cutoff_of(hist_arena_total: npt.ArrayLike, *,
              multiplier: float = CUTOFF_MULTIPLIER) -> float:
    """§3's rule: `mt_cutoff = multiplier x percentile(arena |delta|, 99.99)`."""
    return float(multiplier) * floor_percentile(hist_arena_total)


def scan(video: str, pose: npt.ArrayLike, *, dilate_px: float,
         max_frames: int = 0) -> dict[str, Any]:
    """One sequential pass. Returns the histograms every later number needs.

    Videos are read **sequentially and read-only**: a random seek costs 5-36 ms
    against 0.5-0.65 ms for a sequential frame, measured, so nothing here seeks.

    `raw_identical` is computed on the **unblurred** greys, because a duplicate
    video frame is an exact equality of the decoded frames and blurring can only
    hide the difference that would disprove it.
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
        grey_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = grey_old.shape
        blur_old = cv2.GaussianBlur(grey_old.astype(np.float64), (0, 0), SIGMA)
        mask_old = exclusion_mask(p[0], dilate_px, (h, w)) if p.shape[0] else \
            np.ones((h, w), dtype=bool)

        cap_n = p.shape[0] if max_frames <= 0 else min(p.shape[0], max_frames)
        hist_full = np.zeros((cap_n, N_BINS), dtype=np.int64)
        hist_arena = np.zeros((cap_n, N_BINS), dtype=np.int64)
        n_arena = np.zeros(cap_n, dtype=np.int64)
        identical = np.zeros(cap_n, dtype=bool)
        t = 1
        while t < cap_n:
            ok, frame = cap.read()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            identical[t] = bool(np.array_equal(grey, grey_old))
            blur = cv2.GaussianBlur(grey.astype(np.float64), (0, 0), SIGMA)
            q = quantise(np.abs(blur - blur_old))
            hist_full[t] = np.bincount(q.ravel(), minlength=N_BINS)[:N_BINS]
            mask = exclusion_mask(p[t], dilate_px, (h, w))
            arena = ~(mask | mask_old)           # outside the animal in BOTH
            if arena.any():
                sel = q[arena]
                hist_arena[t] = np.bincount(sel, minlength=N_BINS)[:N_BINS]
                n_arena[t] = int(sel.size)
            grey_old, blur_old, mask_old = grey, blur, mask
            t += 1
    finally:
        cap.release()
    return {"hist_full": hist_full[:t], "hist_arena": hist_arena[:t],
            "n_arena": n_arena[:t], "identical": identical[:t],
            "n_frames": int(t), "height": int(h), "width": int(w),
            "n_pixels": int(h * w), "video": os.path.basename(video)}
