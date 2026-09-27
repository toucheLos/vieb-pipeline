"""The pose repaired with SAM as a second detector. `results/REPAIR_PREREGISTRATION.md`.

Disposition's pose, then a **mask veto** (a keypoint more than `VETO_BL` body
lengths from SAM's carried mask is set missing), then holding of missing runs of
at most `MAX_HOLD` frames, then `median_0.25`, filtered over held values that are
never stored.

The veto is deliberately far. Near-edge head keypoints outside the mask are
usually SAM clipping the head (`AGREEMENT.md` §2), not DLC erring; 0.3 bl
catches only keypoints off the animal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from vieb.clean import arms
from vieb.pixel import sam

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

VETO_BL = 0.3
MAX_HOLD = 3
MEDIAN_S = 0.25


class MaskDistance:
    """Distance from any point to SAM's mask, carried to any frame.

    Each accepted keyframe's mask gets a distance transform over its crop,
    padded so distances out to `pad_px` are exact. Beyond the pad the distance
    is reported as `pad_px` (at least that far). A frame is carried from its
    nearer bracketing keyframe, shifted by the linearly interpolated centroid
    of the two -- the rule every SAM stage has used. A frame not bracketed by
    two accepted keyframes has no mask.
    """

    def __init__(self, z: Any, pad_px: float) -> None:
        import cv2

        self.every = int(z["every"])
        self.pad = int(np.ceil(pad_px)) + 2
        self.crops = sam.unpack_crops(z)
        self.dt: dict[int, tuple[int, int, F64]] = {}
        self.cent: dict[int, tuple[float, float]] = {}
        for t, (x0, y0, c) in self.crops.items():
            padded = np.pad(c, self.pad).astype(np.uint8)
            d = cv2.distanceTransform((1 - padded).astype(np.uint8),
                                      cv2.DIST_L2, 5)
            self.dt[t] = (x0 - self.pad, y0 - self.pad, np.asarray(d, np.float64))
            ys, xs = np.nonzero(c)
            self.cent[t] = (float(xs.mean() + x0), float(ys.mean() + y0))

    def has(self, t: int) -> bool:
        k0 = (t // self.every) * self.every
        return k0 in self.dt and (t == k0 or k0 + self.every in self.dt)

    def at(self, t: int, pts: npt.ArrayLike) -> F64:
        """Distances of `pts` (``(K, 2)``) to frame t's mask; NaN if no mask."""
        p = np.asarray(pts, dtype=np.float64)
        out = np.full(p.shape[0], np.nan)
        if not self.has(t):
            return out
        k0 = (t // self.every) * self.every
        if t == k0:
            k, dx, dy = k0, 0.0, 0.0
        else:
            k1 = k0 + self.every
            f = (t - k0) / self.every
            cx = (1 - f) * self.cent[k0][0] + f * self.cent[k1][0]
            cy = (1 - f) * self.cent[k0][1] + f * self.cent[k1][1]
            k = k0 if f <= 0.5 else k1
            dx, dy = cx - self.cent[k][0], cy - self.cent[k][1]
        ox, oy, d = self.dt[k]
        for i, q in enumerate(p):
            if not np.isfinite(q).all():
                continue
            x = int(round(float(q[0] - dx))) - ox
            y = int(round(float(q[1] - dy))) - oy
            if 0 <= y < d.shape[0] and 0 <= x < d.shape[1]:
                out[i] = float(d[y, x])
            else:
                out[i] = float(self.pad)
        return out


def veto(pose: npt.ArrayLike, dist: MaskDistance, bl: float, *,
         veto_bl: float = VETO_BL) -> tuple[B1, B1]:
    """(vetoed ``(T, K)``, has_mask ``(T,)``): keypoints far off the mask."""
    p = np.asarray(pose, dtype=np.float64)
    T, K = p.shape[:2]
    vet = np.zeros((T, K), dtype=bool)
    has = np.zeros(T, dtype=bool)
    for t in range(T):
        if not dist.has(t):
            continue
        has[t] = True
        d = dist.at(t, p[t])
        vet[t] = np.isfinite(d) & (d > veto_bl * bl)
    return vet, has


def hold_short(missing: npt.ArrayLike, *, max_hold: int = MAX_HOLD) -> B1:
    """Missing keypoint-frames that stay missing: runs longer than `max_hold`.

    Shorter runs are held (interpolated) by the caller; a run touching either
    end of the recording has no bracket and stays missing.
    """
    m = np.asarray(missing, dtype=bool)
    keep = np.zeros_like(m)
    T = m.shape[0]
    for k in range(m.shape[1]):
        col = m[:, k]
        if not col.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, col.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if (b - a) > max_hold or a == 0 or b == T:
                keep[a:b, k] = True
    return keep


def held_and_stays(pose: npt.ArrayLike, dlc_missing: npt.ArrayLike,
                   vetoed: npt.ArrayLike) -> tuple[F64, B1]:
    """REPAIR §1 steps 2-3: every missing value held, and which stay missing.

    Missing is DLC's own flag or the veto. `held` interpolates every gap (the
    filter needs a contiguous series); `stays` marks the runs longer than
    `MAX_HOLD`, which are re-set missing after any filtering.
    """
    missing = np.asarray(dlc_missing, dtype=bool) | np.asarray(vetoed, dtype=bool)
    return arms.held_array(np.asarray(pose, dtype=np.float64), missing), hold_short(missing)


def unsmoothed(pose: npt.ArrayLike, dlc_missing: npt.ArrayLike,
               vetoed: npt.ArrayLike) -> F64:
    """The repaired pose BEFORE `median_0.25` (MOSEQ §1): short gaps held,
    long gaps NaN. `repair()` is `rolling_median` of the same held series."""
    held, stays = held_and_stays(pose, dlc_missing, vetoed)
    out = held.copy()
    out[stays] = np.nan
    return out


def repair(pose: npt.ArrayLike, dlc_missing: npt.ArrayLike,
           dist: MaskDistance | None, bl: float, fps: float, *,
           use_veto: bool = True) -> dict[str, Any]:
    """REPAIR §1 steps 2-4 on one recording's disposition pose.

    `dlc_missing` is DLC's own missing flag (``(T, K)``). With `use_veto`
    False (or no masks) this is the registered fallback, disposition +
    `median_0.25`.
    """
    p = np.asarray(pose, dtype=np.float64)
    T, K = p.shape[:2]
    if use_veto and dist is not None:
        vetoed, has_mask = veto(p, dist, bl)
    else:
        vetoed, has_mask = np.zeros((T, K), bool), np.zeros(T, bool)
    held, stays = held_and_stays(p, dlc_missing, vetoed)
    smooth = arms.rolling_median(held, None, fps, window_s=MEDIAN_S)
    out = smooth.copy()
    out[stays] = np.nan
    # `pose_held` is the filter's output before long runs are re-set missing.
    # Metrics that need a contiguous series (violation, spectrum) read it; it is
    # never stored as the arm.
    return {"pose": out, "pose_held": smooth, "missing_kp": stays,
            "vetoed": vetoed, "has_mask": has_mask}
