"""The mask stream and head coverage. STATES §1-§2.

Computed on SAM's accepted keyframes (every 6th frame) and carried to every
frame by linear interpolation between the two bracketing accepted keyframes --
the rule every SAM stage has used. A frame not bracketed has no mask features.

Channels (body lengths / seconds, so scale-free across boxes):
  0 area / bl^2                  -- rearing and stretching change it
  1 elongation                   -- sqrt(major / minor) second moment
  2 centroid speed, bl/s          -- from the interpolated centroid
  3 turning rate, rad/s           -- of the major axis, with half-turn continuity
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from vieb.clean import repair as rp
from vieb.pixel import register as rg, sam
from vieb.tok import ego as tego

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

CHANNELS = ("mask_area_bl2", "mask_elongation", "mask_speed_bls",
            "mask_turn_rads")
HEAD_TOL_BL = 0.05


def _moments(c: B1) -> tuple[float, float, float, float, float]:
    """(cx, cy, theta, area, elongation) of one crop, crop coordinates."""
    cx, cy, th, area = rg.mask_pose(c)
    ys, xs = np.nonzero(c)
    cov = np.cov(np.vstack([xs, ys])) if xs.size > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))
    elong = float(np.sqrt(ev[1] / ev[0])) if ev[0] > 0 else float("nan")
    return cx, cy, th, area, elong


def mask_stream(z: Any, n_frames: int, bl: float, fps: float) -> tuple[F64, B1]:
    """(``(T, 4)`` features, ``(T,)`` valid) for one recording's masks npz."""
    every = int(z["every"])
    crops = sam.unpack_crops(z)
    keys = sorted(crops)
    kx = np.full((len(keys), 5), np.nan)
    prev = None
    for i, t in enumerate(keys):
        x0, y0, c = crops[t]
        cx, cy, th, area, el = _moments(c)
        th = rg.continuous(th, prev)
        prev = th
        kx[i] = (cx + x0, cy + y0, th, area / bl ** 2, el)
    feats = np.full((n_frames, 4), np.nan)
    valid = np.zeros(n_frames, dtype=bool)
    cent = np.full((n_frames, 3), np.nan)
    pos = {t: i for i, t in enumerate(keys)}
    for t in range(n_frames):
        k0 = (t // every) * every
        k1 = k0 + every
        if k0 not in pos or (t != k0 and k1 not in pos):
            continue
        a = kx[pos[k0]]
        b = a if t == k0 else kx[pos[k1]]
        f = 0.0 if t == k0 else (t - k0) / every
        v = (1 - f) * a + f * b
        cent[t] = v[:3]
        feats[t, 0], feats[t, 1] = v[3], v[4]
        valid[t] = True
    d = np.full(n_frames, np.nan)
    d[1:] = np.hypot(np.diff(cent[:, 0]), np.diff(cent[:, 1])) * fps / bl
    dth = np.full(n_frames, np.nan)
    dth[1:] = np.diff(cent[:, 2]) * fps
    feats[:, 2], feats[:, 3] = d, dth
    valid &= np.isfinite(feats).all(axis=1)
    return feats, valid


def head_coverage(z: Any, pose: npt.ArrayLike, bl: float) -> tuple[B1, B1]:
    """(covered, defined): both ears within HEAD_TOL_BL of the carried mask."""
    p = np.asarray(pose, dtype=np.float64)
    dist = rp.MaskDistance(z, pad_px=HEAD_TOL_BL * bl + 4)
    covered = np.zeros(p.shape[0], dtype=bool)
    defined = np.zeros(p.shape[0], dtype=bool)
    ears = [tego.LEFT_EAR, tego.RIGHT_EAR]
    for t in range(p.shape[0]):
        if not dist.has(t):
            continue
        dd = dist.at(t, p[t, ears])
        if np.isfinite(dd).all():
            defined[t] = True
            covered[t] = bool((dd <= HEAD_TOL_BL * bl).all())
    return covered, defined
