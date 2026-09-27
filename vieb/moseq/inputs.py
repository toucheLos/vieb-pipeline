"""MOSEQ §1: the pose MoSeq sees -- repaired, not smoothed -- and its surrogates.

kpms reads coordinates plus a confidence per keypoint. Where a keypoint is
missing (DLC's flag, the mask veto, or a frame disposition abstained on) it
gets confidence 0, which kpms treats as unobserved: it interpolates the
coordinate and its noise model ignores it. A NaN coordinate is never passed.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vieb.clean import repair as rp

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

KINDS = ("real", "white", "phase")


def real(pose: npt.ArrayLike, dlc_missing: npt.ArrayLike, vetoed: npt.ArrayLike,
         abstain: npt.ArrayLike, conf: npt.ArrayLike) -> tuple[F64, F64]:
    """``(coords (T, K, 2), conf (T, K))`` for one recording.

    Abstained frames (disposition's bone-violating runs longer than 3 frames)
    are missing on every keypoint, as `ego` treats them (D30). Coordinates
    left missing are linearly filled only so the array is finite; their
    confidence is 0.
    """
    miss = np.asarray(dlc_missing, dtype=bool) | np.asarray(abstain, bool)[:, None]
    un = rp.unsmoothed(pose, miss, vetoed)
    gone = ~np.isfinite(un).all(axis=-1)
    c = np.asarray(conf, dtype=np.float64).copy()
    c[gone | np.asarray(vetoed, bool)] = 0.0
    return fill(un), np.clip(np.nan_to_num(c), 0.0, 1.0)


def fill(pose: npt.ArrayLike) -> F64:
    """Linear interpolation over NaN per keypoint coordinate, edges held."""
    p = np.asarray(pose, dtype=np.float64).copy()
    t = np.arange(p.shape[0])
    for k in range(p.shape[1]):
        for d in range(p.shape[2]):
            col = p[:, k, d]
            ok = np.isfinite(col)
            if ok.all():
                continue
            p[:, k, d] = np.interp(t, t[ok], col[ok]) if ok.any() else 0.0
    return p


def surrogate(coords: npt.ArrayLike, conf: npt.ArrayLike, kind: str,
              rng: np.random.Generator) -> tuple[F64, F64]:
    """The `kind` surrogate of one recording's filled coordinates.

    The real recording's confidence trace is kept exactly, so the surrogate has
    the same unobserved keypoint-frames; their coordinates are re-filled from
    the surrogate's own neighbours, as for the real arm.
    """
    from typing import Callable, cast

    from recur.recurrence import surrogate as sg

    c = np.asarray(conf, dtype=np.float64)
    if kind == "real":
        return np.asarray(coords, dtype=np.float64), c
    gen = cast("Callable[..., F64]",
               {"white": sg.white_surrogate, "phase": sg.phase_surrogate}[kind])
    s = np.asarray(gen(np.asarray(coords, dtype=np.float64), rng), dtype=np.float64)
    s[c == 0] = np.nan
    return fill(s), c
