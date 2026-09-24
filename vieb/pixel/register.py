"""Crop registration that reads no keypoints. `results/STABILISE_PREREGISTRATION.md`.

## Why this module exists

`DEVIATIONS.md` D21: the incumbent egocentric crop (`head._ego_warp`) is rotated
and translated **per frame from DLC pose**, so skull jitter on a rigid triangle
moves the crop, and differencing reads the movement as motion inside a still
animal. The principle this module implements is fixed in §0:

> Keypoints may **locate** a region. They may not **register** a frame.

## The arms, one pass each over the same frames

* **K** -- the incumbent, reproduced with `head.scan_head`'s arithmetic in the
  same order so §7 can demand agreement to 1e-9, not "close".
* **B** -- a background-subtraction mask. Centroid and second-moment axis give a
  rigid warp; the axis's 180 degree ambiguity is resolved by continuity, so no
  head/tail sign is ever needed.
* **P** -- phase-correlation-initialised ECC of frame t onto t-1 inside B's box
  (Amendment 1; the log-polar rotation first registered could not see it).
  It uses B's mask to know WHERE to look and never B's orientation.

For B and P the disc is placed by the **window-median** skull (or hip) position
in B's frame: one value per window, which frame-to-frame jitter cannot move.

## Two passes, because a window-median needs the whole window

Pass 1 builds the mask, its pose and its area for every frame -- the area
refusal (§1) needs the recording's median, and the disc centres need every
frame of their window. Pass 2 differences. `open_frames` is a callable
returning a fresh iterator of grey frames, so the same code runs on a video and
on §5-§6's planted sequences.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from vieb.pixel import head as hd, motion as mo

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

#: §1 B.1: frames sampled, evenly, for the per-pixel median background.
BG_FRAMES = 101
#: §1 B.3: one opening with this square kernel, then the largest component.
OPEN_KERNEL = 3
#: §1: a frame's mask area outside this multiple of the median is refused.
AREA_RANGE = (0.5, 2.0)
#: §1 P.1: B's box is padded by this many body lengths.
PAD_BL = 0.25
#: §1 P.4: below these the estimate is the identity. A float tolerance.
SNAP_PX = 1e-3
SNAP_DEG = 1e-3
#: Amendment 1: ECC's stopping rule and pre-smoothing, fixed.
ECC_CRITERIA = (3, 100, 1e-6)                  # TERM_CRITERIA_EPS | COUNT
ECC_GAUSS = 1
#: §1 refusals, at recording and at arm level.
MAX_NAN_FRAC = 0.10
MAX_REFUSED_FRAC = 0.20
ARMS = ("K", "B", "P")
REGIONS = (("head", hd.SKULL), ("hip", hd.HIPS))

Frames = Callable[[], Iterator[npt.NDArray[Any]]]


def blur(grey: npt.ArrayLike) -> F64:
    """`motion.scan`'s blur, so every arm differences the same image."""
    import cv2

    return np.asarray(cv2.GaussianBlur(np.asarray(grey).astype(np.float64),
                                       (0, 0), mo.SIGMA), dtype=np.float64)


def median_background(blurred: Sequence[npt.ArrayLike]) -> F64:
    """§1 B.1: the per-pixel median of already-blurred frames."""
    return np.asarray(np.median(np.stack([np.asarray(b, dtype=np.float64)
                                          for b in blurred]), axis=0),
                      dtype=np.float64)


def animal_mask(blurred: npt.ArrayLike, bg: npt.ArrayLike,
                cutoff: float) -> B1 | None:
    """§1 B.2-3: |frame - bg| > the recording's own cutoff, opened, largest part."""
    import cv2

    fg = (np.abs(np.asarray(blurred, dtype=np.float64)
                 - np.asarray(bg, dtype=np.float64)) > float(cutoff))
    fg8 = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg8, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(lab == k, dtype=bool)


def mask_pose(mask: npt.ArrayLike) -> tuple[float, float, float, float]:
    """§1 B.4: centroid, major-axis angle (radians, mod pi) and area."""
    import cv2

    m = cv2.moments(np.asarray(mask, dtype=np.uint8), binaryImage=True)
    area = float(m["m00"])
    if area <= 0:
        return (np.nan, np.nan, np.nan, 0.0)
    cx, cy = m["m10"] / area, m["m01"] / area
    th = 0.5 * float(np.arctan2(2.0 * m["mu11"], m["mu20"] - m["mu02"]))
    return (float(cx), float(cy), th, area)


def continuous(theta: float, prev: float | None) -> float:
    """§1 B.4: whichever of theta, theta + pi is nearer the previous angle."""
    if prev is None or not np.isfinite(prev):
        return float(theta)
    best = float(theta)
    for c in (theta, theta + np.pi):
        d = np.angle(np.exp(1j * (c - prev)))
        if abs(d) < abs(np.angle(np.exp(1j * (best - prev)))):
            best = float(c)
    # Carry the angle unwrapped, so the choice never flips on a +-pi boundary.
    return float(prev + np.angle(np.exp(1j * (best - prev))))


def frame_matrix(cx: float, cy: float, ang_rad: float,
                 shape: tuple[int, int]) -> F64:
    """The incumbent's warp convention (`head._ego_warp`), from any pose.

    Rotate by `ang` about the origin point, then put that point at the image
    centre. Shared by K's convention and B's, so the two are the same frame
    whenever their poses agree.
    """
    import cv2

    h, w = int(shape[0]), int(shape[1])
    m = cv2.getRotationMatrix2D((float(cx), float(cy)),
                                float(np.degrees(ang_rad)), 1.0)
    m[0, 2] += w / 2.0 - cx
    m[1, 2] += h / 2.0 - cy
    return np.asarray(m, dtype=np.float64)


def warp(img: npt.ArrayLike, m: npt.ArrayLike, shape: tuple[int, int]) -> F64:
    import cv2

    h, w = int(shape[0]), int(shape[1])
    return np.asarray(cv2.warpAffine(np.asarray(img, dtype=np.float64),
                                     np.asarray(m, dtype=np.float64), (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0.0), dtype=np.float64)


def apply(m: npt.ArrayLike, pts: npt.ArrayLike) -> F64:
    """An affine 2x3 applied to ``(N, 2)`` points."""
    a = np.asarray(m, dtype=np.float64)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.asarray(p @ a[:, :2].T + a[:, 2], dtype=np.float64)


def invert(m: npt.ArrayLike) -> F64:
    import cv2

    return np.asarray(cv2.invertAffineTransform(
        np.asarray(m, dtype=np.float64)), dtype=np.float64)


def disc(cx: float, cy: float, r: float, shape: tuple[int, int]) -> B1 | None:
    """A filled disc, or None when it misses the image entirely."""
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    h, w = int(shape[0]), int(shape[1])
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    out = np.zeros((h, w), dtype=bool)
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
    out[y0:y1, x0:x1] = (yy ** 2 + xx ** 2) <= r ** 2
    return out if out.any() else None


def rigid_between(prev: npt.ArrayLike, cur: npt.ArrayLike
                  ) -> tuple[float, float, float, F64] | None:
    """§1 P.2-3 as amended (D22): register `cur` onto `prev`, rigidly.

    Translation is initialised by Hann-windowed phase correlation, then
    `findTransformECC` (`MOTION_EUCLIDEAN`, unmasked) refines rotation and
    translation. Returns ``(angle_deg, tx, ty, W)`` with `W` the 2x3 ECC warp in
    crop coordinates -- ``cur(W x) ~ prev(x)`` -- so applying it to `cur` with
    `WARP_INVERSE_MAP` lands `cur` on `prev`. None when ECC fails: the pair is
    refused, which counts toward §1's refusals.

    The log-polar rotation the registration first specified missed synthetic
    rotations by 2-14 degrees on an animal-sized crop; ECC recovers them to
    within 0.03 degrees. Amendment 1.
    """
    import cv2

    # COPIES, not views. OpenCV 5.0.0's `phaseCorrelate` writes the window into
    # its inputs in place whenever the crop is already a DFT-optimal size, and a
    # crop of a frame is a view: it silently altered the frame the NEXT pair is
    # differenced against. Found by §7's incumbent check on a real recording.
    a = np.array(prev, dtype=np.float64, copy=True)
    b = np.array(cur, dtype=np.float64, copy=True)
    h, w = a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)
    (sx, sy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), win)
    W0 = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]], dtype=np.float32)
    try:
        _, W1 = cv2.findTransformECC(  # type: ignore[call-overload]
            a.astype(np.float32), b.astype(np.float32), W0,
            cv2.MOTION_EUCLIDEAN, ECC_CRITERIA, None, ECC_GAUSS)
    except cv2.error:
        return None
    Wm = np.asarray(W1, dtype=np.float64)
    if not np.isfinite(Wm).all():
        return None
    ang = float(np.degrees(np.arctan2(Wm[1, 0], Wm[0, 0])))
    tx, ty = float(Wm[0, 2]), float(Wm[1, 2])
    if abs(ang) < SNAP_DEG and np.hypot(tx, ty) < SNAP_PX:
        return 0.0, 0.0, 0.0, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return ang, tx, ty, Wm


def to_frame(Wc: npt.ArrayLike, x0: int, y0: int) -> F64:
    """A crop-coordinate warp as the same warp in full-frame coordinates."""
    Wm = np.asarray(Wc, dtype=np.float64)
    R, t = Wm[:, :2], Wm[:, 2]
    o = np.array([float(x0), float(y0)])
    return np.asarray(np.column_stack([R, t + o - R @ o]), dtype=np.float64)


def warp_inverse(img: npt.ArrayLike, Wf: npt.ArrayLike,
                 shape: tuple[int, int]) -> F64:
    """``out(x) = img(Wf x)``: `cur` carried onto `prev`'s coordinates."""
    import cv2

    h, w = int(shape[0]), int(shape[1])
    return np.asarray(cv2.warpAffine(
        np.asarray(img, dtype=np.float64), np.asarray(Wf, dtype=np.float64),
        (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0), dtype=np.float64)


def rigid_matrix(ang_deg: float, dx: float, dy: float,
                 centre: tuple[float, float]) -> F64:
    """The full-frame 2x3 for `rigid_between`'s answer about `centre`."""
    import cv2

    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                float(ang_deg), 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    return np.asarray(m, dtype=np.float64)


def _box(mask: B1, pad: float, shape: tuple[int, int]
         ) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = int(shape[0]), int(shape[1])
    p = int(round(pad))
    x0, x1 = max(0, int(xs.min()) - p), min(w, int(xs.max()) + p + 1)
    y0, y1 = max(0, int(ys.min()) - p), min(h, int(ys.max()) + p + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _centroid(p: F64, pts: tuple[int, ...]) -> F64 | None:
    q = p[list(pts)]
    ok = np.isfinite(q).all(axis=1)
    if int(ok.sum()) < hd.MIN_SKULL_POINTS:
        return None
    return np.asarray(q[ok].mean(axis=0), dtype=np.float64)


def _close(it: Iterator[Any]) -> None:
    close = getattr(it, "close", None)
    if close is not None:
        close()


def scan_register(open_frames: Frames, pose: npt.ArrayLike, *,
                  bg: npt.ArrayLike, cutoff: float, radii_px: Sequence[float],
                  dilate_px: float, win: int,
                  body_length_px: float) -> dict[str, Any]:
    """Pass 1 then pass 2 over `open_frames()`. Every arm, every radius.

    Returns per-frame arrays keyed ``"{arm}|{region}|{r}"`` holding the RAW
    in-disc mean |delta| (index t is the pair t-1 -> t, index 0 NaN), plus
    ``"arena"``, ``"identical"``, mask diagnostics and P's estimates. The K keys
    ``"K|head|r" - arena`` reproduce `head.scan_head`'s ``energy_ego|r``.
    """
    p = np.asarray(pose, dtype=np.float64)
    bgf = np.asarray(bg, dtype=np.float64)
    rad = [float(r) for r in radii_px]

    # ---- pass 1: the mask, its pose and area, for every frame ----------------
    cx_l: list[float] = []
    cy_l: list[float] = []
    th_l: list[float] = []
    area_l: list[float] = []
    boxes: list[tuple[int, int, int, int] | None] = []
    shape: tuple[int, int] | None = None
    prev_th: float | None = None
    it1 = open_frames()
    for grey in it1:
        if len(cx_l) >= p.shape[0]:
            break
        b = blur(grey)
        shape = (int(b.shape[0]), int(b.shape[1]))
        m = animal_mask(b, bgf, cutoff)
        if m is None:
            cx_l.append(np.nan)
            cy_l.append(np.nan)
            th_l.append(np.nan)
            area_l.append(0.0)
            boxes.append(None)
            continue
        cx, cy, th, ar = mask_pose(m)
        th = continuous(th, prev_th) if np.isfinite(th) else th
        prev_th = th if np.isfinite(th) else prev_th
        cx_l.append(cx)
        cy_l.append(cy)
        th_l.append(th)
        area_l.append(ar)
        boxes.append(_box(m, PAD_BL * body_length_px, shape))
    # Release pass 1's decoder BEFORE pass 2 opens its own. Left suspended, a
    # second concurrent decode of the same file returned different pixels on 12
    # of 6,302 frames of a real recording (found by §7's incumbent check).
    _close(it1)
    if shape is None:
        raise SystemExit("no frames")
    n = len(cx_l)
    area = np.asarray(area_l)
    med = float(np.median(area[area > 0])) if (area > 0).any() else np.nan
    ok = ((area >= AREA_RANGE[0] * med) & (area <= AREA_RANGE[1] * med)
          & np.isfinite(np.asarray(th_l)))
    mB = [frame_matrix(cx_l[t], cy_l[t], th_l[t], shape) if ok[t] else None
          for t in range(n)]

    # Window-median disc centres in B's frame: one per (window, region).
    n_win = (n + win - 1) // win
    centre: dict[str, F64] = {}
    for name, pts in REGIONS:
        c = np.full((n_win, 2), np.nan)
        for k in range(n_win):
            got = []
            for t in range(k * win, min(n, (k + 1) * win)):
                mt = mB[t]
                q = _centroid(p[t], pts) if t < p.shape[0] else None
                if mt is None or q is None:
                    continue
                got.append(apply(mt, q)[0])
            if got:
                c[k] = np.median(np.asarray(got), axis=0)
        centre[name] = c

    # ---- pass 2: differences ------------------------------------------------
    out: dict[str, Any] = {}
    for arm in ARMS:
        for name, _ in REGIONS:
            for r in rad:
                out[f"{arm}|{name}|{r}"] = np.full(n, np.nan)
    arena = np.full(n, np.nan)
    identical = np.zeros(n, dtype=bool)
    pest = np.full((n, 3), np.nan)
    h, w = shape
    it = open_frames()
    t = 0
    grey_old: npt.NDArray[Any] | None = None
    b_old = mask_old = warpK_old = warpB_old = None
    for grey in it:
        if t >= n:
            break
        g = np.asarray(grey)
        b = blur(g)
        mask = mo.exclusion_mask(p[t], dilate_px, (h, w))
        warpK = hd._ego_warp(b, p[t], (h, w))
        mt = mB[t]
        warpB = warp(b, mt, shape) if mt is not None else None
        if t == 0:
            grey_old, b_old, mask_old = g, b, mask
            warpK_old, warpB_old = warpK, warpB
            t += 1
            continue
        assert b_old is not None and mask_old is not None
        assert grey_old is not None
        identical[t] = bool(np.array_equal(g, grey_old))
        dif = np.abs(b - b_old)
        arena_px = ~(mask | mask_old)
        if arena_px.any():
            arena[t] = float(dif[arena_px].mean())
        # K: `scan_head`'s own loop, same order, same operations.
        edif = (np.abs(warpK - warpK_old)
                if warpK is not None and warpK_old is not None else None)
        if edif is not None:
            for name, pts in REGIONS:
                for r in rad:
                    dm = hd._fixed_disc(p[t], pts, r, (h, w))
                    if dm is not None:
                        out[f"K|{name}|{r}"][t] = float(edif[dm].mean())
        k = t // win
        # B: both frames in B's frame, the disc fixed for the window.
        if warpB is not None and warpB_old is not None:
            bdif = np.abs(warpB - warpB_old)
            for name, _ in REGIONS:
                c = centre[name][k]
                for r in rad:
                    dm = disc(c[0], c[1], r, shape)
                    if dm is not None:
                        out[f"B|{name}|{r}"][t] = float(bdif[dm].mean())
        # P: frame t registered onto t-1 by phase correlation in B's box.
        box = boxes[t - 1]
        if ok[t - 1] and ok[t] and box is not None:
            x0, y0, x1, y1 = box
            est = rigid_between(b_old[y0:y1, x0:x1], b[y0:y1, x0:x1])
            pdif: F64 | None = None
            if est is not None:
                ang, tx, ty, Wc = est
                pest[t] = (ang, tx, ty)
                if ang == 0.0 and tx == 0.0 and ty == 0.0:
                    pdif = np.abs(b - b_old)
                else:
                    pdif = np.abs(warp_inverse(b, to_frame(Wc, x0, y0), shape)
                                  - b_old)
            mprev = mB[t - 1]
            assert mprev is not None
            inv = invert(mprev)
            for name, _ in REGIONS:
                if pdif is None:
                    break
                c = apply(inv, centre[name][k])[0]
                for r in rad:
                    dm = disc(c[0], c[1], r, shape)
                    if dm is not None:
                        out[f"P|{name}|{r}"][t] = float(pdif[dm].mean())
        # B and P are refused on a pair when either frame's mask is refused.
        if not (ok[t] and ok[t - 1]):
            for arm in ("B", "P"):
                for name, _ in REGIONS:
                    for r in rad:
                        out[f"{arm}|{name}|{r}"][t] = np.nan
        grey_old, b_old, mask_old = g, b, mask
        warpK_old, warpB_old = warpK, warpB
        t += 1
    _close(it)
    out.update({"arena": arena[:t], "identical": identical[:t],
                "n_frames": int(t), "mask_area": area[:t],
                "mask_ok": ok[:t], "mask_theta": np.asarray(th_l)[:t],
                "p_estimate": pest[:t],
                "nan_frac": float(1.0 - ok[:t].mean()) if t else 1.0})
    for key in list(out):
        if "|" in key:
            out[key] = np.asarray(out[key])[:t]
    return out
