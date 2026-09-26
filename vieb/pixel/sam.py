"""A mask track from a segmenter, not a background. `STABILISE3_PREREGISTRATION.md` §1.

SAM runs on every `STRIDE`-th frame, prompted by its own last accepted mask's
padded box. Keypoints **seed** it only at the first keyframe and after a refusal:
a located region telling it which object to cut, never a registration. Between
keyframes the mask pose is interpolated, and a frame is refused if either
bracketing keyframe is.

The segmenter is passed in as `predict(rgb, box) -> (mask, iou)`, so this module
never imports torch and its logic is tested with a stand-in.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from vieb.pixel import register as rg
from vieb.tok import ego as tego

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

#: §1, fixed.
STRIDE = 3
PAD_BL = 0.25
IOU_MIN = 0.80
AREA_RANGE = rg.AREA_RANGE

Predict = Callable[[npt.NDArray[Any], F64], tuple[B1, float]]
#: POINT_PROMPTS_GUARDED: the keypoints offered as positive point prompts, how
#: near the box mask a keypoint must lie to be used, and the area cap.
GUARD_PROMPTS = (tego.NOSE, tego.CENTER, tego.TAIL_BASE)
GUARD_DIST_BL = 0.2
GUARD_AREA = 1.3


def keypoint_box(pose_frame: npt.ArrayLike, pad_px: float) -> F64 | None:
    """The seed: the frame's keypoint bounding box, padded. §0's 'locate'."""
    p = np.asarray(pose_frame, dtype=np.float64)
    ok = np.isfinite(p).all(axis=1)
    if int(ok.sum()) < 3:
        return None
    q = p[ok]
    return np.array([q[:, 0].min() - pad_px, q[:, 1].min() - pad_px,
                     q[:, 0].max() + pad_px, q[:, 1].max() + pad_px])


def mask_box(mask: B1, pad_px: float) -> F64 | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.array([xs.min() - pad_px, ys.min() - pad_px,
                     xs.max() + pad_px, ys.max() + pad_px], dtype=np.float64)


def sam_track(open_rgb: Callable[[], Iterator[npt.NDArray[Any]]],
              pose: npt.ArrayLike, predict: Predict, *,
              body_length_px: float, n_frames: int | None = None,
              prompt_mode: str = "propagate") -> dict[str, Any]:
    """Keyframe SAM, then interpolation, in `register.mask_track`'s format.

    `prompt_mode`: ``"propagate"`` (STABILISE 3: the last accepted mask's box,
    keypoints only to seed) or ``"keypoint"`` (STABILISE 4 §1: the padded
    keypoint box at EVERY keyframe, and a keyframe is refused unless its mask
    centroid lies inside that box).

    Extra keys: ``key_t`` (keyframe indices), ``key_iou``, ``key_seeded``,
    ``key_accepted``, and ``key_crops`` -- ``(x0, y0, bool crop)`` per accepted
    keyframe, for gate 0's disc-on-animal check without holding full masks.
    """
    p = np.asarray(pose, dtype=np.float64)
    pad = PAD_BL * float(body_length_px)
    n_max = p.shape[0] if n_frames is None else min(int(n_frames), p.shape[0])
    key_t: list[int] = []
    key_iou: list[float] = []
    key_seeded: list[bool] = []
    key_ok: list[bool] = []
    key_pose: list[tuple[float, float, float, float]] = []
    key_bbox: list[F64 | None] = []
    crops: dict[int, tuple[int, int, B1]] = {}
    shape: tuple[int, int] | None = None
    prompt: F64 | None = None
    prev_th: float | None = None
    n = 0
    it = open_rgb()
    try:
        for t, rgb in enumerate(it):
            if t >= n_max:
                break
            n = t + 1
            if shape is None:
                shape = (int(rgb.shape[0]), int(rgb.shape[1]))
            if t % STRIDE:
                continue
            if prompt_mode == "keypoint":
                seeded = True
                box = keypoint_box(p[t], pad)
            else:
                seeded = prompt is None
                box = keypoint_box(p[t], pad) if seeded else prompt
            key_t.append(t)
            key_seeded.append(seeded)
            if box is None:
                key_iou.append(np.nan)
                key_ok.append(False)
                key_pose.append((np.nan, np.nan, np.nan, 0.0))
                key_bbox.append(None)
                prompt = None
                continue
            m, iou = predict(rgb, box)
            m = np.asarray(m, dtype=bool)
            key_iou.append(float(iou))
            cx, cy, th, ar = rg.mask_pose(m)
            good = float(iou) >= IOU_MIN and ar > 0 and np.isfinite(th)
            if good and prompt_mode == "keypoint":
                good = bool(box[0] <= cx <= box[2] and box[1] <= cy <= box[3])
            if good:
                th = rg.continuous(th, prev_th)
                prev_th = th
                prompt = mask_box(m, pad)
                ys, xs = np.nonzero(m)
                y0, x0 = int(ys.min()), int(xs.min())
                crops[t] = (x0, y0, m[y0:int(ys.max()) + 1, x0:int(xs.max()) + 1])
            else:
                prompt = None            # re-seed at the next keyframe
            key_ok.append(bool(good))
            key_pose.append((cx, cy, th, ar))
            key_bbox.append(mask_box(m, pad) if good else None)
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            close()
    if shape is None:
        raise SystemExit("no frames")
    kt = np.asarray(key_t, dtype=np.int64)
    kok = np.asarray(key_ok, dtype=bool)
    kp = np.asarray(key_pose, dtype=np.float64).reshape(-1, 4)
    # Post-hoc area refusal against the recording's own accepted median.
    if kok.any():
        med = float(np.median(kp[kok, 3]))
        kok &= (kp[:, 3] >= AREA_RANGE[0] * med) & (kp[:, 3] <= AREA_RANGE[1] * med)
    for j, t in enumerate(kt):
        if not kok[j]:
            crops.pop(int(t), None)
    cx_a = np.full(n, np.nan)
    cy_a = np.full(n, np.nan)
    th_a = np.full(n, np.nan)
    area = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    boxes: list[tuple[int, int, int, int] | None] = [None] * n
    h, w = shape
    for j in range(kt.size):
        t0 = int(kt[j])
        last = j + 1 >= kt.size
        # A frame is refused unless BOTH bracketing keyframes are accepted.
        # After the last keyframe only the keyframe itself is bracketed.
        if not kok[j] or (not last and not kok[j + 1]):
            continue
        j1 = j if last else j + 1
        t1 = t0 + 1 if last else int(kt[j1])
        b0, b1 = key_bbox[j], key_bbox[j1]
        assert b0 is not None and b1 is not None
        for t in range(t0, min(t1, n)):
            f = 0.0 if last else (t - t0) / float(t1 - t0)
            cx_a[t], cy_a[t], th_a[t], area[t] = (1 - f) * kp[j] + f * kp[j1]
            bb = (1 - f) * b0 + f * b1
            x0, y0 = max(0, int(np.floor(bb[0]))), max(0, int(np.floor(bb[1])))
            x1 = min(w, int(np.ceil(bb[2])) + 1)
            y1 = min(h, int(np.ceil(bb[3])) + 1)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                boxes[t] = (x0, y0, x1, y1)
                ok[t] = True
    return {"cx": cx_a, "cy": cy_a, "th": th_a, "area": area, "ok": ok,
            "boxes": boxes, "shape": shape, "n": n,
            "key_t": kt, "key_iou": np.asarray(key_iou), "key_accepted": kok,
            "key_seeded": np.asarray(key_seeded, dtype=bool),
            "key_crops": crops}


def on_animal(track: dict[str, Any], centre_img: F64) -> tuple[int, int]:
    """Gate 0: how many accepted keyframes have `centre_img[t]` inside the mask.

    `centre_img` is ``(n, 2)`` image-coordinate head-disc centres. Returns
    ``(inside, scored)``.
    """
    inside = scored = 0
    for t, (x0, y0, crop) in track["key_crops"].items():
        c = centre_img[t]
        if not np.isfinite(c).all():
            continue
        scored += 1
        xi, yi = int(round(c[0])) - x0, int(round(c[1])) - y0
        if 0 <= yi < crop.shape[0] and 0 <= xi < crop.shape[1] and crop[yi, xi]:
            inside += 1
    return inside, scored


def sam_predictor(checkpoint: str, device: str = "cuda") -> Predict:
    """The real segmenter, §1: vit_b, one box, single mask, inference mode."""
    return sam_point_predictor(checkpoint, device)


def sam_point_predictor(checkpoint: str, device: str = "cuda"
                        ) -> Callable[..., tuple[B1, float]]:
    """`sam_predictor`, optionally with positive POINT prompts beside the box.

    ``predict(rgb, box, points=None)``: with `points` (``(k, 2)`` image
    coordinates) each is passed as a foreground point (label 1). With none it
    is exactly `sam_predictor`'s call, so every earlier stage is unchanged.
    """
    import torch
    from segment_anything import (  # type: ignore[import-untyped]
        SamPredictor, sam_model_registry)

    model = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device).eval()
    pr = SamPredictor(model)

    def predict(rgb: npt.NDArray[Any], box: F64,
                points: npt.ArrayLike | None = None) -> tuple[B1, float]:
        kw: dict[str, Any] = {}
        if points is not None:
            pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            kw = {"point_coords": pts, "point_labels": np.ones(pts.shape[0])}
        with torch.inference_mode():
            pr.set_image(np.ascontiguousarray(rgb))
            m, sc, _ = pr.predict(box=np.asarray(box, dtype=np.float64),
                                  multimask_output=False, **kw)
        return np.asarray(m[0], dtype=bool), float(sc[0])
    return predict


def mask_at(track: dict[str, Any], t: int, *, dilate_px: float = 0.0,
            erode_px: float = 0.0) -> B1 | None:
    """STABILISE 5 §1: SAM's mask for frame `t`, image-derived throughout.

    The nearest accepted keyframe's mask, translated by the track's own
    interpolated centroid shift from that keyframe to `t` (rounded to whole
    pixels), then dilated by `dilate_px` or eroded by `erode_px` (Amendment 1:
    eroded, so ECC never sees the boundary moving against the floor). None
    when `t` is refused, no accepted keyframe exists, or erosion empties it.
    No keypoint enters.
    """
    import cv2

    if not bool(track["ok"][t]) or not track["key_crops"]:
        return None
    keys = np.fromiter(track["key_crops"].keys(), dtype=np.int64)
    k = int(keys[np.argmin(np.abs(keys - t))])
    if not (np.isfinite(track["cx"][k]) and np.isfinite(track["cx"][t])):
        return None
    x0, y0, crop = track["key_crops"][k]
    dx = int(round(float(track["cx"][t] - track["cx"][k])))
    dy = int(round(float(track["cy"][t] - track["cy"][k])))
    h, w = track["shape"]
    out = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(crop)
    ys = ys + y0 + dy
    xs = xs + x0 + dx
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    out[ys[keep], xs[keep]] = True
    m8 = out.astype(np.uint8)
    if dilate_px > 0:
        r = int(round(dilate_px))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m8 = np.asarray(cv2.dilate(m8, ker), dtype=np.uint8)
    if erode_px > 0:
        r = int(round(erode_px))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m8 = np.asarray(cv2.erode(m8, ker), dtype=np.uint8)
    res = np.asarray(m8 > 0, dtype=bool)
    return res if res.any() else None


def _passes(m: B1, iou: float, box: F64) -> bool:
    """STABILISE 7's online rules: predicted IoU, non-empty, centroid in box."""
    if iou < IOU_MIN or not m.any():
        return False
    cx, cy, _, _ = rg.mask_pose(m)
    return bool(box[0] <= cx <= box[2] and box[1] <= cy <= box[3])


def guarded_mask(rgb: npt.NDArray[Any], pose_frame: npt.ArrayLike, bl: float,
                 predict: Callable[..., tuple[B1, float]]
                 ) -> tuple[B1 | None, dict[str, Any]]:
    """`results/POINT_PROMPTS_GUARDED.md`: each detector checks the other.

    1. SAM's mask from the padded keypoint box; refused (None) unless it passes
       STABILISE 7's online rules.
    2. Nose, centre and tail base are offered as positive point prompts, but
       only those within `GUARD_DIST_BL` body lengths of that mask -- a
       keypoint out on the arena wall is never given to SAM.
    3. The point mask replaces the box mask only if it passes the same rules
       and its area is at most `GUARD_AREA` times the box mask's.

    `predict(rgb, box, points=None)` is `sam_point_predictor`'s call. The
    recording-relative area rule is the caller's, once every keyframe is in.
    """
    import cv2

    p = np.asarray(pose_frame, dtype=np.float64)
    box = keypoint_box(p, PAD_BL * float(bl))
    info: dict[str, Any] = {"iou_box": np.nan, "iou_points": np.nan,
                            "used_points": False, "n_points": 0}
    if box is None:
        info["reason"] = "no_box"
        return None, info
    mb, iou_b = predict(rgb, box)
    mb = np.asarray(mb, dtype=bool)
    info["iou_box"] = float(iou_b)
    if not _passes(mb, float(iou_b), box):
        info["reason"] = "box_refused"
        return None, info
    dist = cv2.distanceTransform((~mb).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mb.shape
    cand = []
    for k in GUARD_PROMPTS:
        q = p[k]
        if not np.isfinite(q).all():
            continue
        x, y = int(round(float(q[0]))), int(round(float(q[1])))
        if 0 <= y < h and 0 <= x < w and dist[y, x] <= GUARD_DIST_BL * float(bl):
            cand.append(q)
    info["n_points"] = len(cand)
    if len(cand) < 2:
        return mb, info
    mp, iou_p = predict(rgb, box, points=np.asarray(cand))
    mp = np.asarray(mp, dtype=bool)
    info["iou_points"] = float(iou_p)
    if (_passes(mp, float(iou_p), box)
            and float(mp.sum()) <= GUARD_AREA * float(mb.sum())):
        info["used_points"] = True
        return mp, info
    return mb, info


def pack_crops(crops: dict[int, tuple[int, int, B1]]) -> dict[str, np.ndarray]:
    """Bit-pack ``{t: (x0, y0, bool crop)}`` for storage (STABILISE 7 layout)."""
    ts = sorted(crops)
    lens, bits, boxes = [], [], []
    for t in ts:
        x0, y0, c = crops[t]
        b = np.packbits(np.asarray(c, dtype=bool).ravel())
        bits.append(b)
        lens.append(b.size)
        boxes.append((x0, y0, c.shape[0], c.shape[1]))
    return {"mask_t": np.asarray(ts, dtype=np.int64),
            "mask_box": np.asarray(boxes, dtype=np.int64).reshape(-1, 4),
            "mask_offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            "mask_bits": (np.concatenate(bits) if bits
                          else np.zeros(0, dtype=np.uint8))}


def unpack_crops(z: Any) -> dict[int, tuple[int, int, B1]]:
    """The inverse of `pack_crops`, from a mapping with its four keys."""
    out: dict[int, tuple[int, int, B1]] = {}
    for i, t in enumerate(np.asarray(z["mask_t"])):
        x0, y0, h, w = (int(v) for v in np.asarray(z["mask_box"])[i])
        a, b = int(z["mask_offsets"][i]), int(z["mask_offsets"][i + 1])
        bits = np.asarray(z["mask_bits"])[a:b]
        out[int(t)] = (x0, y0,
                       np.unpackbits(bits)[:h * w].reshape(h, w).astype(bool))
    return out
