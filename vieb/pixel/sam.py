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

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

#: §1, fixed.
STRIDE = 3
PAD_BL = 0.25
IOU_MIN = 0.80
AREA_RANGE = rg.AREA_RANGE

Predict = Callable[[npt.NDArray[Any], F64], tuple[B1, float]]


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
    import torch
    from segment_anything import (  # type: ignore[import-untyped]
        SamPredictor, sam_model_registry)

    model = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device).eval()
    pr = SamPredictor(model)

    def predict(rgb: npt.NDArray[Any], box: F64) -> tuple[B1, float]:
        with torch.inference_mode():
            pr.set_image(np.ascontiguousarray(rgb))
            m, sc, _ = pr.predict(box=np.asarray(box, dtype=np.float64),
                                  multimask_output=False)
        return np.asarray(m[0], dtype=bool), float(sc[0])
    return predict
