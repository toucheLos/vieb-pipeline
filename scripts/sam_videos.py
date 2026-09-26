"""Full-length overlay videos: SAM's mask and the DLC keypoints on every frame.

    sbatch jobs/sam_videos.slurm        # CPU; ten recordings
    # writes work/sam_videos/<id>.mp4, <id>.jpg and manifest.json

For a reader to judge by eye how the two detectors behave on real video.
Descriptive only: nothing here enters a gate.

**What is drawn is what STABILISE 7 used**, reconstructed from its saved
records (`work/stabilise7/rec/*.npz`), with no new SAM call:

* **SAM mask (red).** On a keyframe (every third frame), SAM's own accepted
  mask. Between keyframes, the nearest accepted keyframe's mask shifted by the
  linearly interpolated centroid of its two bracketing keyframes, which is the
  rule `sam.mask_at` applies. A frame whose bracketing keyframes were not both
  accepted was refused by the pipeline. It is shown with no mask and the word
  REFUSED.
* **DLC keypoints and skeleton** (`spine.clean` pose), in `recur`'s style.
  Missing points are not drawn.
* **Status bar:** time, frame, and whether the mask is a keyframe (with SAM's
  predicted IoU), carried, or refused.

**Selection, fixed here:** the recording SAM drifted on in STABILISE 3; the
recording STABILISE 7 refused most; the smoke-test recording; and seven more,
one per animal, drawn at seed 0 from animals not already used, alternating
contexts A and B.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.render import video as vid                                # noqa: E402
from recur.util import log                                           # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.tok import config                                          # noqa: E402

SEED = 0
N_RANDOM = 7
FIXED = ("20251117_Box_3_CFD_Day_3_(Context_A)_975",   # drifted in STABILISE 3
         "20251117_Box_2_CFD_Day_3_(Context_A)_104")   # the smoke-test recording
CRF = 28
STRIDE = 3
REC = os.path.join(config.REPO, "work", "stabilise7", "rec")
OUT = os.path.join(config.REPO, "work", "sam_videos")


def _masks(z) -> dict[int, tuple[int, int, np.ndarray]]:
    out = {}
    for i, t in enumerate(z["mask_t"]):
        x0, y0, h, w = (int(v) for v in z["mask_box"][i])
        bits = z["mask_bits"][int(z["mask_offsets"][i]):int(z["mask_offsets"][i + 1])]
        out[int(t)] = (x0, y0, np.unpackbits(bits)[:h * w].reshape(h, w).astype(bool))
    return out


def _centroid(entry) -> tuple[float, float]:
    x0, y0, m = entry
    ys, xs = np.nonzero(m)
    return float(xs.mean() + x0), float(ys.mean() + y0)


def _mask_for(t: int, masks: dict, cent: dict, shape) -> tuple[np.ndarray | None, str]:
    """(full-frame mask or None, state) for frame t, by STABILISE 7's rule."""
    k0 = (t // STRIDE) * STRIDE
    k1 = k0 + STRIDE
    if t == k0:
        if k0 not in masks:
            return None, "refused"
        k, state, (dx, dy) = k0, "keyframe", (0.0, 0.0)
    else:
        if k0 not in masks or k1 not in masks:
            return None, "refused"
        f = (t - k0) / STRIDE
        c = ((1 - f) * cent[k0][0] + f * cent[k1][0],
             (1 - f) * cent[k0][1] + f * cent[k1][1])
        k = k0 if f <= 0.5 else k1
        dx, dy = c[0] - cent[k][0], c[1] - cent[k][1]
        state = "carried"
    x0, y0, m = masks[k]
    h, w = shape
    full = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(m)
    ys = ys + y0 + int(round(dy))
    xs = xs + x0 + int(round(dx))
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    full[ys[keep], xs[keep]] = True
    return full, state


def _choose() -> list[str]:
    man = json.load(open(os.path.join(config.REPO, "work", "pixel",
                                      "manifest.json"), encoding="utf-8"))
    meta = {r["recording_id"]: r for r in man["recordings"] if r.get("in_pilot")}
    have = {f[:-4] for f in os.listdir(REC) if f.endswith(".npz")}
    refused = {}
    for rid in have:
        with np.load(os.path.join(REC, f"{rid}.npz")) as z:
            refused[rid] = float(z["nan_frac"])
    most = max(refused, key=refused.get)
    chosen = list(dict.fromkeys(list(FIXED[:1]) + [most] + list(FIXED[1:])))
    used = {meta[r]["animal"] for r in chosen}
    rng = np.random.default_rng(SEED)
    pool = sorted(have - set(chosen))
    rng.shuffle(pool)
    want = ["A", "B"]
    while len(chosen) < len(FIXED) + 1 + N_RANDOM and pool:
        ctx = want[(len(chosen)) % 2]
        pick = next((r for r in pool if meta[r]["animal"] not in used
                     and ctx in meta[r]["context"]), None)
        if pick is None:
            pick = next((r for r in pool if meta[r]["animal"] not in used), None)
        if pick is None:
            break
        chosen.append(pick)
        used.add(meta[pick]["animal"])
        pool.remove(pick)
    return chosen


def render(rid: str) -> dict:
    import cv2

    z = dict(np.load(os.path.join(REC, f"{rid}.npz")))
    masks = _masks(z)
    cent = {t: _centroid(e) for t, e in masks.items()}
    iou = np.asarray(z["key_iou"], dtype=np.float64)
    pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
    fps = spine.fps()
    cap = cv2.VideoCapture(vid.video_path(rid))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, f"{rid}.mp4")
    jpg = os.path.join(OUT, f"{rid}.jpg")
    bar = 30
    proc = subprocess.Popen(
        [vid.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h + bar}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
        stdin=subprocess.PIPE)
    counts = {"keyframe": 0, "carried": 0, "refused": 0}
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or pose.shape[0], pose.shape[0])
    poster_t = n // 10
    t = 0
    try:
        while t < n:
            ok, fr = cap.read()
            if not ok:
                break
            m, state = _mask_for(t, masks, cent, (h, w))
            counts[state] += 1
            if m is not None:
                red = np.array([40, 40, 230], dtype=np.float64)
                fr[m] = (0.62 * fr[m] + 0.38 * red).astype(np.uint8)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(fr, cnts, -1, (40, 40, 255), 1, cv2.LINE_AA)
            pts = pose[t]
            ok_pts = np.isfinite(pts).all(axis=1)
            if ok_pts.all():
                vid.draw_skeleton(fr, pts, cv2)
            else:
                for p in pts[ok_pts]:
                    cv2.circle(fr, tuple(int(v) for v in p), 3, vid.JOINT, -1,
                               cv2.LINE_AA)
            top = np.zeros((bar, w, 3), dtype=np.uint8)
            s = t / fps
            if state == "keyframe":
                label = f"SAM keyframe  IoU {iou[t // STRIDE]:.2f}"
                col = (120, 230, 120)
            elif state == "carried":
                label = "SAM mask carried from nearest keyframe"
                col = (200, 200, 200)
            else:
                label = "REFUSED by the pipeline - no mask"
                col = (60, 60, 255)
            cv2.putText(top, f"{int(s // 60)}:{s % 60:05.2f}  frame {t}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(top, label, (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 1, cv2.LINE_AA)
            out = np.vstack([top, fr])
            if t == poster_t:
                cv2.imwrite(jpg, out, [cv2.IMWRITE_JPEG_QUALITY, 82])
            assert proc.stdin is not None
            proc.stdin.write(out.tobytes())
            t += 1
    finally:
        cap.release()
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait()
    return {"recording_id": rid, "frames": t, "fps": fps,
            "seconds": round(t / fps, 1),
            "share_keyframe": round(counts["keyframe"] / max(1, t), 4),
            "share_carried": round(counts["carried"] / max(1, t), 4),
            "share_refused": round(counts["refused"] / max(1, t), 4),
            "on_animal": [int(z["gate0_inside"]), int(z["gate0_scored"])],
            "mb": round(os.path.getsize(mp4) / 1e6, 2)}


def main() -> int:
    rows = []
    for rid in _choose():
        r = render(rid)
        rows.append(r)
        log(f"  {rid}: {r}")
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "work/stabilise7/rec", "crf": CRF, "videos": rows},
                  fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
