"""Evidence images for STABILISE 1-4, for a reader to check by eye.

    sbatch jobs/stabilise_figures.slurm      # needs a GPU for SAM

Writes into ``results/stabilise4/``:

* ``masks_<rid>.png`` -- six keyframes: SAM's mask (red) prompted from the
  padded keypoint box (yellow), the DLC keypoints (green) and the head-disc
  centre (blue cross), for three recordings -- the one STABILISE 3's
  propagated prompt drifted on, a typical one, and one STABILISE 4 refused.
* ``plant_<rid>.png`` -- one fast pair of gate 4's planted trajectory: the
  pasted frame, then |difference| unregistered, after SP (ECC in SAM's box),
  after K (pose warp, real jitter) and after the TRUE inverse transform
  (the oracle), on one colour scale. SP's image looking like the unregistered
  one is its gate 4 failure, seen directly.

Descriptive only: nothing here enters a gate.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.render import video as vid                                # noqa: E402
from recur.util import frames, log                                   # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import head as hd, motion as mo, register as rg, sam  # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stabilise4 as s4                                              # noqa: E402

st, s2, s3 = s4.st, s4.s2, s4.s3
OUT = os.path.join(config.REPO, "results", "stabilise4")
DRIFTED = "20251117_Box_3_CFD_Day_3_(Context_A)_975"
KEYS_SHOWN = (0, 60, 120, 180, 240, 297)


def _pick() -> list[str]:
    """The drifted recording, the median-refusal one, and the most refused."""
    rows = []
    for f in glob.glob(os.path.join(config.REPO, "work", "stabilise4", "rec",
                                    "*.npz")):
        with np.load(f) as z:
            rows.append((float(z["nan_frac"]), os.path.basename(f)[:-4]))
    rows.sort()
    return [DRIFTED, rows[len(rows) // 2][1], rows[-1][1]]


def masks_figure(rid: str, predict) -> str:
    import cv2

    pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
    bl = float(np.nanmedian(tego.body_length(pose)))
    fr = []
    cap = cv2.VideoCapture(vid.video_path(rid))
    for _ in range(max(KEYS_SHOWN) + 1):
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    shown = {}

    def rec_predict(rgb, box):
        m, iou = predict(rgb, box)
        shown[len(shown) * sam.STRIDE] = (m, box, iou)
        return m, iou
    tr = sam.sam_track(lambda: iter(fr), pose[:len(fr)], rec_predict,
                       body_length_px=bl, prompt_mode="keypoint")
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr]
    s = rg.scan_track(lambda: iter(grey), pose[:len(fr)], tr,
                      radii_px=[bl * st.HEADLINE_BL],
                      dilate_px=bl * mo.DILATE_BODY_LENGTHS,
                      win=int(frames(st.WIN_S, spine.fps())))
    tiles = []
    for t in KEYS_SHOWN:
        if t not in shown:
            continue
        m, box, iou = shown[t]
        img = fr[t].copy()
        img[m] = (0.55 * img[m] + 0.45 * np.array([230, 40, 40])).astype(np.uint8)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                      (250, 220, 0), 2)
        for q in pose[t]:
            if np.isfinite(q).all():
                cv2.circle(img, (int(q[0]), int(q[1])), 3, (40, 230, 40), -1)
        c = s["centre_head_img"][t]
        if np.isfinite(c).all():
            cv2.drawMarker(img, (int(c[0]), int(c[1])), (40, 120, 255),
                           cv2.MARKER_CROSS, 16, 2)
        acc = bool(tr["key_accepted"][t // sam.STRIDE])
        cv2.rectangle(img, (0, 0), (330, 34), (0, 0, 0), -1)
        cv2.putText(img, f"t={t}  IoU {iou:.2f}  {'kept' if acc else 'REFUSED'}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (400, 300)))
    while len(tiles) < 6:
        tiles.append(np.zeros((300, 400, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    path = os.path.join(OUT, f"masks_{rid}.png")
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return path


def plant_figure(rid: str, predict) -> str | None:
    import cv2

    gg = st._load("grooming_gate")
    pp = st._load("pixel_pilot")
    pix = st._pixel(rid)
    pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
    bl = float(np.nanmedian(tego.body_length(pose)))
    video = vid.video_path(rid)
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    speed = np.asarray(gg._speed(_animal(rid), rid), dtype=np.float64)
    still = pp._still_frames(_animal(rid), rid, int(pix["n_frames"]))
    immobile = st._immobile(pix, still)
    inp = st._plant_inputs(pose, speed, immobile, w)
    if inp is None:
        return None
    sg = st._frames_at(video, [inp["src"]])[0]
    sp = pose[inp["src"]]
    m0, _ = predict(s3._as_rgb([sg])[0], sam.keypoint_box(sp, sam.PAD_BL * bl))
    nf = min(int(pix["n_frames"]), pose.shape[0])
    idx = np.unique(np.round(np.linspace(0, nf - 1, s2.BG_SAMPLES)).astype(int))
    raw = st._frames_at(video, idx)
    bg_raw, _ = rg.masked_median_background(
        [np.asarray(f, dtype=np.float64) for f in raw],
        [pose[i] for i in idx[:len(raw)]], dilate_px=bl * mo.DILATE_BODY_LENGTHS,
        min_samples=s2.MIN_SAMPLES)
    fr, poses, Ms = st._plant_frames(sg, sp, bg_raw, m0, bl, inp, moving=True,
                                     hz=None, amp=None, fps=fps)
    step = np.hypot(*np.diff(np.asarray([M[:, 2] for M in Ms]), axis=0).T)
    t = int(np.argmax(step)) + 1                      # the fastest pair
    tr = sam.sam_track(lambda: iter(s3._as_rgb(fr)), poses, predict,
                       body_length_px=bl, prompt_mode="keypoint")
    a, b = rg.blur(fr[t - 1]), rg.blur(fr[t])
    shape = a.shape
    raw_d = np.abs(b - a)
    box = tr["boxes"][t - 1]
    sp_d = np.full(shape, np.nan)
    if box is not None:
        x0, y0, x1, y1 = box
        est = rg.rigid_between(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
        if est is not None:
            sp_d = np.abs(rg.warp_inverse(b, rg.to_frame(est[3], x0, y0), shape)
                          - a)
    wk0, wk1 = hd._ego_warp(a, poses[t - 1], shape), hd._ego_warp(b, poses[t], shape)
    k_d = np.abs(wk1 - wk0) if wk0 is not None and wk1 is not None else raw_d * np.nan
    o0 = rg.warp(a, rg.invert(Ms[t - 1]), shape)
    o1 = rg.warp(b, rg.invert(Ms[t]), shape)
    o_d = np.abs(o1 - o0)
    vmax = float(np.nanpercentile(raw_d, 99.5)) or 1.0

    def heat(d, label):
        x = np.clip(np.nan_to_num(d, nan=0.0) / vmax, 0, 1)
        im = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        im = cv2.resize(im, (400, 300))
        cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2)
        return im
    src_tile = cv2.resize(cv2.cvtColor(fr[t], cv2.COLOR_GRAY2BGR), (400, 300))
    cv2.rectangle(src_tile, (0, 0), (400, 34), (0, 0, 0), -1)
    cv2.putText(src_tile, f"planted frame t={t}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    row1 = np.hstack([src_tile, heat(raw_d, "unregistered |diff|"),
                      heat(sp_d, "SP: ECC in SAM box")])
    # Row 2: the gate-4 disc (0.2 bl) itself, zoomed, with its mean |diff| --
    # the full-frame panels saturate, and a registered-looking animal can still
    # leave most of the difference inside the disc.
    skull = sp[list(hd.SKULL)].mean(axis=0)
    r4 = s3.GATE4_BL * bl
    head_t1 = rg.apply(Ms[t - 1], skull)[0]

    def zoom(d, c, label):
        half = int(2.5 * r4)
        x, y = int(round(c[0])), int(round(c[1]))
        pad = np.pad(np.nan_to_num(d, nan=0.0), half)
        crop = pad[y:y + 2 * half, x:x + 2 * half]
        dm = rg.disc(float(c[0]), float(c[1]), r4, d.shape)
        mean = float(np.nanmean(d[dm])) if dm is not None else float("nan")
        xz = np.clip(crop / vmax, 0, 1)
        im = cv2.applyColorMap((xz * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        im = cv2.resize(im, (400, 300), interpolation=cv2.INTER_NEAREST)
        cv2.ellipse(im, (200, 150), (int(400 * r4 / (2 * half)),
                                      int(300 * r4 / (2 * half))), 0, 0, 360,
                    (80, 255, 80), 2)
        cv2.rectangle(im, (0, 0), (400, 34), (0, 0, 0), -1)
        cv2.putText(im, f"{label}: disc mean {mean:.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        return im
    row2 = np.hstack([zoom(raw_d, head_t1, "unregistered"),
                      zoom(sp_d, head_t1, "SP"),
                      zoom(o_d, skull, "oracle")])
    row3 = np.hstack([heat(k_d, "K: pose warp (body frame)"),
                      heat(o_d, "oracle: true transform"),
                      np.zeros((300, 400, 3), np.uint8)])
    path = os.path.join(OUT, f"plant_{rid}.png")
    cv2.imwrite(path, np.vstack([row1, row2, row3]))
    return path


def _animal(rid: str) -> str:
    for r in st.manifest()["recordings"]:
        if r["recording_id"] == rid:
            return r["animal"]
    raise KeyError(rid)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    s3._check_checkpoint()
    predict = sam.sam_predictor(s3.CHECKPOINT)
    for rid in _pick():
        log(f"  masks {masks_figure(rid, predict)}")
    for rid in _pick()[:2]:
        log(f"  plant {plant_figure(rid, predict)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
