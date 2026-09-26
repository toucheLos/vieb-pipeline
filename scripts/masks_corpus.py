"""Guarded SAM masks for the whole corpus: every 6th frame of all 3,846 recordings.

    sbatch jobs/masks_corpus.slurm               # 16 GPU shards, resumable

Plan Stage A1. `sam.guarded_mask` (`results/POINT_PROMPTS_GUARDED.md`): SAM from
the padded keypoint box, nose/centre/tail offered as point prompts only within
0.2 bl of that mask, and the point mask kept only if it is at most 1.3x the box
mask's area. The recording-relative area rule (0.5-2x the median over the
recording's online-accepted keyframes, STABILISE 7) is applied here once every
keyframe is in.

Writes `work/masks/<rid>.npz`: for every keyframe `key_t`, `iou_box`,
`iou_points`, `used_points`, `n_points`, `online_ok`, `accepted`, `area`; and
the accepted masks bit-packed (`sam.pack_crops`). Prompts use `spine.clean`'s
pose, as every SAM stage has.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.render import video as vid                                # noqa: E402
from recur.util import log                                           # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import sam                                           # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stabilise3 as s3                                              # noqa: E402

EVERY = 6
OUT = os.path.join(config.REPO, "work", "masks")


def run_one(rid: str, predict) -> dict:
    import cv2

    pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
    bl = float(np.nanmedian(tego.body_length(pose)))
    rows: list[dict] = []
    crops: dict[int, tuple[int, int, np.ndarray]] = {}
    cap = cv2.VideoCapture(vid.video_path(rid))
    t = 0
    try:
        while t < pose.shape[0]:
            ok, fr = cap.read()
            if not ok:
                break
            if t % EVERY == 0:
                m, info = sam.guarded_mask(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
                                           pose[t], bl, predict)
                row = {"t": t, **{k: info[k] for k in
                                  ("iou_box", "iou_points", "used_points",
                                   "n_points")},
                       "online_ok": m is not None,
                       "area": float(m.sum()) if m is not None else 0.0}
                rows.append(row)
                if m is not None:
                    ys, xs = np.nonzero(m)
                    y0, x0 = int(ys.min()), int(xs.min())
                    crops[t] = (x0, y0, m[y0:int(ys.max()) + 1,
                                          x0:int(xs.max()) + 1])
            t += 1
    finally:
        cap.release()
    area = np.array([r["area"] for r in rows])
    online = np.array([r["online_ok"] for r in rows], dtype=bool)
    med = float(np.median(area[online])) if online.any() else np.nan
    accepted = online & (area >= sam.AREA_RANGE[0] * med) & (area <= sam.AREA_RANGE[1] * med)
    kept = {r["t"]: crops[r["t"]] for r, a in zip(rows, accepted) if a}
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, f".{rid}.tmp.npz")
    np.savez_compressed(
        tmp, key_t=np.array([r["t"] for r in rows], dtype=np.int64),
        iou_box=np.array([r["iou_box"] for r in rows]),
        iou_points=np.array([r["iou_points"] for r in rows]),
        used_points=np.array([r["used_points"] for r in rows], dtype=bool),
        n_points=np.array([r["n_points"] for r in rows], dtype=np.int64),
        online_ok=online, accepted=accepted, area=area,
        body_length=bl, every=EVERY, n_frames=t, **sam.pack_crops(kept))
    os.replace(tmp, os.path.join(OUT, f"{rid}.npz"))
    return {"keyframes": len(rows), "accepted": int(accepted.sum()),
            "used_points": int(sum(r["used_points"] for r in rows))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--of", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    s3._check_checkpoint()
    predict = sam.sam_point_predictor(s3.CHECKPOINT)
    rids = sorted(spine.recording_ids())
    mine = rids[a.shard::a.of]
    if a.limit:
        mine = mine[:a.limit]
    log(f"  shard {a.shard}/{a.of}: {len(mine)} recordings")
    for n, rid in enumerate(mine, 1):
        if os.path.exists(os.path.join(OUT, f"{rid}.npz")):
            continue
        t0 = time.time()
        r = run_one(rid, predict)
        log(f"  {n}/{len(mine)} {rid}: {r} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
