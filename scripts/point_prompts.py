"""Does prompting SAM with keypoints as POINTS stop it cutting off the head?

    sbatch jobs/point_prompts.slurm              # GPU; 12 shards
    python3 scripts/point_prompts.py --phase combine

Descriptive; every metric below is fixed in this docstring and committed before
the first run. `results/AGREEMENT.md` §2 found SAM's box-prompted mask leaves the
nose and ears outside on 13-16% of keyframes and the body centre on 0.5%.

**The two prompts, compared on the same frames** (every 30th frame of all 300
STABILISE 7 recordings, all of which are STABILISE 7 keyframes):

* **box**: STABILISE 7's saved mask (padded keypoint box), no new SAM call;
* **points**: the same box **plus the NOSE, CENTER and TAIL_BASE keypoints as
  positive points** (label 1), run here.

Each is accepted by STABILISE 7's own rules: predicted IoU >= 0.80, mask
centroid inside the padded box, and area within 0.5-2x the recording's median
over its accepted sampled frames.

**Scored on keypoints SAM was NOT given.** Prompting SAM with a keypoint and
then asking whether that keypoint is inside the mask is circular, so:

* **Primary:** the share of the two EARS (never prompted) outside the mask,
  0.05 bl edge tolerance, on frames where **both** variants are accepted.
  Paired per animal: box minus points, with an animal-level bootstrap (2,000
  replicates, seed 0).
* **Secondary, also unprompted:** left and right hip.
* **Reported and flagged as circular:** nose, centre and tail base.
* **Over-inclusion guard:** mask area / body length^2 (median and 95th
  percentile), and the share of frames each variant has refused. A point
  prompt that pulls in glare or the arena shows up here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot                                      # noqa: E402
from recur.render import video as vid                                # noqa: E402
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import register as rg, sam                           # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agreement as ag                                               # noqa: E402
import stabilise3 as s3                                              # noqa: E402

REC = os.path.join(config.REPO, "work", "stabilise7", "rec")
OUT = os.path.join(config.REPO, "work", "point_prompts")
EVERY = 30
PROMPTS = (tego.NOSE, tego.CENTER, tego.TAIL_BASE)
EARS = (tego.LEFT_EAR, tego.RIGHT_EAR)
HIPS = (tego.LEFT_HIP, tego.RIGHT_HIP)
NAMES = ("left_ear", "right_ear", "nose", "center", "left_hip", "right_hip",
         "tail_base")
TOL_BL = 0.05
SEED = 0
N_BOOT = 2000


def _outside(m_full: np.ndarray, pts: np.ndarray, tol_px: int) -> np.ndarray:
    import cv2

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (2 * tol_px + 1, 2 * tol_px + 1))
    md = cv2.dilate(m_full.astype(np.uint8), ker) > 0
    h, w = md.shape
    out = np.full(pts.shape[0], np.nan)
    for k, q in enumerate(pts):
        if not np.isfinite(q).all():
            continue
        x, y = int(round(q[0])), int(round(q[1]))
        out[k] = float(not (0 <= y < h and 0 <= x < w and md[y, x]))
    return out


def _accept(masks: dict, boxes: dict, ious: dict) -> set[int]:
    """STABILISE 7's rules: IoU, centroid in box, area vs the median."""
    ok = [t for t in masks
          if ious[t] >= sam.IOU_MIN and masks[t].any()
          and boxes[t][0] <= rg.mask_pose(masks[t])[0] <= boxes[t][2]
          and boxes[t][1] <= rg.mask_pose(masks[t])[1] <= boxes[t][3]]
    if not ok:
        return set()
    med = float(np.median([masks[t].sum() for t in ok]))
    return {t for t in ok
            if sam.AREA_RANGE[0] * med <= masks[t].sum() <= sam.AREA_RANGE[1] * med}


def run_one(rid: str, predict) -> dict:
    import cv2

    z = dict(np.load(os.path.join(REC, f"{rid}.npz")))
    saved = ag._unpack(z)
    pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
    bl = float(np.nanmedian(tego.body_length(pose)))
    tol = int(max(1, round(TOL_BL * bl)))
    pad = sam.PAD_BL * bl
    cap = cv2.VideoCapture(vid.video_path(rid))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pm: dict[int, np.ndarray] = {}
    pb: dict[int, np.ndarray] = {}
    pi: dict[int, float] = {}
    t = 0
    try:
        while t < pose.shape[0]:
            ok, fr = cap.read()
            if not ok:
                break
            if t % EVERY == 0:
                p = pose[t]
                box = sam.keypoint_box(p, pad)
                pts = p[list(PROMPTS)]
                pts = pts[np.isfinite(pts).all(axis=1)]
                if box is not None and pts.shape[0] >= 2:
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    m, iou = predict(rgb, box, points=pts)
                    pm[t], pb[t], pi[t] = np.asarray(m, bool), box, float(iou)
            t += 1
    finally:
        cap.release()
    acc_p = _accept(pm, pb, pi)
    rows = []
    for t in sorted(pm):
        full_b = None
        if t in saved:
            x0, y0, mc = saved[t]
            full_b = np.zeros((h, w), dtype=bool)
            full_b[y0:y0 + mc.shape[0], x0:x0 + mc.shape[1]] = mc
        row = {"t": t, "box_ok": full_b is not None, "pt_ok": t in acc_p,
               "area_box": float(full_b.sum()) / bl ** 2 if full_b is not None else np.nan,
               "area_pt": float(pm[t].sum()) / bl ** 2}
        if full_b is not None:
            row["out_box"] = _outside(full_b, pose[t], tol).tolist()
        row["out_pt"] = _outside(pm[t], pose[t], tol).tolist()
        rows.append(row)
    return {"recording_id": rid, "body_length": bl, "rows": rows}


def shard(a) -> int:
    s3._check_checkpoint()
    predict = sam.sam_point_predictor(s3.CHECKPOINT)
    os.makedirs(OUT, exist_ok=True)
    rids = sorted(f[:-4] for f in os.listdir(REC) if f.endswith(".npz"))
    mine = rids[a.shard::a.of]
    for n, rid in enumerate(mine, 1):
        path = os.path.join(OUT, f"{rid}.json")
        if os.path.exists(path):
            continue
        res = run_one(rid, predict)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        both = [r for r in res["rows"] if r["box_ok"] and r["pt_ok"]]
        eb = np.nanmean([np.nanmean([r["out_box"][k] for k in EARS]) for r in both]) if both else np.nan
        ep = np.nanmean([np.nanmean([r["out_pt"][k] for k in EARS]) for r in both]) if both else np.nan
        log(f"  {n}/{len(mine)} {rid}: {len(both)} paired frames, ears outside "
            f"box {eb:.3f} -> points {ep:.3f}")
    return 0


def combine(a) -> int:
    man = json.load(open(os.path.join(config.REPO, "work", "pixel",
                                      "manifest.json"), encoding="utf-8"))
    animal = {r["recording_id"]: r["animal"] for r in man["recordings"]}
    per: dict[str, dict[str, list]] = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
        acc = per.setdefault(animal[d["recording_id"]],
                             {"box": [], "pt": [], "area_box": [], "area_pt": [],
                              "n": 0, "box_refused": 0, "pt_refused": 0})
        for r in d["rows"]:
            acc["n"] += 1
            acc["box_refused"] += int(not r["box_ok"])
            acc["pt_refused"] += int(not r["pt_ok"])
            if r["box_ok"] and r["pt_ok"]:
                acc["box"].append(r["out_box"])
                acc["pt"].append(r["out_pt"])
                acc["area_box"].append(r["area_box"])
                acc["area_pt"].append(r["area_pt"])
    res: dict = {"every": EVERY, "prompts": [NAMES[k] for k in PROMPTS]}

    def interval(vals: dict[str, float]) -> dict:
        v = {k: x for k, x in vals.items() if np.isfinite(x)}
        return boot.animal_interval(list(v.values()), list(v), how="mean",
                                    n_boot=N_BOOT, seed=SEED)

    keep = {an: v for an, v in per.items() if len(v["box"]) >= 10}
    for label, ks in (("ears", EARS), ("hips", HIPS)):
        for arm in ("box", "pt"):
            res[f"{label}_outside_{arm}"] = interval(
                {an: float(np.nanmean(np.asarray(v[arm])[:, list(ks)]))
                 for an, v in keep.items()})
        res[f"{label}_outside_box_minus_points"] = interval(
            {an: float(np.nanmean(np.asarray(v["box"])[:, list(ks)])
                       - np.nanmean(np.asarray(v["pt"])[:, list(ks)]))
             for an, v in keep.items()})
    res["per_keypoint_outside"] = {
        NAMES[k]: {arm: float(np.nanmean(np.concatenate(
            [np.asarray(v[arm])[:, k] for v in keep.values()])))
            for arm in ("box", "pt")} | {"circular": k in PROMPTS}
        for k in range(len(NAMES))}
    for arm in ("box", "pt"):
        allv = np.concatenate([v[f"area_{arm}"] for v in keep.values()])
        res[f"area_over_bl2_{arm}"] = {"median": float(np.nanmedian(allv)),
                                       "p95": float(np.nanpercentile(allv, 95))}
        res[f"refused_{arm}"] = interval(
            {an: v[f"{arm}_refused"] / v["n"] for an, v in per.items() if v["n"]})
    res["n_animals"] = len(keep)
    res["n_paired_frames"] = int(sum(len(v["box"]) for v in keep.values()))
    for k, v in res.items():
        if isinstance(v, dict) and "point" in v:
            log(f"  {k:32s} {v['point']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
    log(f"  per keypoint: {json.dumps(res['per_keypoint_outside'])}")
    log(f"  area/bl^2 box {res['area_over_bl2_box']}  points {res['area_over_bl2_pt']}")
    out = a.out or config.PATHS.result("point_prompts.json")
    write_json({**provenance.header(anchors.LUNA, stage="point_prompts",
                                    unverified="a descriptive comparison"),
                "inherited_digest": spine.digest(),
                "checkpoint_sha256": s3.CKPT_SHA256, "results": res}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("combine",), default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.phase == "combine":
        return combine(a)
    if a.shard is None:
        raise SystemExit("pass --shard I --of N, or --phase combine")
    return shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
