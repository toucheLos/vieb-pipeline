"""Do SAM's mask and the DLC keypoints agree, frame by frame, and who is wrong when not?

    sbatch jobs/agreement.slurm                  # CPU; 12 shards
    python3 scripts/agreement.py --phase combine

Descriptive: nothing here enters a registered gate, and no threshold below is
tuned. They are fixed in this docstring, committed before the first run.

Inputs: STABILISE 7's saved SAM keyframe masks (`work/stabilise7/rec`, every
third frame, accepted keyframes only; SAM prompted from the padded keypoint box)
and the cleaned DLC pose (`spine.clean`). No new SAM call.

**Per keyframe, three agreement measures** (the raw, un-eroded mask):

* `kp_in`: finite keypoints inside the mask dilated by **0.05 body lengths**
  (an ear or nose tip sits on the edge), as a share of finite keypoints;
* `dist`: mask centroid to keypoint centroid, in body lengths;
* `angle`: mask major axis against the NOSE-to-TAIL_BASE line, mod 180, degrees.

**They agree** iff `kp_in >= 6/7`, `dist <= 0.25 bl` and `angle <= 30 deg`.

**When they do not, each detector is checked against itself**, never against
the other:

* keypoints *plausible*: nose-to-tail length within **0.7-1.3x** the
  recording's median, and every skull bone within **+-30%** of its median;
* mask *plausible*: area within **0.7-1.4x** the recording's median accepted
  area.

| keypoints plausible | mask plausible | verdict when they disagree |
|---|---|---|
| yes | no | **SAM suspect** |
| no | yes | **DLC suspect** |
| no | no | **both off** |
| yes | yes | **unresolved** (both look sane and still disagree) |

A keyframe with no accepted mask is **SAM refused**; one with fewer than 3
finite keypoints is **DLC missing**.

**Added after the first run (descriptive, not a threshold):** which keypoints
fall outside the mask, per keypoint, over all keyframes and over the
disagreeing ones. Inspection of example frames showed the "DLC suspect"
verdicts were mostly the head sticking out of SAM's mask, so the per-keypoint
share is the direct measure. The self-plausibility attribution above is kept
as registered in this docstring, and reported with that caveat.
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
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import register as rg                                # noqa: E402
from vieb.qc import bones as bn                                      # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

REC = os.path.join(config.REPO, "work", "stabilise7", "rec")
OUT = os.path.join(config.REPO, "work", "agreement")
STRIDE = 3
TOL_BL = 0.05
AGREE_KP = 6 / 7
AGREE_DIST_BL = 0.25
AGREE_ANGLE = 30.0
BL_RANGE = (0.7, 1.3)
SKULL_TOL = 0.30
AREA_RANGE = (0.7, 1.4)
CLASSES = ("agree", "sam_suspect", "dlc_suspect", "both_off", "unresolved",
           "sam_refused", "dlc_missing")
SEED = 0
N_BOOT = 2000


def _unpack(z) -> dict[int, tuple[int, int, np.ndarray]]:
    out = {}
    for i, t in enumerate(z["mask_t"]):
        x0, y0, h, w = (int(v) for v in z["mask_box"][i])
        bits = z["mask_bits"][int(z["mask_offsets"][i]):int(z["mask_offsets"][i + 1])]
        out[int(t)] = (x0, y0, np.unpackbits(bits)[:h * w].reshape(h, w).astype(bool))
    return out


def classify(rid: str) -> dict:
    import cv2

    z = dict(np.load(os.path.join(REC, f"{rid}.npz")))
    masks = _unpack(z)
    pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
    blen = tego.body_length(pose)
    bl = float(np.nanmedian(blen))
    skull = bn.bone_lengths(pose, bn.SKULL)
    skull_med = np.nanmedian(skull, axis=0)
    areas = np.array([m.sum() for _, _, m in masks.values()], dtype=np.float64)
    area_med = float(np.median(areas)) if areas.size else np.nan
    r = int(max(1, round(TOL_BL * bl)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    n_key = int(z["key_accepted"].size)
    keys = np.arange(n_key) * STRIDE
    keys = keys[keys < pose.shape[0]]
    cls = np.empty(keys.size, dtype=object)
    kp_in = np.full(keys.size, np.nan)
    dist = np.full(keys.size, np.nan)
    angle = np.full(keys.size, np.nan)
    outside = np.full((keys.size, pose.shape[1]), np.nan)
    for i, t in enumerate(keys):
        p = pose[t]
        fin = np.isfinite(p).all(axis=1)
        if int(fin.sum()) < 3:
            cls[i] = "dlc_missing"
            continue
        if int(t) not in masks:
            cls[i] = "sam_refused"
            continue
        x0, y0, m = masks[int(t)]
        mh, mw = m.shape
        pad = cv2.copyMakeBorder(m.astype(np.uint8), r, r, r, r,
                                 cv2.BORDER_CONSTANT, value=0)
        md = cv2.dilate(pad, ker) > 0
        q = p[fin]
        xi = np.round(q[:, 0] - x0 + r).astype(int)
        yi = np.round(q[:, 1] - y0 + r).astype(int)
        ok = (xi >= 0) & (xi < md.shape[1]) & (yi >= 0) & (yi < md.shape[0])
        inside = np.zeros(q.shape[0], dtype=bool)
        inside[ok] = md[yi[ok], xi[ok]]
        kp_in[i] = float(inside.mean())
        outside[i, np.flatnonzero(fin)] = (~inside).astype(np.float64)
        cx, cy, th, area = rg.mask_pose(m)
        dist[i] = float(np.hypot(cx + x0 - q[:, 0].mean(),
                                 cy + y0 - q[:, 1].mean())) / bl
        if np.isfinite(p[list(tego.AXIS)]).all():
            h_kp = float(tego.heading(p[None])[0])
            d = np.degrees(np.angle(np.exp(2j * (th - h_kp)))) / 2
            angle[i] = abs(d)
        agree = (kp_in[i] >= AGREE_KP - 1e-9 and dist[i] <= AGREE_DIST_BL
                 and np.isfinite(angle[i]) and angle[i] <= AGREE_ANGLE)
        if agree:
            cls[i] = "agree"
            continue
        kp_ok = (np.isfinite(blen[t])
                 and BL_RANGE[0] * bl <= blen[t] <= BL_RANGE[1] * bl
                 and np.isfinite(skull[t]).all()
                 and bool(np.all(np.abs(skull[t] / skull_med - 1) <= SKULL_TOL)))
        m_ok = AREA_RANGE[0] * area_med <= area <= AREA_RANGE[1] * area_med
        cls[i] = ("unresolved" if kp_ok and m_ok else
                  "sam_suspect" if kp_ok else
                  "dlc_suspect" if m_ok else "both_off")
    counts = {c: int((cls == c).sum()) for c in CLASSES}
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{rid}.npz"), key_t=keys,
                        cls=np.asarray([CLASSES.index(c) for c in cls]),
                        kp_in=kp_in, dist=dist, angle=angle, body_length=bl,
                        outside=outside)
    dis = np.array([c not in ("agree", "sam_refused", "dlc_missing") for c in cls])
    return {"counts": counts, "n": int(keys.size),
            "outside_all": np.nansum(outside, axis=0).tolist(),
            "scored_all": np.isfinite(outside).sum(axis=0).tolist(),
            "outside_dis": np.nansum(outside[dis], axis=0).tolist(),
            "scored_dis": np.isfinite(outside[dis]).sum(axis=0).tolist(),
            "kp_in_median": float(np.nanmedian(kp_in)),
            "dist_median": float(np.nanmedian(dist)),
            "angle_median": float(np.nanmedian(angle))}


def shard(a) -> int:
    rids = sorted(f[:-4] for f in os.listdir(REC) if f.endswith(".npz"))
    mine = rids[a.shard::a.of]
    for n, rid in enumerate(mine, 1):
        res = classify(rid)
        with open(os.path.join(OUT, f"{rid}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        log(f"  {n}/{len(mine)} {rid}: {res['counts']}")
    return 0


def combine(a) -> int:
    man = json.load(open(os.path.join(config.REPO, "work", "pixel",
                                      "manifest.json"), encoding="utf-8"))
    animal = {r["recording_id"]: r["animal"] for r in man["recordings"]}
    per: dict[str, dict[str, int]] = {}
    recs = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json"):
            continue
        rid = f[:-5]
        d = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
        recs[rid] = d
        acc = per.setdefault(animal[rid], {c: 0 for c in CLASSES} | {"n": 0})
        for c in CLASSES:
            acc[c] += d["counts"][c]
        acc["n"] += d["n"]
    res: dict = {"n_recordings": len(recs), "n_animals": len(per),
                 "n_keyframes": int(sum(v["n"] for v in per.values()))}
    for c in CLASSES:
        units = [v[c] / v["n"] for v in per.values() if v["n"]]
        b = boot.animal_interval(units, list(per), how="mean", n_boot=N_BOOT,
                                 seed=SEED)
        res[c] = b
        log(f"  {c:12s} {100 * b['point']:.2f}% [{100 * b['lo']:.2f}, "
            f"{100 * b['hi']:.2f}]")
    # Of the keyframes where both detectors produced something, how often do
    # they agree -- the share that matters for using them as two checks.
    both = {an: v["n"] - v["sam_refused"] - v["dlc_missing"]
            for an, v in per.items()}
    units = [per[an]["agree"] / both[an] for an in per if both[an] > 0]
    res["agree_given_both_present"] = boot.animal_interval(
        units, [an for an in per if both[an] > 0], how="mean", n_boot=N_BOOT,
        seed=SEED)
    b = res["agree_given_both_present"]
    log(f"  agree | both present: {100 * b['point']:.2f}% [{100 * b['lo']:.2f}, "
        f"{100 * b['hi']:.2f}]")
    names = ("left_ear", "right_ear", "nose", "center", "left_hip", "right_hip",
             "tail_base")
    oa = np.sum([recs[r]["outside_all"] for r in recs], axis=0)
    sa = np.sum([recs[r]["scored_all"] for r in recs], axis=0)
    od = np.sum([recs[r]["outside_dis"] for r in recs], axis=0)
    sd = np.sum([recs[r]["scored_dis"] for r in recs], axis=0)
    res["keypoint_outside_mask"] = {
        n: {"all_keyframes": float(oa[k] / sa[k]),
            "disagreeing_keyframes": float(od[k] / sd[k]) if sd[k] else None}
        for k, n in enumerate(names)}
    for n, v in res["keypoint_outside_mask"].items():
        log(f"  outside mask  {n:10s} all {100 * v['all_keyframes']:.2f}%  "
            f"disagreeing {100 * (v['disagreeing_keyframes'] or 0):.2f}%")
    worst = sorted(recs, key=lambda r: recs[r]["counts"]["agree"] / max(1, recs[r]["n"]))
    res["least_agreeing_recordings"] = [
        {"recording_id": r, "agree_share": recs[r]["counts"]["agree"] / recs[r]["n"],
         "counts": recs[r]["counts"]} for r in worst[:5]]
    out = a.out or config.PATHS.result("agreement.json")
    write_json({**provenance.header(anchors.LUNA, stage="agreement",
                                    unverified="a descriptive diagnostic"),
                "inherited_digest": spine.digest(),
                "thresholds": {"tol_bl": TOL_BL, "agree_kp": AGREE_KP,
                               "agree_dist_bl": AGREE_DIST_BL,
                               "agree_angle_deg": AGREE_ANGLE,
                               "bl_range": BL_RANGE, "skull_tol": SKULL_TOL,
                               "area_range": AREA_RANGE},
                "results": res}, out)
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
