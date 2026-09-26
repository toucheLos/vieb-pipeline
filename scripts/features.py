"""STATES §1: per-animal feature files -- the pose stream and the mask stream.

    python3 scripts/features.py --shard I --of N      # after ego.py on the adopted arm
    # writes work/features/<animal>.npz

Pose stream: the 17 ego channels of `work/ego/<arm>__bodylen__<animal>.npz`,
where <arm> is REPAIR's adopted arm (`results/repair.json`), `repaired` or its
registered fallback `repaired_noveto`. Mask stream: `vieb.feat.streams.mask_stream`
on `work/masks/<rid>.npz`, in the animal's own body lengths (`ell_a`). Also:
per-frame head coverage (both ears on the carried mask), the RAW ego speed
(for the leak gate's immobile frames, the census definition: raw ego speed 0),
and each recording's context and day.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import labels as lab                                      # noqa: E402
from recur.util import log                                           # noqa: E402
from vieb.feat import streams                                        # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.tok import config, quantize as qz                          # noqa: E402

OUT = os.path.join(config.REPO, "work", "features")


def adopted_arm() -> str:
    p = config.PATHS.result("repair.json")
    with open(p, encoding="utf-8") as fh:
        res = json.load(fh)["results"]
    return "repaired" if res["adopted_arm"] == "repaired" else "repaired_noveto"


def _ctx_day(rid: str) -> tuple[str, int]:
    m = re.search(r"Day_(\d+)_\(Context_([AB])", rid)
    return (m.group(2), int(m.group(1))) if m else ("?", -1)


def build(tag: str, arm: str) -> dict:
    ego_dir = os.path.join(config.REPO, "work", "ego")
    with np.load(os.path.join(ego_dir, f"{arm}__bodylen__{tag}.npz"),
                 allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        valid = z["valid"].astype(bool)
        b = z["bounds"].astype(np.int64)
        rids = [str(r) for r in z["recording_ids"]]
        ell = float(z["ell_a"])
    raw_speed = np.full(X.shape[0], np.nan, dtype=np.float32)
    raw_path = os.path.join(ego_dir, f"raw__bodylen__{tag}.npz")
    if os.path.exists(raw_path):
        with np.load(raw_path, allow_pickle=False) as z:
            if [str(r) for r in z["recording_ids"]] == rids:
                raw_speed = qz.speed(z["X"]).astype(np.float32)
    rep_pose_path = os.path.join(config.REPO, "work",
                                 "repaired" if arm == "repaired" else "repaired_noveto",
                                 f"{tag}.npz")
    with np.load(rep_pose_path, allow_pickle=False) as z:
        pose_all = z["pose"].astype(np.float64)
    fps = spine.fps()
    xm = np.full((X.shape[0], len(streams.CHANNELS)), np.nan, dtype=np.float32)
    vm = np.zeros(X.shape[0], dtype=bool)
    cov = np.zeros(X.shape[0], dtype=bool)
    dfn = np.zeros(X.shape[0], dtype=bool)
    ctx, day = [], []
    for i, rid in enumerate(rids):
        lo, hi = int(b[i]), int(b[i + 1])
        c, d = _ctx_day(rid)
        ctx.append(c)
        day.append(d)
        mpath = os.path.join(config.REPO, "work", "masks", f"{rid}.npz")
        if not os.path.exists(mpath):
            continue
        mz = dict(np.load(mpath))
        f, v = streams.mask_stream(mz, hi - lo, ell, fps)
        xm[lo:hi], vm[lo:hi] = f.astype(np.float32), v
        cv_, df_ = streams.head_coverage(mz, np.nan_to_num(pose_all[lo:hi]), ell)
        cov[lo:hi], dfn[lo:hi] = cv_, df_
    return {"X_pose": X, "valid_pose": valid, "X_mask": xm, "valid_mask": vm,
            "head_covered": cov, "head_defined": dfn, "raw_speed": raw_speed,
            "bounds": b, "recording_ids": np.array(rids), "ell_a": ell,
            "context": np.array(ctx), "day": np.array(day, dtype=np.int64),
            "pose_arm": np.array(arm)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    arm = adopted_arm()
    tags = sorted({lab.animal_tag(r) for r in spine.recording_ids()})[a.shard::a.of]
    if a.limit:
        tags = tags[:a.limit]
    os.makedirs(OUT, exist_ok=True)
    for tag in tags:
        out = os.path.join(OUT, f"{tag}.npz")
        if os.path.exists(out):
            continue
        np.savez_compressed(out, **build(tag, arm))
        log(f"  {tag}: arm {arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
