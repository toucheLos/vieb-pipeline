"""Phase A. What the cleaning actually changes, layer by layer.

    python3 scripts/effect.py --write-grid
    sbatch --array=0-29%12 jobs/effect.slurm
    python3 scripts/effect.py --combine

Sharded by animal, because `ell_a` -- the only unit in which two differently
sized mice are comparable -- is a per-animal constant.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits             # noqa: E402
from recur.geom import represent as rep                            # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json      # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.qc import bones, effect                                  # noqa: E402
from vieb.tok import config, ego                                   # noqa: E402

ALL_BONES = bones.SKULL + bones.TRUNK
EPS = 0.10


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def violation_mask(pose, pairs, keep, l_hat) -> np.ndarray:
    """The Step 1 skull mask at the primary eps, for the cross-tab."""
    lengths = bones.metric_lengths(pose, bones.SKULL, "raw", pairs=pairs, keep=keep)
    return bones.frame_mask(bones.violations(lengths, l_hat, EPS))


def shard(args, tag: str) -> int:
    paths = config.PATHS
    out_dir = os.path.join(os.path.dirname(paths.bones_dir), "effect")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    mine = animals_of(spine.recording_ids())[tag]
    kp = list(anchors.LUNA.keypoints)
    pairs = bones.pair_indices(anchors.LUNA.n_kept_keypoints)

    # ell_a first: one number per animal, over every recording it has.
    poses, usable = [], []
    for rid in mine:
        d = spine.clean(rid)
        poses.append(d["pose"].astype(np.float64))
        usable.append(spine.representation(rid)["usable"].astype(bool))
    ell = ego.ell_a(poses, usable)

    # Per-animal reference lengths for the skull, so the violation mask in the
    # cross-tab is the same object Step 1 reported.
    pooled = bones.metric_lengths(np.concatenate(poses), bones.SKULL, "raw",
                                  pairs=pairs)
    keep = bones.rigid_pairs(bones.log_lengths(np.concatenate(poses), pairs), pairs)
    l_hat = np.array([bones.reference_length(pooled[:, m], EPS)["l_hat"]
                      for m in range(len(bones.SKULL))])
    del pooled

    rows = []
    for n, rid in enumerate(mine):
        d = spine.clean(rid)
        raw = spine.raw_pose(rid)["pose"].astype(np.float64)
        arrays = {"raw": raw,
                  "pose_unfiltered": d["pose_unfiltered"].astype(np.float64),
                  "pose": d["pose"].astype(np.float64),
                  "pose_butterworth": d["pose_butterworth"].astype(np.float64)}
        flags = {
            "bone_flagged": d["bone_flagged"].astype(bool),
            "interpolated": d["interpolated"].astype(bool),
            "missing": d["missing"].astype(bool),
            "eps_violation": violation_mask(arrays["pose_unfiltered"], pairs,
                                            keep, l_hat),
        }
        row: dict = {"recording_id": rid, "animal": tag, "ell_a": ell,
                     "n_frames": int(raw.shape[0]), "layers": {},
                     "flag_mass": {k: float(np.asarray(v).mean())
                                   for k, v in flags.items()}}
        for layer in effect.LAYERS:
            a, b = effect.LAYER_ARRAYS[layer]
            disp = effect.displacement(arrays[a], arrays[b])
            row["layers"][layer] = {
                "all": effect.summarize(disp, ell=ell),
                "by_keypoint": effect.by_keypoint(disp, kp),
                "by_flag": effect.by_flag(disp, flags),
            }
            # Does it change behaviour, or only position?
            sb, tb = rep.raw_kinematics(arrays[a], fps, axis=ego.AXIS, causal=True)
            sa, ta = rep.raw_kinematics(arrays[b], fps, axis=ego.AXIS, causal=True)
            row["layers"][layer]["kinematics"] = effect.kinematic_shift(sb, sa, tb, ta)
        rows.append(row)
        log(f"[{tag}] {n + 1}/{len(mine)} {rid}")

    np.savez_compressed(
        os.path.join(out_dir, f"{tag}.npz"),
        recording_ids=np.array(mine), ell_a=np.array(ell),
        rows_json=np.array(json.dumps(rows)),
        inherited_digest=np.array(spine.digest()))
    log(f"[{tag}] ell_a={ell:.2f} peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _pool(rows, layer, path, weight="n_keypoint_frames"):
    """Frame-weighted mean of a nested scalar across recordings."""
    num = den = 0.0
    for r in rows:
        cur = r["layers"][layer]
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        w = float(r["layers"][layer]["all"].get(weight, 0) or 0)
        if isinstance(cur, (int, float)) and np.isfinite(cur) and w:
            num += float(cur) * w
            den += w
    return num / den if den else float("nan")


def combine(args) -> int:
    paths = config.PATHS
    out_dir = os.path.join(os.path.dirname(paths.bones_dir), "effect")
    shards = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            rows += json.loads(str(z["rows_json"]))
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    log(f"{len(shards)} shards, {len(rows)} recordings")

    kp = list(anchors.LUNA.keypoints)
    layers: dict = {}
    reads: dict = {}
    for layer in effect.LAYERS:
        med = [r["layers"][layer]["all"]["px"]["median"] for r in rows]
        p90 = [r["layers"][layer]["all"]["px"]["p90"] for r in rows]
        mx = [r["layers"][layer]["all"]["px"]["max"] for r in rows]
        bl = [r["layers"][layer]["all"].get("body_lengths", {}).get("median", np.nan)
              for r in rows]
        animals = [r["animal"] for r in rows]
        weights = [r["layers"][layer]["all"]["n_keypoint_frames"] for r in rows]
        thresholds = {
            t: boot.animal_interval(
                [r["layers"][layer]["all"]["frac_moved_over"][t] for r in rows],
                animals, weights=weights, how="wmean")
            for t in map(str, effect.THRESHOLDS_PX)}
        # `_pool` averages; an infinite concentration would poison it, so the
        # flag-level concentration is recomputed from the pooled means instead.
        flagged = {
            name: {k: _pool(rows, layer, ["by_flag", name, k])
                   for k in ("median_inside", "median_outside", "mean_inside",
                             "mean_outside", "concentration")}
            for name in ("bone_flagged", "interpolated", "missing", "eps_violation")}
        for name, cell in flagged.items():
            mo, mi = cell["median_outside"], cell["median_inside"]
            if np.isfinite(mo) and mo > 1e-12:
                cell["concentration"], cell["concentration_on"] = mi / mo, "median"
            else:
                ao, ai = cell["mean_outside"], cell["mean_inside"]
                cell["concentration_on"] = "mean"
                cell["concentration"] = (ai / ao if np.isfinite(ao) and ao > 1e-12
                                         else (float("inf") if ai > 1e-12 else float("nan")))
                # `util.write_json` nulls every non-finite float, so an infinite
                # concentration -- which MEANS "this layer touches only the
                # frames it flagged" -- would round-trip as `null` and read to a
                # consumer as "not measured". The meaning gets its own boolean.
                cell["touches_only_flagged"] = bool(np.isinf(cell["concentration"]))
                cell["n_inside"] = int(_pool(rows, layer, ["by_flag", name, "n_inside"]) or 0)
        layers[layer] = {
            "median_px": boot.animal_interval(med, animals, weights=weights, how="wmean"),
            "p90_px": boot.animal_interval(p90, animals, weights=weights, how="wmean"),
            "max_px": describe(mx),
            "median_body_lengths": boot.animal_interval(bl, animals, weights=weights,
                                                        how="wmean"),
            "frac_moved_over": thresholds,
            "by_flag": flagged,
            "by_keypoint": [
                {"keypoint": k,
                 "median_px": float(np.median(
                     [r["layers"][layer]["by_keypoint"][i]["px"]["median"]
                      for r in rows]))}
                for i, k in enumerate(kp)],
            "kinematics": {
                m: {"ratio_q": [float(np.nanmedian(
                        [r["layers"][layer]["kinematics"][m].get("ratio", [np.nan] * 4)[j]
                         for r in rows])) for j in range(4)],
                    "quantiles": [0.5, 0.9, 0.99, 0.999]}
                for m in ("speed", "turn")},
        }
        obj = {"dataset": "luna", "arm": f"effect_{layer}", "split": "all",
               "eps": EPS}
        reads[f"concentration_{layer}"] = effect.concentration_read(
            layer, {k: v for k, v in flagged["bone_flagged"].items()},
            scored_object=obj, n_effective=len(shards))

    doc = {
        **anchors.header(anchors.LUNA, stage="effect", observed={
            "n_recordings": len(rows),
            "n_frames": sum(r["n_frames"] for r in rows),
            "fps": spine.fps()}),
        "inherited_digest": spine.digest(),
        "reads": {k: v.to_dict() for k, v in reads.items()},
        "eps": EPS,
        "baseline": ("raw_pose.npz -- NOT pose_unfiltered, which is the gap "
                     "policy's output and already carries its interpolants on "
                     "~1.34% of keypoint-frames"),
        "layers": layers,
        "flag_mass": {k: float(np.mean([r["flag_mass"][k] for r in rows]))
                      for k in rows[0]["flag_mass"]},
        "ell_a": describe([r["ell_a"] for r in rows]),
        "by_split": {name: describe(
            [r["layers"]["wiener"]["all"]["px"]["median"]
             for r in rows if of.get(r["animal"]) == name])
            for name in ("tune", "fit", "report")},
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    for k, v in reads.items():
        log(v.line())
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("effect")
    with open(path, "w") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("effect.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("effect")) as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "effect")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
