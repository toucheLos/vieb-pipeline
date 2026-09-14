"""Step 1. The bone-length violation sweep, and the ExBias R^2 join. BLOCKING.

    python3 scripts/bones.py --write-grid
    sbatch --array=0-29%12 jobs/bones.slurm
    python3 scripts/bones.py --combine

The grid is emitted here rather than by `scripts/grid.py`, which is Stage 4's
sweep and knows about windows and nulls. This one is a list of animal tags, and
the two have nothing in common but the file format.

Sharded by **animal**, not by recording, because the reference length is fitted
per animal per bone on that animal's own frames. A corpus-wide reference would
flag small mice everywhere -- a systematic exclusion correlated with the animal,
which is the shape of a confound rather than of a quality gate. shapeflow's own
cleaning stage shards by animal for the same reason.

Nothing downstream of this is interpretable until it lands. A tracking artifact
is a short, highly stereotyped excursion with predictable duration, which is
exactly what a cause-specific hazard model rewards.
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

from recur import anchors, boot, labels as lab, splits            # noqa: E402
from vieb.io import spine                                        # noqa: E402
from recur.qc import swap
from vieb import seeds
from vieb.qc import bones                                  # noqa: E402
from vieb.tok import config                                      # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json     # noqa: E402

ALL_BONES = bones.SKULL + bones.TRUNK
N_SKULL = len(bones.SKULL)
GROUPS = {"skull": list(range(N_SKULL)),
          "trunk": list(range(N_SKULL, len(ALL_BONES)))}


def animals_of(ids) -> dict:
    """``{animal_tag: [recording_id]}``, in the corpus's own order."""
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def segment_join(rid: str, mask, paths: config.Paths) -> dict | None:
    """Per-segment violation rate against that segment's ExBias R^2.

    `fit_bounds` rather than `bounds`: it is the interval the coefficients and
    the R^2 were actually computed on, with ExBias's two-frame guard excluding
    the genuine transition frames at either end. Joining the rate to a wider
    interval than the R^2 describes would blur the very association being tested.
    """
    path = paths.segments(rid)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if int(z["n_frames"]) != int(mask.shape[0]):
            # Frame counts must agree exactly. They do on this corpus; a
            # mismatch means the two repos are looking at different ingests and
            # the join would silently index one recording's segments into
            # another's frames.
            return {"skipped": "frame count mismatch",
                    "exbias_n_frames": int(z["n_frames"]),
                    "clean_n_frames": int(mask.shape[0])}
        return {
            "rate": bones.segment_rates(mask, z["fit_bounds"]).tolist(),
            "fit_r2": z["fit_r2"].astype(np.float64).tolist(),
            "fit_r2_adj": z["fit_r2_adj"].astype(np.float64).tolist(),
            "durations": z["durations"].astype(np.float64).tolist(),
        }


def shard(args) -> int:
    """One animal, end to end.

    An animal takes ~3 s, so an array task takes a STRIDED SLICE of the grid
    rather than one line: 298 tasks of three seconds each is mostly scheduler
    overhead, and strided rather than contiguous so a task's wall clock does not
    depend on whether its block happened to hold the long recordings.
    """
    paths = config.PATHS
    swap.check_order(anchors.LUNA.keypoints)
    os.makedirs(paths.bones_dir, exist_ok=True)

    ids = spine.recording_ids()
    by_animal = animals_of(ids)
    if args.animal not in by_animal:
        raise SystemExit(f"{args.animal!r} is not an animal in this corpus")
    mine = by_animal[args.animal]
    pairs = bones.pair_indices(anchors.LUNA.n_kept_keypoints)

    # ---- pass 1: pooled lengths, for the per-animal reference -------------
    pose_by_rec: dict = {}
    flags_by_rec: dict = {}
    for rid in mine:
        d = spine.clean(rid)
        pose_by_rec[rid] = {a: d[config.POSE_KEY[a]].astype(np.float64)
                            for a in config.POSE_ARMS}
        flags_by_rec[rid] = {
            "bone_flagged": d["bone_flagged"].astype(bool),
            "interpolated": d["interpolated"].any(axis=1),
            "missing": d["missing"].any(axis=1),
        }
    n_frames = {rid: pose_by_rec[rid]["unfiltered"].shape[0] for rid in mine}

    keep: dict = {}
    pooled: dict = {}
    for arm in config.POSE_ARMS:
        stacked = np.concatenate([pose_by_rec[r][arm] for r in mine], axis=0)
        # The rigid half of all 21 pairs, from this animal's own frames. Used
        # only to estimate the per-frame common scale for the `scalefree` metric.
        keep[arm] = bones.rigid_pairs(bones.log_lengths(stacked, pairs), pairs)
        for metric in bones.METRICS:
            pooled[(arm, metric)] = bones.metric_lengths(
                stacked, ALL_BONES, metric, pairs=pairs, keep=keep[arm])

    # l_hat[arm][metric][eps] -> one reference per bone. eps enters the
    # reference because the trimming set is defined by it.
    l_hat: dict = {}
    ref_detail: dict = {}
    for (arm, metric), lengths in pooled.items():
        for eps in bones.EPS:
            refs, det = [], []
            for m in range(len(ALL_BONES)):
                r = bones.reference_length(lengths[:, m], eps)
                refs.append(r["l_hat"])
                det.append(r)
            l_hat[(arm, metric, eps)] = np.asarray(refs, dtype=np.float64)
            ref_detail[f"{arm}|{metric}|{eps}"] = det
    del pooled

    # ---- pass 2: per recording -------------------------------------------
    # `abs(hash(args.animal))` here was salted per interpreter, so WHICH
    # recordings entered the shuffled-keypoint ceiling changed on every run.
    # See SEED_AUDIT.md for the blast radius.
    rng = np.random.default_rng(
        [config.CEILING_SEED, seeds.stable_seed(args.animal, modulus=2 ** 31)])
    ceiling_for = set(rng.choice(len(mine), size=min(config.CEILING_PER_ANIMAL,
                                                     len(mine)),
                                 replace=False).tolist())
    rows = []
    masks: dict = {(arm, metric, eps, g): []
                   for arm in config.POSE_ARMS for metric in bones.METRICS
                   for eps in bones.EPS for g in GROUPS}
    for n, rid in enumerate(mine):
        row: dict = {"recording_id": rid, "animal": args.animal,
                     "n_frames": int(n_frames[rid]), "cells": {}, "join": {}}
        for arm in config.POSE_ARMS:
            pose = pose_by_rec[rid][arm]
            for metric in bones.METRICS:
                lengths = bones.metric_lengths(pose, ALL_BONES, metric,
                                               pairs=pairs, keep=keep[arm])
                for eps in bones.EPS:
                    viol = bones.violations(lengths, l_hat[(arm, metric, eps)], eps)
                    for g, cols in GROUPS.items():
                        m = bones.frame_mask(viol[:, cols])
                        masks[(arm, metric, eps, g)].append(m)
                        cell = {"rate": float(m.mean())}
                        if g == "skull":
                            cell["attribution"] = bones.attribution(
                                viol[:, cols], bones.SKULL,
                                anchors.LUNA.n_kept_keypoints)
                            for other, om in flags_by_rec[rid].items():
                                cell[f"overlap_{other}"] = bones.overlap(
                                    m, om, other_name=other)
                            # Excess length on frames that were MEASURED, not
                            # filled. The cell with no benign explanation.
                            measured = ~(flags_by_rec[rid]["interpolated"]
                                         | flags_by_rec[rid]["missing"])
                            cell["rate_new_only_measured"] = float(
                                (m & measured & ~flags_by_rec[rid]["bone_flagged"]).mean())
                        if n in ceiling_for and arm == "unfiltered":
                            cell["shuffled_ceiling"] = bones.shuffled_ceiling(
                                pose, [ALL_BONES[c] for c in cols],
                                l_hat[(arm, metric, eps)][cols], eps,
                                np.random.default_rng([config.CEILING_SEED, n]),
                                metric=metric, pairs=pairs, keep=keep[arm])
                        row["cells"][f"{arm}|{metric}|{eps}|{g}"] = cell
        # The join runs on the PRIMARY cell only: unfiltered pose, raw metric,
        # skull group. Every eps, because the branch may move the cut.
        for eps in bones.EPS:
            j = segment_join(
                rid, masks[("unfiltered", "raw", eps, "skull")][-1], paths)
            if j is not None:
                row["join"][str(eps)] = j
        rows.append(row)
        log(f"[{args.animal}] {n + 1}/{len(mine)} {rid}")

    store = {f"viol|{arm}|{metric}|{eps}|{g}":
             np.packbits(np.concatenate(v))
             for (arm, metric, eps, g), v in masks.items()}
    bounds = np.concatenate([[0], np.cumsum([n_frames[r] for r in mine])])
    np.savez_compressed(
        paths.bones_shard(args.animal),
        recording_ids=np.array(mine), bounds=bounds.astype(np.int64),
        eps=np.asarray(bones.EPS), metrics=np.array(bones.METRICS),
        pose_arms=np.array(config.POSE_ARMS),
        bone_pairs=np.asarray(ALL_BONES, dtype=np.int64), n_skull=N_SKULL,
        keep_pairs=np.array([np.isin(np.arange(len(pairs)), keep[a])
                             for a in config.POSE_ARMS]),
        l_hat=np.stack([l_hat[(a, m, e)] for a in config.POSE_ARMS
                        for m in bones.METRICS for e in bones.EPS]),
        rows_json=np.array(json.dumps({"rows": rows, "reference": ref_detail})),
        inherited_digest=np.array(spine.digest()), **store)
    log(f"wrote {paths.bones_shard(args.animal)}  peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _cells(rows, key: str, field: str = "rate"):
    """One value per recording for a cell key, with its frame weight.

    A recording missing the field is dropped rather than filled: only a seeded
    handful per animal carries a `shuffled_ceiling`, and a zero there would
    average a ceiling that was never computed into one that was.
    """
    have = [r for r in rows if field in r["cells"].get(key, {})]
    return (np.asarray([r["cells"][key][field] for r in have], dtype=np.float64),
            np.asarray([r["n_frames"] for r in have], dtype=np.float64))


def _pooled(rows, key: str, field: str = "rate") -> float:
    v, w = _cells(rows, key, field)
    return float((v * w).sum() / w.sum()) if w.sum() else float("nan")


def _overlap_totals(rows, key: str, other: str) -> dict:
    """The 2x2 summed over recordings, then re-derived as rates.

    Summed rather than averaged: a rate-of-rates weights a 4,000-frame recording
    like a 6,000-frame one, and the cells have to stay a partition of the frames
    for the marginals to reconcile.
    """
    tot = {k: 0 for k in ("both", "new_only", "other_only", "neither", "n_frames")}
    for r in rows:
        o = r["cells"].get(key, {}).get(f"overlap_{other}")
        if o:
            for k in tot:
                tot[k] += int(o[k])
    n = tot["n_frames"] or 1
    union = tot["both"] + tot["new_only"] + tot["other_only"]
    new = tot["both"] + tot["new_only"]
    oth = tot["both"] + tot["other_only"]
    return {
        "other_name": other, **tot,
        "rate_both": tot["both"] / n, "rate_new_only": tot["new_only"] / n,
        "rate_other_only": tot["other_only"] / n,
        "rate_new": new / n, "rate_other": oth / n,
        "jaccard": tot["both"] / union if union else float("nan"),
        "p_other_given_new": tot["both"] / new if new else float("nan"),
        "p_new_given_other": tot["both"] / oth if oth else float("nan"),
    }


def combine(args) -> int:
    paths = config.PATHS
    anchor = anchors.LUNA
    shards = sorted(glob.glob(os.path.join(paths.bones_dir, "*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {paths.bones_dir}")

    rows: list = []
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            rows += json.loads(str(z["rows_json"]))["rows"]
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    log(f"{len(shards)} shards, {len(rows)} recordings")

    # ---- the eps curves ---------------------------------------------------
    curves: dict = {}
    for arm in config.POSE_ARMS:
        for metric in bones.METRICS:
            for g in GROUPS:
                cells = []
                for eps in bones.EPS:
                    key = f"{arm}|{metric}|{eps}|{g}"
                    rate = _pooled(rows, key)
                    per_rec, _ = _cells(rows, key)
                    ceil_v, _ = _cells(rows, key, "shuffled_ceiling")
                    cells.append({
                        "eps": eps, "rate": rate,
                        "rate_p50": float(np.median(per_rec)) if per_rec.size else float("nan"),
                        "rate_p95": float(np.quantile(per_rec, 0.95)) if per_rec.size else float("nan"),
                        "n_recordings_above_1pct": int((per_rec > 0.01).sum()),
                        "n_recordings": int(per_rec.size),
                        "shuffled_ceiling": float(np.mean(ceil_v)) if ceil_v.size else float("nan"),
                        "n_ceiling": int(ceil_v.size),
                    })
                curves[f"{arm}|{metric}|{g}"] = cells

    # ---- the 2x2, per (eps, metric), on the primary pose arm --------------
    overlaps = {
        f"{metric}|{eps}|{other}": _overlap_totals(
            rows, f"unfiltered|{metric}|{eps}|skull", other)
        for metric in bones.METRICS for eps in bones.EPS
        for other in ("bone_flagged", "interpolated", "missing")}
    measured_only = {
        f"{metric}|{eps}": _pooled(rows, f"unfiltered|{metric}|{eps}|skull",
                                   "rate_new_only_measured")
        for metric in bones.METRICS for eps in bones.EPS}

    # ---- attribution, pooled ---------------------------------------------
    attribution = {}
    for metric in bones.METRICS:
        for eps in bones.EPS:
            key = f"unfiltered|{metric}|{eps}|skull"
            share = np.zeros(anchor.n_kept_keypoints)
            for r in rows:
                a = r["cells"].get(key, {}).get("attribution")
                if a:
                    share += np.asarray(a["share"]) * a["n_violating_bone_frames"]
            tot = share.sum()
            attribution[f"{metric}|{eps}"] = {
                "keypoints": list(anchor.keypoints),
                "share": (share / tot).tolist() if tot else share.tolist()}

    # ---- the join ---------------------------------------------------------
    join_by_eps: dict = {}
    reads: dict = {}
    for eps in bones.EPS:
        rate, r2, r2a, dur, animals = [], [], [], [], []
        skipped = 0
        for r in rows:
            j = r["join"].get(str(eps))
            if not j or "skipped" in j:
                skipped += 1 if j else 0
                continue
            rate += j["rate"]
            r2 += j["fit_r2"]
            r2a += j["fit_r2_adj"]
            dur += j["durations"]
            animals += [r["animal"]] * len(j["rate"])
        rate, r2 = np.asarray(rate), np.asarray(r2)
        r2a, dur = np.asarray(r2a), np.asarray(dur)
        if rate.size == 0:
            join_by_eps[str(eps)] = {"n_segments": 0, "skipped": skipped}
            continue
        rhos, kept = bones.per_animal_spearman(
            rate, r2, animals, min_segments=config.MIN_SEGMENTS_PER_ANIMAL)
        ci = boot.animal_interval(rhos, kept, how="mean")
        # The same correlation on the duration-free statistic. `rate` divides by
        # the segment's own length, so it is coarse on short segments and its
        # rho mixes the association with whatever duration does across R^2. The
        # binary indicator does not, and on this corpus the two disagree.
        any_rhos, any_kept = bones.per_animal_spearman(
            (rate > 0).astype(np.float64), r2, animals,
            min_segments=config.MIN_SEGMENTS_PER_ANIMAL)
        any_ci = boot.animal_interval(any_rhos, any_kept, how="mean")
        dur_rhos, dur_kept = bones.per_animal_spearman(
            dur, r2, animals, min_segments=config.MIN_SEGMENTS_PER_ANIMAL)
        # The number the branch is actually read on. ExBias's R^2 is largely a
        # readout of segment length, and the violation rate's denominator IS the
        # segment length, so the marginal rho is confounded on both sides.
        part_rhos, part_kept = bones.per_animal_partial_spearman(
            rate, r2, dur, animals,
            min_segments=config.MIN_SEGMENTS_PER_ANIMAL)
        part_ci = boot.animal_interval(part_rhos, part_kept, how="mean")
        join_by_eps[str(eps)] = {
            "n_segments": int(rate.size), "n_recordings_skipped": skipped,
            "mean_fit_r2": float(np.mean(r2)),
            "mean_duration_s": float(np.nanmean(dur)),
            "frac_fit_r2_below_0.5": float((r2 < 0.5).mean()),
            "frac_fit_r2_adj_below_0.5": float((r2a < 0.5).mean()),
            "deciles": bones.r2_deciles(rate, r2, durations=dur),
            "deciles_adjusted": bones.r2_deciles(rate, r2a, durations=dur),
            "rho": ci, "n_animals_correlated": len(kept),
            "rho_partial_duration_controlled": part_ci,
            "rho_any_violation": any_ci,
            "rho_duration_vs_r2": boot.animal_interval(dur_rhos, dur_kept,
                                                       how="mean"),
        }

    primary_eps = 0.10
    pj = join_by_eps.get(str(primary_eps), {})
    obj = {"dataset": "luna", "arm": "bones", "split": "all",
           "pose": "unfiltered", "eps": primary_eps}
    join_read = bones.join_read(
        pj.get("rho", {"point": float("nan")}), pj.get("deciles", {}),
        scored_object={**obj, "null": "exbias_fit_r2"},
        n_effective=pj.get("n_animals_correlated", 1) or 1,
        partial_ci=pj.get("rho_partial_duration_controlled"))
    reads["r2_join"] = join_read

    for metric in bones.METRICS:
        for g in GROUPS:
            reads[f"{g}_{metric}"] = bones.curve_read(
                curves[f"unfiltered|{metric}|{g}"], group=g, metric=metric,
                scored_object={**obj, "metric": metric, "group": g},
                n_effective=len(shards),
                join=join_read if metric == "raw" else None)

    head = curves["unfiltered|raw|skull"][bones.EPS.index(primary_eps)]
    reads["shuffled_ceiling"] = bones.ceiling_read(
        head["rate"], head["shuffled_ceiling"],
        scored_object={**obj, "metric": "raw", "group": "skull"},
        n_effective=head["n_ceiling"] or 1)

    by_split = {name: describe([
        r["cells"][f"unfiltered|raw|{primary_eps}|skull"]["rate"]
        for r in rows if of.get(r["animal"]) == name])
        for name in ("tune", "fit", "report")}
    worst = sorted(rows, key=lambda r: -r["cells"][
        f"unfiltered|raw|{primary_eps}|skull"]["rate"])[:25]

    doc = {
        **anchors.header(anchor, stage="bones", observed={
            "n_recordings": len(rows),
            "n_frames": sum(r["n_frames"] for r in rows),
            "fps": spine.fps()}),
        "inherited_digest": spine.digest(),
        "reads": {k: v.to_dict() for k, v in reads.items()},
        "primary": {"pose_arm": "unfiltered", "metric": "raw",
                    "group": "skull", "eps": primary_eps,
                    "why": ("unfiltered because it is the measurement and the "
                            "array shapeflow's own bone_flagged was computed "
                            "on, which is what makes the 2x2 like-for-like; "
                            "skull because trunk bones flex")},
        "curves": curves,
        "overlap": overlaps,
        "rate_new_only_on_measured_frames": measured_only,
        "attribution": attribution,
        "r2_join": join_by_eps,
        "by_split": by_split,
        "worst_recordings": [
            {"recording_id": r["recording_id"], "animal": r["animal"],
             "rate": r["cells"][f"unfiltered|raw|{primary_eps}|skull"]["rate"]}
            for r in worst],
        "bones": {"skull": [list(b) for b in bones.SKULL],
                  "trunk": [list(b) for b in bones.TRUNK],
                  "keypoints": list(anchor.keypoints)},
        "eps_swept": list(bones.EPS),
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    for name, rd in reads.items():
        log(f"{name}: {rd.line()}")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    """One animal tag per line, sorted, so `--array=17` is always the same animal."""
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("bones")
    with open(path, "w") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def slice_of_grid(task: int, n_tasks: int) -> list:
    path = config.PATHS.grid("bones")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run `--write-grid` first")
    with open(path) as fh:
        tags = [ln.strip() for ln in fh if ln.strip()]
    return tags[task::n_tasks]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default=None,
                   help="one animal_tag; overrides --task/--n-tasks")
    p.add_argument("--task", type=int, default=None,
                   help="array task index; takes a strided slice of the grid")
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="redo shards that already exist")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("bones.json")
        return combine(a)

    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        tags = slice_of_grid(a.task, a.n_tasks)
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")

    for tag in tags:
        if not a.force and os.path.exists(config.PATHS.bones_shard(tag)):
            log(f"[{tag}] shard exists, skipping")
            continue
        a.animal = tag
        shard(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
