r"""Combine Step A's shards into the gate: did the distance make the clumps?

Reuses `combine_recur.paired` -- Q1's `paired_excess` imported from
`recur/scripts/q1.py` -- so the excess under a new metric is computed by the same
code object as the excess under the old one. A second implementation could
disagree with the published number, and then a metric effect and an
implementation difference would be indistinguishable.

The clump stage is `vocab.components` at the same theta derivation, so a change
in clump count is a change in the distance and not in the clusterer.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np
from recur import anchors
from recur.read import Read
from recur.util import log, write_json

from vieb.io import spine
from vieb.seg import combine_recur as cr, dist as ds, recur as rc, vocab as vb

Detail = dict[str, Any]

__all__ = ["main", "metric_read"]

#: The time-normalised numbers Step A is read against, from `seg_recur.json` and
#: `seg_vocab.json`. Quoted, never recomputed.
WARP = "time-normalised"


def metric_read(now: Detail, before: Detail, *, group: str, metric: str,
                scored_object: Detail, n_effective: int) -> Read:
    """Did the excess and the clumps survive the change of ruler?

    Three outcomes, registered before either metric ran. The one that ends the
    route is the excess collapsing: segment recurrence would then have been a
    property of the warp, and there is nothing downstream worth doing.
    """
    ci = now.get("excess")
    if not ci:
        return Read("INCONCLUSIVE", scored_object,
                    f"{group}/{metric} produced no excess interval",
                    n_effective=n_effective, detail=dict(now))
    lo, point = float(ci["lo"]), float(ci["point"])
    was = float(before.get("excess_point", float("nan")))
    detail: Detail = {**now, "warp_excess": was,
                      "warp_clumps": before.get("n_clumps")}
    survives = lo > 0.0
    clumps, null_clumps = int(now.get("n_clumps", 0)), int(now.get("null_clumps", 0))
    tail = (f"{point * 100:+.4f}% [{lo * 100:+.4f}%, "
            f"{float(ci['hi']) * 100:+.4f}%] against {was * 100:+.4f}% under the "
            f"{WARP} distance; {clumps} clumps against the nulls' {null_clumps}")
    if not survives:
        return Read("FAIL", scored_object,
                    f"THE EXCESS COLLAPSES on {group} under {metric}: {tail}. "
                    f"Segment recurrence depended on the warp, so it was a "
                    f"property of the ruler and not of the animal",
                    n_effective=n_effective, detail=detail)
    if clumps <= null_clumps:
        return Read("FAIL", scored_object,
                    f"the excess survives on {group} under {metric} but THE "
                    f"CLUMPS DO NOT: {tail}. The 19 clumps the {WARP} distance "
                    f"found were tempo-invariance, and the vocabulary claim "
                    f"weakens materially",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"both survive on {group} under {metric}: {tail}. The grouping "
                f"is not an artifact of time-normalisation",
                n_effective=n_effective, detail=detail)


def _load(d: str, group: str, metric: str, arm: str, unit: str) -> Any:
    p = os.path.join(d, f"{group}__{metric}__{arm}__{unit}.npz")
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def _graph(z: Any, theta: float) -> Any:
    """The clump graph, restricted to report queries as Step 3's was.

    `i_cross` indexes the full 298-animal bank; the queries are the report
    split. An edge whose partner is not itself a query has no row to point at,
    so it is dropped rather than silently reindexed -- Step 3 built its graph on
    a report-only bank, and the comparison is only a metric comparison if the
    population is the same.
    """
    q_idx = np.asarray(z["q_idx"], dtype=np.int64)
    pos = {int(g): r for r, g in enumerate(q_idx.tolist())}
    # ALL `KEEP` neighbours, not just the nearest: Step 3 built its graph with
    # ten edges per node, and a one-edge graph would have a different component
    # count for reasons that have nothing to do with the metric.
    ii = np.asarray(z["nn_idx"], dtype=np.int64)
    local = np.vectorize(lambda v: pos.get(int(v), -1))(ii).astype(np.int64)
    nn = np.asarray(z["nn_all"], dtype=np.float64)
    return vb.components(local, nn, theta=theta, min_size=vb.MIN_CLUMP)


def _warp_reference() -> Detail:
    """The time-normalised numbers, read from the committed results."""
    out: Detail = {}
    for name, key in (("seg_recur.json", "excess"), ("seg_vocab.json", "vocab")):
        p = os.path.join(spine.REPO if hasattr(spine, "REPO") else ".",
                         "results", name)
        p = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "results", name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            out[key] = json.load(fh)
    return out


def main(args: Any, shard_dir: str) -> int:
    metas = sorted(glob.glob(os.path.join(shard_dir, "*__*.json")))
    if not metas:
        raise SystemExit(f"no shards in {shard_dir}")
    ref = _warp_reference()
    groups: Detail = {}
    reads: Detail = {}
    for mp in metas:
        with open(mp, encoding="utf-8") as fh:
            meta = json.load(fh)
        group, metric = str(meta["group"]), str(meta["metric"])
        key = f"{group}|{metric}"
        o = _load(shard_dir, group, metric, "corpus", "segment")
        if o is None:
            continue
        cell: Detail = {"group": group, "metric": metric,
                        "n_bank": {k: v.get("n_bank")
                                   for k, v in meta["arms"].items()},
                        "recovery": (meta["arms"].get("corpus|segment", {})
                                     .get("recovery")),
                        "excess_by_null": {}, "duration": {}}
        for null in rc.NULLS:
            for unit in ("segment", "window"):
                n = _load(shard_dir, group, metric, null, unit)
                obs = _load(shard_dir, group, metric, "corpus", unit)
                if n is None or obs is None:
                    continue
                cell["excess_by_null"][f"{unit}|{null}"] = cr.paired(
                    obs, n, seed=0,
                    scale_obs=meta["arms"][f"corpus|{unit}"]["scale"],
                    scale_null=meta["arms"][f"{null}|{unit}"]["scale"])
        # The duration ratio between matched partners: the number that says
        # whether the metric was creating the tempo-invariance.
        ok = np.asarray(o["nn_len"]) > 0
        gap = np.abs(np.log(np.asarray(o["q_len"])[ok])
                     - np.log(np.asarray(o["nn_len"])[ok]))
        cell["duration"] = {"n": int(ok.sum()),
                            "mean_abs_log_gap": float(gap.mean()),
                            "ratio": float(np.exp(gap.mean()))}
        # Clumps, at the same theta derivation as Step 3.
        prim = cell["excess_by_null"].get(f"segment|{rc.NULLS[0]}")
        theta = float(prim["theta"]) if prim else float("nan")
        lab, summ = _graph(o, theta)
        null_clumps = 0
        for null in rc.NULLS:
            nz = _load(shard_dir, group, metric, null, "segment")
            if nz is None:
                continue
            _nl, ns = _graph(nz, theta)
            null_clumps = max(null_clumps, int(ns["n_clumps"]))
        cell["clumps"] = {**summ, "null_clumps": null_clumps}

        was_e = float("nan")
        was_c = None
        rg = (ref.get("excess", {}).get("groups", {}) or {}).get(f"{group}|k3")
        if rg:
            was_e = float(rg["excess"][f"segment|{rc.NULLS[0]}"]["delta"]["point"])
        vg = (ref.get("vocab", {}).get("groups", {}) or {}).get(f"{group}|k3")
        if vg:
            was_c = int(vg["corpus"]["n_clumps"])
        now = {"excess": prim["delta"] if prim else None,
               "n_clumps": int(summ["n_clumps"]),
               "null_clumps": null_clumps,
               "unassigned_fraction": float(summ["unassigned_fraction"]),
               "duration_ratio": cell["duration"]["ratio"]}
        obj = {"dataset": "luna", "arm": "seg_dist", "group": group,
               "metric": metric, "pose_arm": "raw", "split": "report",
               "null": rc.NULLS[0]}
        reads[key] = metric_read(
            now, {"excess_point": was_e, "n_clumps": was_c},
            group=group, metric=metric, scored_object=obj,
            n_effective=int(prim["delta"]["n_animals"]) if prim else 1).to_dict()
        log(f"  {key}: " + reads[key]["verdict"])
        groups[key] = cell

    write_json({**anchors.header(anchors.LUNA, stage="seg_dist_combine",
                                 unverified="scored on the report split"),
                "inherited_digest": spine.digest(),
                "registration": "results/DISTANCE_PREREGISTRATION.md",
                "primary_metric": "open", "diagnostic_metric": "union",
                "compared_against": WARP,
                "reads": reads, "groups": groups},
               args.out)
    log(f"wrote {args.out}")
    return 0
