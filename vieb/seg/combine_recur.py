r"""Combine Step 2's per-group shards into the gate.

`paired_excess` is **imported from `recur/scripts/q1.py`**, not reimplemented.
That is the point: the statistic has to be the same object Q1's +1.639% was
computed with, and a second implementation could disagree with the published
one. `recur/scripts` is not a package, so it is put on `sys.path` and imported
by module name -- the alternative, copying thirty lines of threshold and scale
logic into this repository, is exactly how two numbers that should be identical
start to drift.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

import numpy as np
from recur import anchors
from recur.util import log, write_json

from vieb.io import spine
from vieb.seg import recur as rc

Detail = dict[str, Any]

__all__ = ["main", "paired"]


def _q1() -> Any:
    """Q1's driver as a module, so `paired_excess` is the same code object."""
    root = os.environ.get("VIEB_RECUR", "/home/tul26194/recur")
    p = os.path.join(root, "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import q1  # type: ignore[import-not-found]  # noqa: PLC0415
    return q1


def paired(obs: Any, null: Any, *, scale_obs: float, scale_null: float,
           seed: int = 0) -> Detail:
    """Q1's `paired_excess`, each arm divided by its own ambient scale.

    The scales are passed in rather than read off the npz because they are
    scalars and the shard carries only arrays; they live in the group's JSON
    beside the bank sizes. Passing them explicitly also makes it impossible to
    normalise two arms by one scale, which is the failure the normalisation
    exists to prevent.
    """
    q1 = _q1()
    out: Detail = q1.paired_excess(obs, null, seed=seed,
                                   scale_obs=float(scale_obs),
                                   scale_null=float(scale_null))
    return out


def _load(d: str, group: str, k: float, arm: str, unit: str,
          suffix: str = "") -> Any:
    p = os.path.join(d, f"{group}__k{k:g}{suffix}__{arm}__{unit}.npz")
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def _matched(d: str, group: str, k: float) -> Detail:
    """The declared post-hoc bank-size control, if its run exists.

    Not the registered primary. Nearest-neighbour distance falls as a bank
    grows, and because theta comes from the NULL's own quantile a smaller null
    bank inflates the excess in the direction that flatters the corpus. The
    first run showed `microstate` producing 25% fewer segments than the corpus
    on `shape`, so every arm is re-run with its bank cut to the smallest.
    """
    mp = os.path.join(d, f"{group}__k{k:g}__matched.json")
    if not os.path.exists(mp):
        return {}
    with open(mp, encoding="utf-8") as fh:
        meta = json.load(fh)
    out: Detail = {"matched_bank": meta.get("matched_bank"), "excess": {}}
    for unit in ("segment", "window"):
        for null in rc.NULLS:
            o = _load(d, group, k, "corpus", unit, "__matched")
            n = _load(d, group, k, null, unit, "__matched")
            if o is None or n is None:
                continue
            out["excess"][f"{unit}|{null}"] = paired(
                o, n, seed=0,
                scale_obs=_scale(meta, f"corpus|{unit}"),
                scale_null=_scale(meta, f"{null}|{unit}"))
    seg = out["excess"].get(f"segment|{rc.NULLS[0]}")
    win = out["excess"].get(f"window|{rc.NULLS[0]}")
    if seg and win:
        out["gate_delta"] = rc.excess_delta(seg, win, seed=0)
    return out


def _exactness(meta: Detail) -> Detail:
    """The gate split by quantity, because the statistic reads only one of them.

    `exactness_read` gates jointly on `d_cross` and `d_within` and returns FAIL
    if either drifts. `paired_excess` reads **only `d_cross`**; `d_within` is a
    reported diagnostic that already carries a stated upper-bound limitation.

    Splitting is not weakening the gate -- the joint verdict is kept and
    reported -- it is saying which quantity each number depends on. The
    `d_within` drift is a near-tie: the within-animal neighbour is chosen by a
    Gram-form argmin and its index is never returned, so two same-animal units
    at nearly equal distance can be ordered differently in float32 and float64
    without `index_agreement`, which compares only `i_cross`, registering it.
    """
    worst_c = worst_w = 0.0
    tol = 1e-4
    for v in meta["arms"].values():
        det = v.get("exactness", {}).get("detail", {}) or {}
        worst_c = max(worst_c, float(det.get("max_abs_diff_d_cross", 0.0)))
        worst_w = max(worst_w, float(det.get("max_abs_diff_d_within", 0.0)))
        tol = float(det.get("tolerance", tol))
    return {"worst_d_cross": worst_c, "worst_d_within": worst_w,
            "tolerance": tol,
            "d_cross_within_tolerance": bool(worst_c <= tol),
            "d_within_within_tolerance": bool(worst_w <= tol),
            "statistic_reads": "d_cross only"}


def _scale(meta: Detail, name: str) -> float:
    return float(meta["arms"][name]["scale"])


def _floor(d: str, group: str, k: float, meta: Detail, seed: int) -> Detail:
    """The planted dose-response, scored against `microstate`, as registered."""
    null = _load(d, group, k, "microstate", "segment")
    s_null = _scale(meta, "microstate|segment")
    rows: list[Detail] = []
    smallest: float | None = None
    cells = {key[3:]: v for key, v in (meta.get("planted") or {}).items()
             if key.startswith("occ")}
    for occ, info in sorted(cells.items(), key=lambda kv: float(kv[0])):
        obs = _load(d, group, k, f"planted{float(occ):g}", "segment")
        if obs is None or null is None:
            continue
        got = paired(obs, null, seed=seed,
                     scale_obs=float(info["scale"]), scale_null=s_null)
        realized = float(info.get("realized_occupancy", float("nan")))
        rec = bool(float(got["delta"]["lo"]) > 0.0)
        rows.append({"requested_occupancy": float(occ),
                     "realized_occupancy": realized,
                     "delta": got["delta"], "recovered": rec})
        if rec and smallest is None:
            smallest = realized
    return {"ladder": rows, "smallest_recovered": smallest,
            "planted_into": (meta.get("planted") or {}).get("planted_into"),
            "note": ("planted into the NULL and scored against the unplanted "
                     "null, so the background carries no recurrence of its "
                     "own. Reported against the REALIZED fraction, never the "
                     "request: `plant` places whole instances and the two "
                     "diverge at long templates")}


def main(args: Any, shard_dir: str) -> int:
    metas = sorted(p for p in glob.glob(os.path.join(shard_dir, "*__k*.json"))
                   if not p.endswith("__matched.json"))
    if not metas:
        raise SystemExit(f"no group shards in {shard_dir}")
    groups: Detail = {}
    reads: Detail = {}
    for mp in metas:
        with open(mp, encoding="utf-8") as fh:
            meta = json.load(fh)
        group, k = str(meta["group"]), float(meta["k_mad"])
        key = f"{group}|k{k:g}"
        arms = {u: {a: _load(shard_dir, group, k, a, u) for a in
                    ("corpus",) + rc.NULLS} for u in ("segment", "window")}
        if arms["segment"]["corpus"] is None:
            log(f"  {key}: no corpus segment shard, skipping")
            continue
        cell: Detail = {"group": group, "k_mad": k,
                        "counts": {n: v.get("counts")
                                   for n, v in meta["arms"].items()},
                        "pca": {n: v.get("pca_read", {}).get("verdict")
                                for n, v in meta["arms"].items()},
                        "exactness": {n: v.get("exactness", {}).get("verdict")
                                      for n, v in meta["arms"].items()},
                        "exactness_by_quantity": _exactness(meta),
                        "n_bank": {n: v.get("n_bank")
                                   for n, v in meta["arms"].items()},
                        "excess": {}, "duration_match": {}}
        for null in rc.NULLS:
            for unit in ("segment", "window"):
                o, n = arms[unit]["corpus"], arms[unit][null]
                if o is None or n is None:
                    continue
                cell["excess"][f"{unit}|{null}"] = paired(
                    o, n, seed=0, scale_obs=_scale(meta, f"corpus|{unit}"),
                    scale_null=_scale(meta, f"{null}|{unit}"))
            os_, ns_ = arms["segment"]["corpus"], arms["segment"][null]
            if os_ is not None and ns_ is not None:
                cell["duration_match"][null] = rc.duration_match(
                    os_["q_len"], ns_["q_len"])

        rng = np.random.default_rng(0)
        o = arms["segment"]["corpus"]
        ok = np.asarray(o["nn_len"]) > 0
        cell["length_match_control"] = rc.length_match_control(
            np.asarray(o["q_len"])[ok], np.asarray(o["nn_len"])[ok], rng)
        dw, dc = np.asarray(o["d_within"]), np.asarray(o["d_cross"])
        fin = np.isfinite(dw) & np.isfinite(dc)
        cell["within_vs_cross"] = {
            "median_d_within": float(np.median(dw[fin])),
            "median_d_cross": float(np.median(dc[fin])),
            "frac_within_closer": float((dw[fin] < dc[fin]).mean()),
            "limitation": (
                "d_within is an UPPER BOUND on within-animal contamination: "
                "search's exclusion is a start-offset test, and two "
                "unequal-length segments can overlap on most of their extent "
                "while their starts are far apart. The headline is unaffected "
                "-- paired_excess reads only d_cross, which the exclusion never "
                "touches")}

        floor = _floor(shard_dir, group, k, meta, seed=0)
        cell["planted_floor"] = floor
        cell["bank_size_control"] = _matched(shard_dir, group, k)
        obj = {"dataset": "luna", "arm": "seg_recur", "group": group,
               "k_mad": k, "pose_arm": "raw", "split": "report",
               "distance": "time-normalised"}
        for null in rc.NULLS:
            got = cell["excess"].get(f"segment|{null}")
            if got:
                reads[f"{key}|{null}"] = rc.recur_read(
                    got, null=null, unit="segment",
                    scored_object={**obj, "null": null},
                    n_effective=int(got["delta"]["n_animals"])).to_dict()
        seg = cell["excess"].get(f"segment|{rc.NULLS[0]}")
        win = cell["excess"].get(f"window|{rc.NULLS[0]}")
        if seg and win:
            d = rc.excess_delta(seg, win, seed=0)
            cell["gate_delta"] = d
            rd = rc.gate_read(d, floor, group=group, scored_object=obj,
                              n_effective=int(d.get("n_animals", 0) or 1))
            reads[f"{key}|gate"] = rd.to_dict()
            log(rd.line())
        groups[key] = cell

    doc = {**anchors.header(anchors.LUNA, stage="seg_recur_combine",
                            unverified="scored on the report split"),
           "inherited_digest": spine.digest(),
           "registration": "results/SEGRECUR_PREREGISTRATION.md",
           "preprocessing_freeze": "F3", "split": "report",
           "distance_primary": "time-normalised to 40 points",
           "nulls": list(rc.NULLS),
           "comparator": ("a length-matched windowed arm in the IDENTICAL ego "
                          "space, lengths drawn from the same arm's clean "
                          "segment durations and its count matched, so the "
                          "only difference left is where the edges fall"),
           "reads": reads, "groups": groups}
    write_json(doc, args.out)
    log(f"wrote {args.out}")
    return 0
