"""Step B. Can the detector find what is demonstrably there? Diagnose, then sweep.

    python3 scripts/detector_sweep.py --diagnose --group shape
    sbatch --array=0-17 jobs/detector_sweep.slurm
    python3 scripts/detector_sweep.py --combine

READ results/DETECTOR_PREREGISTRATION.md FIRST. The gate is named there:
planted-instance isolation >= 20% at the primary planted duration, and no
parameter is tuned until it passes.

## The number this exists for

Of planted instances at 10% occupancy, 2.3% became their own segment and 66.7%
were merely overlapped by a spanning segment. The detector misses 97.7% of events
that are demonstrably present, and every coverage figure downstream is bounded by
that.

## Why the diagnosis comes first

2.3% is one number covering four different failures that point at four different
fixes. If most instances produce NO boundary at either edge, the criterion cannot
see a smooth crossfaded block and no threshold sweep will help; if both edges are
found and then merged, it is the refractory period. The four bins are asserted to
sum to the total so a missing case cannot hide in rounding.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                           # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.util import frames, log, write_json                      # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seg import breaks as bk, embed, planted as pl             # noqa: E402
from vieb.tok import config                                         # noqa: E402

SEED = 0
#: The tolerance for "a boundary landed on this edge". The same +/-2 frames
#: BREAKS.md uses for its channel-group Jaccard, so the number is comparable to
#: one already published rather than to a new convention.
EDGE_TOL = 2
#: The registered gate.
ISOLATION_TARGET = 0.20
PRIMARY_OCC = 0.1
#: The sweep. Both already exist in scripts/breaks.py and are reused.
DERIV_SWEEP: tuple[float, ...] = (0.067, 0.133, 0.267)
DEGREE_SWEEP: tuple[int, ...] = (2, 3)
DURATIONS_S: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


def out_dir() -> str:
    return os.path.join(config.PATHS.tok_dir, "detector")


def _sr():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_recur.py")
    spec = importlib.util.spec_from_file_location("sr_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["seg_recur"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def classify(rows, mask, *, lo: int, hi: int, peaks) -> dict:
    """Each planted instance into exactly one of four bins.

    The bins are the four ways the detector can fail an instance, and they
    implicate different fixes. Asserted to sum to the total, so a case that fits
    none cannot vanish into the arithmetic.
    """
    m = np.asarray(mask, dtype=bool)
    pk = np.asarray(peaks, dtype=np.int64)
    run = np.diff(np.concatenate([[0], m[lo:hi].astype(np.int8), [0]]))
    starts = np.flatnonzero(run == 1) + lo
    stops = np.flatnonzero(run == -1) + lo
    bins = {"isolated": 0, "no_edge": 0, "one_edge": 0, "merged": 0,
            "split": 0}
    for s, e in zip(starts.tolist(), stops.tolist()):
        has_s = bool(np.any(np.abs(pk - s) <= EDGE_TOL))
        has_e = bool(np.any(np.abs(pk - e) <= EDGE_TOL))
        inside = int(np.sum((pk > s + EDGE_TOL) & (pk < e - EDGE_TOL)))
        exact = [r for r in rows
                 if abs(int(r["start"]) - s) <= EDGE_TOL
                 and abs(int(r["stop"]) - e) <= EDGE_TOL]
        if exact:
            bins["isolated"] += 1
        elif not has_s and not has_e:
            bins["no_edge"] += 1
        elif has_s != has_e:
            bins["one_edge"] += 1
        elif inside > 0:
            bins["split"] += 1
        else:
            bins["merged"] += 1
    total = int(starts.size)
    assert sum(bins.values()) == total, (
        f"the four failure bins plus isolated sum to {sum(bins.values())} "
        f"against {total} planted instances; a case fits none of them")
    return {**bins, "n_instances": total}


def cell(args, tags, idx, *, fps, sd, sr, deriv: float, degree: int,
         duration_s: float, occ: float, into: str = "microstate") -> dict:
    """One (deriv_sec, degree, planted duration) cell of the sweep."""
    h = max(2, frames(deriv, fps))
    guard = bk.guard_frames(fps, deriv_sec=deriv)
    min_gap = bk.min_segment_frames(degree, guard)
    w_tpl = frames(duration_s, fps)
    donors = []
    for tag in list(tags)[:3]:
        a = sr.load_arm(into, tag, sd)
        donors.append((a["X"], ~a["abstain"]))
    tpl = pl.ego_template(np.concatenate([d[0] for d in donors]), w_tpl,
                          np.random.default_rng(seeds.stable_seed(SEED, "tpl")),
                          valid=np.concatenate([d[1] for d in donors]))
    del donors
    agg = {"isolated": 0, "no_edge": 0, "one_edge": 0, "merged": 0,
           "split": 0, "n_instances": 0}
    for tag in tags:
        a = sr.load_arm(into, tag, sd)
        rng = np.random.default_rng(
            seeds.stable_seed(SEED, f"det{duration_s:g}{occ:g}", tag))
        a["X"], mask, _meta = pl.ego_plant(a["X"], tpl, rng,
                                           bounds=a["bounds"], occupancy=occ,
                                           blocked=a["abstain"])
        sub = a["X"][:, list(idx)]
        for r in range(a["bounds"].shape[0] - 1):
            lo, hi = int(a["bounds"][r]), int(a["bounds"][r + 1])
            d = bk.discontinuity(sub[lo:hi], h)
            pk = bk.boundaries(d, bk.mad_threshold(d, bk.K_MAD),
                               min_gap=min_gap,
                               blocked=a["abstain"][lo:hi]) + lo
            tab = bk.segment_table(a["X"], pk, lo=lo, hi=hi,
                                   abstain=a["abstain"], guard=guard,
                                   degree=degree, fps=fps)
            rows = embed.mark_edges(tab, a["abstain"], lo=lo, hi=hi)
            got = classify(rows, mask, lo=lo, hi=hi, peaks=pk)
            for k in agg:
                agg[k] += got[k]
        del a
    n = max(agg["n_instances"], 1)
    return {"deriv_sec": deriv, "degree": degree, "duration_s": duration_s,
            "occupancy": occ, "h": h, "guard": guard, "min_gap": min_gap,
            "template_frames": w_tpl, "planted_into": into,
            **agg,
            "isolation": agg["isolated"] / n,
            "shares": {k: agg[k] / n for k in
                       ("isolated", "no_edge", "one_edge", "merged", "split")}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="shape", choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=18)
    p.add_argument("--n-animals", type=int, default=12)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    os.makedirs(out_dir(), exist_ok=True)
    if a.combine:
        from vieb.seg import detector
        a.out = a.out or config.PATHS.result("detector.json")
        return detector.combine(a, out_dir())

    sr = _sr()
    fps = spine.fps()
    sd = sr.basis_sd()
    idx = bk.CHANNEL_GROUPS[a.group]
    tags = sorted(sr.animals_of(spine.recording_ids()))[:a.n_animals]

    grid = [(d, g, t) for d in DERIV_SWEEP for g in DEGREE_SWEEP
            for t in DURATIONS_S]
    if a.diagnose:
        grid = [(bk.DERIV_SEC, 3, 0.5)]
    elif a.task is not None:
        grid = grid[a.task::a.n_tasks]
    rows = []
    for deriv, degree, dur in grid:
        got = cell(a, tags, idx, fps=fps, sd=sd, sr=sr, deriv=deriv,
                   degree=degree, duration_s=dur, occ=PRIMARY_OCC)
        got["group"] = a.group
        rows.append(got)
        log(f"  [{a.group}] deriv={deriv} degree={degree} dur={dur}s  "
            f"isolated {got['isolation']:.1%}  "
            f"no_edge {got['shares']['no_edge']:.1%}  "
            f"one_edge {got['shares']['one_edge']:.1%}  "
            f"merged {got['shares']['merged']:.1%}  "
            f"split {got['shares']['split']:.1%}  "
            f"(n={got['n_instances']})")
    tag = "diagnose" if a.diagnose else f"task{a.task:02d}"
    write_json({**provenance.header(anchors.LUNA, stage="detector",
                                 unverified="a subset of animals per cell"),
                "inherited_digest": spine.digest(),
                "registration": "results/DETECTOR_PREREGISTRATION.md",
                "edge_tolerance_frames": EDGE_TOL,
                "isolation_target": ISOLATION_TARGET,
                "n_animals": len(tags), "cells": rows},
               os.path.join(out_dir(), f"{a.group}__{tag}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
