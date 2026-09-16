r"""Step B's gate: can the detector find what is demonstrably there?

Registered in `results/DETECTOR_PREREGISTRATION.md`.

The detector isolates **2.3%** of planted instances as their own segment and
merely overlaps 66.7% of them, so it misses 97.7% of events that are
demonstrably present. Every coverage figure downstream is bounded by that, which
is why this gate sits between the metric question and any attempt to raise
coverage.

The diagnosis matters more than the rate. Four failures hide inside one number
and implicate different fixes: **no boundary at either edge** says the criterion
cannot see the insertion and no threshold will help; **merged** says the
refractory period swallowed a real edge. Reporting only "2.3%" would have sent
the next person to sweep thresholds whatever the cause.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Mapping, Sequence

from recur import anchors
from recur.read import Read
from recur.util import log, write_json

from vieb.io import spine

Detail = dict[str, Any]

__all__ = ["combine", "isolation_read"]

#: Registered before the sweep ran.
TARGET = 0.20
#: What the current detector does, for the record.
BASELINE = 0.023


def isolation_read(best: Mapping[str, Any], *, scored_object: Detail,
                   n_effective: int, target: float = TARGET) -> Read:
    """Does the best cell in the sweep isolate enough of what is planted?

    **A refusal is a correct outcome.** If no cell reaches the target, the
    detector cannot find what is demonstrably there and Step D does not run --
    and if the dominant failure is `no_edge`, that is a statement about the
    criterion rather than about its parameters, so no further sweeping is
    indicated either.
    """
    if not best:
        return Read("INCONCLUSIVE", scored_object,
                    "no sweep cell produced a planted instance to classify",
                    n_effective=n_effective, detail={})
    iso = float(best["isolation"])
    sh = best.get("shares", {})
    worst = max(((k, v) for k, v in sh.items() if k != "isolated"),
                key=lambda kv: kv[1], default=("none", 0.0))
    detail: Detail = {k: v for k, v in best.items() if k != "shares"}
    detail["shares"] = dict(sh)
    detail["baseline_isolation"] = BASELINE
    where = (f"deriv_sec {best['deriv_sec']}, degree {best['degree']}, "
             f"planted duration {best['duration_s']} s")
    tail = (f"{iso:.1%} of {int(best['n_instances']):,} planted instances became "
            f"their own segment at {where}, against {BASELINE:.1%} for the "
            f"detector of record; the dominant failure is {worst[0]} at "
            f"{worst[1]:.1%}")
    if iso >= target:
        return Read("PASS", scored_object,
                    f"the detector clears the registered gate: {tail}",
                    n_effective=n_effective, detail=detail)
    if worst[0] == "no_edge":
        return Read("FAIL", scored_object,
                    f"THE CRITERION CANNOT SEE THE INSERTION: {tail}, against a "
                    f"{target:.0%} target. The dominant failure is that NO "
                    f"boundary appears at either edge of a planted instance, so "
                    f"this is a statement about the acceleration-mismatch "
                    f"criterion and not about its parameters -- no further "
                    f"threshold sweep is indicated",
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"the detector does not clear the registered gate: {tail}, "
                f"against a {target:.0%} target. Coverage computed on this "
                f"detector is a statement about the detector",
                n_effective=n_effective, detail=detail)


def combine(args: Any, shard_dir: str) -> int:
    rows: list[Detail] = []
    for p in sorted(glob.glob(os.path.join(shard_dir, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            rows += json.load(fh).get("cells", [])
    if not rows:
        raise SystemExit(f"no cells in {shard_dir}")
    prim = [r for r in rows if float(r["duration_s"]) == 0.5]
    best = max(prim or rows, key=lambda r: float(r["isolation"]))
    obj = {"dataset": "luna", "arm": "detector", "group": best.get("group"),
           "pose_arm": "raw", "planted_into": best.get("planted_into"),
           "edge_tolerance_frames": 2}
    rd = isolation_read(best, scored_object=obj,
                        n_effective=int(best["n_instances"]))
    log(rd.line())
    # The resolution the sweep buys, per cell: the shortest planted duration a
    # cell isolates at the target rate. A rate alone does not say what the
    # detector can see.
    res: Detail = {}
    for r in rows:
        key = f"deriv{r['deriv_sec']:g}_deg{r['degree']}"
        if float(r["isolation"]) >= TARGET:
            cur = res.get(key)
            if cur is None or float(r["duration_s"]) < cur:
                res[key] = float(r["duration_s"])
    write_json({**anchors.header(anchors.LUNA, stage="detector_combine",
                                 unverified="a subset of animals per cell"),
                "inherited_digest": spine.digest(),
                "registration": "results/DETECTOR_PREREGISTRATION.md",
                "target": TARGET, "baseline_isolation": BASELINE,
                "reads": {"isolation": rd.to_dict()},
                "best_cell": best,
                "smallest_isolated_duration_s": res or None,
                "cells": sorted(rows, key=lambda r: -float(r["isolation"]))},
               args.out)
    log(f"wrote {args.out}")
    return 0
