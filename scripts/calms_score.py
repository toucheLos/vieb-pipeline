"""STEP 1 §4-7: score every instrument × condition, then decide. Reads labels.

    python3 scripts/calms_score.py cell --instrument M --cond m1     # one cell file
    python3 scripts/calms_score.py combine                           # -> results/step1/recovery.json

A *condition* is a level (`m0` ... `m3`) or a control (`white`, `phase`); it
pools that condition's five seeds. Instruments: D (boundary F1, no map), C and M
(event F1 through the many-to-one map).

This script and `vieb/calms/score.py` are the only code that reads CalMS21
labels, and they run only after every instrument's states exist.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                           # noqa: E402
from recur.read import Read                                         # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.calms import degrade as dg, io as cio                     # noqa: E402
from vieb.calms import score as sc                                  # noqa: E402
from vieb.tok import config                                         # noqa: E402

WORK = os.path.join(config.REPO, "work", "calms")
OUT = os.path.join(config.REPO, "results", "step1")
PARTS = os.path.join(OUT, "partitions.json")
CELLS = os.path.join(WORK, "score")
INSTRUMENTS = ("D", "C", "M")
CONDS = tuple(f"m{m:g}" for m in dg.LEVELS) + ("white", "phase")
CAVEAT = ("unit = sequence; a resident can recur across sequences undetectably, "
          "so intervals may be too narrow")
#: (label set, map partition, eval partition, targets)
TASKS = (("t1", "map_t1", "eval_t1", sc.T1_LABELS), ("g", "map_g", "eval_g", ("groom",)))


def arm_names(cond: str) -> list[str]:
    return [f"{cond}_s{s}" for s in dg.SEEDS]


def states_of(instrument: str, arm: str, seed: int) -> dict[str, np.ndarray]:
    if instrument == "M":
        p = os.path.join(WORK, "moseq", "labels", f"{arm}_s{seed}.npz")
        with np.load(p) as z:
            s, b = z["syllable"].astype(np.int64), z["bounds"]
            return {str(v): s[b[i]:b[i + 1]] for i, v in enumerate(z["recording_ids"])}
    key = {"C": ("C.npz", "s_"), "D": ("D.npz", "b_")}[instrument]
    with np.load(os.path.join(WORK, "states", arm, key[0])) as z:
        return {str(v): np.asarray(z[key[1] + str(v)], dtype=np.int64) for v in z["videos"]}


def ceilings(parts: dict) -> dict[str, dict[str, float]]:
    """Annotators 3 and 4 on their shared videos, both directions averaged; groom 1."""
    a3, a4 = sc.labels("a3"), sc.labels("a4")
    ids = parts["ceiling"]
    tol = int(round(sc.TOL_S * sc.FPS))
    out: dict[str, dict[str, float]] = {}
    for lab in sc.T1_LABELS:
        cls = sc.class_id("a3", lab)
        ev = [sc.f1(np.sum([sc.event_counts(x[v], y[v], cls) for v in ids], axis=0))
              for x, y in ((a3, a4), (a4, a3))]
        bd = [sc.f1(np.sum([sc.boundary_counts(sc.true_boundaries(x[v], cls),
                                               sc.true_boundaries(y[v], cls), tol)
                            for v in ids], axis=0)) for x, y in ((a3, a4), (a4, a3))]
        out[lab] = {"event": float(np.mean(ev)), "boundary": float(np.mean(bd)),
                    "n_videos": len(ids)}
    out["groom"] = {"event": 1.0, "boundary": 1.0, "n_videos": 0,
                    "note": "no second annotator: ceiling = 1 (D-S3)"}
    return out


def cell(a) -> int:
    parts = cio.load(PARTS)
    ceil = ceilings(parts)
    arms = arm_names(a.cond)
    st = [states_of(a.instrument, arm, s) for arm, s in zip(arms, dg.SEEDS)]
    res: dict[str, Any] = {"instrument": a.instrument, "cond": a.cond, "labels": {}}
    for label_set, mp, ev, targets in TASKS:
        y = sc.labels(label_set)
        map_ids, eval_ids = parts[mp], parts[ev]
        ids = sorted(set(map_ids) | set(eval_ids))

        def counts(ys: dict) -> list[sc.Counts]:
            if a.instrument == "D":
                return [sc.boundary_pass(s, ys, eval_ids, label_set, targets) for s in st]
            return [sc.event_pass(s, ys, map_ids, eval_ids, label_set, targets) for s in st]

        obs = counts(y)
        rng = np.random.default_rng(sc.SEED)
        null = {lab: [] for lab in targets}
        for _ in range(a.shifts):
            ys = sc.shifted(y, ids, rng)
            cs = counts(ys)
            for lab in targets:
                null[lab].append(sc.seed_mean_f1([c[lab] for c in cs]))
        boot = sc.boot_index(len(eval_ids))
        metric = "boundary" if a.instrument == "D" else "event"
        for lab in targets:
            sup = sc.support(y, eval_ids, label_set, lab)
            entry: dict[str, Any] = {"support": sup, "metric": metric,
                                     "n_eval": len(eval_ids), "n_map": len(map_ids)}
            if sup["ok"]:
                entry.update(sc.cell([c[lab] for c in obs], np.asarray(null[lab]),
                                     ceil[lab][metric], boot))
            else:
                entry["score"] = sc.seed_mean_f1([c[lab] for c in obs])
            entry["null"] = np.asarray(null[lab]).tolist()
            if a.instrument != "D":
                cls = sc.class_id(label_set, lab)
                entry["nmi"] = float(np.mean([sc.nmi(
                    np.concatenate([y[v] == cls for v in eval_ids]),
                    np.concatenate([s[v] for v in eval_ids])) for s in st]))
            res["labels"][lab] = entry
    os.makedirs(CELLS, exist_ok=True)
    write_json(res, os.path.join(CELLS, f"{a.instrument}_{a.cond}.json"))
    log(f"{a.instrument} {a.cond}: " + ", ".join(
        f"{k} {v.get('normalised', float('nan')):+.3f}" for k, v in res["labels"].items()))
    return 0


def combine(_a) -> int:
    parts = cio.load(PARTS)
    with open(os.path.join(OUT, "noise.json"), encoding="utf-8") as fh:
        nz = json.load(fh)
    cells: dict = {}
    for p in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        cells[(c["instrument"], c["cond"])] = c
    missing = [(i, c) for i in INSTRUMENTS for c in CONDS if (i, c) not in cells]
    assert not missing, f"cells missing: {missing}"

    # Controls (§6): fail iff the rule declares anything recoverable on either.
    ctrl = {(i, c, lab): v for (i, c), d in cells.items() if c in ("white", "phase")
            for lab, v in d["labels"].items() if "recoverable" in v}
    controls_fail = any(v["recoverable"] for v in ctrl.values())

    sweep = {(i, lab, float(c[1:])): v for (i, c), d in cells.items() if c.startswith("m")
             for lab, v in d["labels"].items() if "recoverable" in v}
    gl = {lab for (i, c), d in cells.items() for lab, v in d["labels"].items()
          if not v["support"]["ok"]}
    decision = sc.decide(sweep, native=0.0, at=1.0, controls_fail=controls_fail,
                         r_ge_1=bool(nz["r_ge_1"]), groom_grid_limited="groom" in gl)

    levels = list(dg.LEVELS)
    curves: dict = {}
    reads: list = []
    for i in INSTRUMENTS:
        for lab in sc.T1_LABELS + ("groom",):
            if lab in gl:
                sup = cells[(i, "m0")]["labels"][lab]["support"]
                reads.append(Read("GRID_LIMITED", {"instrument": i, "label": lab,
                                                   "arm": "primary"},
                                  f"support {sup['events']} events in {sup['units']} "
                                  f"evaluation sequences, below {sc.MIN_EVENTS} / "
                                  f"{sc.MIN_UNITS}; {CAVEAT}",
                                  n_effective=max(1, int(sup["units"]))).to_dict())
                continue
            rc = [bool(sweep[(i, lab, m)]["recoverable"]) for m in levels]
            curves[f"{i}|{lab}"] = {
                "levels": levels, "reachable": [bool(nz["reachable"][f"{m:g}"]) for m in levels],
                "normalised": [sweep[(i, lab, m)]["normalised"] for m in levels],
                "ci": [sweep[(i, lab, m)]["ci"] for m in levels],
                "score": [sweep[(i, lab, m)]["score"] for m in levels],
                "chance": [sweep[(i, lab, m)]["chance"] for m in levels],
                "recoverable": rc, "breakdown": sc.breakdown(levels, rc)}
            for m in levels:
                v = sweep[(i, lab, m)]
                verdict = "PASS" if v["recoverable"] else (
                    "INCONCLUSIVE" if v["straddles"] else "FAIL")
                n_eval = len(parts["eval_g" if lab == "groom" else "eval_t1"])
                reads.append(Read(verdict, {"instrument": i, "label": lab, "level": m,
                                            "arm": "primary", "metric": cells[(i, f"m{m:g}")]
                                            ["labels"][lab]["metric"]},
                                  f"normalised {v['normalised']:+.3f} [{v['ci'][0]:+.3f}, "
                                  f"{v['ci'][1]:+.3f}] vs bar {sc.BAR}; score {v['score']:.3f} "
                                  f"vs null p95 {v['null_p95']:.3f}; {CAVEAT}",
                                  n_effective=n_eval).to_dict())
    write_json({**provenance.header(anchors.LUNA, stage="calms_step1",
                                    unverified="CalMS21 external; instruments at their commits"),
                "registration": "results/step1/REGISTRATION.md (+ Amendment 1)",
                "decision": decision, "controls_fail": controls_fail,
                "controls": {f"{i}|{c}|{lab}": {k: v[k] for k in ("normalised", "ci", "score",
                                                                   "null_p95", "recoverable")}
                             for (i, c, lab), v in ctrl.items()},
                "r": nz["r"], "gain": nz["gain"], "grid_limited": sorted(gl),
                "curves": curves, "reads": reads,
                "ceilings": ceilings(parts),
                "nmi": {f"{i}|{c}|{lab}": v.get("nmi") for (i, c), d in cells.items()
                        for lab, v in d["labels"].items() if i != "D"}},
               os.path.join(OUT, "recovery.json"))
    log(f"decision: {decision}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("cell", "combine"))
    p.add_argument("--instrument", choices=INSTRUMENTS)
    p.add_argument("--cond", choices=CONDS)
    p.add_argument("--index", type=int, default=None, help="array index over instrument x cond")
    p.add_argument("--shifts", type=int, default=sc.N_SHIFTS)
    a = p.parse_args(argv)
    if a.phase == "cell" and a.index is not None:
        a.instrument, a.cond = [(i, c) for i in INSTRUMENTS for c in CONDS][a.index]
    return {"cell": cell, "combine": combine}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
