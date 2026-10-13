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
from vieb.calms import arms as A, degrade as dg, io as cio          # noqa: E402
from vieb.calms import score as sc                                  # noqa: E402
from vieb.tok import config                                         # noqa: E402

WORK = os.path.join(config.REPO, "work", "calms")
OUT = os.path.join(config.REPO, "results", "step1")
PARTS = os.path.join(OUT, "partitions.json")
CELLS = os.path.join(WORK, "score")
INSTRUMENTS = ("D", "C", "M")
CONDS = tuple(f"m{m:g}" for m in dg.LEVELS) + ("white", "phase")
PRIMARY_FAMILIES = ("m", "white", "phase")
CAVEAT = ("unit = sequence; a resident can recur across sequences undetectably, "
          "so intervals may be too narrow")
#: (label set, map partition, eval partition, targets)
TASKS = (("t1", "map_t1", "eval_t1", sc.T1_LABELS), ("g", "map_g", "eval_g", ("groom",)))


def family_of(cond: str) -> str:
    for fam, (tpl, levels, _i) in A.FAMILIES.items():
        if any(A.replace(tpl, level=m).cond == cond for m in levels):
            return fam
    raise KeyError(cond)


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
    fam = family_of(a.cond)
    tpl = A.FAMILIES[fam][0]
    step = int(round(dg.FPS / tpl.fps))
    seeds = A.seeds(fam, a.instrument)
    st = [states_of(a.instrument, f"{a.cond}_s{s}", s) for s in seeds]
    res: dict[str, Any] = {"instrument": a.instrument, "cond": a.cond, "family": fam,
                           "seeds": list(seeds), "fps": tpl.fps, "labels": {}}
    for label_set, mp, ev, targets in TASKS:
        y = {v: lab_[::step] for v, lab_ in sc.labels(label_set).items()}
        map_ids, eval_ids = parts[mp], parts[ev]
        ids = sorted(set(map_ids) | set(eval_ids))
        if a.instrument != "D":
            for s_ in st:
                for v in ids:
                    assert s_[v].shape == y[v].shape, (a.cond, v, s_[v].shape, y[v].shape)

        def counts(ys: dict) -> list[sc.Counts]:
            if a.instrument == "D":
                return [sc.boundary_pass(s, ys, eval_ids, label_set, targets, fps=tpl.fps)
                        for s in st]
            return [sc.event_pass(s, ys, map_ids, eval_ids, label_set, targets) for s in st]

        obs = counts(y)
        rng = np.random.default_rng(sc.SEED)
        null = {lab: [] for lab in targets}
        null_ps: dict[str, list] = {lab: [] for lab in targets}
        for _ in range(a.shifts):
            ys = sc.shifted(y, ids, rng)
            cs = counts(ys)
            for lab in targets:
                null[lab].append(sc.seed_mean_f1([c[lab] for c in cs]))
                null_ps[lab].append([sc.f1(c[lab].sum(0)) for c in cs])
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
            entry["null_per_seed"] = null_ps[lab]
            entry["counts"] = [c[lab].tolist() for c in obs]
            entry["eval_ids"] = list(eval_ids)
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
        if c.get("family", family_of(c["cond"])) in PRIMARY_FAMILIES:
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


def _cells() -> dict:
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        out[(c["instrument"], c["cond"])] = c
    return out


def secondary(_a) -> int:
    """ABLATION_REGISTRATION A: curves for j, d and k, with Reads. Descriptive."""
    cells = _cells()
    with open(os.path.join(OUT, "noise.json"), encoding="utf-8") as fh:
        nz = json.load(fh)
    parts = cio.load(PARTS)
    out: dict = {"arms": {}, "reads": []}
    for fam in ("j", "d", "k"):
        tpl, levels, inst = A.FAMILIES[fam]
        for i in inst:
            for lab in sc.T1_LABELS + ("groom",):
                rows = [cells[(i, A.replace(tpl, level=m).cond)]["labels"][lab] for m in levels]
                if not rows[0]["support"]["ok"]:
                    continue
                rc = [bool(r["recoverable"]) for r in rows]
                out["arms"][f"{fam}|{i}|{lab}"] = {
                    "levels": list(levels), "normalised": [r["normalised"] for r in rows],
                    "ci": [r["ci"] for r in rows], "recoverable": rc,
                    "breakdown": sc.breakdown(list(levels), rc),
                    "seeds": list(A.seeds(fam, i))}
                n_eval = len(parts["eval_g" if lab == "groom" else "eval_t1"])
                for m, r in zip(levels, rows):
                    verdict = "PASS" if r["recoverable"] else (
                        "INCONCLUSIVE" if r["straddles"] else "FAIL")
                    out["reads"].append(Read(verdict, {"instrument": i, "label": lab,
                                                       "level": m, "arm": fam},
                                             f"normalised {r['normalised']:+.3f} "
                                             f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]; {CAVEAT}",
                                             n_effective=n_eval).to_dict())
    write_json({**provenance.header(anchors.LUNA, stage="calms_secondary",
                                    unverified="CalMS21 external; descriptive"),
                "registration": "results/step1/ABLATION_REGISTRATION.md A", "r": nz["r"],
                **out}, os.path.join(OUT, "secondary.json"))
    log(f"secondary: {len(out['arms'])} curves, {len(out['reads'])} reads")
    return 0


def _norm(entry: dict, seeds: list[int], all_seeds: list[int], ceiling: float,
          boot: np.ndarray) -> tuple[float, np.ndarray]:
    pos = [all_seeds.index(s) for s in seeds]
    per_seed = [np.asarray(entry["counts"][k]) for k in pos]
    nps = np.asarray(entry["null_per_seed"], dtype=np.float64)[:, pos]
    return sc.normalised_reps(per_seed, nps, ceiling, boot)


def ablation(_a) -> int:
    """ABLATION_REGISTRATION B: paired Delta per decision, behaviour and level."""
    cells = _cells()
    parts = cio.load(PARTS)
    ceil = ceilings(parts)
    labs = ("investigation", "mount", "groom")
    boots = {"t1": sc.boot_index(len(parts["eval_t1"]), seed=0),
             "g": sc.boot_index(len(parts["eval_g"]), seed=1)}
    boot0 = {"t1": boots["t1"], "g": sc.boot_index(len(parts["eval_g"]), seed=0)}
    # Smoothing inside the nulls: does a median-smoothed control fire the rule?
    med_ctrl_fire = {i: any(v.get("recoverable") for c in ("a-med-white", "a-med-phase")
                            for v in cells.get((i, c), {"labels": {}})["labels"].values())
                     for i in INSTRUMENTS}
    rows: list = []
    for fam, decision in A.ABLATION_DECISION.items():
        tpl, _levels, inst = A.FAMILIES[fam]
        for i in inst:
            for m in A.ABLATION_LEVELS:
                V = cells[(i, A.replace(tpl, level=m).cond)]
                R = cells[(i, f"m{m:g}")]
                seeds = list(A.seeds(fam, i))
                per: dict = {}
                pooled_reps = []
                for lab in labs:
                    ve, re_ = V["labels"][lab], R["labels"][lab]
                    if not ve["support"]["ok"]:
                        continue
                    ls = "g" if lab == "groom" else "t1"
                    metric = ve["metric"]
                    nv, rv = _norm(ve, seeds, V["seeds"], ceil[lab][metric], boot0[ls])
                    nr, rr = _norm(re_, seeds, R["seeds"], ceil[lab][metric], boot0[ls])
                    d = rv - rr
                    lo, hi = (float(x) for x in np.quantile(d, [0.025, 0.975]))
                    per[lab] = {"delta": nv - nr, "ci": [lo, hi], "variant": nv, "reference": nr,
                                "abs_delta_lower": sc.abs_delta_lower(lo, hi)}
                    _pv, pvr = _norm(ve, seeds, V["seeds"], ceil[lab][metric], boots[ls])
                    _pr, prr = _norm(re_, seeds, R["seeds"], ceil[lab][metric], boots[ls])
                    pooled_reps.append(pvr - prr)
                pd_ = np.mean(pooled_reps, axis=0)
                plo, phi = (float(x) for x in np.quantile(pd_, [0.025, 0.975]))
                nar = fam == "a-med-m" and med_ctrl_fire[i]
                rows.append({"decision": decision, "family": fam, "instrument": i, "level": m,
                             "seeds": seeds, "per_label": per,
                             "pooled": {"delta": float(np.mean([v["delta"] for v in per.values()])),
                                        "ci": [plo, phi],
                                        "abs_delta_lower": sc.abs_delta_lower(plo, phi)},
                             "verdict": "NOT_A_RESULT" if nar else "MEASURED",
                             "reason": ("a median-smoothed control fired the recoverable rule"
                                        if nar else "")})
    rows.append({"decision": "smoothing: Wiener", "verdict": "NOT_A_RESULT",
                 "reason": "no committed implementation for non-Luna data (GAPS G11)"})
    ranked = sorted([r for r in rows if r["verdict"] == "MEASURED"],
                    key=lambda r: -r["pooled"]["abs_delta_lower"])
    write_json({**provenance.header(anchors.LUNA, stage="calms_ablation",
                                    unverified="CalMS21 external; descriptive"),
                "registration": "results/step1/ABLATION_REGISTRATION.md B",
                "median_controls_fire": med_ctrl_fire, "rows": rows,
                "ranking_pooled": [{k: r[k] for k in ("decision", "instrument", "level")}
                                   | {"abs_delta_lower": r["pooled"]["abs_delta_lower"],
                                      "delta": r["pooled"]["delta"]} for r in ranked]},
               os.path.join(OUT, "ablation.json"))
    log(f"ablation: {len(rows)} rows; top: {ranked[0]['decision'] if ranked else None}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("cell", "combine", "secondary", "ablation", "cells"))
    p.add_argument("--instrument", choices=INSTRUMENTS)
    p.add_argument("--cond", default=None)
    p.add_argument("--index", type=int, default=None, help="array index into `cells`")
    p.add_argument("--shifts", type=int, default=sc.N_SHIFTS)
    a = p.parse_args(argv)
    cl = [(i, c) for i, _f, c in A.conds()]
    if a.phase == "cells":
        print("\n".join(f"{i} {c}" for i, c in cl))
        return 0
    if a.phase == "cell" and a.index is not None:
        a.instrument, a.cond = cl[a.index]
    return {"cell": cell, "combine": combine, "secondary": secondary,
            "ablation": ablation}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
