"""STEP 3A Amendment 1: scorer check, speed baseline, and a bout-aware probe.

Registered in `results/step3a/REGISTRATION.md` Amendment 1 (9a77cc5), post-result.
Supervised and diagnostic; it does not change the Step 1 verdict.

    python3 scripts/calms_bout_probe.py scorer          # A1.1: identity + flicker (raw)
    python3 scripts/calms_bout_probe.py cell --index 3  # one (feature set, arm, task) cell
    python3 scripts/calms_bout_probe.py combine         # -> results/step3a/bout_probe.json

Feature sets: B (speed and stillness only, 6 features), R (17-D ego + window
stats, as 3A.1), L (MoSeq latent + window stats, as 3A.1). The readout
(median smoothing w, threshold tau, minimum bout d) is chosen by out-of-fold
event F1 on the map partition, folds by sequence, and refitted inside every
circular-shift null replicate.
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

from scipy.ndimage import median_filter                            # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402

from recur import anchors                                          # noqa: E402
from recur.read import Read                                        # noqa: E402
from recur.util import log, write_json                             # noqa: E402
from vieb import provenance                                        # noqa: E402
from vieb.calms import io as cio                                   # noqa: E402
from vieb.calms import score as sc                                 # noqa: E402
from vieb.hsmm import mouse as M                                   # noqa: E402
from vieb.tok import config                                        # noqa: E402

WORK = os.path.join(config.REPO, "work", "calms")
OUT = os.path.join(config.REPO, "results", "step3a")
CELLS_DIR = os.path.join(WORK, "bout_probe")
PARTS = os.path.join(config.REPO, "results", "step1", "partitions.json")
FPS = 30.0
SEED = 0
N_SHIFTS = 100
N_BOOT = 2000
FLICKER_P = (0.01, 0.05, 0.10, 0.20)
TAUS = tuple(round(0.05 * k, 2) for k in range(1, 19))           # 0.05 .. 0.90
WS = (0.0, 0.2, 0.5, 1.0)                                        # median smoothing, s
DS = (0.0, 0.2, 0.5, 1.0)                                        # minimum bout, s
N_FOLDS = {"t1": 5, "g": 3}
SETS = ("B", "R", "L")
ARMS = ("m0_s0", "m1_s0", "white_s0")
TASKS = {"t1": ("map_t1", "eval_t1", sc.T1_LABELS), "g": ("map_g", "eval_g", ("groom",))}
CELLS = [(f, a, t) for f in SETS for a in ARMS for t in TASKS]
LOG_FLOOR = 1e-3                                                 # BL/s, M's grid floor
CAVEAT = ("unit = sequence; a resident can recur across sequences undetectably, "
          "so intervals may be too narrow")


def probe_mod():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calms_probe.py")
    spec = importlib.util.spec_from_file_location("calms_probe_mod", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["calms_probe"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def speed_X(arm: str, v: str, pm) -> np.ndarray:
    """B: log10 centroid speed and pose-change rate (w = 0.2 s), with +-0.5 s mean and SD."""
    with np.load(os.path.join(WORK, "raw", arm, f"{v}.npz")) as z:
        pose = np.asarray(z["pose"], dtype=np.float64)
    assert np.isfinite(pose).all(), v
    vv, rr = M.signal(pose, cio.LUNA_ELL_PX, FPS)
    x = np.log10(np.maximum(np.stack([vv, rr], axis=1), LOG_FLOOR))
    return np.asarray(pm.window_stats(x))


def features(fset: str, arm: str, vids: list[str], parts: dict) -> dict[str, np.ndarray]:
    pm = probe_mod()
    if fset == "B":
        return {v: speed_X(arm, v, pm) for v in vids}
    return dict(pm.features(fset, arm, vids, parts))


# --------------------------------------------------------------------------
# Readout
# --------------------------------------------------------------------------

def odd(w_s: float) -> int:
    return int(w_s * FPS) // 2 * 2 + 1


def readout(p: np.ndarray, w: float, tau: float, d: float) -> np.ndarray:
    """Boolean prediction for ONE sequence: median smoothing, threshold, minimum bout."""
    q = median_filter(p, size=odd(w), mode="nearest") if w > 0 else p
    m = q > tau
    if d > 0:
        n = int(round(d * FPS))
        for lo, hi in sc.runs(m).tolist():
            if hi - lo < n:
                m[lo:hi] = False
    return m


def grid_f1(probs: dict[str, np.ndarray], truth: dict[str, np.ndarray]) -> tuple[tuple, float]:
    """The readout with the best pooled event F1; ties to smaller w, smaller d, larger tau."""
    best, bf = (WS[0], max(TAUS), DS[0]), -1.0
    for w in WS:
        sm = {v: (median_filter(p, size=odd(w), mode="nearest") if w > 0 else p)
              for v, p in probs.items()}
        for d in DS:
            for tau in sorted(TAUS, reverse=True):
                c = np.zeros(3)
                for v, q in sm.items():
                    m = readout(q, 0.0, tau, d)
                    c += sc.event_counts(truth[v], m, 1)
                f = sc.f1(c)
                if f > bf:
                    best, bf = (w, tau, d), f
    return best, bf


def fit_predict(F: dict, yb: dict, fit_ids: list[str], pred_ids: list[str]) -> dict[str, np.ndarray]:
    X = np.concatenate([F[v] for v in fit_ids])
    t = np.concatenate([yb[v] for v in fit_ids]).astype(int)
    if t.min() == t.max():
        return {v: np.full(F[v].shape[0], float(t[0])) for v in pred_ids}
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X), t)
    return {v: clf.predict_proba(scaler.transform(F[v]))[:, 1] for v in pred_ids}


def run_probe(F: dict, yb: dict, map_ids: list[str], eval_ids: list[str], k: int
              ) -> tuple[np.ndarray, dict, tuple, float]:
    """OOF readout choice on map, refit on all map, scored per eval sequence."""
    folds = np.arange(len(map_ids)) % k
    oof: dict[str, np.ndarray] = {}
    for f in range(k):
        tr = [v for v, g in zip(map_ids, folds) if g != f]
        te = [v for v, g in zip(map_ids, folds) if g == f]
        oof.update(fit_predict(F, yb, tr, te))
    choice, f_oof = grid_f1(oof, {v: yb[v] for v in map_ids})
    pe = fit_predict(F, yb, map_ids, eval_ids)
    rows = np.asarray([sc.event_counts(yb[v], readout(pe[v], *choice), 1) for v in eval_ids],
                      dtype=np.float64)
    return rows, pe, choice, f_oof


def cell(a) -> int:
    fset, arm, task = CELLS[a.index]
    mp, ev, targets = TASKS[task]
    parts = cio.load(PARTS)
    map_ids, eval_ids = list(parts[mp]), list(parts[ev])
    F = features(fset, arm, sorted(set(map_ids) | set(eval_ids)), parts)
    y = {v: np.asarray(a_)[: F[v].shape[0]] for v, a_ in sc.labels(task).items() if v in F}
    for v in map_ids + eval_ids:
        assert F[v].shape[0] == y[v].shape[0], (v, F[v].shape, y[v].shape)
    with open(os.path.join(config.REPO, "results", "step1", "recovery.json"), encoding="utf-8") as fh:
        ceil = json.load(fh)["ceilings"]
    ids = sorted(set(map_ids) | set(eval_ids))
    boot = sc.boot_index(len(eval_ids))
    res: dict = {"set": fset, "arm": arm, "task": task, "eval_ids": eval_ids, "labels": {}}
    probs_out: dict = {}
    for lab in targets:
        cls = sc.class_id(task, lab)
        yb = {v: np.asarray(y[v]) == cls for v in ids}
        obs, pe, choice, f_oof = run_probe(F, yb, map_ids, eval_ids, N_FOLDS[task])
        rng = np.random.default_rng(SEED)
        null, null_choice = [], []
        for _ in range(a.shifts):
            ys = sc.shifted(y, ids, rng)
            ysb = {v: np.asarray(ys[v]) == cls for v in ids}
            c, _p, ch, _f = run_probe(F, ysb, map_ids, eval_ids, N_FOLDS[task])
            null.append(sc.f1(c.sum(0)))
            null_choice.append(list(ch))
        t = np.concatenate([yb[v] for v in eval_ids])
        pc = np.concatenate([pe[v] for v in eval_ids])
        sup = sc.support(y, eval_ids, task, lab)
        entry = {"support": sup, "n_eval": len(eval_ids), "readout": {"w_s": choice[0], "tau": choice[1],
                 "d_s": choice[2], "oof_map_f1": f_oof},
                 "auroc": float(roc_auc_score(t, pc)) if 0 < t.mean() < 1 else None,
                 "pr_auc": float(average_precision_score(t, pc)) if t.any() else None,
                 "prevalence": float(t.mean()), "null": null, "null_readouts": null_choice,
                 "event_counts": obs.tolist()}
        if sup["ok"]:
            entry.update(sc.cell([obs], np.asarray(null), ceil[lab]["event"], boot))
        else:
            entry["score"] = sc.f1(obs.sum(0))
        res["labels"][lab] = entry
        probs_out[lab] = pc.astype(np.float32)
        log(f"{fset} {arm} {lab}: readout {choice}, score {entry.get('score'):.3f} "
            f"norm {entry.get('normalised', float('nan')):+.3f} auroc {entry['auroc']} pr {entry['pr_auc']}")
    os.makedirs(CELLS_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(CELLS_DIR, f"{fset}_{arm}_{task}_probs.npz"),
                        lengths=np.asarray([y[v].shape[0] for v in eval_ids]), **probs_out)
    write_json(res, os.path.join(CELLS_DIR, f"{fset}_{arm}_{task}.json"))
    return 0


# --------------------------------------------------------------------------
# A1.1: the scorer
# --------------------------------------------------------------------------

def flip(mask: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    k = int(round(p * mask.size))
    idx = rng.choice(mask.size, size=k, replace=False)
    out[idx] = ~out[idx]
    return out


def scorer_check(readouts: dict | None = None) -> dict:
    parts = cio.load(PARTS)
    tol = int(round(sc.TOL_S * FPS))
    out: dict = {"identity": {}, "flicker": {}}
    for task, (mp, ev, targets) in TASKS.items():
        y = sc.labels(task)
        for lab in targets:
            cls = sc.class_id(task, lab)
            for part in (mp, ev):
                ids = parts[part]
                ce = np.sum([sc.event_counts(y[v], y[v], cls) for v in ids], axis=0)
                cb = np.sum([sc.boundary_counts(sc.true_boundaries(y[v], cls),
                                                sc.true_boundaries(y[v], cls), tol) for v in ids], axis=0)
                out["identity"][f"{lab}|{part}"] = {"event_f1": sc.f1(ce), "boundary_f1": sc.f1(cb),
                                                     "n_events": int(ce[1])}
            rows = {}
            for p in FLICKER_P:
                rng = np.random.default_rng(SEED)
                raw, sm = np.zeros(3), np.zeros(3)
                for v in parts[ev]:
                    tb = np.asarray(y[v]) == cls
                    f = flip(tb, p, rng)
                    raw += sc.event_counts(tb, f, 1)
                    if readouts and lab in readouts:
                        sm += sc.event_counts(tb, readout(f.astype(np.float64), *readouts[lab]), 1)
                rows[str(p)] = {"raw_f1": sc.f1(raw)} | ({"after_readout_f1": sc.f1(sm)}
                                                         if readouts and lab in readouts else {})
            out["flicker"][lab] = rows
    ident = [v for k, v in out["identity"].items()]
    out["identity_pass"] = bool(all(v["event_f1"] == 1.0 and v["boundary_f1"] == 1.0
                                    for v in ident if v["n_events"] > 0))
    return out


def scorer(_a) -> int:
    s = scorer_check()
    log(json.dumps(s, indent=1))
    return 0


# --------------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------------

def seq_boot(lengths: np.ndarray, truth: np.ndarray, probs: dict[str, np.ndarray]) -> dict:
    """Sequence bootstrap of AUROC and PR-AUC per set, and paired deltas against B."""
    cuts = np.r_[0, np.cumsum(lengths)]
    segs = [np.arange(cuts[i], cuts[i + 1]) for i in range(lengths.size)]
    rng = np.random.default_rng(SEED)
    reps: dict = {k: {"auroc": [], "pr_auc": []} for k in probs}
    for _ in range(N_BOOT):
        idx = np.concatenate([segs[i] for i in rng.integers(0, lengths.size, lengths.size)])
        t = truth[idx]
        if not (0 < t.mean() < 1):
            continue
        for k, p in probs.items():
            reps[k]["auroc"].append(roc_auc_score(t, p[idx]))
            reps[k]["pr_auc"].append(average_precision_score(t, p[idx]))
    out: dict = {}
    for k, p in probs.items():
        out[k] = {m: {"point": float(f(truth, p)),
                      "ci": [float(x) for x in np.quantile(reps[k][m], [0.025, 0.975])]}
                  for m, f in (("auroc", roc_auc_score), ("pr_auc", average_precision_score))}
    for k in probs:
        if k == "B":
            continue
        d = np.asarray(reps[k]["pr_auc"]) - np.asarray(reps["B"]["pr_auc"])
        da = np.asarray(reps[k]["auroc"]) - np.asarray(reps["B"]["auroc"])
        out[f"{k}-B"] = {"pr_auc": {"point": out[k]["pr_auc"]["point"] - out["B"]["pr_auc"]["point"],
                                    "ci": [float(x) for x in np.quantile(d, [0.025, 0.975])]},
                         "auroc": {"point": out[k]["auroc"]["point"] - out["B"]["auroc"]["point"],
                                   "ci": [float(x) for x in np.quantile(da, [0.025, 0.975])]}}
    out["n_boot_used"] = len(reps["B"]["auroc"])
    return out


def combine(_a) -> int:
    cells: dict = {}
    for p in sorted(glob.glob(os.path.join(CELLS_DIR, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        cells[(c["set"], c["arm"], c["task"])] = c
    missing = [k for k in CELLS if k not in cells]
    assert not missing, missing
    parts = cio.load(PARTS)
    ctrl_fire = any(v.get("recoverable") for (f, a, t), c in cells.items() if a == "white_s0"
                    for v in c["labels"].values())
    ro = {lab: tuple(v["readout"][k] for k in ("w_s", "tau", "d_s"))
          for t in TASKS for lab, v in cells[("R", "m0_s0", t)]["labels"].items()}
    scor = scorer_check(ro)
    frame: dict = {}
    for arm in ("m0_s0", "m1_s0"):
        for task, (_mp, ev, targets) in TASKS.items():
            y = sc.labels(task)
            Z = {f: np.load(os.path.join(CELLS_DIR, f"{f}_{arm}_{task}_probs.npz")) for f in SETS}
            for lab in targets:
                cls = sc.class_id(task, lab)
                lengths = np.asarray(Z["B"]["lengths"])
                truth = np.concatenate([np.asarray(y[v])[:n] == cls for v, n in zip(parts[ev], lengths)])
                frame[f"{arm}|{lab}"] = seq_boot(lengths, truth,
                                                 {f: np.asarray(Z[f][lab], np.float64) for f in SETS})
                log(f"frame metrics {arm} {lab} done")
    table, reads, readings = {}, [], {}
    for (f, arm, task), c in cells.items():
        for lab, v in c["labels"].items():
            table[f"{f}|{arm}|{lab}"] = {k: v.get(k) for k in ("normalised", "ci", "score", "null_p95",
                                                               "recoverable", "auroc", "pr_auc",
                                                               "readout", "support")}
            if arm == "white_s0" or "recoverable" not in v:
                continue
            verdict = ("NOT_A_RESULT" if ctrl_fire or not scor["identity_pass"] else
                       "PASS" if v["recoverable"] else "INCONCLUSIVE" if v["straddles"] else "FAIL")
            reads.append(Read(verdict, {"set": f, "arm": arm, "label": lab, "probe": "logistic+fitted readout"},
                              f"normalised {v['normalised']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]; "
                              f"readout {v['readout']}; {CAVEAT}", n_effective=v["n_eval"]).to_dict())
    for k, m in frame.items():
        arm, lab = k.split("|")
        for f in ("R", "L"):
            d = m[f"{f}-B"]["pr_auc"]
            reads.append(Read("PASS" if d["ci"][0] > 0 else "FAIL",
                              {"set": f, "arm": arm, "label": lab, "metric": "delta_pr_auc_vs_speed"},
                              f"PR-AUC {m[f]['pr_auc']['point']:.3f} vs B {m['B']['pr_auc']['point']:.3f}; "
                              f"delta {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]; {CAVEAT}",
                              n_effective=len(parts["eval_g" if lab == "groom" else "eval_t1"])).to_dict())
    for arm in ("m0_s0", "m1_s0"):
        for lab in sc.T1_LABELS + ("groom",):
            task = "g" if lab == "groom" else "t1"
            if not scor["identity_pass"]:
                readings[f"{arm}|{lab}"] = "scorer identity check failed: earlier 3A event verdicts void"
                continue
            if ctrl_fire:
                readings[f"{arm}|{lab}"] = "NOT_A_RESULT (control fired)"
                continue
            rec = any(cells[(f, arm, task)]["labels"][lab].get("recoverable") for f in ("R", "L"))
            beyond = any(frame[f"{arm}|{lab}"][f"{f}-B"]["pr_auc"]["ci"][0] > 0 for f in ("R", "L"))
            readings[f"{arm}|{lab}"] = (
                "the unsupervised method loses information the features contain" if rec else
                "information at frame level beyond speed, but a linear probe cannot assemble it into bouts"
                if beyond else "the representation, under a linear probe, lacks it beyond speed")
    write_json({**provenance.header(anchors.LUNA, stage="calms_bout_probe",
                                    unverified="CalMS21 external; post-result Amendment 1; supervised, diagnostic"),
                "registration": "results/step3a/REGISTRATION.md Amendment 1",
                "scorer": scor, "control_fires": ctrl_fire, "readings": readings,
                "frame_metrics": frame, "table": table, "reads": reads},
               os.path.join(OUT, "bout_probe.json"))
    log(f"identity {scor['identity_pass']}; control fires {ctrl_fire}; readings {readings}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("scorer", "cell", "combine"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--shifts", type=int, default=N_SHIFTS)
    a = p.parse_args(argv)
    return {"scorer": scorer, "cell": cell, "combine": combine}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
