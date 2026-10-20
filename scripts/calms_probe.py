"""STEP 3A: a linear probe on the pipeline's features, and stage-by-stage localisation.

Registered in `results/step3a/REGISTRATION.md` (629d3ba). Reads CalMS21 labels
(a scoring script, like `calms_score.py`).

    python3 scripts/calms_probe.py cell --index 3      # one (stage/feature set, arm) cell
    python3 scripts/calms_probe.py combine             # -> results/step3a/probe.json

Cells: ``(stage, arm)`` for stage in {R, L} x arm in {m0_s0, m1_s0, white_s0}
and stage in {SEG, EMB} x arm in {m0_s0, m1_s0}.
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

from scipy.ndimage import uniform_filter1d                         # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402

from recur import anchors                                          # noqa: E402
from recur.read import Read                                        # noqa: E402
from recur.util import log, write_json                             # noqa: E402
from vieb import provenance                                        # noqa: E402
from vieb.calms import io as cio                                   # noqa: E402
from vieb.calms import score as sc                                 # noqa: E402
from vieb.seg import breaks as bk, embed                           # noqa: E402
from vieb.tok import config, ego                                   # noqa: E402

WORK = os.path.join(config.REPO, "work", "calms")
OUT = os.path.join(config.REPO, "results", "step3a")
CELLS_DIR = os.path.join(WORK, "probe")
PARTS = os.path.join(config.REPO, "results", "step1", "partitions.json")
FPS = 30.0
WIN_S = 0.5
N_SHIFTS = 200
N_EMB_PC = 20
SEED = 0
CELLS = ([("R", a) for a in ("m0_s0", "m1_s0", "white_s0")]
         + [("L", a) for a in ("m0_s0", "m1_s0", "white_s0")]
         + [("SEG", a) for a in ("m0_s0", "m1_s0")]
         + [("EMB", a) for a in ("m0_s0", "m1_s0")])
TASKS = (("t1", "map_t1", "eval_t1", sc.T1_LABELS), ("g", "map_g", "eval_g", ("groom",)))
CAVEAT = ("unit = sequence; a resident can recur across sequences undetectably, "
          "so intervals may be too narrow")


def _module(name: str):
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def window_stats(x: np.ndarray) -> np.ndarray:
    """Per channel: centred +-WIN_S mean and SD, within ONE recording."""
    w = 2 * int(round(WIN_S * FPS)) + 1
    m = uniform_filter1d(x, size=w, axis=0, mode="nearest")
    m2 = uniform_filter1d(x * x, size=w, axis=0, mode="nearest")
    return np.concatenate([x, m, np.sqrt(np.maximum(m2 - m * m, 0.0))], axis=1)


def ego_X(arm: str, v: str, sd: np.ndarray, *, with_abstain: bool = False):
    with np.load(os.path.join(WORK, "raw", arm, f"{v}.npz")) as z:
        pose = np.asarray(z["pose"], dtype=np.float64)
        ab = np.asarray(z["abstain"], dtype=bool)
    X, valid = ego.transform(pose, cio.LUNA_ELL_PX, FPS)
    Xs = X / sd[None, :]
    return (Xs, ab | ~valid) if with_abstain else Xs


def features(stage: str, arm: str, vids: list[str], parts: dict) -> dict[str, np.ndarray]:
    sr = _module("seg_recur")
    sd = sr.basis_sd()
    if stage == "R":
        return {v: window_stats(ego_X(arm, v, sd)) for v in vids}
    if stage == "L":
        with np.load(os.path.join(WORK, "moseq", "labels", f"{arm}_s0.npz")) as z:
            x, b = np.asarray(z["x"], np.float64), z["bounds"]
            ids = [str(r) for r in z["recording_ids"]]
        return {v: window_stats(x[b[i]:b[i + 1]]) for i, v in enumerate(ids) if v in set(vids)}
    if stage == "SEG":
        with np.load(os.path.join(WORK, "states", arm, "D.npz")) as z:
            D = {str(v): np.asarray(z[f"b_{v}"], np.int64) for v in z["videos"]}
        out = {}
        for v in vids:
            X = ego_X(arm, v, sd)
            cuts = np.r_[0, np.sort(D[v]), X.shape[0]]
            F = np.zeros_like(X)
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi > lo:
                    F[lo:hi] = X[lo:hi].mean(axis=0)
            out[v] = F
        return out
    if stage == "EMB":
        idx = bk.CHANNEL_GROUPS["shape"]
        segs: dict[str, list] = {}
        mats: dict[str, np.ndarray] = {}
        for v in vids:
            X, ab = ego_X(arm, v, sd, with_abstain=True)
            a = {"X": X, "abstain": ab, "bounds": np.array([0, X.shape[0]])}
            rows = sr.segments_of(a, idx, fps=FPS, k_mad=3.0)
            sel = [r for r, k in zip(rows, embed.selectable(rows)) if k]
            segs[v] = [(int(r["start"]), int(r["stop"])) for r in sel]
            if sel:
                f, _s = embed.stack_segments(X, sel, cols=idx, n=embed.SEG_GRID)
                mats[v] = f.reshape(len(sel), -1).astype(np.float64)
            else:
                mats[v] = np.zeros((0, embed.SEG_GRID * len(idx)))
        mp = sorted(set(parts["map_t1"]) | set(parts["map_g"]))
        M = np.concatenate([mats[v] for v in mp if mats[v].size])
        mu = M.mean(0)
        _u, _s, vt = np.linalg.svd(M - mu, full_matrices=False)
        comp = vt[:N_EMB_PC]
        out = {}
        for v in vids:
            T = ego_X(arm, v, sd).shape[0]
            F = np.zeros((T, N_EMB_PC + 1))
            for (lo, hi), row in zip(segs[v], mats[v]):
                F[lo:hi, :N_EMB_PC] = (row - mu) @ comp.T
                F[lo:hi, N_EMB_PC] = 1.0
            out[v] = F
        return out
    raise ValueError(stage)


def probe_counts(F: dict, y: dict, map_ids: list[str], eval_ids: list[str], cls: int
                 ) -> tuple[np.ndarray, float]:
    """Fit on map frames, predict eval; per-eval-sequence event counts and frame AUROC."""
    Xm = np.concatenate([F[v] for v in map_ids])
    ym = np.concatenate([np.asarray(y[v]) == cls for v in map_ids]).astype(int)
    if ym.min() == ym.max():
        return np.asarray([sc.event_counts(y[v], np.zeros_like(y[v]), cls) for v in eval_ids],
                          dtype=np.float64), float("nan")
    scaler = StandardScaler().fit(Xm)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(Xm), ym)
    rows, probs, truth = [], [], []
    for v in eval_ids:
        p = clf.predict_proba(scaler.transform(F[v]))[:, 1]
        yhat = np.where(p > 0.5, cls, -1)
        rows.append(sc.event_counts(y[v], yhat, cls))
        probs.append(p)
        truth.append(np.asarray(y[v]) == cls)
    t = np.concatenate(truth)
    auc = float(roc_auc_score(t, np.concatenate(probs))) if 0 < t.mean() < 1 else float("nan")
    return np.asarray(rows, dtype=np.float64), auc


def cell(a) -> int:
    stage, arm = CELLS[a.index]
    parts = cio.load(PARTS)
    vids = cio.scored(parts)
    F = features(stage, arm, vids, parts)
    res: dict = {"stage": stage, "arm": arm, "labels": {}}
    with open(os.path.join(config.REPO, "results", "step1", "recovery.json"), encoding="utf-8") as fh:
        ceil = json.load(fh)["ceilings"]
    for label_set, mp, ev, targets in TASKS:
        y = {v: a_[: F[v].shape[0]] if v in F else a_ for v, a_ in sc.labels(label_set).items()}
        map_ids, eval_ids = parts[mp], parts[ev]
        for v in map_ids + eval_ids:
            assert F[v].shape[0] == y[v].shape[0], (v, F[v].shape, y[v].shape)
        ids = sorted(set(map_ids) | set(eval_ids))
        boot = sc.boot_index(len(eval_ids))
        for lab in targets:
            cls = sc.class_id(label_set, lab)
            obs, auc = probe_counts(F, y, map_ids, eval_ids, cls)
            rng = np.random.default_rng(SEED)
            null = []
            for _ in range(a.shifts):
                ys = sc.shifted(y, ids, rng)
                c, _a = probe_counts(F, ys, map_ids, eval_ids, cls)
                null.append(sc.f1(c.sum(0)))
            sup = sc.support(y, eval_ids, label_set, lab)
            entry = {"support": sup, "auroc": auc, "n_eval": len(eval_ids)}
            if sup["ok"]:
                entry.update(sc.cell([obs], np.asarray(null), ceil[lab]["event"], boot))
            else:
                entry["score"] = sc.f1(obs.sum(0))
            entry["null"] = null
            res["labels"][lab] = entry
            log(f"{stage} {arm} {lab}: {entry.get('normalised', float('nan')):+.3f} auc {auc:.3f}")
    os.makedirs(CELLS_DIR, exist_ok=True)
    write_json(res, os.path.join(CELLS_DIR, f"{stage}_{arm}.json"))
    return 0


def combine(_a) -> int:
    cells = {}
    for p in sorted(glob.glob(os.path.join(CELLS_DIR, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        cells[(c["stage"], c["arm"])] = c
    missing = [k for k in CELLS if k not in cells]
    assert not missing, missing
    ctrl_fire = any(v.get("recoverable") for st in ("R", "L")
                    for v in cells[(st, "white_s0")]["labels"].values())
    with open(os.path.join(config.REPO, "results", "step1", "recovery.json"), encoding="utf-8") as fh:
        s1 = json.load(fh)
    reads, table = [], {}
    for (stage, arm), c in cells.items():
        for lab, v in c["labels"].items():
            key = f"{stage}|{arm}|{lab}"
            table[key] = {k: v.get(k) for k in ("normalised", "ci", "score", "null_p95",
                                                "recoverable", "auroc", "support")}
            if arm == "white_s0" or "recoverable" not in v:
                continue
            verdict = ("NOT_A_RESULT" if ctrl_fire else
                       "PASS" if v["recoverable"] else "INCONCLUSIVE" if v["straddles"] else "FAIL")
            reads.append(Read(verdict, {"stage": stage, "arm": arm, "label": lab, "probe": "logistic"},
                              f"normalised {v['normalised']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]; "
                              f"AUROC {v['auroc']:.3f}; {CAVEAT}", n_effective=v["n_eval"]).to_dict())
    readings = {}
    for lab in sc.T1_LABELS + ("groom",):
        r0 = cells[("R", "m0_s0")]["labels"][lab]
        if "recoverable" not in r0:
            readings[lab] = "GRID_LIMITED"
            continue
        inst = any(s1["curves"].get(f"{i}|{lab}", {}).get("recoverable", [False])[0] for i in ("D", "C", "M"))
        readings[lab] = ("NOT_A_RESULT (control fired)" if ctrl_fire else
                         "method loses information the representation holds" if r0["recoverable"] and not inst
                         else "the representation is the problem" if not r0["recoverable"] and not inst
                         else "instruments recover it")
    write_json({**provenance.header(anchors.LUNA, stage="calms_probe",
                                    unverified="CalMS21 external; post-result branch 3A"),
                "registration": "results/step3a/REGISTRATION.md", "control_fires": ctrl_fire,
                "readings_native": readings, "table": table, "reads": reads},
               os.path.join(OUT, "probe.json"))
    log(f"control fires: {ctrl_fire}; readings {readings}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("cell", "combine"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--shifts", type=int, default=N_SHIFTS)
    a = p.parse_args(argv)
    return {"cell": cell, "combine": combine}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
