"""MOSEQ §3-7: the gates, the reported syllable statistics, and the context test.

    python3 scripts/moseq_gates.py score --model real_s0     # one model; CPU array
    python3 scripts/moseq_gates.py combine                    # -> results/moseq.json

`score` reads `work/moseq/export/<model>.npz` and `work/moseq/labels/<model>.npz`
(both written by `moseq_kpms.py` in the kpms venv). It fits the single-regime
AR(3) baseline on MoSeq's own inferred latents for the fit recordings and
scores both models on its inferred report latents with `vieb.moseq.forward`
(Amendment 2). It writes per-recording sums to
`work/moseq/scores/<model>.json`. `combine` reads those, the report syllables
(`work/moseq/labels/`), and the pixel freezing (`work/moseq/freeze/`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab                       # noqa: E402
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.moseq import forward as fw                                 # noqa: E402
from vieb.tok import config                                          # noqa: E402

WORK = os.path.join(config.REPO, "work", "moseq")
REAL = tuple(f"real_s{s}" for s in range(5))
WHITE = tuple(f"white_s{s}" for s in range(3))
PHASE = tuple(f"phase_s{s}" for s in range(3))
MODELS = REAL + WHITE + PHASE
NLAGS = 3
MARGIN = 0.05            # §3, nats/frame
AMI_BAR = 0.70           # §4
IN_USE = 0.005           # §5, kpms convention
REFUSE_SHARE = 0.05      # §6 as amended
LEAK_BAR = 0.60          # §7
N_BOOT, SEED = 2000, 0
DAYS, CONTEXTS = (3, 4, 5, 6, 7), ("A", "B")
#: CLR needs a positive count; one frame per syllable, fixed before any label
#: existed. Not in the registration, which says only "CLR-transformed".
CLR_PSEUDOCOUNT = 1.0


# ---- score: one model -------------------------------------------------------

def score(a) -> int:
    z = np.load(os.path.join(WORK, "export", f"{a.model}.npz"))
    lz = np.load(os.path.join(WORK, "labels", f"{a.model}.npz"))
    assert [str(r) for r in lz["recording_ids"]] == [str(r) for r in z["rids_report"]]
    assert np.array_equal(lz["bounds"], z["bounds_report"])
    fit_runs = fw.runs(z["broken_fit"], z["bounds_fit"])
    ar_Ab, ar_Q = fw.fit_ar(z["x_fit"], fit_runs, NLAGS)
    rep_runs = fw.runs(z["broken_report"], z["bounds_report"])
    m, b, rec = fw.score(lz["x"], rep_runs, z["Ab"], z["Q"], z["pi"], ar_Ab, ar_Q)
    rids = [str(r) for r in z["rids_report"]]
    fin = np.isfinite(m) & np.isfinite(b)
    per = {}
    for i, rid in enumerate(rids):
        sel = rec == i
        ok = sel & fin
        per[rid] = {"n_scored": int(sel.sum()), "n_nonfinite": int((sel & ~fin).sum()),
                    "sum_gain": float((m[ok] - b[ok]).sum()),
                    "sum_moseq": float(m[ok].sum()), "sum_ar": float(b[ok].sum())}
    n_total = int(z["bounds_report"][-1])
    os.makedirs(os.path.join(WORK, "scores"), exist_ok=True)
    with open(os.path.join(WORK, "scores", f"{a.model}.json"), "w") as fh:
        json.dump({"model": a.model, "n_report_frames": n_total,
                   "n_selected": int(m.size), "n_nonfinite": int((~fin).sum()),
                   "per_recording": per}, fh)
    log(f"  {a.model}: {m.size:,} frames scored of {n_total:,}; "
        f"gain {np.mean(m[fin] - b[fin]):.4f} nats/frame")
    return 0


# ---- combine ----------------------------------------------------------------

def _scores(model: str) -> dict:
    with open(os.path.join(WORK, "scores", f"{model}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _per_animal_gain(sc: dict) -> dict[str, float]:
    tot: dict[str, list[float]] = {}
    for rid, r in sc["per_recording"].items():
        n = r["n_scored"] - r["n_nonfinite"]
        t = tot.setdefault(lab.animal_tag(rid), [0.0, 0.0])
        t[0] += r["sum_gain"]
        t[1] += n
    return {k: v[0] / v[1] for k, v in tot.items() if v[1] > 0}


def gate1() -> dict:
    g = {m: _per_animal_gain(_scores(m)) for m in MODELS}
    animals = sorted(set(g["real_s0"]) & set(g["white_s0"]) & set(g["phase_s0"]))
    arr = {m: np.array([g[m][k] for k in animals]) for m in ("real_s0", "white_s0", "phase_s0")}

    def stat(ix):
        return arr["real_s0"][ix].mean() - max(arr["white_s0"][ix].mean(),
                                               arr["phase_s0"][ix].mean())
    rng = np.random.default_rng(SEED)
    reps = np.array([stat(rng.integers(0, len(animals), len(animals)))
                     for _ in range(N_BOOT)])
    point = float(stat(np.arange(len(animals))))
    lo, hi = (float(v) for v in np.quantile(reps, [0.025, 0.975]))
    per_model = {m: boot.animal_interval(list(g[m].values()), list(g[m].keys()),
                                         how="mean", n_boot=N_BOOT, seed=SEED)
                 for m in MODELS}
    refused = {}
    for m in MODELS:
        sc = _scores(m)
        refused[m] = {"selected_share": sc["n_selected"] / sc["n_report_frames"],
                      "nonfinite_share_of_selected":
                          sc["n_nonfinite"] / max(sc["n_selected"], 1)}
    return {"difference": {"point": point, "lo": lo, "hi": hi, "n_animals": len(animals),
                           "statistic": "gain_real - max(gain_white, gain_phase), seed 0"},
            "pass": bool(lo > MARGIN), "margin": MARGIN,
            "gain_by_model": per_model, "scoring": refused}


def _labels(model: str) -> dict:
    z = np.load(os.path.join(WORK, "labels", f"{model}.npz"))
    return {"syllable": z["syllable"].astype(np.int64), "bounds": z["bounds"],
            "rids": [str(r) for r in z["recording_ids"]]}


def gate2() -> dict:
    from sklearn.metrics import adjusted_mutual_info_score as ami

    lab_ = {m: _labels(m)["syllable"] for m in MODELS}
    out = {}
    for fam in (REAL, WHITE, PHASE):
        vals = {f"{p}~{q}": float(ami(lab_[p], lab_[q]))
                for p, q in itertools.combinations(fam, 2)}
        out[fam[0].split("_")[0]] = {"pairs": vals,
                                     "median": float(np.median(list(vals.values())))}
    out["pass"] = bool(out["real"]["median"] >= AMI_BAR)
    out["bar"] = AMI_BAR
    return out


def dwell_stats(model: str) -> dict:
    L = _labels(model)
    s, b = L["syllable"], L["bounds"]
    frac = np.bincount(s) / s.size
    runs = []
    for i in range(len(b) - 1):
        seg = s[b[i]:b[i + 1]]
        cut = np.flatnonzero(np.diff(seg)) + 1
        runs.append(np.diff(np.r_[0, cut, seg.size]))
    r = np.concatenate(runs).astype(np.float64)
    top = np.sort(frac)[::-1]
    return {"syllables_in_use": int((frac > IN_USE).sum()),
            "syllables_used": int((frac > 0).sum()),
            "median_dwell_frames": float(np.median(r)),
            "dwell_var_over_mean2": float(r.var() / r.mean() ** 2),
            "top10_share": float(top[:10].sum())}


def context() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    L = _labels("real_s0")
    s, b, rids = L["syllable"], L["bounds"], L["rids"]
    use = np.flatnonzero(np.bincount(s) / s.size > IN_USE)
    rows = []
    for i, rid in enumerate(rids):
        p = lab.parse(rid)
        if str(p["context_letter"]) not in CONTEXTS or int(p["day"]) not in DAYS:
            continue
        fpath = os.path.join(WORK, "freeze", f"{rid}.npz")
        if not os.path.exists(fpath):
            continue
        frz = np.load(fpath)["freeze"].astype(bool)
        seg = s[b[i]:b[i + 1]]
        n = min(frz.size, seg.size)
        frz, seg = frz[:n], seg[:n]

        def clr(sel):
            c = np.array([(seg[sel] == k).sum() for k in use], dtype=np.float64)
            lc = np.log(c + CLR_PSEUDOCOUNT)
            return lc - lc.mean()
        rows.append({"rid": rid, "animal": lab.animal_tag(rid),
                     "y": int(str(p["context_letter"]) == "B"),
                     "F": float(frz.mean()), "S_move": clr(~frz), "S_freeze": clr(frz),
                     "n_freeze": int(frz.sum())})
    y = np.array([r["y"] for r in rows])
    an = np.array([r["animal"] for r in rows])
    F = np.array([[r["F"]] for r in rows])
    S = np.array([r["S_move"] for r in rows])
    Sf = np.array([r["S_freeze"] for r in rows])
    uniq = np.unique(an)
    perm = np.random.default_rng(SEED).permutation(uniq)
    folds = np.array_split(perm, 5)

    def per_animal_ba(X):
        pred = np.zeros_like(y)
        for f in folds:
            te = np.isin(an, f)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[~te], y[~te])
            pred[te] = m.predict(X[te])
        ba = {}
        for k in uniq:
            sel = an == k
            if len(set(y[sel])) < 2:
                continue
            ba[k] = 0.5 * sum((pred[sel & (y == c)] == c).mean() for c in (0, 1))
        return ba
    ba_F, ba_FS = per_animal_ba(F), per_animal_ba(np.hstack([F, S]))
    ba_S, ba_leak = per_animal_ba(S), per_animal_ba(Sf)
    ks = sorted(set(ba_F) & set(ba_FS))
    diff = boot.animal_interval([ba_FS[k] - ba_F[k] for k in ks], ks, how="mean",
                                n_boot=N_BOOT, seed=SEED)
    iv = {name: boot.animal_interval(list(d.values()), list(d.keys()), how="mean",
                                     n_boot=N_BOOT, seed=SEED)
          for name, d in (("F", ba_F), ("F+S", ba_FS), ("S", ba_S),
                          ("S_on_freezing_frames", ba_leak))}
    leak = iv["S_on_freezing_frames"]["lo"] > LEAK_BAR
    adds = diff["lo"] > 0
    verdict = ("CONFOUNDED" if adds and leak else "ADDS" if adds
               else "REPRODUCES FREEZING")
    return {"verdict": verdict, "ba_gain_F+S_over_F": diff, "balanced_accuracy": iv,
            "leak": bool(leak), "n_recordings": len(rows), "n_animals": int(uniq.size),
            "syllables_in_use": int(use.size), "freeze_fraction_mean": float(F.mean()),
            "clr_pseudocount": CLR_PSEUDOCOUNT}


def combine(a) -> int:
    g1, g2 = gate1(), gate2()
    nonfin = max(v["nonfinite_share_of_selected"] for v in g1["scoring"].values())
    if nonfin > REFUSE_SHARE:
        verdict = "NOT_A_RESULT"
    elif g1["pass"] and g2["pass"]:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    failing = [n for n, g in (("gate1_likelihood", g1), ("gate2_seed_ami", g2))
               if not g["pass"]]
    res = {"verdict": verdict, "failing": failing, "gate1": g1, "gate2": g2,
           "syllables": {m: dwell_stats(m) for m in MODELS},
           "context": context()}
    res["context"]["read_as"] = ("evidence about behaviour" if verdict == "PASS"
                                 else "descriptive only (MOSEQ did not PASS)")
    write_json({**provenance.header(anchors.LUNA, stage="moseq",
                                    unverified="a report-split test"),
                "inherited_digest": spine.digest(),
                "registration": "results/MOSEQ_PREREGISTRATION.md + Amendment 1",
                "results": res}, config.PATHS.result("moseq.json"))
    log(f"  gate 1 diff {g1['difference']['point']:.4f} "
        f"[{g1['difference']['lo']:.4f}, {g1['difference']['hi']:.4f}] "
        f"-> {'PASS' if g1['pass'] else 'FAIL'}")
    log(f"  gate 2 median AMI {g2['real']['median']:.3f} -> "
        f"{'PASS' if g2['pass'] else 'FAIL'}")
    log(f"  verdict {verdict}; context {res['context']['verdict']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("score", "combine"))
    p.add_argument("--model", choices=MODELS)
    a = p.parse_args(argv)
    return {"score": score, "combine": combine}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
