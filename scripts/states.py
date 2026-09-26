"""STATES: behaviour as maximally predictive states and a transfer operator.

    python3 scripts/states.py --phase leak           # §2 gates on tune
    python3 scripts/states.py --phase select         # §3: standardise, K*, tau*
    python3 scripts/states.py --phase fit            # §3.6: microstates, P, timescales
    python3 scripts/states.py --phase surrogate --kind phase   # one per kind
    python3 scripts/states.py --phase combine        # §4: gates, states, outputs

READ results/STATES_PREREGISTRATION.md FIRST. Everything is fitted on tune and
fit and reported on report; every surrogate runs through this same code from
standardisation onward (§4 gate 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, splits                              # noqa: E402
from recur.null import microstate as ms                              # noqa: E402
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance, seeds                                   # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.states import embed, micro, operator as op                 # noqa: E402
from vieb.tok import config, surrogate as sr                         # noqa: E402

# Environment overrides exist ONLY for the synthetic smoke test
# (tests/test_states_pipeline.py); a real run leaves them unset.
FEAT = os.environ.get("VIEB_STATES_FEAT", os.path.join(config.REPO, "work", "features"))
WORK = os.environ.get("VIEB_STATES_WORK", os.path.join(config.REPO, "work", "states"))
SEED = 0
N_BOOT = int(os.environ.get("VIEB_STATES_NBOOT", 2000))
K_GRID = (1, 2, 4, 8, 16, 32)
TAU_GRID = (1, 2, 4, 8, 16, 32, 64)
K_TOL = 0.01
#: Amendment 1: the prediction horizon beyond the window's end, frames (0.5 s).
H_FUTURE = 15
TAU_TOL = 0.10
N_TS = 11                     # t_2 .. t_11
MEMBERSHIP = 0.8
LEAK_BAR = 0.60
AMI_BAR = 0.7
PLANT_BAR = 0.8
PLANT_SHARE = 0.20
PLANT_MEAN_S = 2.0
FIT_ROWS = int(os.environ.get("VIEB_STATES_FITROWS", 600_000))
SURROGATES = ("phase", "var5", "microstate", "microstate0", "white")
#: §4 gate 3: the two channels the plant moves -- a body-extension proxy (the
#: nose's egocentric x) and speed (v_x).
PLANT_CHANNELS = (4, 14)


# ---- data -------------------------------------------------------------------

def _split_tags() -> dict[str, list[str]]:
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    tags = sorted(f[:-4] for f in os.listdir(FEAT) if f.endswith(".npz"))
    out: dict[str, list[str]] = {"tune": [], "fit": [], "report": []}
    for t in tags:
        if of.get(t) in out:
            out[of[t]].append(t)
    return out


def load(split: str, use_mask: bool) -> dict:
    """One split's stream, concatenated: X, valid, bounds, animal per recording."""
    xs, vs, bs, an, ctx, day, rspeed, cov, dfn = [], [], [0], [], [], [], [], [], []
    for tag in _split_tags()[split]:
        with np.load(os.path.join(FEAT, f"{tag}.npz"), allow_pickle=False) as z:
            x = z["X_pose"]
            v = z["valid_pose"].copy()
            if use_mask:
                x = np.concatenate([x, z["X_mask"]], axis=1)
                v &= z["valid_mask"]
            b = z["bounds"]
            xs.append(np.nan_to_num(x).astype(np.float32))
            vs.append(v)
            base = bs[-1]
            bs += [base + int(e) for e in b[1:]]
            an += [tag] * (b.size - 1)
            ctx += [str(c) for c in z["context"]]
            day += [int(d) for d in z["day"]]
            rspeed.append(z["raw_speed"])
            cov.append(z["head_covered"])
            dfn.append(z["head_defined"])
    return {"X": np.concatenate(xs), "valid": np.concatenate(vs),
            "bounds": np.asarray(bs, dtype=np.int64), "animal": np.asarray(an),
            "context": np.asarray(ctx), "day": np.asarray(day),
            "raw_speed": np.concatenate(rspeed),
            "head_covered": np.concatenate(cov), "head_defined": np.concatenate(dfn)}


def _save(name: str, obj: dict) -> None:
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _read(name: str) -> dict:
    with open(os.path.join(WORK, name), encoding="utf-8") as fh:
        return json.load(fh)


# ---- §2 leak gates ----------------------------------------------------------

def leak(a) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    res = {}
    for stream, use_mask in (("pose", False), ("mask", True)):
        d = load("tune", use_mask=use_mask)
        cols = slice(0, 17) if stream == "pose" else slice(17, None)
        per_rec = np.repeat(np.arange(d["bounds"].size - 1), np.diff(d["bounds"]))
        still = d["valid"] & (d["raw_speed"] == 0)
        y = (d["context"][per_rec] == "A")[still]
        x = d["X"][still][:, cols]
        grp = d["animal"][per_rec][still]
        animals = np.unique(grp)
        rng = np.random.default_rng(SEED)
        folds = np.array_split(rng.permutation(animals), 5)
        scores, who = [], []
        for f in folds:
            te = np.isin(grp, f)
            if te.sum() == 0 or len(np.unique(y[~te])) < 2:
                continue
            clf = LogisticRegression(max_iter=1000).fit(x[~te], y[~te])
            for an in f:
                m = grp == an
                if m.sum() >= 50 and len(np.unique(y[m])) == 2:
                    scores.append(balanced_accuracy_score(y[m], clf.predict(x[m])))
                    who.append(an)
        b = boot.animal_interval(scores, who, how="mean", n_boot=N_BOOT, seed=SEED)
        res[stream] = {"balanced_accuracy": b, "n_still_frames": int(still.sum()),
                       "dropped": bool(b["lo"] > LEAK_BAR)}
        log(f"  leak {stream}: {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
            f" -> {'DROPPED' if res[stream]['dropped'] else 'kept'}")
    res["use_mask"] = not res["mask"]["dropped"]
    if res["pose"]["dropped"]:
        res["stop"] = "the pose stream itself decodes context during immobility"
    _save("leak.json", res)
    return 0


# ---- standardisation, embedding, microstates --------------------------------

def standardiser(d: dict) -> tuple[np.ndarray, np.ndarray]:
    x = d["X"][d["valid"]].astype(np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def _std(d: dict, mu, sd) -> dict:
    return {**d, "X": ((d["X"] - mu) / sd).astype(np.float32)}


def _labels(m: micro.Micro, d: dict) -> np.ndarray:
    out = np.full(d["X"].shape[0], -1, dtype=np.int64)
    b = d["bounds"]
    for i, e, ok in embed.recordings(d["X"], d["valid"], b, m.k):
        out[b[i]:b[i + 1]] = m.assign(e, ok)
    return out


def _future(d: dict) -> np.ndarray:
    """Features H_FUTURE frames ahead, within the recording; NaN past its end
    or where invalid."""
    x = d["X"].astype(np.float64)
    out = np.full_like(x, np.nan)
    b = d["bounds"]
    for i in range(b.size - 1):
        lo, hi = int(b[i]), int(b[i + 1])
        if hi - lo > H_FUTURE:
            out[lo:hi - H_FUTURE] = x[lo + H_FUTURE:hi]
            bad = ~d["valid"][lo + H_FUTURE:hi]
            out[lo:hi - H_FUTURE][bad] = np.nan
    return out


def _subset(d: dict, animals) -> dict:
    per_rec = np.repeat(np.arange(d["bounds"].size - 1), np.diff(d["bounds"]))
    keep_rec = np.isin(d["animal"], list(animals))
    keep = keep_rec[per_rec]
    lens = np.diff(d["bounds"])[keep_rec]
    return {**d, "X": d["X"][keep], "valid": d["valid"][keep],
            "bounds": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            "animal": d["animal"][keep_rec], "context": d["context"][keep_rec],
            "day": d["day"][keep_rec]}


def select(a) -> int:
    lk = _read("leak.json")
    if "stop" in lk:
        raise SystemExit(lk["stop"])
    tune = load("tune", lk["use_mask"])
    mu, sd = standardiser(tune)
    tune = _std(tune, mu, sd)
    animals = np.unique(tune["animal"])
    rng = np.random.default_rng(SEED)
    halves = np.array_split(rng.permutation(animals), 2)
    ce = {}
    for k in K_GRID:
        vals = []
        for h in (0, 1):
            tr, te = _subset(tune, halves[h]), _subset(tune, halves[1 - h])
            rows = embed.sample_rows(tr["X"], tr["valid"], tr["bounds"], k,
                                     FIT_ROWS, np.random.default_rng(SEED))
            m = micro.fit(rows, k, seed=SEED)
            vals.append(micro.predictive_r2(_labels(m, tr), _future(tr),
                                            _labels(m, te), _future(te),
                                            micro.N_MICRO))
        ce[k] = float(np.mean(vals))
        log(f"  K={k:2d}: held-out R^2 of features {H_FUTURE} frames ahead "
            f"{ce[k]:.4f}")
    best = max(ce.values())
    k_star = min(k for k in K_GRID if ce[k] >= best * (1 - K_TOL))
    rows = embed.sample_rows(tune["X"], tune["valid"], tune["bounds"], k_star,
                             FIT_ROWS, np.random.default_rng(SEED))
    m = micro.fit(rows, k_star, seed=SEED)
    lab = _labels(m, tune)
    t2 = {}
    for tau in TAU_GRID:
        p, _ = op.transition_matrix(micro.counts(lab, tune["bounds"], tau,
                                                 micro.N_MICRO))
        t2[tau] = float(op.timescales(p, tau, k=2)[0])
        log(f"  tau={tau:3d}: t2 = {t2[tau]:.1f} frames")
    tau_star = TAU_GRID[-1]
    for i, tau in enumerate(TAU_GRID[:-1]):
        nxt = TAU_GRID[i + 1]
        if abs(t2[nxt] - t2[tau]) / t2[tau] < TAU_TOL:
            tau_star = tau
            break
    _save("select.json", {"use_mask": lk["use_mask"], "mu": mu.tolist(),
                          "sd": sd.tolist(), "predictive_r2": ce,
                          "k_star": k_star, "t2_by_tau": t2,
                          "tau_star": tau_star})
    log(f"  K* = {k_star}, tau* = {tau_star}")
    return 0


# ---- the pipeline, shared by the corpus and every surrogate ------------------

def run_pipeline(tune: dict, fit_d: dict, rep: dict, k: int, tau: int,
                 seed: int = SEED, bootstrap: bool = True) -> dict:
    """Standardised streams in; timescales, P and labels out."""
    rows = embed.sample_rows(tune["X"], tune["valid"], tune["bounds"], k,
                             FIT_ROWS, np.random.default_rng(seed))
    m = micro.fit(rows, k, seed=seed)
    lab_fit, lab_rep = _labels(m, fit_d), _labels(m, rep)
    c_fit = micro.counts(lab_fit, fit_d["bounds"], tau, micro.N_MICRO)
    p_fit, keep_fit = op.transition_matrix(c_fit)
    # per-animal report counts for the animal-level bootstrap of timescales
    per_an: dict[str, np.ndarray] = {}
    b = rep["bounds"]
    for i, an in enumerate(rep["animal"]):
        c = micro.counts(lab_rep[b[i]:b[i + 1]], np.array([0, b[i + 1] - b[i]]),
                         tau, micro.N_MICRO)
        per_an[an] = per_an.get(an, 0) + c
    names = sorted(per_an)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT if bootstrap else 0):
        pick = rng.integers(0, len(names), len(names))
        c = sum(per_an[names[j]] for j in pick)
        p, _ = op.transition_matrix(c)
        boots.append(op.timescales(p, tau, k=N_TS))
    p_rep, _ = op.transition_matrix(sum(per_an.values()))
    point = op.timescales(p_rep, tau, k=N_TS)
    boots = np.asarray(boots) if boots else point[None, :]
    ratios = np.asarray([op.gap_ratios(bt) for bt in boots])
    return {"micro": m, "lab_fit": lab_fit, "lab_rep": lab_rep,
            "p_fit": p_fit, "keep_fit": keep_fit, "p_rep": p_rep,
            "ts_point": point.tolist(),
            "ts_lo": np.quantile(boots, 0.025, axis=0).tolist(),
            "ts_hi": np.quantile(boots, 0.975, axis=0).tolist(),
            "gap_point": op.gap_ratios(point).tolist(),
            "gap_lo": np.nanquantile(ratios, 0.025, axis=0).tolist(),
            "gap_hi": np.nanquantile(ratios, 0.975, axis=0).tolist()}


def _splits_std(sel: dict) -> tuple[dict, dict, dict]:
    mu, sd = np.asarray(sel["mu"]), np.asarray(sel["sd"])
    return tuple(_std(load(s, sel["use_mask"]), mu, sd)
                 for s in ("tune", "fit", "report"))       # type: ignore[return-value]


def fit(a) -> int:
    sel = _read("select.json")
    tune, fit_d, rep = _splits_std(sel)
    r0 = run_pipeline(tune, fit_d, rep, sel["k_star"], sel["tau_star"], SEED)
    r1 = run_pipeline(tune, fit_d, rep, sel["k_star"], sel["tau_star"], SEED + 1,
                      bootstrap=False)
    np.savez_compressed(os.path.join(WORK, "corpus.npz"),
                        lab_fit=r0["lab_fit"], lab_rep=r0["lab_rep"],
                        lab_rep_seed1=r1["lab_rep"], p_fit=r0["p_fit"],
                        keep_fit=r0["keep_fit"], p_rep=r0["p_rep"],
                        p_fit_seed1=r1["p_fit"], keep_fit_seed1=r1["keep_fit"])
    _save("corpus_timescales.json", {k: r0[k] for k in (
        "ts_point", "ts_lo", "ts_hi", "gap_point", "gap_lo", "gap_hi")})
    log(f"  corpus t2..t{N_TS}: {np.round(r0['ts_point'], 1)}")
    return 0


def surrogate(a) -> int:
    sel = _read("select.json")
    tune, fit_d, rep = _splits_std(sel)
    fps = spine.fps()
    part = None
    if a.kind in sr.VISIT_KINDS:
        live = sr.live_columns(tune["X"][tune["valid"]])
        pv = ms.fit_partition(tune["X"], np.ones(tune["X"].shape[1]), np.flatnonzero(live),
                              tune["valid"], n_states=sr.N_MICROSTATES, seed=SEED,
                              logger=log)
        states = np.asarray(ms.assign_states(tune["X"], pv["centroids"],
                                             np.ones(tune["X"].shape[1]),
                                             np.flatnonzero(live)), dtype=np.int64)
        b = tune["bounds"]
        seqs = [np.asarray(ms.visits(states[b[i]:b[i + 1]]), dtype=np.int64)[:, 0]
                for i in range(b.size - 1) if b[i + 1] - b[i] > 1]
        chain = ms.pooled_chain(seqs, sr.N_MICROSTATES, order=1)
        part = {"centroids": pv["centroids"], "sd": np.ones(tune["X"].shape[1]),
                "cols": np.flatnonzero(live), "chain": chain}

    def surr(d: dict, split: str) -> dict:
        rng = np.random.default_rng(seeds.stable_seed(SEED, a.kind, split))
        return {**d, "X": np.asarray(sr.generate(d["X"], d["bounds"], a.kind, rng,
                                                 fps=fps, fit=part)["x"],
                                     dtype=np.float32)}
    r = run_pipeline(surr(tune, "tune"), surr(fit_d, "fit"), surr(rep, "report"),
                     sel["k_star"], sel["tau_star"], SEED)
    _save(f"surrogate_{a.kind}.json", {k: r[k] for k in (
        "ts_point", "ts_lo", "ts_hi", "gap_point", "gap_lo", "gap_hi")})
    log(f"  {a.kind} t2..t{N_TS}: {np.round(r['ts_point'], 1)}")
    return 0


# ---- §4 combine -------------------------------------------------------------

def _ami(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    ok = (x >= 0) & (y >= 0)
    return float(adjusted_mutual_info_score(x[ok], y[ok])) if ok.sum() else float("nan")


def _states(p: np.ndarray, keep: np.ndarray, lab: np.ndarray, m: int) -> np.ndarray:
    chi = op.pcca(op.reversibilise(p), m)
    return op.assign(lab, keep, chi, threshold=MEMBERSHIP)


def _plant(rep: dict, sel: dict) -> dict:
    """§4 gate 3: gamma-dwell switching, +1 SD on two channels, 20% of recordings."""
    fps = spine.fps()
    rng = np.random.default_rng(SEED)
    x = rep["X"].copy()
    truth = np.zeros(x.shape[0], dtype=bool)
    in_plant = np.zeros(x.shape[0], dtype=bool)
    b = rep["bounds"]
    chosen = rng.random(b.size - 1) < PLANT_SHARE
    mean_f = PLANT_MEAN_S * fps
    for i in np.flatnonzero(chosen):
        lo, hi = int(b[i]), int(b[i + 1])
        t, on = lo, bool(rng.random() < 0.5)
        while t < hi:
            n = max(1, int(rng.gamma(4.0, mean_f / 4.0)))
            if on:
                truth[t:min(hi, t + n)] = True
            t += n
            on = not on
        in_plant[lo:hi] = True
    for c in PLANT_CHANNELS:
        x[truth, c] += 1.0
    d = {**rep, "X": x}
    rows = embed.sample_rows(d["X"], d["valid"], d["bounds"], sel["k_star"],
                             FIT_ROWS, np.random.default_rng(SEED))
    m = micro.fit(rows, sel["k_star"], seed=SEED)
    lab = _labels(m, d)
    p, keep = op.transition_matrix(micro.counts(lab, d["bounds"], sel["tau_star"],
                                                micro.N_MICRO))
    best = 0.0
    for n_states in (2, 3, 4):
        st = _states(p, keep, lab, n_states)
        sel_ = in_plant & (st >= 0)
        for s in range(n_states):
            pred = st[sel_] == s
            tru = truth[sel_]
            if tru.any() and (~tru).any():
                ba = 0.5 * (pred[tru].mean() + (~pred[~tru]).mean())
                best = max(best, float(ba), float(1 - ba))
    return {"best_balanced_accuracy": best, "pass": best >= PLANT_BAR,
            "planted_recordings": int(chosen.sum())}


def combine(a) -> int:
    sel = _read("select.json")
    ts = _read("corpus_timescales.json")
    surr = {k: _read(f"surrogate_{k}.json") for k in SURROGATES
            if os.path.exists(os.path.join(WORK, f"surrogate_{k}.json"))}
    if set(surr) != set(SURROGATES):
        raise SystemExit(f"missing surrogates: {set(SURROGATES) - set(surr)}")
    # Amendment 1: gate 1 is on the GAP ratios r_i = t_i / t_(i+1), i = 2..10.
    # A gap after mode i counts iff its report lower bound exceeds every
    # surrogate's upper bound at the same i. M = the largest such i (the number
    # of slow modes before the largest significant gap); none -> CONTINUUM.
    counting = []
    for i in range(N_TS - 2):
        s_hi = max(surr[k]["gap_hi"][i] for k in SURROGATES)
        counting.append(bool(ts["gap_lo"][i] > s_hi))
    passing = [i + 2 for i, c in enumerate(counting) if c]
    M = max(passing) if passing else 1
    res: dict = {"k_star": sel["k_star"], "tau_star": sel["tau_star"],
                 "use_mask": sel["use_mask"], "corpus_timescales": ts,
                 "surrogate_timescales": surr, "counting": counting, "M": M}
    z = dict(np.load(os.path.join(WORK, "corpus.npz")))
    tune, fit_d, rep = _splits_std(sel)
    res["gate3_plant"] = _plant(rep, sel)
    if M == 1:
        res["verdict"] = "CONTINUUM"
    else:
        st_fit_model = _states(z["p_fit"], z["keep_fit"], z["lab_rep"], M)
        p_rep, keep_rep = op.transition_matrix(micro.counts(
            z["lab_rep"], rep["bounds"], sel["tau_star"], micro.N_MICRO))
        st_rep_model = _states(p_rep, keep_rep, z["lab_rep"], M)
        st_seed1 = _states(z["p_fit_seed1"], z["keep_fit_seed1"],
                           z["lab_rep_seed1"], M)
        ami_split = _ami(st_fit_model, st_rep_model)
        ami_seed = _ami(st_fit_model, st_seed1)
        res["gate2"] = {"ami_fit_vs_report": ami_split, "ami_seed0_vs_seed1": ami_seed,
                        "pass": ami_split >= AMI_BAR and ami_seed >= AMI_BAR}
        res["abstain_share"] = float(np.mean(st_fit_model[rep["valid"]] < 0))
        occ = {}
        per_rec = np.repeat(np.arange(rep["bounds"].size - 1), np.diff(rep["bounds"]))
        for s in range(M):
            occ[s] = {f"{c}|day{d}": float(np.mean(st_fit_model[
                (rep["context"][per_rec] == c) & (rep["day"][per_rec] == d)
                & rep["valid"]] == s))
                for c in ("A", "B") for d in sorted(set(rep["day"].tolist()))}
        res["occupancy"] = occ
        np.save(os.path.join(WORK, "report_states.npy"), st_fit_model)
        res["verdict"] = ("PASS" if res["gate2"]["pass"] and res["gate3_plant"]["pass"]
                          else "FAIL")
    log(f"  M = {M}; verdict {res['verdict']}")
    out = a.out or config.PATHS.result("states.json")
    write_json({**provenance.header(anchors.LUNA, stage="states",
                                    unverified="a behaviour-discovery stage"),
                "inherited_digest": spine.digest(),
                "registration": "results/STATES_PREREGISTRATION.md",
                "results": res}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=("leak", "select", "fit", "surrogate", "combine"))
    p.add_argument("--kind", choices=SURROGATES)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    return {"leak": leak, "select": select, "fit": fit, "surrogate": surrogate,
            "combine": combine}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
