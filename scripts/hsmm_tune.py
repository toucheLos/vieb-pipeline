"""STEP 4 C2-C4 on the tune split: emissions, duration fits, selection, controls.

Registered in `results/step4/REGISTRATION.md` (7d26641). Phases:

    python3 scripts/hsmm_tune.py prep                      # signals, flags, freeze surrogate, emissions
    python3 scripts/hsmm_tune.py fit --fold 0 --family weibull   # one CV fit (array job)
    python3 scripts/hsmm_tune.py select                    # family choice; HSMM vs HMM
    python3 scripts/hsmm_tune.py final                     # refit on all tune; decode; imprint; flicker
    python3 scripts/hsmm_tune.py controls                  # jitter-only, planted bouts
    python3 scripts/hsmm_tune.py planted_refit --boot 7    # one bootstrap refit (array job)
    python3 scripts/hsmm_tune.py report                    # Reads -> results/step4/hsmm.json

Animals are keyed by `animal_tag` (298; cohort twins kept apart, as the splits).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from scipy.optimize import minimize                                 # noqa: E402

from recur import anchors, boot, labels as lab, splits              # noqa: E402
from recur.read import Read                                         # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.calms import degrade as dg, io as cio                     # noqa: E402
from vieb.clean import arms                                         # noqa: E402
from vieb.hsmm import core, mouse as M                              # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seeds import stable_seed                                  # noqa: E402
from vieb.tok import config, ego                                    # noqa: E402

WORK = os.path.join(config.REPO, "work", "hsmm")
OUT = os.path.join(config.REPO, "results", "step4")
FPS = 30.0
SEED = 0
K_FOLDS = 5
N_FIT_RECS = 120
FAMILIES = ("geometric", "weibull", "lognormal")
N_STATIONARY = 50
N_PLANTED, PLANT_S = 30, 300.0
PLANT_FREEZE, PLANT_MOVE = (1.5, 3.0), (1.2, 2.0)        # Weibull (k, lambda_s)
PLANT_MOVE_MIN_V = 0.5                                   # BL/s, the snippet rule
N_BOOT_PLANT = 50


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def tune_ids() -> list[str]:
    sp = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    return sorted(r for r in spine.recording_ids() if sp.get(lab.animal_tag(r)) == "tune")


def raw_pose(rid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(held raw pose (T,7,2) px, flag (T,), usable (T,))`` for one recording."""
    d = spine.clean(rid)
    miss = np.asarray(d["missing"], dtype=bool)
    pose = arms.held_array(np.asarray(d["pose_unfiltered"], dtype=np.float64), miss)
    flag = (np.asarray(d["bone_flagged"], bool) | miss.any(axis=1)
            | np.asarray(d["interpolated"], bool).any(axis=1))
    return pose, flag, ~miss.any(axis=1)


def ell_by_animal(ids: list[str]) -> dict[str, float]:
    blocks: dict[str, list] = {}
    usable: dict[str, list] = {}
    for rid in ids:
        p, _f, u = raw_pose(rid)
        blocks.setdefault(lab.animal_tag(rid), []).append(p)
        usable.setdefault(lab.animal_tag(rid), []).append(u)
    return {a: float(ego.ell_a(blocks[a], usable[a])) for a in blocks}


STILL_Q = 0.10     # Amendment 1: the lowest decile of centroid speed


def jitter_freeze(pose: np.ndarray, usable: np.ndarray, ell: float, T: int, rid: str,
                  nz: dict, level: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The freeze surrogate: the recording's median pose, motionless, scaled to
    Luna's median ell, plus Step 1's jitter operator at `level` (1.0 as first
    registered; m_still after Amendment 1). Returns (v, r)."""
    med = np.median(pose[usable] if usable.any() else pose, axis=0)
    still = np.broadcast_to(med * (cio.LUNA_ELL_PX / ell), (T,) + med.shape).copy()
    prof = np.asarray([nz["profile_luna_bl"][k] for k in dg.LUNA_ORDER])
    rng = dg.rng_for(SEED, 1.0, f"hsmm-freeze|{rid}")
    noisy = still + dg.jitter_noise(rng, T, dg.added_bl(level, 0.0, prof, nz["gain"]), nz["s1_px"])
    return M.signal(noisy, cio.LUNA_ELL_PX, FPS)


def still_sigma(ids: list[str], ells: dict[str, float], thr: float) -> tuple[float, dict]:
    """Amendment 1: the skull-bone estimator on still frames only (unflagged,
    centroid speed below the pooled lowest-decile threshold)."""
    from vieb.seg import jitter
    rs, cs = [], []
    for rid in ids:
        d = spine.clean(rid)
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        still = (~np.asarray(z["flag"], bool)) & (np.asarray(z["v"]) <= thr)
        miss = np.asarray(d["missing"], bool) | (~still)[:, None]
        r, c = jitter.residuals(np.asarray(d["pose_unfiltered"], dtype=np.float64),
                                np.asarray(d["conf"], dtype=np.float64), missing=miss,
                                interpolated=np.asarray(d["interpolated"], bool),
                                ell=ells[lab.animal_tag(rid)])
        rs.append(r)
        cs.append(c)
    table = jitter.calibrate(np.concatenate(rs), np.concatenate(cs))
    return dg.count_weighted(table), table


MIN_PAIRS = 200   # Amendment 2: recordings with fewer still pairs are skipped


def _skull_lengths_bl(pose: np.ndarray, ell: float) -> np.ndarray:
    from vieb.qc import bones as bn
    return np.asarray(bn.bone_lengths(pose, bn.SKULL), dtype=np.float64) / float(ell)


def still_increment(ids: list[str], ells: dict[str, float], thr: float) -> tuple[float, int]:
    """Amendment 2: SD of skull-bone-length increments between consecutive still
    frames (both still, no missing / interpolated endpoint), pooled over bones per
    recording; the median over recordings with >= MIN_PAIRS pairs."""
    from vieb.qc import bones as bn
    per = []
    for rid in ids:
        d = spine.clean(rid)
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        still = (~np.asarray(z["flag"], bool)) & (np.asarray(z["v"]) <= thr)
        bad = np.asarray(d["missing"], bool) | np.asarray(d["interpolated"], bool)
        L = _skull_lengths_bl(np.asarray(d["pose_unfiltered"], dtype=np.float64),
                              ells[lab.animal_tag(rid)])
        incs = []
        for m, (i, j) in enumerate(bn.SKULL):
            ok = still & ~bad[:, i] & ~bad[:, j] & np.isfinite(L[:, m])
            pair = ok[1:] & ok[:-1]
            incs.append(np.diff(L[:, m])[pair])
        dL = np.concatenate(incs)
        if dL.size >= MIN_PAIRS:
            per.append(float(np.std(dL)))
    return float(np.median(per)), len(per)


def operator_increment(nz: dict) -> float:
    """The same increment statistic on Step 1's operator at level 1, on the rigid
    calibration pose with Luna's median skull geometry (300,000 frames, seed 0)."""
    skull = [nz["luna_skull_bl"][k] for k in ("left_ear-right_ear", "left_ear-nose", "right_ear-nose")]
    pose = dg.calibration_pose(skull, n_frames=dg.CAL_FRAMES, seed=SEED)
    prof = np.asarray([nz["profile_luna_bl"][k] for k in dg.LUNA_ORDER])
    noisy = pose + dg.jitter_noise(np.random.default_rng(SEED), pose.shape[0],
                                   dg.added_bl(1.0, 0.0, prof, nz["gain"]), nz["s1_px"])
    L = _skull_lengths_bl(noisy, cio.LUNA_ELL_PX)
    return float(np.std(np.diff(L, axis=0)))


def prep(_a) -> int:
    os.makedirs(os.path.join(WORK, "tune"), exist_ok=True)
    ids = tune_ids()
    ells = ell_by_animal(ids)
    with open(os.path.join(OUT.replace("step4", "step1"), "noise.json"), encoding="utf-8") as fh:
        nz = json.load(fh)
    # Pass 1: the signal and flag of every tune recording.
    data_unflagged, all_bins, v_unflagged, nflag, nall = [], [], [], 0, 0
    for i, rid in enumerate(ids):
        pose, flag, _usable = raw_pose(rid)
        v, r = M.signal(pose, ells[lab.animal_tag(rid)], FPS)
        b = M.bins(v, r)
        np.savez_compressed(os.path.join(WORK, "tune", f"{rid}.npz"), bins=b, flag=flag,
                            v=v.astype(np.float32), r=r.astype(np.float32))
        data_unflagged.append(b[~flag])
        v_unflagged.append(v[~flag])
        all_bins.append(b)
        nflag += int(flag.sum())
        nall += int(flag.size)
        if i % 100 == 0:
            log(f"  signal {i}/{len(ids)}")
    # Pass 2 (Amendment 1): still frames and their skull-estimator jitter.
    thr = float(np.quantile(np.concatenate(v_unflagged), STILL_Q))
    sig_still, table = still_sigma(ids, ells, thr)            # Amendment 1, reported only
    s_real, n_rec = still_increment(ids, ells, thr)            # Amendment 2
    s_op1 = operator_increment(nz)
    m_still = s_real / s_op1
    log(f"still threshold v <= {thr:.4f} BL/s; A1 sigma_still {sig_still:.5f}; "
        f"A2 s_real {s_real:.5f} (n {n_rec}), s_op(1) {s_op1:.5f}, m_inc {m_still:.4f}")
    # Pass 3: the freeze surrogate at m_still.
    fz, below = [], []
    for i, rid in enumerate(ids):
        pose, _flag, usable = raw_pose(rid)
        ell = ells[lab.animal_tag(rid)]
        fv, fr = jitter_freeze(pose, usable, ell, pose.shape[0], rid, nz, level=m_still)
        fb = M.bins(fv, fr)
        z = dict(np.load(os.path.join(WORK, "tune", f"{rid}.npz")))
        z["freeze_bins"] = fb
        np.savez_compressed(os.path.join(WORK, "tune", f"{rid}.npz"), **z)
        fz.append(fb)
        unf = ~np.asarray(z["flag"], bool)
        below.append(float((np.asarray(z["r"])[unf] < np.percentile(fr, 5)).mean()))
        if i % 100 == 0:
            log(f"  freeze surrogate {i}/{len(ids)}")
    p_freeze = M.histogram(np.concatenate(fz))
    counts = np.bincount(np.concatenate(data_unflagged), minlength=M.N_BINS ** 2)
    p_move, pi, it = M.em_move(counts, p_freeze)
    p_marg = M.histogram(np.concatenate(all_bins))
    q = nflag / nall
    np.savez_compressed(os.path.join(WORK, "emissions.npz"), p_freeze=p_freeze, p_move=p_move,
                        p_marginal=p_marg, q_flag=q)
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_emissions",
                                    unverified="tune split only"),
                "registration": "results/step4/REGISTRATION.md (+ Amendment 1)",
                "n_recordings": len(ids), "n_animals": len(ells), "n_frames": nall,
                "flag_rate": q, "em_pi_freeze": pi, "em_iterations": it,
                "still": {"quantile": STILL_Q, "v_threshold_bl_s": thr,
                          "amendment1_sigma_still_bl": sig_still,
                          "sigma_bar_luna_bl": dg.sigma_bar_luna(),
                          "s_real_increment_bl": s_real, "n_recordings_increment": n_rec,
                          "s_operator_level1_bl": s_op1,
                          "m_still": m_still, "m_rule": "Amendment 2: m_inc = s_real / s_op(1)",
                          "table_counts": table["counts"],
                          "share_real_unflagged_r_below_surrogate_p5_median": float(np.median(below))},
                "ell_median_px": float(np.median(list(ells.values()))),
                "w_s": M.W_S, "d_max_s": M.D_MAX_S, "grid": [M.LOG_LO, M.LOG_HI, M.N_BINS]},
               os.path.join(OUT, "hsmm_emissions.json"))
    log(f"prep: {len(ids)} recordings, flag rate {q:.4f}, EM pi_freeze {pi:.3f} ({it} it); "
        f"real r below surrogate p5 (median) {np.median(below):.3f}")
    return 0


def emissions() -> M.Emissions:
    z = np.load(os.path.join(WORK, "emissions.npz"))
    return M.Emissions(p_freeze=z["p_freeze"], p_move=z["p_move"], p_marginal=z["p_marginal"],
                       q_flag=float(z["q_flag"]))


def loglik_of(ids: list[str], em: M.Emissions) -> dict[str, np.ndarray]:
    out = {}
    for rid in ids:
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        out[rid] = em.loglik(z["bins"], z["flag"])
    return out


def folds(ids: list[str]) -> dict[str, int]:
    animals = sorted({lab.animal_tag(r) for r in ids})
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(animals))
    return {a: i % K_FOLDS for i, a in enumerate(order)}


def fit_subset(ids: list[str], seed_tag: str) -> list[str]:
    """Two recordings per animal where available, to N_FIT_RECS (random.Random(0))."""
    rng = random.Random(f"{SEED}|{seed_tag}")
    by: dict[str, list[str]] = {}
    for r in ids:
        by.setdefault(lab.animal_tag(r), []).append(r)
    out: list[str] = []
    for a in sorted(by):
        rs = sorted(by[a])
        rng.shuffle(rs)
        out += rs[:2]
    rng.shuffle(out)
    return sorted(out[:N_FIT_RECS])


def fit_params(les: list[np.ndarray], family: str, theta_init: np.ndarray | None = None,
               fixed: dict | None = None) -> tuple[np.ndarray, float, int]:
    """Maximise the HSMM marginal likelihood (Nelder-Mead, maxiter 400)."""
    n = sum(x.shape[0] for x in les)
    th0 = M.theta0(family) if theta_init is None else theta_init
    free = np.ones(th0.size, bool) if not fixed else np.asarray([i not in fixed for i in range(th0.size)])

    def full(x: np.ndarray) -> np.ndarray:
        th = th0.copy()
        th[free] = x
        for i, v in (fixed or {}).items():
            th[i] = v
        return th

    def nll(x: np.ndarray) -> float:
        try:
            m = M.build(full(x), family, FPS)
            ll, _f, _s = core.forward(m, les)
            return float(-ll.sum() / n)
        except (FloatingPointError, AssertionError):
            return 1e6

    res = minimize(nll, th0[free], method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-7})
    return full(res.x), float(-res.fun), int(res.nit)


def fit(a) -> int:
    os.makedirs(os.path.join(WORK, "fits"), exist_ok=True)
    ids = tune_ids()
    fo = folds(ids)
    em = emissions()
    train = [r for r in ids if fo[lab.animal_tag(r)] != a.fold]
    test = [r for r in ids if fo[lab.animal_tag(r)] == a.fold]
    sub = fit_subset(train, f"fold{a.fold}")
    L = loglik_of(sub, em)
    theta, ll_train, nit = fit_params([L[r] for r in sub], a.family)
    m = M.build(theta, a.family, FPS)
    Lt = loglik_of(test, em)
    per_animal: dict[str, list[float]] = {}
    lls, _f, _s = core.forward(m, [Lt[r] for r in test])
    for r, ll in zip(test, lls):
        per_animal.setdefault(lab.animal_tag(r), []).append(float(ll))
        per_animal[lab.animal_tag(r)].append(int(Lt[r].shape[0]))
    out = {a_: {"ll": float(sum(v[0::2])), "n": int(sum(v[1::2]))} for a_, v in per_animal.items()}
    dur, ex = M.unpack(theta, a.family)
    write_json({"fold": a.fold, "family": a.family, "theta": theta.tolist(), "dur": dur,
                "exit": ex.tolist(), "ll_train_per_frame": ll_train, "nit": nit,
                "n_fit_recordings": len(sub), "test_animals": out},
               os.path.join(WORK, "fits", f"{a.family}_f{a.fold}.json"))
    log(f"fold {a.fold} {a.family}: train ll/frame {ll_train:.5f}, {nit} it; dur {dur}")
    return 0


def select(_a) -> int:
    rows = {}
    for f in sorted(glob.glob(os.path.join(WORK, "fits", "*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for a_, v in d["test_animals"].items():
            rows.setdefault(d["family"], {})[a_] = v["ll"] / v["n"]
    assert set(rows) == set(FAMILIES) and all(len(rows[f]) == 60 for f in FAMILIES), {
        f: len(v) for f, v in rows.items()}
    animals = sorted(rows["geometric"])
    mean = {f: float(np.mean([rows[f][a_] for a_ in animals])) for f in FAMILIES}
    best = "weibull" if mean["weibull"] >= mean["lognormal"] else "lognormal"
    gain = np.asarray([rows[best][a_] - rows["geometric"][a_] for a_ in animals])
    ci = boot.animal_interval(gain, animals, how="mean", n_boot=2000, seed=SEED)
    lo, hi = float(ci["lo"]), float(ci["hi"])
    keep = bool(lo > 0)
    read = Read("PASS" if keep else "FAIL",
                {"arm": "hsmm_vs_hmm", "family": best, "split": "tune", "cv": f"{K_FOLDS}-fold animal"},
                f"held-out gain of the {best} HSMM over a plain HMM: {gain.mean():+.5f} nats/frame "
                f"[{lo:+.5f}, {hi:+.5f}]; the semi-Markov part is "
                f"{'kept' if keep else 'DROPPED (geometric fits as well)'}",
                n_effective=len(animals)).to_dict()
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_select", unverified="tune split only"),
                "mean_heldout_ll_per_frame": mean, "family": best, "keep_semimarkov": keep,
                "gain_mean": float(gain.mean()), "gain_ci": [float(lo), float(hi)],
                "chosen": best if keep else "geometric", "read": read},
               os.path.join(OUT, "hsmm_select.json"))
    log(f"select: {mean}; chosen {best if keep else 'geometric'} (gain {gain.mean():+.5f} [{lo:+.5f}, {hi:+.5f}])")
    return 0


def chosen() -> tuple[str, np.ndarray]:
    with open(os.path.join(WORK, "final_fit.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    return d["family"], np.asarray(d["theta"])


def switch_rate(state: np.ndarray) -> float:
    return float(np.count_nonzero(np.diff(state)) / (state.size / FPS))


def bout_lengths(state: np.ndarray, s: int) -> np.ndarray:
    return np.asarray([(e - b) / FPS for k, b, e in core.runs(state) if k == s])


def final(_a) -> int:
    ids = tune_ids()
    em = emissions()
    with open(os.path.join(OUT, "hsmm_select.json"), encoding="utf-8") as fh:
        fam = json.load(fh)["chosen"]
    sub = fit_subset(ids, "final")
    L = loglik_of(ids, em)
    theta, llf, nit = fit_params([L[r] for r in sub], fam)
    with open(os.path.join(WORK, "final_fit.json"), "w", encoding="utf-8") as fh:
        json.dump({"family": fam, "theta": theta.tolist(), "ll_per_frame": llf, "nit": nit}, fh)
    m = M.build(theta, fam, FPS)
    rng = np.random.default_rng(SEED)
    probe = sorted(rng.choice(ids, 100, replace=False).tolist())
    real, shuf, circ, raw_sw, sm_sw = [], [], [], [], []
    per_ctx: dict = {}
    for rid in ids:
        le = L[rid]
        _ll, post = core.smooth(m, [le])
        st = post[0].argmax(1)
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        np.savez_compressed(os.path.join(WORK, "tune", f"{rid}_post.npz"), post=post[0].astype(np.float32))
        info = lab.parse(rid)
        if info["protocol"] == "CFD":
            k = info["context_letter"]
            fl = np.asarray(z["flag"], bool)
            per_ctx.setdefault(k, {"all": [], "unflagged": []})
            per_ctx[k]["all"].append(float(post[0][:, M.FREEZE].mean()))
            per_ctx[k]["unflagged"].append(float(post[0][~fl, M.FREEZE].mean()) if (~fl).any() else np.nan)
        if rid in probe:
            real.append(bout_lengths(st, M.FREEZE))
            raw_sw.append(switch_rate(le.argmax(1)))
            sm_sw.append(switch_rate(st))
            r_ = np.random.default_rng(stable_seed("hsmm-imprint", SEED, rid))
            _l, ps = core.smooth(m, [le[r_.permutation(le.shape[0])]])
            shuf.append(bout_lengths(ps[0].argmax(1), M.FREEZE))
            _l, pc = core.smooth(m, [np.roll(le, int(r_.integers(1, le.shape[0])), axis=0)])
            circ.append(bout_lengths(pc[0].argmax(1), M.FREEZE))
    from scipy.stats import ks_2samp

    def q(x):
        x = np.concatenate(x) if x else np.zeros(0)
        return {"n_bouts": int(x.size), "median_s": float(np.median(x)) if x.size else None,
                "p10_s": float(np.quantile(x, 0.1)) if x.size else None,
                "p90_s": float(np.quantile(x, 0.9)) if x.size else None}, x
    qr, xr = q(real)
    qs, xs = q(shuf)
    qc, xc = q(circ)
    imprint = {"real": qr, "frame_shuffled": qs, "circular_shift": qc,
               "ks_real_vs_shuffled": float(ks_2samp(xr, xs).statistic) if xs.size and xr.size else None,
               "ks_real_vs_circular": float(ks_2samp(xr, xc).statistic) if xc.size and xr.size else None,
               "median_ratio_shuffled_over_real": (qs["median_s"] / qr["median_s"]
                                                   if qs["median_s"] and qr["median_s"] else None)}
    ctx = {k: {kk: float(np.nanmean(vv)) for kk, vv in v.items()} | {"n_recordings": len(v["all"])}
           for k, v in per_ctx.items()}
    dur, ex = M.unpack(theta, fam)
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_final", unverified="tune split only"),
                "family": fam, "durations": dict(zip(M.STATES, dur)), "exit": ex.tolist(),
                "ll_per_frame": llf, "n_fit_recordings": len(sub), "n_decoded": len(ids),
                "prior_imprint": imprint,
                "flicker": {"raw_switches_per_s": float(np.mean(raw_sw)),
                            "smoothed_switches_per_s": float(np.mean(sm_sw)), "n_recordings": len(probe)},
                "freeze_fraction_by_context": ctx},
               os.path.join(OUT, "hsmm_final.json"))
    log(f"final {fam}: dur {dur}; imprint {imprint}; ctx {ctx}")
    return 0


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

def planted_sessions(ids: list[str], nz: dict, em: M.Emissions) -> list[dict]:
    """30 synthetic 5-min sessions with known freeze / move bouts (registered)."""
    rng = np.random.default_rng(SEED)
    T = int(round(PLANT_S * FPS))
    srcs = sorted(rng.choice(ids, N_PLANTED, replace=False).tolist())
    out = []
    ells = ell_by_animal(sorted({r for r in srcs}))
    for i, rid in enumerate(srcs):
        pose, flag, usable = raw_pose(rid)
        ell = ells[lab.animal_tag(rid)]
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        fast = np.asarray(z["v"]) > PLANT_MOVE_MIN_V
        # candidate move snippets: maximal runs where v stays above the registered speed
        cand = [(b, e) for k, b, e in core.runs(fast.astype(int)) if k == 1 and e - b >= 15]
        med = np.median(pose[usable] if usable.any() else pose, axis=0)
        frames_, truth = [], []
        s = 0
        while sum(len(x) for x in frames_) < T:
            k_, lam = PLANT_FREEZE if s == 0 else PLANT_MOVE
            n = max(1, int(round(lam * rng.weibull(k_) * FPS)))
            if s == 0:
                frames_.append(np.broadcast_to(med, (n,) + med.shape).copy())
            else:
                b, e = cand[int(rng.integers(len(cand)))]
                seg = pose[b:e]
                reps = int(np.ceil(n / seg.shape[0]))
                piece = np.concatenate([seg] * reps)[:n]
                frames_.append(piece - piece[0].mean(0) + med.mean(0))
            truth += [s] * n
            s = 1 - s
        P = np.concatenate(frames_)[:T]
        tr = np.asarray(truth[:T])
        # Luna jitter on the freeze frames; move snippets carry their own real jitter
        prof = np.asarray([nz["profile_luna_bl"][k] for k in dg.LUNA_ORDER])
        with open(os.path.join(OUT, "hsmm_emissions.json"), encoding="utf-8") as fh:
            m_still = float(json.load(fh)["still"]["m_still"])           # Amendment 1
        jit = dg.jitter_noise(dg.rng_for(SEED, 1.0, f"plant|{i}"), T,
                              dg.added_bl(m_still, 0.0, prof, nz["gain"]), nz["s1_px"]) * (ell / cio.LUNA_ELL_PX)
        P = P + jit * (tr == 0)[:, None, None]
        v, r = M.signal(P, ell, FPS)
        out.append({"le": em.loglik(M.bins(v, r), np.zeros(T, bool)), "truth": tr})
    return out


def controls(_a) -> int:
    ids = tune_ids()
    em = emissions()
    fam, theta = chosen()
    m = M.build(theta, fam, FPS)
    with open(os.path.join(OUT.replace("step4", "step1"), "noise.json"), encoding="utf-8") as fh:
        nz = json.load(fh)
    # Jitter-only stationary
    rng = np.random.default_rng(SEED)
    stat_ids = sorted(rng.choice(ids, N_STATIONARY, replace=False).tolist())
    fr = []
    for rid in stat_ids:
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        le = em.loglik(z["freeze_bins"], np.zeros(z["freeze_bins"].size, bool))
        _l, post = core.smooth(m, [le])
        fr.append(float(post[0][:, M.FREEZE].mean()))
    stat_pass = bool(np.mean(fr) >= 0.95 and min(fr) >= 0.90)
    # Planted bouts
    ps = planted_sessions(ids, nz, em)
    acc = []
    for s in ps:
        _l, post = core.smooth(m, [s["le"]])
        acc.append(float(((post[0][:, M.FREEZE] > 0.5) == (s["truth"] == 0)).mean()))
    np.savez_compressed(os.path.join(WORK, "planted.npz"),
                        **{f"le{i}": s["le"] for i, s in enumerate(ps)},
                        **{f"tr{i}": s["truth"] for i, s in enumerate(ps)})
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_controls", unverified="tune split"),
                "model": {"family": fam, "theta": theta.tolist()},
                "stationary": {"freeze_fraction": fr, "mean": float(np.mean(fr)), "min": float(min(fr)),
                               "pass": stat_pass, "bar": "mean >= 0.95 and every recording >= 0.90"},
                "planted": {"frame_accuracy": acc, "mean": float(np.mean(acc)),
                            "pass_accuracy": bool(np.mean(acc) >= 0.90),
                            "truth_freeze": list(PLANT_FREEZE), "truth_move": list(PLANT_MOVE)}},
               os.path.join(WORK, "controls_part1.json"))
    log(f"stationary mean {np.mean(fr):.4f} min {min(fr):.4f} pass {stat_pass}; planted acc {np.mean(acc):.4f}")
    return 0


def planted_refit(a) -> int:
    """Refit freeze and move durations on (a bootstrap of) the planted sessions.
    Boot 0 = the sessions as drawn; the lost state's parameters stay at the final fit."""
    os.makedirs(os.path.join(WORK, "planted_boot"), exist_ok=True)
    z = np.load(os.path.join(WORK, "planted.npz"))
    n = len([k for k in z.files if k.startswith("le")])
    idx = (np.arange(n) if a.boot == 0 else
           np.random.default_rng(SEED + a.boot).integers(0, n, n))
    les = [z[f"le{i}"] for i in idx]
    fam, theta = chosen()
    if fam == "geometric":
        raise SystemExit("planted refit is registered for the chosen semi-Markov family")
    th0 = M.theta0(fam)
    th0[4:6] = theta[4:6]
    fixed = {4: theta[4], 5: theta[5], 6: theta[6], 7: theta[7], 8: theta[8]}
    th, ll, nit = fit_params(les, fam, theta_init=th0, fixed=fixed)
    dur, _ex = M.unpack(th, fam)
    with open(os.path.join(WORK, "planted_boot", f"b{a.boot}.json"), "w", encoding="utf-8") as fh:
        json.dump({"boot": a.boot, "freeze": dur[0], "move": dur[1], "nit": nit}, fh)
    log(f"planted boot {a.boot}: freeze {dur[0]} move {dur[1]}")
    return 0


def report(_a) -> int:
    with open(os.path.join(WORK, "controls_part1.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    fam = c["model"]["family"]
    bs = []
    for f in sorted(glob.glob(os.path.join(WORK, "planted_boot", "b*.json"))):
        with open(f, encoding="utf-8") as fh:
            bs.append(json.load(fh))
    point = next(b for b in bs if b["boot"] == 0)
    reps = [b for b in bs if b["boot"] != 0]
    rec: dict = {}
    for st, truth in (("freeze", PLANT_FREEZE), ("move", PLANT_MOVE)):
        arr = np.asarray([b[st] for b in reps])
        lo, hi = np.quantile(arr, 0.025, axis=0), np.quantile(arr, 0.975, axis=0)
        inside = [bool(lo[j] <= truth[j] <= hi[j]) for j in range(2)]
        rec[st] = {"truth": list(truth), "point": point[st], "lo": lo.tolist(), "hi": hi.tolist(),
                   "inside": inside}
    planted_pass = bool(c["planted"]["pass_accuracy"] and all(all(v["inside"]) for v in rec.values()))
    reads = [
        Read("PASS" if c["stationary"]["pass"] else "FAIL",
             {"arm": "jitter_only_stationary", "family": fam, "split": "tune"},
             f"mean smoothed P(freeze) {c['stationary']['mean']:.4f}, min {c['stationary']['min']:.4f} "
             f"(bar: mean >= 0.95, every recording >= 0.90)", n_effective=len(c["stationary"]["freeze_fraction"])).to_dict(),
        Read("PASS" if planted_pass else "FAIL",
             {"arm": "planted_bouts", "family": fam, "split": "tune"},
             f"frame accuracy {c['planted']['mean']:.4f} (bar 0.90); durations recovered inside the "
             f"95% bootstrap interval: {json.dumps({k: v['inside'] for k, v in rec.items()})}",
             n_effective=len(c["planted"]["frame_accuracy"])).to_dict()]
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_controls", unverified="tune split"),
                "registration": "results/step4/REGISTRATION.md", "stationary": c["stationary"],
                "planted": {**c["planted"], "recovery": rec, "n_boot": len(reps), "pass": planted_pass},
                "reads": reads}, os.path.join(OUT, "hsmm_controls.json"))
    log(f"controls: stationary {c['stationary']['pass']}, planted {planted_pass}")
    return 0


def diagnose(_a) -> int:
    """Pre-run diagnostic (found in the smoke test, NOT a registered result):
    does the registered freeze emission -- a motionless pose plus measured
    tracking noise -- overlap the stillest real frames? 30 tune recordings,
    seed 0. Compared: the registered Step 1 operator at 1.0x, and shapeflow's
    native noise spectrum (`noise.draw`, scale 1)."""
    from vieb.seg import noise
    ids = tune_ids()
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(ids, 30, replace=False).tolist())
    ells = ell_by_animal(pick)
    with open(os.path.join(OUT.replace("step4", "step1"), "noise.json"), encoding="utf-8") as fh:
        nz = json.load(fh)
    rows = []
    for rid in pick:
        pose, _flag, us = raw_pose(rid)
        ell = ells[lab.animal_tag(rid)]
        v, r = M.signal(pose, ell, FPS)
        med = np.median(pose[us], axis=0)
        T = pose.shape[0]
        still = np.broadcast_to(med, (T,) + med.shape).copy()
        nv, nr = M.signal(still + noise.draw(np.random.default_rng(stable_seed("diag", rid)), T, 7, fps=FPS),
                          ell, FPS)
        sv, sr = jitter_freeze(pose, us, ell, T, rid, nz)
        rows.append({"recording_id": rid,
                     "data_r_p5": float(np.percentile(r, 5)), "data_v_p5": float(np.percentile(v, 5)),
                     "native_r_median": float(np.median(nr)), "native_v_median": float(np.median(nv)),
                     "step1_r_median": float(np.median(sr)), "step1_v_median": float(np.median(sv)),
                     "share_data_r_below_native_p5": float((r < np.percentile(nr, 5)).mean()),
                     "share_data_r_below_step1_p5": float((r < np.percentile(sr, 5)).mean())})
    med_ = {k: float(np.median([x[k] for x in rows])) for k in rows[0] if k != "recording_id"}
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_floor_diagnostic",
                                    unverified="pre-run diagnostic; not a registered result"),
                "question": "does a motionless pose plus measured tracking noise overlap the stillest real frames?",
                "n_recordings": len(rows), "medians_over_recordings": med_, "rows": rows},
               os.path.join(OUT, "hsmm_floor_diagnostic.json"))
    log(f"diagnostic: {med_}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prep", "fit", "select", "final", "controls",
                                     "planted_refit", "report", "diagnose"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--family", choices=FAMILIES, default="weibull")
    p.add_argument("--index", type=int, default=None, help="fit array index: fold*3 + family")
    p.add_argument("--boot", type=int, default=0)
    a = p.parse_args(argv)
    if a.phase == "fit" and a.index is not None:
        a.fold, a.family = a.index // len(FAMILIES), FAMILIES[a.index % len(FAMILIES)]
    return {"prep": prep, "fit": fit, "select": select, "final": final, "controls": controls,
            "planted_refit": planted_refit, "report": report, "diagnose": diagnose}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
