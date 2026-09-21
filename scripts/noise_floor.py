"""How often does the boundary detector fire on nothing but tracking noise?

    python3 scripts/noise_floor.py [--limit N] [--out PATH]

READ results/NOISEFLOOR_PREREGISTRATION.md FIRST.

Three phases, one calibration shared between them:

1. **Calibrate.** Skull-bone residuals give the per-keypoint jitter amplitude
   as a function of DLC confidence. The skull is rigid, so all of its
   length variance is tracking noise -- unlike a stillness-based estimate,
   which is circular.
2. **The floor.** A constant pose plus measured jitter, through the identical
   frozen detector. Every boundary it finds is noise by construction.
3. **The corpus, stratified jointly** by speed x confidence x arena, against
   that floor in each cell. Joint because low confidence concentrates at the
   wall where the mouse rears, so a marginal cannot tell a noisy still mouse
   from a rearing one.

`tune` split only: the floor is meant to select a parameter later, and
choosing a parameter against a number is fitting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb import seeds                                                # noqa: E402
from vieb.clean import arms as clean_arms                             # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.qc import concentration as cc                               # noqa: E402
from vieb.seg import breaks as bk, floor as fl, jitter as jit         # noqa: E402
from vieb.tok import config, ego, quantize as qz                      # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
K_MAD = bk.K_MAD
CENTRE = 3
#: Quantile grids for the joint stratification. Speed and confidence in fifths,
#: arena in thirds -- 75 cells. Finer on arena would leave most cells below the
#: registered 20,000-frame floor and refuse them; the 10-decile arena profile is
#: reported separately as a marginal so the published decile structure stays
#: comparable.
N_SPEED, N_CONF, N_EDGE = 5, 5, 3


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="animals, 0 = all")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("noise_floor.json")

    fps = spine.fps()
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    by_animal: dict[str, list[str]] = {}
    for rid in spine.recording_ids():
        tag = lab.animal_tag(rid)
        if split_of.get(tag) == SPLIT:
            by_animal.setdefault(tag, []).append(rid)
    tags = sorted(by_animal)
    if a.limit:
        tags = tags[:a.limit]
    n_rec = sum(len(by_animal[t]) for t in tags)
    log(f"  {len(tags)} {SPLIT} animals, {n_rec} recordings")

    # ---- phase 1: calibrate -------------------------------------------
    res_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    ells: dict[str, float] = {}
    for tag in tags:
        with np.load(ego_path(tag), allow_pickle=False) as z:
            got = str(z["pose_arm"]) if "pose_arm" in z.files else "<absent>"
            if got != POSE_ARM:
                raise SystemExit(f"[{tag}] shard says pose_arm={got!r}, "
                                 f"expected {POSE_ARM!r}")
            ells[tag] = float(z["ell_a"])
        for rid in by_animal[tag]:
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            r, c = jit.residuals(held, d["conf"], missing=d["missing"],
                                 interpolated=d["interpolated"],
                                 ell=ells[tag])
            if r.size:
                res_all.append(r.astype(np.float32))
                conf_all.append(c.astype(np.float32))
    resid = np.concatenate(res_all) if res_all else np.zeros(0)
    confs = np.concatenate(conf_all) if conf_all else np.zeros(0)
    del res_all, conf_all
    table = jit.calibrate(resid, confs)
    obj = {"dataset": "luna", "arm": "noise_floor", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "k_mad": K_MAD,
           "deriv_sec": bk.DERIV_SEC, "degree": bk.DEGREE}
    cal = jit.calibration_read(table, scored_object={**obj, "arm": "jitter"},
                               n_effective=len(tags))
    log("  " + cal.line())
    if cal.verdict != "PASS":
        write_json({**anchors.header(anchors.LUNA, stage="noise_floor",
                                     unverified="an instrument probe on tune"),
                    "inherited_digest": spine.digest(),
                    "registration":
                        "results/NOISEFLOOR_PREREGISTRATION.md",
                    "calibration": table,
                    "reads": {"calibration": cal.to_dict()}}, out)
        log(f"  wrote {out}")
        return 1
    del resid, confs

    # ---- phase 2: the floor, and phase 3: the corpus, per animal -------
    per_arm: dict[str, list[float]] = {k: [] for k in fl.ARMS}
    who: list[str] = []
    n_pk_arm: dict[str, int] = {k: 0 for k in fl.ARMS}
    n_sec_arm: dict[str, float] = {k: 0.0 for k in fl.ARMS}
    sp_all, cf_all, eg_all, pk_all, an_all = [], [], [], [], []
    corpus_rate: list[float] = []

    for n, tag in enumerate(tags, 1):
        ell = ells[tag]
        with np.load(ego_path(tag), allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float64)
            bounds = z["bounds"].astype(np.int64)
            valid = z["valid"].astype(bool)
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as z:
            abstain = z["abstain"].astype(bool) | ~valid

        # the corpus arm, at the frozen detector
        Xs = X / sd[None, :]
        got = fl.rate_of(Xs, bounds, abstain, idx=idx, fps=fps, k_mad=K_MAD)
        corpus_rate.append(float(got["rate_per_s"]))

        # per-frame covariates and boundary flags, for the stratification
        speed = qz.speed(X)
        conf_min = np.full(X.shape[0], np.nan)
        edge = np.full(X.shape[0], np.nan)
        is_pk = np.zeros(X.shape[0], dtype=bool)
        h = max(2, int(round(bk.DERIV_SEC * fps)))
        noisy: dict[str, list[np.ndarray]] = {k: [] for k in fl.ARMS}
        rng = np.random.default_rng(seeds.stable_seed(SEED, f"floor|{tag}"))
        for r, rid in enumerate(by_animal[tag]):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            conf_min[lo:hi] = np.asarray(d["conf"], float).min(axis=1)
            edge[lo:hi] = cc.edgeness(held[:, CENTRE])
            usable = spine.representation(rid)["usable"].astype(bool)
            if hi - lo >= 2 * h + 1:
                pk = fl.peaks_of(Xs[lo:hi], idx=idx, fps=fps, k_mad=K_MAD,
                                 blocked=abstain[lo:hi])
                is_pk[lo + pk] = True
            # the injection arms: a constant pose carrying only jitter
            base = fl.static_pose(held, usable)
            for arm in fl.ARMS:
                sc = fl.SCALES[arm]
                pose = base if sc == 0.0 else (
                    base + jit.draw(rng, d["conf"], table, scale=sc) * ell)
                Xi, vi = ego.transform(pose, ell, fps, usable=usable)
                noisy[arm].append(Xi / sd[None, :])
        for arm in fl.ARMS:
            Xi = np.concatenate(noisy[arm], axis=0)
            g = fl.rate_of(Xi, bounds, abstain, idx=idx, fps=fps, k_mad=K_MAD)
            per_arm[arm].append(float(g["rate_per_s"]))
            n_pk_arm[arm] += int(g["n_peaks"])
            n_sec_arm[arm] += float(g["eligible_seconds"])
        del noisy
        who.append(tag)

        elig = ~abstain
        elig[:h] = False
        elig[-h:] = False
        sp_all.append(speed[elig].astype(np.float32))
        cf_all.append(conf_min[elig].astype(np.float32))
        eg_all.append(edge[elig].astype(np.float32))
        pk_all.append(is_pk[elig])
        an_all.append(np.full(int(elig.sum()), tag))
        if n % 10 == 0 or n == len(tags):
            log(f"  {n}/{len(tags)} animals")

    speed_a = np.concatenate(sp_all).astype(np.float64)
    conf_a = np.concatenate(cf_all).astype(np.float64)
    edge_a = np.concatenate(eg_all).astype(np.float64)
    peak_a = np.concatenate(pk_all)
    anim_a = np.concatenate(an_all)
    del sp_all, cf_all, eg_all, pk_all, an_all

    reads: dict = {"calibration": cal.to_dict()}
    rates: dict = {}
    for arm in fl.ARMS:
        ci = boot.animal_interval(per_arm[arm], who, how="mean",
                                  n_boot=fl.N_BOOT, seed=SEED)
        secs = n_sec_arm[arm]
        rates[arm] = {"n_peaks": n_pk_arm[arm], "eligible_seconds": secs,
                      "rate_per_s": (n_pk_arm[arm] / secs) if secs > 0
                      else float("nan"), "ci": dict(ci)}
    ci_c = boot.animal_interval(corpus_rate, who, how="mean",
                                n_boot=fl.N_BOOT, seed=SEED)
    rates["corpus"] = {"ci": dict(ci_c),
                       "rate_per_s": float(np.mean(corpus_rate))}

    st = fl.static_read(rates["static"], scored_object={**obj, "arm": "static"},
                        n_effective=len(tags))
    reads["static"] = st.to_dict()
    log("  " + st.line())

    fr = fl.floor_read(rates["jitter1.0"], fps=fps,
                       scored_object={**obj, "arm": "floor"},
                       n_effective=len(tags))
    reads["floor"] = fr.to_dict()
    log("  " + fr.line())

    dr = fl.dose_read(rates, scored_object={**obj, "arm": "dose"},
                      n_effective=len(tags))
    reads["dose"] = dr.to_dict()
    log("  " + dr.line())

    # The headline: the whole corpus against the whole floor, same detector,
    # same split, same standardising SD.
    whole = fl.separation_read(
        {"refused": False, "n_frames": int(peak_a.size),
         "n_peaks": int(peak_a.sum()), "ci": rates["corpus"]["ci"]},
        rates["jitter1.0"], label="corpus overall",
        scored_object={**obj, "arm": "corpus_vs_floor"},
        n_effective=len(tags))
    reads["corpus_vs_floor"] = whole.to_dict()
    log("  " + whole.line())

    # ---- the joint stratification --------------------------------------
    s_edges = np.quantile(speed_a, np.linspace(0, 1, N_SPEED + 1))
    c_edges = np.quantile(conf_a, np.linspace(0, 1, N_CONF + 1))
    e_edges = np.quantile(edge_a, np.linspace(0, 1, N_EDGE + 1))
    for e in (s_edges, c_edges, e_edges):
        e[0], e[-1] = -np.inf, np.inf
    cells = fl.cell_index(speed_a, conf_a, edge_a, speed_edges=s_edges,
                          conf_edges=c_edges, edge_edges=e_edges)
    n_cells = N_SPEED * N_CONF * N_EDGE
    rows = fl.cell_rates(cells, peak_a, anim_a, n_cells=n_cells, fps=fps)
    n_ref = sum(1 for r in rows if r["refused"])
    log(f"  {n_cells - n_ref} of {n_cells} joint cells above the "
        f"{fl.MIN_CELL_FRAMES:,}-frame floor; {n_ref} refused")

    # §5: the marginals, at full resolution, beside the joint grid. The arena
    # one is cut on the PUBLISHED decile edges so it stays comparable with
    # `CONCENTRATION.md` rather than being a fresh quantile grid.
    def _marginal(v, edges, name):
        b = np.clip(np.searchsorted(edges, v, "right") - 1, 0,
                    len(edges) - 2)
        got = []
        for k in range(len(edges) - 1):
            m = b == k
            n = int(m.sum())
            got.append({"bin": k, "n_frames": n,
                        "rate_per_s": (float(peak_a[m].sum()) / (n / fps))
                        if n >= fl.MIN_CELL_FRAMES else float("nan"),
                        "refused": n < fl.MIN_CELL_FRAMES})
        return {"axis": name, "edges": list(map(float, edges)), "bins": got}

    with open(config.PATHS.result("concentration.json"),
              encoding="utf-8") as fh:
        prof = json.load(fh)["edge_profile"]
    dec_edges = np.asarray([b["lo_iqr"] for b in prof]
                           + [float("inf")], dtype=np.float64)
    dec_edges[0] = -np.inf
    marginals = [_marginal(speed_a, s_edges, "speed_quintile"),
                 _marginal(conf_a, c_edges, "confidence_quintile"),
                 _marginal(edge_a, dec_edges, "arena_decile_published")]

    # The registered cell: slowest speed, lowest confidence, most central.
    target = (0 * N_CONF + 0) * N_EDGE + 0
    sep = fl.separation_read(rows[target], rates["jitter1.0"],
                             label="still x centre x low-confidence",
                             scored_object={**obj, "arm": "separation"},
                             n_effective=len(tags))
    reads["separation"] = sep.to_dict()
    log("  " + sep.line())

    write_json({**anchors.header(anchors.LUNA, stage="noise_floor",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/NOISEFLOOR_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(tags), "seed": SEED,
                "group": GROUP, "k_mad": K_MAD, "deriv_sec": bk.DERIV_SEC,
                "degree": bk.DEGREE, "fps": fps,
                "nms_ceiling_per_s": fl.nms_ceiling(fps),
                "calibration": table,
                "rates": rates,
                "strata": {"n_speed": N_SPEED, "n_conf": N_CONF,
                           "n_edge": N_EDGE,
                           "speed_edges": s_edges.tolist(),
                           "conf_edges": c_edges.tolist(),
                           "edge_edges": e_edges.tolist(),
                           "target_cell": target,
                           "marginals": marginals,
                           "n_refused": n_ref, "cells": rows},
                "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
