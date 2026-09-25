"""Step 2. Are the boundaries real? The gate.

    sbatch --array=0-29%12 jobs/seg_validate.slurm --shard
    sbatch            jobs/seg_validate.slurm --separability
    python3 scripts/seg_validate.py --combine

Registered in `results/SEGMENTATION_PREREGISTRATION.md`, committed before any
surrogate for this stage was generated.

`BREAKS.md` reports a boundary rate and calls it no evidence, because a
changepoint detector fires on noise. This stage puts four structureless signals
through the **identical** pipeline and asks whether the detector fires more on
the corpus — across a threshold sweep, and in places not explained by arena
position, body size, animal or session.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits              # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.util import describe, frames, log, peak_rss_gb           # noqa: E402
from recur.util import write_json                                   # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.audit import separable as sep                             # noqa: E402
from vieb.clean import arms as clean_arms                           # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.qc import concentration as cc                             # noqa: E402
from vieb.seg import breaks as bk, validate as va                   # noqa: E402
from vieb.tok import config, surrogate as sr                        # noqa: E402

POSE_ARM, SCALE_ARM = "raw", "bodylen"
ARMS: tuple[str, ...] = ("corpus", "phase", "var5", "ou", "white")
GROUP, DERIV_SEC, DEGREE = "both", 0.133, 3
CENTRE = 3
SEED = 0
#: Windows per animal per class for the separability probe, and the window
#: length in seconds. Frame-level features would put a linear probe at chance
#: against `phase` by construction.
SEP_PER_ANIMAL = 200
SEP_WINDOW_S = 0.4


def out_dir() -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_validate")


def ego_path(arm: str, tag: str) -> str:
    name = f"{POSE_ARM}__{SCALE_ARM}__{tag}.npz"
    if arm == "corpus":
        return os.path.join(config.REPO, "work", "ego", name)
    return os.path.join(config.REPO, "work", "surrogate", arm, "ego", name)


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.REPO, "work", "tok", "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def load_arm(arm: str, tag: str, sd: np.ndarray) -> dict:
    with np.load(ego_path(arm, tag), allow_pickle=False) as z:
        if arm == "corpus":
            got = str(z["pose_arm"]) if "pose_arm" in z.files else "<absent>"
            if got != POSE_ARM:
                raise SystemExit(f"[{tag}] corpus shard says pose_arm={got!r}")
        out = {"X": np.asarray(z["X"], dtype=np.float64) / sd[None, :],
               "bounds": z["bounds"].astype(np.int64),
               "ell_a": float(z["ell_a"]),
               "recording_ids": [str(v) for v in z["recording_ids"]]}
    with np.load(os.path.join(config.REPO, "work", "tok", "abstain",
                              f"{tag}.npz"), allow_pickle=False) as z:
        out["abstain"] = z["abstain"].astype(bool)
    return out


def shard(args, tag: str) -> int:
    os.makedirs(out_dir(), exist_ok=True)
    fps = spine.fps()
    sd = basis_sd()
    h = max(2, frames(DERIV_SEC, fps))
    guard = bk.guard_frames(fps, deriv_sec=DERIV_SEC)
    min_gap = bk.min_segment_frames(DEGREE, guard)
    idx = list(bk.CHANNEL_GROUPS[GROUP])

    curves: dict = {}
    for arm in ARMS:
        a = load_arm(arm, tag, sd)
        curves[arm] = va.rate_curve(a["X"][:, idx], a["bounds"], a["abstain"],
                                    h=h, min_gap=min_gap, fps=fps)
        if arm == "corpus":
            corpus = a

    # The artifact ablation: the same curve with nothing blocked.
    free = va.rate_curve(corpus["X"][:, idx], corpus["bounds"],
                         np.zeros(corpus["X"].shape[0], dtype=bool),
                         h=h, min_gap=min_gap, fps=fps)

    # Per-recording: boundary rate, arena position, and the segment table the
    # length-matched control and the duration distribution are built from.
    rng = np.random.default_rng(seeds.stable_seed(SEED, "segval", tag))
    per_rec, dur, adj, wins, cmp_n, dwin = [], [], [], 0, 0, []
    for r, rid in enumerate(corpus["recording_ids"]):
        lo, hi = int(corpus["bounds"][r]), int(corpus["bounds"][r + 1])
        sub = corpus["X"][lo:hi, idx]
        d = bk.discontinuity(sub, h)
        pk = bk.boundaries(d, bk.mad_threshold(d, bk.K_MAD), min_gap=min_gap,
                           blocked=corpus["abstain"][lo:hi])
        rows = bk.segment_table(corpus["X"], pk + lo, lo=lo, hi=hi,
                                abstain=corpus["abstain"], guard=guard,
                                degree=DEGREE, fps=fps)
        ctl = va.length_matched_control(corpus["X"], rows, lo, hi, rng,
                                        degree=DEGREE, n_draws=2)
        if np.isfinite(ctl["win_rate"]):
            wins += int(round(ctl["win_rate"] * ctl["n_compared"]))
            cmp_n += int(ctl["n_compared"])
            dwin.append(float(ctl["mean_delta_r2_adj"]))
        clean = [w for w in rows if w["abstain_frac"] == 0.0]
        dur += [w["duration_s"] for w in clean]
        adj += [w["fit_r2_adj"] for w in clean if np.isfinite(w["fit_r2_adj"])]
        # Arena position: the centroid's own cloud, as `CONCENTRATION.md` does.
        cl = spine.clean(rid)
        held = clean_arms.held_array(cl["pose_unfiltered"].astype(np.float64),
                                     cl["missing"].astype(bool))
        edge = cc.edgeness(held[:, CENTRE])
        per_rec.append({
            "recording_id": rid, "animal": tag,
            "n_frames": hi - lo, "n_peaks": int(pk.size),
            "rate": float(pk.size / ((hi - lo) / fps)),
            "mean_edgeness": float(np.nanmean(edge)),
            "ell_a": corpus["ell_a"],
            **{k: v for k, v in _session(rid).items()},
        })

    write_json({"animal": tag, "pose_arm": POSE_ARM,
                "cell": {"group": GROUP, "deriv_sec": DERIV_SEC,
                         "degree": DEGREE, "h": h, "guard": guard},
                "curves": {a: {"n_peaks": c["n_peaks"], "seconds": c["seconds"]}
                           for a, c in curves.items()},
                "ablation_unblocked": {"n_peaks": free["n_peaks"],
                                       "seconds": free["seconds"]},
                "length_matched": {"n_compared": cmp_n,
                                   "win_rate": (wins / cmp_n) if cmp_n
                                   else float("nan"),
                                   "mean_delta_r2_adj": (float(np.mean(dwin))
                                                         if dwin else float("nan"))},
                "duration_clean_s": describe(np.asarray(dur)) if dur else {},
                "fit_r2_adj_clean": describe(np.asarray(adj)) if adj else {},
                "per_recording": per_rec,
                "inherited_digest": spine.digest()},
               os.path.join(out_dir(), f"{tag}.json"))
    c = curves["corpus"]["rate"] if "rate" in curves["corpus"] else {}
    log(f"[{tag}] corpus {c.get(3.0, float('nan')):.3f} b/s at k=3, "
        f"phase {curves['phase']['rate'][3.0]:.3f}, "
        f"white {curves['white']['rate'][3.0]:.3f}, "
        f"length-matched win {(wins / cmp_n) if cmp_n else float('nan'):.3f}, "
        f"peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _session(rid: str) -> dict:
    try:
        p = lab.parse(rid)
        return {"day": str(p.get("day")), "context": str(p.get("context")),
                "date": str(p.get("date"))}
    except Exception:                                    # noqa: BLE001
        return {"day": "?", "context": "?", "date": "?"}


def separability(args) -> int:
    """Can a probe tell corpus windows from surrogate windows?

    **Window features, not frames**, because a `phase` surrogate matches every
    first- and second-order moment and a probe on single frames is at chance by
    construction.

    **And roughness features on top of the raw window, because the negative
    control proved the raw window was not enough.** The first run of this probe
    returned `white` — i.i.d. noise — as *not separable* from the corpus at
    balanced accuracy 0.523 and **AUC 0.496**, which is chance. White noise is
    obviously distinguishable from a smooth trajectory, so that PASS was a fact
    about the instrument, not about the null: `separability` fits a **logistic
    regression**, which is linear in its features, and the difference between a
    smooth signal and a rough one lives in the *second* moment of the
    increments — a quadratic function of the raw window that no linear model can
    form.

    So each window carries, alongside its raw frames, the per-channel
    `log` mean squared first difference. That is the smoothness the whole
    segmentation arm is about, made visible to a linear model. `white` should now
    fail and `phase` should still pass, since phase matches the increment
    spectrum exactly — and if `phase` fails too, the nulls are off-manifold and
    the boundary comparison is uninterpretable.

    The control was what caught this, which is the argument for having run a
    null whose answer is known.
    """
    tags = sorted(animals_of(spine.recording_ids()))
    # ONE generator threaded through the arms in ARMS order, which is what the
    # first run did. Re-seeding per arm would pair the windows and change every
    # number already on disk.
    rng = np.random.default_rng(SEED)
    raw, rough, grp = {}, {}, {}
    for arm in ARMS:
        raw[arm], rough[arm], grp[arm] = sep_windows(arm, tags, rng)
    log(f"corpus windows: {raw['corpus'].shape} raw + "
        f"{rough['corpus'].shape} roughness")
    reads, out = {}, {}
    for kind in ARMS[1:]:
        x = np.concatenate([
            np.concatenate([raw["corpus"], rough["corpus"]], axis=1),
            np.concatenate([raw[kind], rough[kind]], axis=1)])
        is_surr = np.concatenate([np.zeros(raw["corpus"].shape[0], bool),
                                  np.ones(raw[kind].shape[0], bool)])
        res = sep.separability(x, is_surr, grp["corpus"] + grp[kind], seed=SEED)
        obj = {"dataset": "luna", "arm": "seg_manifold", "surrogate": kind,
               "pose_arm": POSE_ARM, "window_s": SEP_WINDOW_S,
               "features": int(x.shape[1])}
        rd = va.manifold_read(res, kind=kind, scored_object=obj,
                              n_effective=len(tags))
        reads[kind] = rd.to_dict()
        out[kind] = res
        log(rd.line())
    write_json({**provenance.header(anchors.LUNA, stage="seg_manifold",
                                 unverified="windows subsampled per animal"),
                "inherited_digest": spine.digest(),
                "window_s": SEP_WINDOW_S, "per_animal": SEP_PER_ANIMAL,
                "features": ("raw stacked window plus per-channel log mean "
                             "squared first difference; the roughness terms "
                             "exist because a linear probe on the raw window "
                             "returned i.i.d. WHITE NOISE as inseparable at "
                             "AUC 0.496"),
                "reads": reads, "results": out,
                "peak_rss_gb": peak_rss_gb()},
               os.path.join(out_dir(), "_separability.json"))
    return 0


def sep_windows(arm: str, tags: Sequence[str],
                rng: np.random.Generator) -> tuple:
    """Stacked windows for one arm: raw frames, per-channel roughness, groups.

    Roughness is `log` mean squared first difference per channel. It is a
    *quadratic* function of the window, which a logistic regression cannot form
    from raw frames -- which is why the first run of this probe returned i.i.d.
    white noise as inseparable from the corpus at AUC 0.496.
    """
    fps = spine.fps()
    sd = basis_sd()
    w = frames(SEP_WINDOW_S, fps)
    raw, rough, grp = [], [], []
    for tag in tags:
        a = load_arm(arm, tag, sd)
        x, b = a["X"], a["bounds"]
        starts = []
        for r in range(b.shape[0] - 1):
            lo, hi = int(b[r]), int(b[r + 1])
            if hi - lo > w:
                starts += [(int(s), a["recording_ids"][r])
                           for s in rng.integers(lo, hi - w,
                                                 size=max(1, SEP_PER_ANIMAL
                                                          // (b.shape[0] - 1)))]
        for s, rid in starts:
            win = x[s:s + w]
            d1 = np.diff(win, axis=0)
            raw.append(win.ravel())
            rough.append(np.log(np.maximum((d1 ** 2).mean(axis=0), 1e-12)))
            # Class-prefixed, or corpus and surrogate blocks from one
            # recording collide into a single group and land on the same
            # side of the held-out split.
            grp.append(f"{arm}:{rid}:{s // (w * 8)}")
    return (np.asarray(raw, dtype=np.float64),
            np.asarray(rough, dtype=np.float64), grp)


def diagnose(args) -> int:
    """**Post-hoc. Written after the probe failed on all four nulls.**

    It answers one descriptive question and registers no verdict: *what is the
    probe separating on?* Two candidates, with opposite consequences.

    **A level difference in roughness** would be benign for the gate. The
    boundary threshold is `median(D) + k*1.4826*MAD(D)` of each recording's own
    `D`, so a null that is uniformly rougher or smoother than the corpus is
    standardised back before any boundary is counted. The probe sees a
    difference the gate does not.

    **A difference in how roughness VARIES from window to window** is not
    benign, and is not benign in an interesting way. A corpus with boundaries in
    it is non-stationary: roughness is low inside a smooth stretch and high at a
    kink. A null built by phase randomisation is stationary by construction and
    matches the *average* increment spectrum exactly. So a probe sharp enough to
    see smoothness will separate them **because of the boundaries**, and the
    registered precondition -- a null that no probe can pick out -- would be
    unsatisfiable for any null that lacks the very structure under test.

    Reported, not acted on. Whether the precondition is ill-posed is not a
    question this stage gets to answer about itself.
    """
    os.makedirs(out_dir(), exist_ok=True)
    tags = sorted(animals_of(spine.recording_ids()))
    rng = np.random.default_rng(SEED)
    raw, rough, grp = {}, {}, {}
    for arm in ARMS:
        raw[arm], rough[arm], grp[arm] = sep_windows(arm, tags, rng)

    levels = {arm: {"mean_log_msd": r.mean(axis=0).tolist(),
                    "sd_across_windows": r.std(axis=0).tolist(),
                    "mean_over_channels": float(r.mean()),
                    "sd_over_channels": float(r.std(axis=0).mean())}
              for arm, r in rough.items()}

    parts: dict = {}
    for kind in ARMS[1:]:
        is_surr = np.concatenate([np.zeros(raw["corpus"].shape[0], bool),
                                  np.ones(raw[kind].shape[0], bool)])
        g = grp["corpus"] + grp[kind]
        feats = {
            "raw_only": (raw["corpus"], raw[kind]),
            "roughness_only": (rough["corpus"], rough[kind]),
            # KEPT, AND IT ANSWERS NOTHING. The intent was to remove a level
            # difference and leave a variability one, but both arms are
            # transformed by the SAME affine constants and a logistic
            # regression is affine-invariant, so this is arithmetically
            # identical to `roughness_only` and came back identical to four
            # decimals. It is left in as the record of a null test that could
            # not have failed. The level question is answered by
            # `roughness_levels` below, which is a description rather than a
            # probe.
            "roughness_centred": (
                (rough["corpus"] - rough["corpus"].mean(axis=0))
                / np.maximum(rough["corpus"].std(axis=0), 1e-12),
                (rough[kind] - rough["corpus"].mean(axis=0))
                / np.maximum(rough["corpus"].std(axis=0), 1e-12)),
        }
        parts[kind] = {}
        for name, (a, b) in feats.items():
            res = sep.separability(np.concatenate([a, b]), is_surr, g,
                                   seed=SEED)
            parts[kind][name] = {k: res.get(k) for k in
                                 ("balanced_accuracy", "auc", "n_test")}
            log(f"  {kind:6s} {name:18s} acc="
                f"{res.get('balanced_accuracy', float('nan')):.4f} auc="
                f"{res.get('auc', float('nan')):.4f}")

    write_json({**provenance.header(anchors.LUNA, stage="seg_manifold_diagnose",
                                 unverified="post-hoc; no verdict is attached"),
                "inherited_digest": spine.digest(),
                "post_hoc": ("written after the registered probe failed on all "
                             "four nulls; descriptive only, and it does not "
                             "reopen the gate"),
                "window_s": SEP_WINDOW_S, "per_animal": SEP_PER_ANIMAL,
                "roughness_levels": levels, "by_feature_set": parts,
                "peak_rss_gb": peak_rss_gb()},
               os.path.join(out_dir(), "_separability_diagnose.json"))
    return 0


def combine(args) -> int:
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    shards = sorted(glob.glob(os.path.join(out_dir(), "*.json")))
    shards = [p for p in shards if not os.path.basename(p).startswith("_")]
    if not shards:
        raise SystemExit(f"no shards in {out_dir()}")

    rates: dict = {a: {} for a in ARMS}
    abl: dict = {}
    per_rec, tags, win_n, win_k, dwin = [], [], 0, 0, []
    dur_all, adj_all = [], []
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        tag = doc["animal"]
        if of.get(tag) != args.split:
            continue
        tags.append(tag)
        for arm, c in doc["curves"].items():
            for k, n in c["n_peaks"].items():
                rates[arm].setdefault(float(k), {})[tag] = \
                    float(n) / float(c["seconds"])
        for k, n in doc["ablation_unblocked"]["n_peaks"].items():
            abl.setdefault(float(k), {})[tag] = \
                float(n) / float(doc["ablation_unblocked"]["seconds"])
        lm = doc["length_matched"]
        if lm["n_compared"]:
            win_n += int(lm["n_compared"])
            win_k += int(round(lm["win_rate"] * lm["n_compared"]))
            dwin.append(float(lm["mean_delta_r2_adj"]))
        per_rec += doc["per_recording"]
        if doc.get("duration_clean_s"):
            dur_all.append(doc["duration_clean_s"])
        if doc.get("fit_r2_adj_clean"):
            adj_all.append(doc["fit_r2_adj_clean"])

    deltas, curve = {}, []
    for k in sorted(rates["corpus"]):
        row = {"k_mad": k, "corpus": float(np.mean(list(rates["corpus"][k].values())))}
        for arm in ARMS[1:]:
            shared = sorted(set(rates["corpus"][k]) & set(rates[arm].get(k, {})))
            d = np.array([rates["corpus"][k][t] - rates[arm][k][t]
                          for t in shared], dtype=np.float64)
            ci = boot.animal_interval(d, shared, how="mean", n_boot=2000,
                                      seed=SEED)
            deltas[f"{arm}|{k}"] = dict(ci)
            row[arm] = float(np.mean(list(rates[arm][k].values())))
            row[f"delta_{arm}"] = float(ci["point"])
            row[f"lo_{arm}"] = float(ci["lo"])
        curve.append(row)

    obj = {"dataset": "luna", "arm": "seg_gate", "split": args.split,
           "pose_arm": POSE_ARM, "group": GROUP, "deriv_sec": DERIV_SEC,
           "degree": DEGREE}
    gate = va.gate_read(deltas, scored_object=obj, n_effective=len(tags))

    doc = {**provenance.header(anchors.LUNA, stage="seg_gate",
                            unverified=f"{args.split} split only"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "split": args.split, "n_animals": len(tags),
           "registration": "results/SEGMENTATION_PREREGISTRATION.md",
           "cell": {"group": GROUP, "deriv_sec": DERIV_SEC, "degree": DEGREE},
           "gate": gate.to_dict(), "curve": curve, "deltas": deltas,
           "length_matched": {
               "n_compared": win_n,
               "win_rate": (win_k / win_n) if win_n else float("nan"),
               "mean_delta_r2_adj": float(np.mean(dwin)) if dwin else float("nan"),
               "exbias_reference": va.EXBIAS_WIN_RATE},
           "ablation": _ablation(rates["corpus"], abl),
           "confounds": _confounds(per_rec),
           "duration_clean_s": _pool(dur_all),
           "fit_r2_adj_clean": _pool(adj_all),
           "peak_rss_gb": peak_rss_gb()}
    write_json(doc, args.out)
    log(gate.line())
    for row in curve:
        log("  k=%-4g corpus %.4f  phase %.4f  var5 %.4f  ou %.4f  white %.4f"
            % (row["k_mad"], row["corpus"], row["phase"], row["var5"],
               row["ou"], row["white"]))
    log(f"  length-matched win rate {doc['length_matched']['win_rate']:.3f} "
        f"over {win_n:,} comparisons (ExBias reference "
        f"{va.EXBIAS_WIN_RATE:.3f})")
    log(f"wrote {args.out}")
    return 0


def _pool(rows) -> dict:
    if not rows:
        return {}
    keys = set().union(*[set(r) for r in rows])
    return {k: float(np.nanmean([r.get(k, np.nan) for r in rows]))
            for k in sorted(keys)}


def _ablation(blocked: dict, free: dict) -> dict:
    out = []
    for k in sorted(blocked):
        shared = sorted(set(blocked[k]) & set(free.get(k, {})))
        if not shared:
            continue
        d = np.array([free[k][t] - blocked[k][t] for t in shared])
        out.append({"k_mad": k,
                    "rate_blocked": float(np.mean([blocked[k][t] for t in shared])),
                    "rate_unblocked": float(np.mean([free[k][t] for t in shared])),
                    "delta": float(np.mean(d))})
    return {"note": ("boundary rate with abstained frames blocked against not "
                     "blocked; the difference is reported rather than the "
                     "better number"), "by_k": out}


def _confounds(per_rec) -> dict:
    from vieb.qc import bones as bn

    rate = np.array([r["rate"] for r in per_rec], dtype=np.float64)
    edge = np.array([r["mean_edgeness"] for r in per_rec], dtype=np.float64)
    ani = [str(r["animal"]) for r in per_rec]
    rho, kept = bn.per_animal_spearman(rate, edge, ani, min_segments=8)
    ci = (dict(boot.animal_interval(rho, kept, how="mean", n_boot=2000, seed=SEED))
          if rho.size >= 2 else {})

    # Arena position by decile of mean edgeness, as CONCENTRATION.md reports it.
    order = np.argsort(edge)
    dec = []
    for i in range(10):
        sel = order[i * len(order) // 10:(i + 1) * len(order) // 10]
        if sel.size:
            dec.append({"decile": i, "n_recordings": int(sel.size),
                        "mean_edgeness": float(np.mean(edge[sel])),
                        "mean_rate": float(np.mean(rate[sel]))})

    by_animal: dict = {}
    for r in per_rec:
        by_animal.setdefault(r["animal"], []).append(r)
    size = np.array([float(v[0]["ell_a"]) for v in by_animal.values()])
    arate = np.array([float(np.mean([x["rate"] for x in v]))
                      for v in by_animal.values()])
    from scipy import stats
    rho_size = float(stats.spearmanr(size, arate).statistic) \
        if size.size > 3 else float("nan")

    groups = {}
    for key in ("animal", "day", "context", "date"):
        k_by: dict = {}
        n_by: dict = {}
        for r in per_rec:
            k_by[r[key]] = k_by.get(r[key], 0) + int(r["n_peaks"])
            n_by[r[key]] = n_by.get(r[key], 0) + int(r["n_frames"])
        groups[key] = cc.grouped(k_by, n_by)

    return {
        "edgeness_spearman_per_animal": {
            "n_animals": int(len(kept)), "mean_rho": float(np.mean(rho))
            if rho.size else float("nan"), "animal_interval": ci},
        "edgeness_deciles": dec,
        "body_size_spearman_across_animals": rho_size,
        "by_group": groups,
        "note": ("violation rate rises 3.7x monotone from arena centre to wall, "
                 "so a boundary rate that tracks wall proximity is finding "
                 "occlusion rather than behaviour"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", action="store_true")
    p.add_argument("--separability", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--split", default="report")
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.separability:
        return separability(a)
    if a.diagnose:
        return diagnose(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("seg_gate.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        tags = sorted(animals_of(spine.recording_ids()))[a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --separability, --diagnose or --combine")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir(), f"{tag}.json")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
