"""Distortion, frailty, and symbol homogeneity -- the resolution measurements.

    python3 scripts/resolution.py --write-grid
    sbatch --array=0-7%4 jobs/resolution.slurm --covariates
    sbatch --array=0-7%4 jobs/resolution.slurm --distortion
    sbatch            jobs/resolution.slurm --frailty
    python3 scripts/resolution.py --combine

All three measurements need the same expensive join -- the egocentric array
against the label shard, per animal, per cell -- so `--covariates` does it once
and writes a **run table**. Everything after reads that.

The run table carries, per run: `code`, `duration`, `recording` (offset per
animal so no two animals are ever adjacent), `animal`, `mean_speed` and
`mean_dist`. Both covariates are **run-level means over the run's own frames**,
never per frame: per frame they would reintroduce the frame-mass bias run-length
encoding exists to remove, since a long freeze would contribute hundreds of slow
frames and a brief dart a handful of fast ones.

`mean_dist` is `NaN` on abstain runs, and deliberately so: an abstained frame
has no centroid, so it has no distance to one, and a zero there would read as a
perfectly quantized frame.
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

from recur import anchors, labels as lab, splits                  # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.util import log, peak_rss_gb, write_json               # noqa: E402
from vieb.io import spine                                         # noqa: E402
from vieb.tok import config, distortion as dz, ego                # noqa: E402
from vieb.tok import frailty as fr, hazard as hz, ladder as ld    # noqa: E402
from vieb.tok import homogeneity as hm                             # noqa: E402
from vieb.tok import quantize as qz, rle                          # noqa: E402

POSE_ARM, SCALE_ARM = "raw", "bodylen"
SEED = 0


def tok_dir(*parts: str) -> str:
    return os.path.join(config.PATHS.tok_dir, *parts)


def ego_shard(tag: str) -> str:
    return os.path.join(config.PATHS.ego_dir,
                        f"{POSE_ARM}__{SCALE_ARM}__{tag}.npz")


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def cell_dir(arm: str, n: int) -> str:
    return tok_dir(f"{arm}_N{n}")


def runs_path(arm: str, n: int) -> str:
    return tok_dir("runs", f"{arm}_N{n}.npz")


def read_model(arm: str, n: int) -> dict:
    with np.load(os.path.join(cell_dir(arm, n), "_model.npz"),
                 allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# 1. The shared pass
# --------------------------------------------------------------------------

def covariates(args) -> int:
    arm, n = args.arm, int(args.n_states)
    os.makedirs(tok_dir("runs"), exist_ok=True)
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    model = read_model(arm, n)
    cent, sd, cols = model["centroids"], model["sd"], model["cols"]
    rng = np.random.default_rng(SEED)

    tags = sorted(animals_of(spine.recording_ids()))
    cols_out: dict = {k: [] for k in ("code", "duration", "recording",
                                      "animal", "split", "mean_speed",
                                      "mean_dist")}
    acc_parts, checks, base = [], [], 0
    for tag in tags:
        with np.load(ego_shard(tag), allow_pickle=False) as z:
            X, valid, bounds = z["X"], z["valid"].astype(bool), z["bounds"]
            n_rec = len(z["recording_ids"])
        with np.load(os.path.join(cell_dir(arm, n), f"{tag}.npz"),
                     allow_pickle=False) as z:
            y = z["labels"]
        with np.load(tok_dir("abstain", f"{tag}.npz"), allow_pickle=False) as z:
            ab = z["abstain"].astype(bool)
        if not (X.shape[0] == y.shape[0] == ab.shape[0]):
            raise SystemExit(f"[{tag}] length mismatch: X {X.shape[0]}, "
                             f"labels {y.shape[0]}, abstain {ab.shape[0]}")

        keep = valid & ~ab & (y >= 0)
        sp = qz.speed(X)
        dist = np.full(X.shape[0], np.nan, dtype=np.float64)
        if keep.any():
            dist[keep] = dz.frame_sq_error(X[keep], y[keep], cent, sd, cols)
            acc_parts.append(dz.accumulate(X, y, cent, sd, cols, n_states=n,
                                           keep=keep))
            checks.append(dz.verify_assignment(
                X[keep], y[keep], model, rng=rng, sp=sp[keep], n_sample=2000))

        r = rle.encode(y, bounds)
        # Per-run means over the run's own frames. `np.add.at` over a run index
        # rather than a Python loop: 5.8M runs at N=256.
        run_of = np.repeat(np.arange(r["code"].shape[0]),
                           r["duration"].astype(np.int64))
        ok = np.isfinite(dist)
        cnt = np.bincount(run_of[ok], minlength=r["code"].shape[0]).astype(np.float64)
        s_sum = np.bincount(run_of, weights=sp, minlength=r["code"].shape[0])
        d_sum = np.bincount(run_of[ok], weights=dist[ok],
                            minlength=r["code"].shape[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sp = s_sum / r["duration"].astype(np.float64)
            mean_d = np.where(cnt > 0, d_sum / np.maximum(cnt, 1.0), np.nan)

        cols_out["code"].append(r["code"])
        cols_out["duration"].append(r["duration"])
        cols_out["recording"].append(r["recording"].astype(np.int64) + base)
        cols_out["animal"].append(np.full(r["code"].shape[0], tag))
        cols_out["split"].append(np.full(r["code"].shape[0], of.get(tag, "?")))
        cols_out["mean_speed"].append(mean_sp)
        cols_out["mean_dist"].append(mean_d)
        base += n_rec

    out = {k: np.concatenate(v) for k, v in cols_out.items()}
    acc = dz.merge(acc_parts)
    agree = float(np.mean([c["frac_agree"] for c in checks]))
    np.savez_compressed(
        runs_path(arm, n), **out,
        sse_channel=acc["sse_channel"], sse_symbol=acc["sse_symbol"],
        n_symbol=acc["n_symbol"], n_frames=np.array(acc["n_frames"]),
        frac_agree=np.array(agree),
        n_checked=np.array(sum(c["n_checked"] for c in checks)),
        arm=np.array(arm), n_states=np.array(n),
        inherited_digest=np.array(spine.digest()))
    log(f"[{arm} N={n}] {out['code'].shape[0]:,} runs, "
        f"{int(acc['n_frames']):,} scored frames, assignment agreement "
        f"{agree:.4%}, peak_rss={peak_rss_gb():.2f} GB")
    return 0


def load_runs(arm: str, n: int) -> dict:
    p = runs_path(arm, n)
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- run `--covariates` first")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# 2. Distortion
# --------------------------------------------------------------------------

def distortion(args) -> int:
    arm, n = args.arm, int(args.n_states)
    os.makedirs(tok_dir("resolution"), exist_ok=True)
    t = load_runs(arm, n)
    model = read_model(arm, n)
    acc = {"sse_channel": t["sse_channel"], "sse_symbol": t["sse_symbol"],
           "n_symbol": t["n_symbol"], "n_frames": int(t["n_frames"]),
           "n_states": n}
    summary = dz.summarise(acc, model["sd"])
    obj = {"dataset": "luna", "arm": arm, "n_states": n, "split": "all",
           "pose_arm": POSE_ARM, "retired": True}
    rd = dz.distortion_read(summary, {"frac_agree": float(t["frac_agree"]),
                                      "n_checked": int(t["n_checked"])},
                            scored_object=obj, n_effective=89)
    doc = {**provenance.header(anchors.LUNA, stage="tok_distortion",
                            unverified="scored on non-abstained frames; the "
                                       "anchor counts all"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "arm": arm, "n_states": n,
           "retired_by_runlength": True,
           "read": rd.to_dict(), "summary": summary,
           "peak_rss_gb": peak_rss_gb()}
    write_json(doc, tok_dir("resolution", f"distortion_{arm}_N{n}.json"))
    log(rd.line())
    return 0


# --------------------------------------------------------------------------
# 3. Frailty
# --------------------------------------------------------------------------

def frailty(args) -> int:
    arm, n = args.arm or "plain", int(args.n_states or 256)
    os.makedirs(tok_dir("resolution"), exist_ok=True)
    t = load_runs(arm, n)
    runs = {"code": t["code"], "duration": t["duration"],
            "recording": t["recording"]}
    d = ld.prepare(runs, n, abstain="conditioned")
    keep = d["keep"]
    animal, split = t["animal"][keep], t["split"][keep]
    speed = t["mean_speed"][keep]

    # Quintile edges on TUNE runs. A slice boundary is a choice and it is made
    # where every other choice on this branch is made.
    edges_q = fr.quintile_edges(speed, split == "tune")
    quint = fr.quintile_of(speed, edges_q)
    edges_d = hz.duration_grid(d["duration"][split == "tune"])
    log(f"[{arm} N={n}] duration grid {edges_d[:-1].tolist()} .. open")
    log(f"[{arm} N={n}] run-speed quintile edges {np.round(edges_q, 5).tolist()}")

    rep = split == "report"
    dur_r, cen_r = d["duration"][rep], d["censored"][rep]
    ani_r, q_r, code_r = animal[rep], quint[rep], d["code"][rep]
    log(f"[{arm} N={n}] report: {int(rep.sum()):,} runs, "
        f"{len(set(ani_r))} animals")

    pooled_c = fr.curve(dur_r, cen_r, edges_d)
    pooled = fr.fall_ratio(pooled_c["hazard"][0])

    slices, reads = {}, {}
    obj = {"dataset": "luna", "arm": arm, "n_states": n, "split": "report",
           "pose_arm": POSE_ARM, "retired": True}

    specs = [("speed_quintile", q_r, fr.N_QUINTILES)]
    top = np.argsort(np.bincount(code_r, minlength=n))[::-1][:10]
    sid = np.full(code_r.shape[0], -1, dtype=np.int64)
    rank = {int(c): i for i, c in enumerate(top)}
    tags = sorted(set(str(a) for a in ani_r))
    a_index = {t: i for i, t in enumerate(tags)}
    for i, c in enumerate(code_r):
        r_ = rank.get(int(c))
        if r_ is not None:
            sid[i] = r_ * len(tags) + a_index[str(ani_r[i])]
    specs.append(("state_x_animal_top10", sid, 10 * len(tags)))

    for name, ids, n_sl in specs:
        c = fr.curve(dur_r, cen_r, edges_d, slice_id=ids, n_slices=n_sl)
        # Every slice is compared to the pool over the elapsed-bin window
        # reportable in BOTH, never over its own range against the pool's. A
        # fast quintile's runs do not reach the late bins, so an unrestricted
        # comparison reads it as flat when its hazard is nothing of the kind.
        spans = [fr.common_span(c["hazard"][s], pooled_c["hazard"][0])
                 for s in range(n_sl)]
        rows = [fr.fall_ratio(c["hazard"][s], span=spans[s])
                if spans[s][0] >= 0 else fr.fall_ratio(c["hazard"][s])
                for s in range(n_sl)]
        pool_rows = [fr.fall_ratio(pooled_c["hazard"][0], span=spans[s])
                     if spans[s][0] >= 0 else {"log_fall": float("nan")}
                     for s in range(n_sl)]
        w = np.array([c["exits"][s].sum() for s in range(n_sl)], dtype=np.float64)
        lf = np.array([r_["log_fall"] for r_ in rows], dtype=np.float64)
        lp = np.array([r_["log_fall"] for r_ in pool_rows], dtype=np.float64)
        ok = np.isfinite(lf) & np.isfinite(lp) & (w > 0)
        within = {"fall": float(np.exp(np.average(lf[ok], weights=w[ok])))
                  if ok.any() else float("nan"),
                  "pooled_fall_on_same_spans": float(
                      np.exp(np.average(lp[ok], weights=w[ok])))
                  if ok.any() else float("nan"),
                  "n_reportable_slices": int(ok.sum()), "n_slices": int(n_sl)}
        pa = fr.retained_by_animal(dur_r, cen_r, ani_r, ids, edges_d,
                                   n_slices=n_sl)
        rd = fr.frailty_read(pa, pooled, within, slice_name=name,
                             scored_object=obj,
                             n_effective=len(set(str(a) for a in ani_r)),
                             seed=SEED)
        reads[name] = rd.to_dict()
        slices[name] = {
            "within": within,
            "per_slice": [{"slice": s, "fall": rows[s]["fall"],
                           "pooled_fall_same_span": pool_rows[s].get("fall"),
                           "span": list(spans[s]),
                           "first_bin": rows[s]["first_bin"],
                           "last_bin": rows[s]["last_bin"],
                           "exits": float(w[s]),
                           "hazard_by_bin": [None if not np.isfinite(v) else float(v)
                                             for v in c["hazard"][s]],
                           "exits_by_bin": c["exits"][s].tolist(),
                           "at_risk_by_bin": c["at_risk"][s].tolist()}
                          for s in range(min(n_sl, fr.N_QUINTILES))],
            "n_animals_usable": int(pa["n_usable"]),
            "n_animals_refused": int(pa["n_refused"])}
        log(rd.line())

    doc = {**provenance.header(anchors.LUNA, stage="tok_frailty",
                            unverified="report split only; the anchor counts all"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "arm": arm, "n_states": n,
           "retired_by_runlength": True,
           "duration_grid": edges_d[:-1].tolist(),
           "speed_quintile_edges": edges_q.tolist(),
           "pooled": {**pooled,
                      "hazard_by_bin": [None if not np.isfinite(v) else float(v)
                                        for v in pooled_c["hazard"][0]],
                      "exits_by_bin": pooled_c["exits"][0].tolist(),
                      "at_risk_by_bin": pooled_c["at_risk"][0].tolist()},
           "slices": slices, "reads": reads,
           "n_report_runs": int(rep.sum()),
           "peak_rss_gb": peak_rss_gb()}
    write_json(doc, tok_dir("resolution", f"frailty_{arm}_N{n}.json"))
    log(f"wrote frailty_{arm}_N{n}.json")
    return 0


# --------------------------------------------------------------------------
# 4. Symbol homogeneity
# --------------------------------------------------------------------------

def homogeneity(args) -> int:
    arm, n = args.arm, int(args.n_states)
    os.makedirs(tok_dir("resolution"), exist_ok=True)
    t = load_runs(arm, n)
    d = ld.prepare({"code": t["code"], "duration": t["duration"],
                    "recording": t["recording"]}, n, abstain="conditioned")
    keep = d["keep"]
    animal, split = t["animal"][keep], t["split"][keep]
    dist = t["mean_dist"][keep]

    edges = hz.duration_grid(d["duration"][split == "tune"])
    rep = split == "report"
    code, dur = d["code"][rep], d["duration"][rep]
    nxt, cen = d["next_state"][rep], d["censored"][rep]
    ani, dst = animal[rep], dist[rep]
    dbin = hz.bin_of(np.maximum(dur - 1, 0), edges)

    # Quartiles are cut within (symbol, animal). An animal whose frames sit
    # systematically far from every centroid would otherwise fill the far
    # quartile of every symbol, and the test would be measuring that animal
    # against the others rather than position within the cell.
    tags, a_idx = np.unique(ani, return_inverse=True)
    group = code.astype(np.int64) * int(tags.shape[0]) + a_idx
    quart = hm.quartiles_within(dst, group)
    log(f"[{arm} N={n}] report {code.shape[0]:,} runs, {tags.shape[0]} animals, "
        f"{int((quart >= 0).sum()):,} runs in a quartile")

    rows = hm.symbol_table(code, nxt, cen, dbin, quart, dur, n_states=n,
                           n_dur_bins=int(edges.shape[0] - 1), seed=SEED)
    log(f"[{arm} N={n}] {len(rows)} of {n} symbols testable at "
        f"{hm.MIN_RUNS_PER_SYMBOL} runs")
    agg = hm.aggregate(rows)
    ci = hm.jackknife(rows, code, nxt, cen, quart, dur, ani, n_states=n,
                      point=float(agg["excess"]), seed=SEED)
    obj = {"dataset": "luna", "arm": arm, "n_states": n, "split": "report",
           "pose_arm": POSE_ARM, "retired": True}
    rd = hm.homogeneity_read(agg, ci, scored_object=obj,
                             n_effective=int(tags.shape[0]))
    doc = {**provenance.header(anchors.LUNA, stage="tok_homogeneity",
                            unverified="report split only; the anchor counts all"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "arm": arm, "n_states": n,
           "retired_by_runlength": True,
           "min_runs_per_symbol": hm.MIN_RUNS_PER_SYMBOL,
           "n_perm": hm.N_PERM, "n_fix": hm.N_FIX,
           "read": rd.to_dict(), "aggregate": agg, "animal_interval": ci,
           "n_symbols_testable": len(rows), "n_symbols": n,
           "per_symbol": rows[:64],
           "peak_rss_gb": peak_rss_gb()}
    write_json(doc, tok_dir("resolution", f"homogeneity_{arm}_N{n}.json"))
    log(rd.line())
    return 0


def write_grid(args) -> int:
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("resolution")
    with open(path, "w", encoding="utf-8") as fh:
        for arm in qz.ARMS:
            for n in qz.STATE_GRID:
                fh.write(f"{arm} {n}\n")
    log(f"wrote {path}: {len(qz.ARMS) * len(qz.STATE_GRID)} cells")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--covariates", action="store_true")
    p.add_argument("--distortion", action="store_true")
    p.add_argument("--frailty", action="store_true")
    p.add_argument("--homogeneity", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--arm", choices=qz.ARMS, default=None)
    p.add_argument("--n-states", type=int, default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    if a.write_grid:
        return write_grid(a)
    if a.frailty:
        return frailty(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("resolution.json")
        return combine(a)
    if a.arm is None or a.n_states is None:
        if a.task is None:
            raise SystemExit("pass --arm/--n-states or --task")
        with open(config.PATHS.grid("resolution"), encoding="utf-8") as fh:
            grid = [ln.split() for ln in fh if ln.strip()]
        a.arm, a.n_states = grid[a.task][0], int(grid[a.task][1])
    if a.covariates:
        return covariates(a)
    if a.distortion:
        return distortion(a)
    if a.homogeneity:
        return homogeneity(a)
    raise SystemExit("pass --write-grid, --covariates, --distortion, "
                     "--homogeneity, --frailty or --combine")


def combine(args) -> int:
    d = tok_dir("resolution")
    cells = []
    for p in sorted(glob.glob(os.path.join(d, "distortion_*.json"))):
        with open(p, encoding="utf-8") as fh:
            cells.append(json.load(fh))
    frail, homo = [], []
    for p in sorted(glob.glob(os.path.join(d, "frailty_*.json"))):
        with open(p, encoding="utf-8") as fh:
            frail.append(json.load(fh))
    for p in sorted(glob.glob(os.path.join(d, "homogeneity_*.json"))):
        with open(p, encoding="utf-8") as fh:
            homo.append(json.load(fh))
    if not cells:
        raise SystemExit(f"no distortion shards in {d}")
    doc = {**provenance.header(anchors.LUNA, stage="tok_resolution",
                            unverified="report split only"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "distortion": cells, "frailty": frail,
           "homogeneity": homo,
           "peak_rss_gb": peak_rss_gb()}
    write_json(doc, args.out)
    for c in cells:
        log(f"{c['arm']}_N{c['n_states']}: D = "
            f"{c['summary']['d_total']:.4f}  {c['read']['verdict']}")
    for f in frail:
        for k, rd in f["reads"].items():
            log(f"{f['arm']}_N{f['n_states']} {k}: {rd['verdict']}")
    for h in homo:
        log(f"{h['arm']}_N{h['n_states']} homogeneity: "
            f"{h['read']['verdict']} excess "
            f"{h['aggregate']['excess']:+.4f} "
            f"[{h['animal_interval']['lo']:+.4f}, "
            f"{h['animal_interval']['hi']:+.4f}]")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
