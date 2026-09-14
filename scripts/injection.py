"""Phase F. Does the arm get closer to the truth, and what does it break?

    python3 scripts/injection.py --write-grid
    sbatch --array=0-29%15 jobs/injection.slurm
    python3 scripts/injection.py --combine --split report

Configuration fixed in `results/INJECTION_PREREGISTRATION.md`, committed before
this ran on the corpus.

Corruption is injected inside clean segments of a recording and the arm is then
run over the **whole** recording, so it gets the temporal context it would have
in production. Scoring happens inside the segments only, because that is the only
place there is a truth to compare against.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits             # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json      # noqa: E402
from vieb import seeds
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.qc import bones, disposition as dp, inject, recover, truth  # noqa: E402
from vieb.tok import config, ego                                   # noqa: E402

EPS = 0.10
SEED = 0
#: The pre-registration listed `wiener` and it is **not** here. It cannot be:
#: `clean_arms.STORED_ARMS` reads the Wiener and Butterworth arrays off disk
#: rather than recomputing them, because this repo never reimplemented
#: shapeflow's filter -- and a benchmark needs to apply the arm to an array it
#: has just corrupted. Writing a fresh Wiener here and calling it "the incumbent"
#: would attribute a benchmark score to a filter that is not the one every
#: existing result was built on. Recorded as a limitation, not silently dropped.
#:
#: `median_0.50` stays despite the movement-retention evidence already arguing
#: against it: ruling an arm out by assertion is what this benchmark exists to
#: avoid doing.
ARMS = ("raw", "median_0.50", "viterbi", "disposition")
UNBENCHMARKABLE = {
    "wiener": ("read off disk from shapeflow, never reimplemented here, so it "
               "cannot be applied to a freshly corrupted array"),
    "butterworth": ("read off disk from shapeflow, never reimplemented here"),
}
PAIRS = bones.pair_indices(anchors.LUNA.n_kept_keypoints)
CONSTRAIN = [PAIRS.index(b) for b in bones.SKULL]


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def disposition_arm(pose: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Phase D's corrector, re-run on a corrupted array.

    Its committed shards hold the corrected *clean* pose and are no use here --
    the whole point is to hand it something broken. This reproduces the shard
    loop's disposition exactly, calling the same functions with the same
    constants, so the arm being benchmarked is the arm that was published.
    """
    t_n = pose.shape[0]
    keep = bones.rigid_pairs(bones.log_lengths(pose, PAIRS), PAIRS)
    bl = bones.bone_lengths(pose, PAIRS)
    logs = np.log(np.where(bl > 0, bl, np.nan))
    scale_pooled = np.nanmedian(logs[:, keep], axis=1)
    sf = bl / np.exp(scale_pooled)[:, None]
    l_hat_sf = np.array([bones.reference_length(sf[:, m], EPS)["l_hat"]
                         for m in range(len(PAIRS))])
    lengths = bones.metric_lengths(pose, bones.SKULL, "raw", pairs=PAIRS,
                                   keep=keep)
    l_skull = np.array([bones.reference_length(lengths[:, m], EPS)["l_hat"]
                        for m in range(len(bones.SKULL))])
    viol = bones.violations(lengths, l_skull, EPS)
    mask = bones.frame_mask(viol)
    correctable, _abstain = dp.envelope(mask)
    susp = dp.suspects(viol, bones.SKULL, conf)
    out = pose.copy()
    for t in np.flatnonzero(correctable):
        s = int(susp[t])
        if s < 0:
            continue
        touching = set(dp.pairs_touching(s, PAIRS))
        keep_s = [m for m in keep if m not in touching]
        if len(keep_s) < dp.MIN_SCALE_PAIRS:
            continue
        donor = dp.donor_frame(mask, t)
        tgt = None if donor < 0 else dp.predict(out, t, donor, s)
        if tgt is None:
            continue
        scale_t = float(np.exp(np.nanmedian(logs[t, keep_s])))
        if not np.isfinite(scale_t):
            continue
        frame, info = dp.project(out, t, s, pairs=PAIRS, constrain=CONSTRAIN,
                                 l_hat_scaled=l_hat_sf * scale_t, eps=EPS,
                                 target=tgt)
        if info["converged"]:
            out[t] = frame
    assert out.shape[0] == t_n
    return out


def _seed_for(tag: str, rid: str) -> int:
    """A per-recording seed that is the same in every process.

    `hash()` on a str is salted per interpreter unless PYTHONHASHSEED is set, so
    `abs(hash((SEED, tag, rid)))` -- what this used through Phase F -- drew a
    different corruption layout on every run despite `SEED = 0`. The benchmark
    was not reproducible and its pre-registration said it was. blake2b is stable
    across processes, machines and versions.
    """
    return seeds.stable_seed(SEED, tag, rid)


def _dir_for(spike_bl: float) -> str:
    """Sweep outputs must not collide. The banked threshold keeps the original
    directory so `results/injection.json` stays reproducible from it."""
    base = os.path.join(os.path.dirname(config.PATHS.bones_dir), "injection")
    if abs(float(spike_bl) - truth.CLEAN_SPIKE_BL) < 1e-12:
        return base
    tag = "off" if float(spike_bl) >= 1e3 else f"{float(spike_bl):g}".replace(".", "")
    return f"{base}_{tag}"


def shard(args, tag: str) -> int:
    out_dir = _dir_for(args.spike_bl)
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    mine = animals_of(spine.recording_ids())[tag]

    held_by, conf_by = {}, {}
    for rid in mine:
        d = spine.clean(rid)
        held_by[rid] = clean_arms.held_array(
            d["pose_unfiltered"].astype(np.float64), d["missing"].astype(bool))
        conf_by[rid] = d["conf"].astype(np.float64)
    ell = ego.ell_a([held_by[r] for r in mine])

    # Pool and segments first, so every arm is scored on the same frames.
    pools, segs_by, speeds = {}, {}, []
    why_any = None
    for rid in mine:
        d = spine.clean(rid)
        mask, why = truth.clean_mask(held_by[rid], conf_by[rid],
                                     d["missing"].astype(bool), ell,
                                     pairs=PAIRS, eps=EPS,
                                     spike_bl=float(args.spike_bl))
        why_any = why_any or why
        segs = truth.segments(mask)
        segs_by[rid] = segs
        pool = np.zeros(held_by[rid].shape[0], dtype=bool)
        for a, b in segs:
            pool[int(a):int(b)] = True
        pools[rid] = pool
        for seg in segs:
            speeds.append(truth.segment_speed(held_by[rid], seg, ell, fps))
    edges = truth.speed_bins(speeds)

    # Two pool diagnostics, both pre-registered.
    #
    # Contamination: the TRUNK violation rate INSIDE the pool. The pool gates on
    # the SKULL bones, so a skull rate would be zero by construction; the trunk
    # is the same instrument on geometry the pool never selected against.
    #
    # Bias: the share of pool frames faster than that animal's own all-frames p90
    # centre speed. Unbiased would be 10%. Self-referential on purpose -- a
    # hardcoded corpus p90 is the class of number this repo just audited.
    trunk_bad = trunk_tot = 0
    fast_in_pool = pool_tot = 0
    for rid in mine:
        held, pool = held_by[rid], pools[rid]
        keep = bones.rigid_pairs(bones.log_lengths(held, PAIRS), PAIRS)
        tl = bones.metric_lengths(held, bones.TRUNK, "raw", pairs=PAIRS, keep=keep)
        t_hat = np.array([bones.reference_length(tl[:, m], EPS)["l_hat"]
                          for m in range(len(bones.TRUNK))])
        tmask = bones.frame_mask(bones.violations(tl, t_hat, EPS))
        trunk_bad += int(tmask[pool].sum())
        trunk_tot += int(pool.sum())
        v = np.linalg.norm(np.diff(held[:, 3], axis=0), axis=-1) * fps / ell
        ok = np.isfinite(v)
        if ok.any():
            p90 = float(np.percentile(v[ok], 90))
            inpool = pool[:-1] & ok
            fast_in_pool += int((v[inpool] > p90).sum())
            pool_tot += int(inpool.sum())
    pool_contam = float(trunk_bad / trunk_tot) if trunk_tot else float("nan")
    fast_share = float(fast_in_pool / pool_tot) if pool_tot else float("nan")
    log(f"[{tag}] pool trunk-violation {pool_contam:.4%}  "
        f"fast share {fast_share:.4%} (10% would be unbiased)")

    # Recording-outer, arm-inner, because every arm must be scored on the SAME
    # keypoint-frames. An arm that interpolates an injected dropout produces a
    # finite value there and gets charged for it; one that leaves NaN is dropped
    # from the mean and pays nothing. Comparing means over different denominators
    # would reward declining to answer, so the scoring mask is the intersection
    # of the frames every arm produced a finite estimate for.
    agg = {a: {"repair_sum": 0.0, "damage_sum": 0.0, "total_sum": 0.0,
               "n_corrupted": 0, "n_clean": 0, "n_unscoreable": 0} for a in ARMS}
    kinds: dict = {a: {} for a in ARMS}
    strata: dict = {a: {} for a in ARMS}
    for rid in mine:
        segs = segs_by[rid]
        if segs.size == 0:
            continue
        rng = np.random.default_rng(_seed_for(tag, rid))
        corrupted, det = inject.corrupt(held_by[rid], segs, rng, ell=ell)
        if not det["events"]:
            continue
        conf = conf_by[rid]
        cleaned = {}
        for arm in ARMS:
            if arm == "raw":
                cleaned[arm] = corrupted
            elif arm == "disposition":
                cleaned[arm] = disposition_arm(corrupted, conf)
            else:
                cleaned[arm] = clean_arms.apply(arm, corrupted, conf, fps)
        errs = {a: recover.errors(held_by[rid], cleaned[a], ell) for a in ARMS}
        common = np.ones_like(errs["raw"], dtype=bool)
        for a in ARMS:
            common &= np.isfinite(errs[a])
        for arm in ARMS:
            s = recover.score(held_by[rid], cleaned[arm], det["mask"],
                              pools[rid], ell, scoreable=common)
            for k in agg[arm]:
                agg[arm][k] += s[k]
            for kind, cell in recover.by_kind(held_by[rid], cleaned[arm],
                                              det["events"], ell,
                                              scoreable=common).items():
                acc = kinds[arm].setdefault(kind, {"sum": 0.0, "injected": 0.0,
                                                   "n": 0})
                acc["sum"] += cell["sum"]
                acc["injected"] += cell["injected_sum"]
                acc["n"] += cell["n"]
            block_err = np.where(common, errs[arm], np.nan)
            for seg in segs:
                a_i, b_i = int(seg[0]), int(seg[1])
                block = block_err[a_i:b_i]
                fin = np.isfinite(block)
                if not fin.any():
                    continue
                idx = truth.assign_bin(
                    truth.segment_speed(held_by[rid], seg, ell, fps), edges)
                cell = strata[arm].setdefault(idx, {"sum": 0.0, "n": 0})
                cell["sum"] += float(block[fin].sum())
                cell["n"] += int(fin.sum())

    rows: list = []
    for arm in ARMS:
        a_agg = agg[arm]
        n_scored = a_agg["n_corrupted"] + a_agg["n_clean"]
        rows.append({
            "arm": arm, "animal": tag, "ell_px": float(ell),
            "repair": (a_agg["repair_sum"] / a_agg["n_corrupted"]
                       if a_agg["n_corrupted"] else float("nan")),
            "damage": (a_agg["damage_sum"] / a_agg["n_clean"]
                       if a_agg["n_clean"] else float("nan")),
            "total_mean": (a_agg["total_sum"] / n_scored
                           if n_scored else float("nan")),
            "total_sum": a_agg["total_sum"], "n_scored": int(n_scored),
            "n_corrupted": int(a_agg["n_corrupted"]),
            "n_clean": int(a_agg["n_clean"]),
            "n_unscoreable": int(a_agg["n_unscoreable"]),
            "by_kind": {k: {"error_bl": v["sum"] / max(1, v["n"]),
                            "injected_bl": v["injected"] / max(1, v["n"]),
                            "n": v["n"]} for k, v in kinds[arm].items()},
            "by_speed_bin": {str(k): v["sum"] / max(1, v["n"])
                             for k, v in sorted(strata[arm].items())},
            "n_by_speed_bin": {str(k): int(v["n"])
                               for k, v in sorted(strata[arm].items())},
        })
        log(f"[{tag}] {arm:12s} repair {rows[-1]['repair']:.4f}  "
            f"damage {rows[-1]['damage']:.4f}  "
            f"total {rows[-1]['total_mean']:.4f}  n={n_scored}")

    write_json({"animal": tag, "eps": EPS, "seed": SEED, "arms": list(ARMS),
                "spike_bl": float(args.spike_bl),
                "pool_bone_violation_rate": pool_contam,
                "frac_pool_above_corpus_p90": float(fast_share),
                "speed_edges": [float(x) for x in edges],
                "pool": why_any or {}, "rates": inject.RATES,
                "inherited_digest": spine.digest(), "rows": rows},
               os.path.join(out_dir, f"{tag}.json"))
    log(f"[{tag}] peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _num(x) -> float:
    """`util.jsonable` writes a non-finite float as JSON null, so a NaN comes
    back as None and `np.isfinite` raises on it. Coerce on read rather than
    letting a missing value read as zero somewhere downstream."""
    return float("nan") if x is None else float(x)


def combine(args) -> int:
    out_dir = _dir_for(args.spike_bl)
    shards = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    pools: list = []
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        rows += doc["rows"]
        if doc.get("pool"):
            pools.append(doc["pool"])
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    sel = [r for r in rows if of.get(r["animal"]) == args.split]
    log(f"{len(shards)} shards, {len(sel)} arm-animals on {args.split}")

    base = {r["animal"]: _num(r["total_mean"])
            for r in sel if r["arm"] == "raw"}
    out_rows, intervals, strata = [], {}, {}
    stratum_n: dict = {}
    stratum_refused: dict = {}
    unplaceable: dict = {}
    for arm in ARMS:
        mine = [r for r in sel if r["arm"] == arm
                and np.isfinite(_num(r["total_mean"]))]
        if not mine:
            continue
        an = [r["animal"] for r in mine]
        w = [r["n_scored"] for r in mine]
        nets = [_num(r["total_mean"]) - base.get(r["animal"], float("nan"))
                for r in mine]
        ci = lambda v: boot.animal_interval(v, an, weights=w, how="wmean")
        # Bin -1 is `truth.assign_bin`'s REFUSAL code for a segment whose speed
        # could not be placed -- not a stratum. It was being sorted first and
        # printed as "the slowest stratum", on 235 keypoint-frames, which is how
        # a 0.0 for viterbi and a -0.0331 for the median got reported as
        # findings. It is excluded here and reported as `unplaceable`.
        raw_by_animal = {q["animal"]: q for q in sel if q["arm"] == "raw"}
        bins = sorted({int(k) for r in mine for k in r["by_speed_bin"]
                       if int(k) >= 0})
        per_bin, per_bin_n, refused = [], [], []
        for b in bins:
            num = den = 0.0
            for r in mine:
                n = float(r.get("n_by_speed_bin", {}).get(str(b), 0))
                if n <= 0:
                    continue
                a_v = _num(r["by_speed_bin"].get(str(b)))
                q = raw_by_animal.get(r["animal"])
                q_v = _num(q["by_speed_bin"].get(str(b))) if q else float("nan")
                if not (np.isfinite(a_v) and np.isfinite(q_v)):
                    continue
                num += n * (a_v - q_v)
                den += n
            per_bin_n.append(int(den))
            if den < recover.MIN_STRATUM_FRAMES:
                # Refused, not reported thin. Registered minimum.
                per_bin.append(float("nan"))
                refused.append(int(b))
            else:
                per_bin.append(float(num / den))
        strata[arm] = per_bin
        stratum_n[arm] = per_bin_n
        stratum_refused[arm] = refused
        unplaceable[arm] = int(sum(
            r.get("n_by_speed_bin", {}).get("-1", 0) for r in mine))
        kinds: dict = {}
        for r in mine:
            for k, cell in r["by_kind"].items():
                acc = kinds.setdefault(k, {"e": 0.0, "i": 0.0, "n": 0})
                acc["e"] += cell["error_bl"] * cell["n"]
                acc["i"] += cell["injected_bl"] * cell["n"]
                acc["n"] += cell["n"]
        out_rows.append({
            "arm": arm,
            "repair": float(np.average([_num(r["repair"]) for r in mine],
                                       weights=w)),
            "damage": float(np.average([_num(r["damage"]) for r in mine],
                                       weights=w)),
            "net": float(np.average(nets, weights=w)),
            "n_animals": len({r["animal"] for r in mine}),
            "n_unscoreable": int(sum(r["n_unscoreable"] for r in mine)),
            "by_kind": {
                k: {"error_bl": v["e"] / max(1, v["n"]),
                    "injected_bl": v["i"] / max(1, v["n"]),
                    "fraction_removed": (1.0 - (v["e"] / max(1e-12, v["i"])))
                    if v["i"] > 0 else float("nan"),
                    "n": v["n"]} for k, v in kinds.items()},
            "net_by_speed_bin": per_bin,
            "n_by_speed_bin": per_bin_n,
            "strata_refused_below_minimum": refused,
            "n_unplaceable_frames": unplaceable[arm],
        })
        intervals[arm] = {"repair": ci([_num(r["repair"]) for r in mine]),
                          "damage": ci([_num(r["damage"]) for r in mine]),
                          "net": ci(nets)}

    rd = recover.recovery_read(out_rows, intervals, strata, scored_object={
        "dataset": "luna", "arm": "injection", "split": args.split, "eps": EPS},
        n_effective=len({r["animal"] for r in sel}))
    doc = {
        **anchors.header(anchors.LUNA, stage="injection",
                         unverified="a pseudo-ground truth pool, not human labels"),
        "inherited_digest": spine.digest(),
        "preregistration": "results/INJECTION_PREREGISTRATION.md",
        "reads": {"recovery": rd.to_dict()},
        "split": args.split, "eps": EPS, "seed": SEED,
        "rates": inject.RATES, "spike_bl": float(args.spike_bl),
        "arms": out_rows, "intervals": intervals,
        "arms_not_benchmarked": UNBENCHMARKABLE,
        "pool": {k: describe([p[k] for p in pools if k in p])
                 for k in ("frac_all", "frac_bone_ok", "frac_continuity_ok",
                           "frac_present", "frac_confident")},
        "min_stratum_frames": recover.MIN_STRATUM_FRAMES,
        "stratum_note": (
            "bin -1 is truth.assign_bin's refusal code for an unplaceable "
            "segment, not a stratum; it is excluded and counted in "
            "n_unplaceable_frames. A stratum below min_stratum_frames is "
            "refused (NaN), never reported thin"),
        "selection_bias": (
            "the pool is selected for cleanliness and is therefore biased "
            "towards slow behaviour, so damage UNDERSTATES what a smoother does "
            "to fast movement. That runs against this phase's expected "
            "conclusion, and it is why every net is stratified by segment speed"),
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(rd.line())
    for r in out_rows:
        log(f"  {r['arm']:12s} repair {r['repair']:.4f}  damage {r['damage']:.4f}"
            f"  net {r['net']:+.4f}  [{intervals[r['arm']]['net']['lo']:+.4f},"
            f"{intervals[r['arm']]['net']['hi']:+.4f}]")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("injection")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--split", default="report")
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--spike-bl", type=float, default=truth.CLEAN_SPIKE_BL,
                   help="pool's continuity criterion; the swept parameter. "
                        "Pass a large value to disable it.")
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("injection.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("injection"), encoding="utf-8") as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = _dir_for(a.spike_bl)
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.json")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
