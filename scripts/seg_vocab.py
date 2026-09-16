"""Step 3. Is there a vocabulary, or a continuum? Both are results.

    sbatch jobs/seg_vocab.slurm --group both
    python3 scripts/seg_vocab.py --combine

READ results/VOCAB_PREREGISTRATION.md FIRST.

Runs only if Step 2's gate PASSED. It rebuilds each arm's bank exactly as
`seg_recur.py` does -- same segments, same exclusions, same full-dimension
embedding -- and then asks a different question of it: not "is the nearest
neighbour close" but "do the close ones form groups".

## No assumed count

Clumps are connected components of a neighbour graph thresholded at **theta**,
the null's own distance quantile that the recurrence statistic already fixed.
Nothing here fits `k` anything, and the linking distance is inherited rather
than chosen. ExBias's clustering stack returned `n_states = 0` and
self-diagnoses why; it is not the model.

## Everything is computed identically on both nulls

A neighbour graph thresholded at a fixed quantile produces components in any
point cloud, a Gaussian one included. Clumpiness a null reproduces is not
clumpiness.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, splits                             # noqa: E402
from recur.recurrence import bank as bk_                            # noqa: E402
from recur.util import log, peak_rss_gb, write_json                 # noqa: E402
from vieb.clean import arms as clean_arms                           # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.qc import concentration as cc                             # noqa: E402
from vieb.seg import breaks as bk, embed, recur as rc, vocab as vb   # noqa: E402
from vieb.tok import config                                         # noqa: E402

sys.path.insert(2, os.path.join(os.environ.get("VIEB_RECUR",
                                               "/home/tul26194/recur"), "scripts"))

ARMS: tuple[str, ...] = ("corpus",) + rc.NULLS
SEED = 0


def out_dir() -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_vocab")


def _sr():
    """`seg_recur.py` as a module: the bank must be the SAME construction.

    Imported rather than reimplemented. A second copy of `build_arm` that
    drifted by one exclusion would make Step 3 describe a different population
    from the one Step 2 gated on.
    """
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_recur.py")
    spec = importlib.util.spec_from_file_location("seg_recur_mod", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["seg_recur"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def knn(B: np.ndarray, animal: np.ndarray, *, k: int, device: str,
        tile: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """`k` nearest CROSS-animal neighbours per row, exact, on the GPU.

    Cross-animal only, for the same reason the recurrence statistic is: a
    segment's own animal supplies near-copies of itself, and a clump built from
    them would be one animal's habit rather than a shared unit.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    X = torch.as_tensor(np.ascontiguousarray(B), dtype=torch.float32, device=dev)
    codes = torch.as_tensor(np.asarray(animal, dtype=np.int64), device=dev)
    sq = (X * X).sum(1)
    n = X.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    dst = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, tile):
        e = min(s + tile, n)
        Q = X[s:e]
        d2 = (Q * Q).sum(1)[:, None] + sq[None, :] - 2.0 * (Q @ X.T)
        d2 = d2.masked_fill(codes[s:e, None] == codes[None, :], float("inf"))
        v, i = torch.topk(d2, k, dim=1, largest=False)
        # The Gram form finds WHICH; it does not get to say HOW FAR. It loses
        # precision by cancellation exactly where the distance is small, which
        # is the only place this statistic looks.
        exact = (Q[:, None, :] - X[i]).pow(2).sum(-1)
        idx[s:e] = i.cpu().numpy()
        dst[s:e] = exact.clamp_min(0).sqrt().double().cpu().numpy()
    return idx, dst


def run_group(args) -> int:
    os.makedirs(out_dir(), exist_ok=True)
    sr = _sr()
    sr.no_tf32()
    fps = spine.fps()
    sd = sr.basis_sd()
    idx_cols = bk.CHANNEL_GROUPS[args.group]
    tags = sorted(sr.animals_of(spine.recording_ids()))
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))

    # theta is INHERITED from Step 2: the null's own quantile of normalised
    # cross-animal NN distance, at NULL_RATE. Not a new free parameter.
    with open(config.PATHS.result("seg_recur.json"), encoding="utf-8") as fh:
        gate = json.load(fh)
    cell = gate["groups"][f"{args.group}|k{args.k_mad:g}"]
    theta_n = float(cell["excess"][f"segment|{rc.NULLS[0]}"]["theta"])
    log(f"  theta (normalised, from {rc.NULLS[0]}) = {theta_n:.6f}")

    out: dict = {"group": args.group, "k_mad": args.k_mad,
                 "theta_normalised": theta_n, "knn": vb.KNN,
                 "min_clump": vb.MIN_CLUMP, "arms": {}}
    for arm in ARMS:
        log(f"  [{args.group}] {arm}")
        b = sr.build_arm(arm, tags, idx_cols, fps=fps, sd=sd, unit="segment",
                         k_mad=args.k_mad)
        b.pop("counts")
        cols = np.arange(b["X"].shape[1])
        full = embed.SEG_GRID * len(idx_cols)
        split = np.asarray([split_of.get(a, "?") for a in b["animal"]])
        tune = np.flatnonzero(split == "tune")
        pca = bk_.fit_pca(b["X"], b["start"][tune], embed.SEG_GRID, cols,
                          n_components=full, sample=200_000, seed=SEED,
                          device=args.device)
        B = bk_.build(b["X"], b["start"], embed.SEG_GRID, cols, pca=pca)
        # `rec` and `frame` are carried through and PERSISTED. The first run
        # dropped them here, which left the clumps unlocatable: a clump could be
        # counted and sized but not looked at, and "what IS this behaviour" had
        # no answer that did not require recomputing the whole arm.
        (B, an, rec, frm, nfr, spl), _o = bk_.sort_by_animal(
            B, b["animal"], b["rec"], b["frame"], b["n_frames"], split)
        # `rec` is an index into THIS animal's recording list, not a global id.
        # It is resolved against the animal's own ego shard at characterisation
        # time rather than stored here: segment bounds are valid only against
        # their own recording's array, and keying on a pooled position is the
        # trap ExBias records paying for.
        del b
        keep = np.flatnonzero(spl == "report")
        B, an, nfr, rec, frm = B[keep], an[keep], nfr[keep], rec[keep], frm[keep]
        uniq = {a: i for i, a in enumerate(sorted(set(an.tolist())))}
        code = np.asarray([uniq[a] for a in an], dtype=np.int64)
        # The arm's own ambient scale, so theta means the same thing here as it
        # did in Step 2. Each arm is divided by its own, never by the corpus's.
        rng = np.random.default_rng(SEED)
        i = rng.integers(0, B.shape[0], 2_000_000)
        j = rng.integers(0, B.shape[0], 2_000_000)
        ok = an[i] != an[j]
        scale = float(np.median(np.linalg.norm(B[i[ok]] - B[j[ok]], axis=1)))
        log(f"    {B.shape[0]:,} report segments, d={B.shape[1]}, "
            f"scale {scale:.4f}")
        nn_i, nn_d = knn(B, code, k=vb.KNN, device=args.device)
        lab, summary = vb.components(nn_i, nn_d / scale, theta=theta_n,
                                     min_size=vb.MIN_CLUMP)
        part = vb.participation(lab, an, n_animals_total=len(uniq))
        # The chaining check runs HERE because the bank is already in memory and
        # a diameter is O(n^2) over a space that would otherwise be rebuilt.
        diam = vb.clump_diameter(B, lab, scale=scale, theta=theta_n, seed=SEED)
        summary["scale"] = scale
        summary["n_animals"] = len(uniq)
        summary.update({k: v for k, v in
                        vb.bimodality(nn_d[:, 0] / scale).items()
                        if k != "n"})
        summary.update({f"gmm_{k}": v for k, v in
                        vb.gmm1d_bic_gain(nn_d[:, 0] / scale,
                                          seed=SEED).items()})
        summary["bic_gain"] = summary.pop("gmm_bic_gain")
        np.savez_compressed(
            os.path.join(out_dir(), f"{args.group}__k{args.k_mad:g}__{arm}.npz"),
            labels=lab, animal=an, n_frames=nfr, rec=rec, frame=frm,
            nn_idx=nn_i.astype(np.int64),
            nn_all=(nn_d / scale).astype(np.float32),
            nn_dist=(nn_d[:, 0] / scale).astype(np.float32))
        out["arms"][arm] = {"summary": summary, "participation": part[:200],
                            "n_participation_rows": len(part),
                            "diameters": diam}
        log(f"    {summary['n_clumps']} clumps, "
            f"{summary['unassigned_fraction']:.1%} unassigned, "
            f"BIC gain {summary['bic_gain']:+.0f}, "
            f"peak_rss={peak_rss_gb():.1f} GB")
    write_json({**anchors.header(anchors.LUNA, stage="seg_vocab",
                                 unverified="one channel group per job"),
                "inherited_digest": spine.digest(),
                "registration": "results/VOCAB_PREREGISTRATION.md",
                "theta_inherited_from": "results/seg_recur.json",
                "peak_rss_gb": peak_rss_gb(), **out},
               os.path.join(out_dir(), f"{args.group}__k{args.k_mad:g}.json"))
    return 0


def characterise(args) -> int:
    """What IS the island? Four checks, and any of them can refuse.

    Counting clumps says there is structure; it does not say what the structure
    is. Three properties of the headline clump make naming it prematurely a
    wrong-object risk: a 122x duration range that the time-normalised primary
    cannot see, members sitting at the linking threshold rather than well inside
    it, and single-linkage components, which chain.

    So this runs BEFORE anything is published, and the clip render is gated on
    it. The order matters: the chaining check comes first, because if the clumps
    are paths through the space then the speed contrast and the session
    composition are describing a path.
    """
    os.makedirs(out_dir(), exist_ok=True)
    sr = _sr()
    fps = spine.fps()
    sd = sr.basis_sd()
    idx_cols = bk.CHANNEL_GROUPS[args.group]
    tags = sorted(sr.animals_of(spine.recording_ids()))
    shard = os.path.join(out_dir(), f"{args.group}__k{args.k_mad:g}__corpus.npz")
    with np.load(shard, allow_pickle=False) as z:
        if "rec" not in z.files:
            raise SystemExit(
                f"{shard} predates the locator fix: it carries "
                f"{sorted(z.files)} and no rec/frame, so its clumps cannot be "
                f"placed in a recording. Re-run --group {args.group} first")
        lab = z["labels"]
        an = np.asarray([str(v) for v in z["animal"]])
        nfr, rec, frm = z["n_frames"], z["rec"], z["frame"]
        nn_idx, nn_all = z["nn_idx"], z["nn_all"]
    with open(os.path.join(out_dir(),
                           f"{args.group}__k{args.k_mad:g}.json"),
              encoding="utf-8") as fh:
        meta = json.load(fh)
    theta = float(meta["theta_normalised"])
    scale = float(meta["arms"]["corpus"]["summary"]["scale"])
    log(f"  {int((lab >= 0).sum()):,} assigned of {lab.size:,}, "
        f"theta {theta:.4f}, scale {scale:.4f}")

    # Rebuild the report-split bank and the per-segment speed in one pass over
    # the animals. The bank is needed for the diameter; the speed is the
    # freezing test; the recording ids resolve `rec`, which is an index into
    # each animal's OWN recording list and never a pooled position.
    rows_speed, rows_sess, order = [], [], []
    # The unwarped channel blocks, kept for the open-end secondary. Only the
    # selected segments, only the report split: about 220 MB on `shape`.
    raw_by_key: dict = {}
    rows_edge: list = []
    edge_cache: dict = {}
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    for tag in tags:
        if split_of.get(tag) != "report":
            continue
        a = sr.load_arm("corpus", tag, sd)
        segs = sr.segments_of(a, idx_cols, fps=fps, k_mad=args.k_mad)
        keep = embed.selectable(segs)
        sel = [r for r, k in zip(segs, keep) if k]
        for r in sel:
            blk = a["X"][int(r["start"]):int(r["stop"])]
            raw_by_key[(tag, int(r["rec"]), int(r["start"]))] = blk[:, list(idx_cols)]
            # Columns 14 and 15 are the translational twist increments; omega
            # (16) is excluded, as `quantize.speed` does -- a spin is not a
            # displacement and adding it would make a turning animal "fast".
            rows_speed.append(float(np.sqrt((blk[:, 14:16] ** 2).sum(1)).mean()))
            rid = a["recording_ids"][int(r["rec"])]
            rows_sess.append(_session_of(rid))
            if rid not in edge_cache:
                cl = spine.clean(rid)
                held = clean_arms.held_array(
                    cl["pose_unfiltered"].astype(np.float64),
                    cl["missing"].astype(bool))
                edge_cache[rid] = cc.edgeness(held[:, 3])
            lo_r = int(a["bounds"][int(r["rec"])])
            seg = edge_cache[rid][int(r["start"]) - lo_r:
                                  int(r["stop"]) - lo_r]
            rows_edge.append(float(np.nanmean(seg)) if seg.size else float("nan"))
            order.append((tag, int(r["rec"]), int(r["start"])))
        del a
        edge_cache.clear()
    key = {k: i for i, k in enumerate(order)}
    want = [key.get((str(x), int(y), int(z2)), -1)
            for x, y, z2 in zip(an, rec, frm)]
    if min(want) < 0:
        raise SystemExit("a shard segment has no match in the rebuilt table: "
                         "the detector is not reproducing its own output")
    speed = np.asarray([rows_speed[i] for i in want], dtype=np.float64)
    sess = [rows_sess[i] for i in want]
    blocks = [raw_by_key[(str(x), int(y), int(z2))]
              for x, y, z2 in zip(an, rec, frm)]
    edges = np.asarray([rows_edge[i] for i in want], dtype=np.float64)

    obj = {"dataset": "luna", "arm": "behaviour", "group": args.group,
           "k_mad": args.k_mad, "pose_arm": "raw", "split": "report",
           "theta": theta}
    n_an = len(set(an.tolist()))

    # 1. CHAINING, first, because it can invalidate the rest. Computed in the
    # group run, where the bank was in memory, and read back here.
    diam = meta["arms"]["corpus"].get("diameters") or []
    rd_chain = vb.chaining_read(diam, scored_object=obj, n_effective=n_an)
    log("  " + rd_chain.line())
    # And the same check scored on the clump a page would be about. The
    # aggregate can PASS on a majority while the headline clump fails, and a
    # claim about one clump is not supported by a statistic over nineteen.
    rd_one = vb.one_clump_chaining_read(diam, clump=0,
                                        scored_object={**obj, "clump": 0},
                                        n_effective=n_an)
    log("  " + rd_one.line())

    # 2. SPEED -- the freezing test.
    contrast = vb.speed_contrast(speed, lab, an, seed=SEED)
    rd_speed = vb.speed_read(contrast, clump=0, scored_object=obj,
                             n_effective=n_an)
    log("  " + rd_speed.line())

    # 3. SESSION composition.
    comp = vb.session_composition(lab, sess, clump=0)

    # 3b. ARENA POSITION, paired within animal. Added because the contact sheet
    # LOOKED like an animal pressed against the wall, and `CONCENTRATION.md`
    # records tracking failure rising 3.7x monotone from arena centre to wall --
    # so a wall-enriched clump would be finding the arena, not behaviour. The
    # eye cannot judge arena position from a crop centred on the animal, which
    # is why this is measured rather than read off the sheet.
    edge_in, edge_out, etags = [], [], []
    for tag in sorted(set(an.tolist())):
        mine = an == tag
        ins = [edges[i] for i in np.flatnonzero(mine) if lab[i] == 0]
        out_ = [edges[i] for i in np.flatnonzero(mine) if lab[i] != 0]
        if ins and out_:
            edge_in.append(float(np.mean(ins)))
            edge_out.append(float(np.mean(out_)))
            etags.append(tag)
    arena: dict = {"n_animals": len(etags)}
    if len(etags) > 1:
        arena.update({
            "edgeness_in_clump0": float(np.mean(edge_in)),
            "edgeness_other_segments": float(np.mean(edge_out)),
            "paired_difference": boot.animal_interval(
                np.asarray(edge_in) - np.asarray(edge_out), etags,
                how="mean", seed=SEED),
            "units": ("IQR of the animal's own centroid cloud, as "
                      "vieb/qc/concentration.py:edgeness defines it")})

    # 4. The registered SECONDARY, owed since SEGRECUR_PREREGISTRATION.md.
    # Only the clump-0 members and their nearest partners are needed, so only
    # the animals holding them are re-read.
    oe = _open_end_pairs(blocks, lab, nn_idx, nn_all, clump=0)

    doc = {**anchors.header(anchors.LUNA, stage="behaviour",
                            unverified="one channel group per run"),
           "inherited_digest": spine.digest(),
           "group": args.group, "k_mad": args.k_mad, "theta": theta,
           "n_assigned": int((lab >= 0).sum()), "n_segments": int(lab.size),
           "reads": {"chaining": rd_chain.to_dict(),
                     "chaining_clump0": rd_one.to_dict(),
                     "speed_clump0": rd_speed.to_dict()},
           "diameters": diam, "speed": contrast,
           "session_clump0": comp, "arena_clump0": arena, "open_end": oe,
           "bone_violations": (
               "vacuous by construction and reported as such: a selected "
               "segment has abstain_frac == 0, and the abstain mask CONTAINS "
               "the skull bone-violation mask, so clump 0 holds zero flagged "
               "frames and so does every other segment. It cannot be a "
               "bone-violation artifact, and that is a property of the "
               "selection rather than a measured contrast"),
           "duration_s": {"clump0": _describe(nfr[lab == 0] / fps),
                          "corpus": _describe(nfr / fps)}}
    write_json(doc, os.path.join(out_dir(),
                                 f"{args.group}__k{args.k_mad:g}__behaviour.json"))
    log(f"wrote {args.group} characterisation")
    return 0


def _open_end_pairs(blocks, labels, nn_idx, nn_all, *, clump: int = 0) -> dict:
    """The registered SECONDARY distance, on the clump's own nearest pairs.

    Owed since `SEGRECUR_PREREGISTRATION.md` registered it and never computed
    until now -- recorded as a deviation rather than quietly supplied. The
    primary time-normalises, so a 0.53 s and a 65 s segment can sit at distance
    zero; the open-end form compares the shared extent without warping and
    cannot. Their disagreement is the measurement.
    """
    lab = np.asarray(labels, dtype=np.int64)
    idx = np.flatnonzero(lab == int(clump))
    if idx.size == 0:
        return {"n": 0, "why": f"clump {clump} is empty"}
    oe, warped, dur_gap = [], [], []
    for i in idx.tolist():
        j = int(np.asarray(nn_idx)[i, 0])
        if j < 0 or j >= len(blocks):
            continue
        a, b = blocks[i], blocks[j]
        oe.append(embed.open_end_distance(a, b))
        warped.append(float(np.asarray(nn_all)[i, 0]))
        dur_gap.append(abs(np.log(max(len(a), 1)) - np.log(max(len(b), 1))))
    got = vb.open_end_agreement(oe, warped)
    got["mean_abs_log_duration_gap_of_pairs"] = (float(np.mean(dur_gap))
                                                 if dur_gap else float("nan"))
    got["clump"] = int(clump)
    return got


def _describe(x) -> dict:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return {}
    q = np.percentile(a, [1, 10, 25, 50, 75, 90, 99])
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()),
            **{f"p{p}": float(v) for p, v in zip((1, 10, 25, 50, 75, 90, 99), q)}}


def _session_of(rid: str) -> dict:
    """Box, day and context parsed from the recording id.

    The same parse `scripts/concentration.py` uses; the ids are structured and
    no artifact carries the decomposition separately.
    """
    box = re.search(r"Box_(\d+)", rid)
    day = re.search(r"Day_(\d+)", rid)
    ctx = re.search(r"Context_([A-Z])", rid)
    date = re.match(r"(\d{8})", os.path.basename(rid))
    return {"box": box.group(1) if box else "?",
            "day": day.group(1) if day else "?",
            "context": ctx.group(1) if ctx else "?",
            "date": date.group(1) if date else "?"}


def combine(args) -> int:
    metas = sorted(glob.glob(os.path.join(out_dir(), "*__k*.json")))
    if not metas:
        raise SystemExit(f"no group shards in {out_dir()}")
    groups: dict = {}
    reads: dict = {}
    for mp in metas:
        with open(mp, encoding="utf-8") as fh:
            meta = json.load(fh)
        key = f"{meta['group']}|k{meta['k_mad']:g}"
        obs = meta["arms"]["corpus"]["summary"]
        nulls = {a: meta["arms"][a]["summary"] for a in rc.NULLS
                 if a in meta["arms"]}
        obj = {"dataset": "luna", "arm": "seg_vocab", "group": meta["group"],
               "k_mad": meta["k_mad"], "pose_arm": "raw", "split": "report",
               "theta": meta["theta_normalised"]}
        n_an = int(obs.get("n_animals", 89))
        reads[f"{key}|clumps"] = vb.clump_read(
            obs, nulls, scored_object=obj, n_effective=n_an).to_dict()
        reads[f"{key}|coverage"] = vb.coverage_read(
            obs, scored_object=obj, n_effective=n_an).to_dict()
        reads[f"{key}|participation"] = vb.participation_read(
            meta["arms"]["corpus"]["participation"], n_animals_total=n_an,
            scored_object=obj, n_effective=n_an).to_dict()
        groups[key] = {"corpus": obs,
                       "nulls": nulls,
                       "participation_top": sorted(
                           meta["arms"]["corpus"]["participation"],
                           key=lambda r: -float(r["animal_fraction"]))[:20],
                       "n_clumps_total": meta["arms"]["corpus"]
                       ["n_participation_rows"]}
        for k, v in reads.items():
            if k.startswith(key):
                log(f"  {k}: {v['verdict']}")
    write_json({**anchors.header(anchors.LUNA, stage="seg_vocab_combine",
                                 unverified="scored on the report split"),
                "inherited_digest": spine.digest(),
                "registration": "results/VOCAB_PREREGISTRATION.md",
                "nulls": list(rc.NULLS), "reads": reads, "groups": groups},
               args.out)
    log(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default=None, choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--k-mad", type=float, default=rc.K_MAD_PRIMARY)
    p.add_argument("--device", default="cuda")
    p.add_argument("--characterise", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.characterise:
        if not a.group:
            raise SystemExit("--characterise needs --group")
        return characterise(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("seg_vocab.json")
        return combine(a)
    if a.group:
        if a.device != "cuda":
            raise SystemExit("--device cuda is asserted, never auto-detected")
        return run_group(a)
    raise SystemExit("pass --group or --combine")


if __name__ == "__main__":
    raise SystemExit(main())
