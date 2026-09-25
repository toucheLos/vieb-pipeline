"""Step A. Does the distance make the clumps? Two metrics, one registered primary.

    sbatch jobs/seg_dist.slurm --group shape --metric open
    sbatch jobs/seg_dist.slurm --group shape --metric union
    python3 scripts/seg_dist.py --combine

READ results/DISTANCE_PREREGISTRATION.md FIRST. The gate: if the excess collapses
under either metric, segment recurrence depended on the warp and the route stops
there. If the excess survives but the clumps collapse, the 19 clumps were
tempo-invariance and the vocabulary claim weakens materially.

## What this shares with Step 2 and Step 3, and why

The segments, the exclusions, the nulls, the windowed control and theta are the
SAME objects -- `seg_recur.build_arm` and `seg_vocab`'s graph are imported, not
reimplemented. Only the distance changes. A second copy that drifted by one
exclusion would compare two populations and call it a metric effect.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, splits                                   # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.recurrence import bank as bk_, search as se              # noqa: E402
from recur.util import log, peak_rss_gb, write_json                 # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seg import breaks as bk, dist as ds, embed                # noqa: E402
from vieb.seg import recur as rc                                    # noqa: E402
from vieb.tok import config                                         # noqa: E402

ARMS: tuple[str, ...] = ("corpus",) + rc.NULLS
SEED = 0
EXACT_BANK = 4000
EXACT_QUERIES = 400


def out_dir() -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_dist")


def _mod(name: str):
    """A sibling script as a module, so its objects are the same objects."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def raw_blocks(sr, arm: str, tags, idx, *, fps: float, sd, k_mad: float,
               unit: str = "segment"):
    """Every selected segment as its UNWARPED channel block, plus its keys.

    The warp is what is under test, so nothing here resamples. Memory: the
    report-split shape arm is about 220 MB of float64.
    """
    blocks, animal, rec, frame, nfr = [], [], [], [], []
    for tag in tags:
        a = sr.load_arm(arm, tag, sd)
        rows, _c = sr.units_for(
            a, idx, fps=fps, k_mad=k_mad, unit=unit,
            rng=np.random.default_rng(
                seeds.stable_seed(SEED, f"{arm}{unit}", tag)))
        for r in rows:
            blocks.append(np.asarray(
                a["X"][int(r["start"]):int(r["stop"])][:, list(idx)],
                dtype=np.float64))
            animal.append(tag)
            rec.append(int(r["rec"]))
            frame.append(int(r["start"]))
            nfr.append(int(r["n_frames"]))
        del a
    return (blocks, np.asarray(animal), np.asarray(rec, dtype=np.int32),
            np.asarray(frame, dtype=np.int64), np.asarray(nfr, dtype=np.int64))


def prefix_bank(blocks, *, device: str):
    """The retrieval space: every segment truncated to PREFIX_L frames.

    Rectangular, so `bank.build` and `search.search` run unmodified. Truncation
    drops nothing -- PREFIX_L is the identifiability floor and no selected
    segment is shorter.
    """
    L = ds.PREFIX_L
    C = blocks[0].shape[1]
    out = np.empty((len(blocks) * L, C), dtype=np.float32)
    for i, b in enumerate(blocks):
        if b.shape[0] < L:
            raise SystemExit(
                f"segment {i} has {b.shape[0]} frames, below the "
                f"identifiability floor {L}: the retrieval space would have to "
                f"pad, and a padded prefix is not the segment")
        out[i * L:(i + 1) * L] = b[:L]
    return out, np.arange(len(blocks), dtype=np.int64) * L


def knn_prefix(Xs, start, animal, *, device: str, k: int):
    """Exact k nearest CROSS-animal neighbours in the prefix space."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    cols = np.arange(Xs.shape[1])
    B = bk_.build(Xs, start, ds.PREFIX_L, cols, pca=None)
    uniq = {a: i for i, a in enumerate(sorted(set(animal.tolist())))}
    code = torch.as_tensor(np.asarray([uniq[a] for a in animal], dtype=np.int64),
                           device=device)
    X = torch.as_tensor(np.ascontiguousarray(B), dtype=torch.float32,
                        device=device)
    sq = (X * X).sum(1)
    n = X.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        Q = X[s:e]
        d2 = (Q * Q).sum(1)[:, None] + sq[None, :] - 2.0 * (Q @ X.T)
        d2 = d2.masked_fill(code[s:e, None] == code[None, :], float("inf"))
        idx[s:e] = torch.topk(d2, k, dim=1, largest=False)[1].cpu().numpy()
    return idx


def run(args) -> int:
    os.makedirs(out_dir(), exist_ok=True)
    sr = _mod("seg_recur")
    sr.no_tf32()
    fps = spine.fps()
    sd = sr.basis_sd()
    idx = bk.CHANNEL_GROUPS[args.group]
    tags = sorted(sr.animals_of(spine.recording_ids()))
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    out: dict = {"group": args.group, "metric": args.metric,
                 "k_mad": args.k_mad, "prefix_L": ds.PREFIX_L,
                 "candidates": ds.CANDIDATES, "arms": {}}

    for arm in ARMS:
        for unit in ("segment", "window"):
            name = f"{arm}|{unit}"
            log(f"  [{args.group}/{args.metric}] {name}")
            blocks, an, rec, frm, nfr = raw_blocks(
                sr, arm, tags, idx, fps=fps, sd=sd, k_mad=args.k_mad, unit=unit)
            Xs, start = prefix_bank(blocks, device=args.device)
            cand = knn_prefix(Xs, start, an, device=args.device,
                              k=ds.CANDIDATES)
            del Xs
            split = np.asarray([split_of.get(a, "?") for a in an])
            q = np.flatnonzero(split == "report")
            nn_i = np.full((q.size, ds.KEEP), -1, dtype=np.int64)
            nn_d = np.full((q.size, ds.KEEP), np.inf, dtype=np.float64)
            for r, i in enumerate(q.tolist()):
                nn_i[r], nn_d[r] = ds.rerank(blocks, i, cand[i],
                                             metric=args.metric)
            # The arm's own ambient scale: RANDOM cross-animal pairs, no
            # nearest-neighbour feedback. Its absence is what made D8's
            # withdrawn row meaningless.
            rng = np.random.default_rng(SEED)
            fn = ds.metric_fn(args.metric)
            pairs = []
            while len(pairs) < 20000:
                i, j = rng.integers(0, len(blocks), 2)
                if an[i] == an[j]:
                    continue
                v = fn(blocks[i], blocks[j])
                if np.isfinite(v):
                    pairs.append(v)
            scale = float(np.median(pairs))
            rec_read = None
            if arm == "corpus" and unit == "segment":
                rec_read = exactness(blocks, an, q, args.metric, args.group)
            np.savez_compressed(
                os.path.join(out_dir(),
                             f"{args.group}__{args.metric}__{arm}__{unit}.npz"),
                # RAW distances, not divided by the ambient scale. The scale
                # sits in the group JSON and `paired_excess` applies it exactly
                # once. Dividing here as well made theta a quantile of
                # d/scale^2 while the clump graph compared d/scale -- two units
                # in one comparison, which is the failure DEVIATIONS.md D8
                # records and which produced a graph with zero edges.
                d_cross=nn_d[:, 0].astype(np.float32),
                nn_idx=nn_i, nn_all=nn_d.astype(np.float32),
                d_within=np.full(q.size, np.nan, dtype=np.float32),
                # `i_cross` indexes the FULL bank (all 298 animals) while the
                # queries are the report split, so a clump graph built from it
                # would index out of its own array. `q_idx` is what lets the
                # combine step map a neighbour back to a query row -- and Step
                # 3 built its graph on the report-only bank, so the comparison
                # has to be restricted the same way to stay like-for-like.
                i_cross=nn_i[:, 0], q_idx=q, q_animal=an[q], q_len=nfr[q],
                nn_len=np.where(nn_i[:, 0] >= 0,
                                nfr[np.maximum(nn_i[:, 0], 0)], -1))
            out["arms"][name] = {
                "scale": scale, "n_bank": len(blocks), "n_queries": int(q.size),
                "n_unmatched": int((nn_i[:, 0] < 0).sum()),
                **({"recovery": rec_read} if rec_read else {})}
            log(f"    {len(blocks):,} units, {q.size:,} queries, "
                f"scale {scale:.4f}, peak_rss={peak_rss_gb():.1f} GB")
            del blocks
    write_json({**provenance.header(anchors.LUNA, stage="seg_dist",
                                 unverified="one group and metric per job"),
                "inherited_digest": spine.digest(),
                "registration": "results/DISTANCE_PREREGISTRATION.md",
                "pose_arm": "raw", "split": "report",
                "peak_rss_gb": peak_rss_gb(), **out},
               os.path.join(out_dir(), f"{args.group}__{args.metric}.json"))
    return 0


def exactness(blocks, an, q, metric: str, group: str) -> dict:
    """The retrieval measured against a full exact pass on a reduced bank."""
    rng = np.random.default_rng(SEED)
    n = min(EXACT_BANK, len(blocks))
    take = np.sort(rng.choice(len(blocks), size=n, replace=False))
    sub = [blocks[i] for i in take.tolist()]
    sub_an = an[take]
    qs = np.flatnonzero(np.isin(take, q))[:EXACT_QUERIES]
    if qs.size < 10:
        return {"verdict": "INCONCLUSIVE", "why": "too few report queries"}
    ex_i, ex_d = ds.pairwise(sub, qs, sub_an, metric=metric)
    L = ds.PREFIX_L
    C = sub[0].shape[1]
    Xs = np.empty((n * L, C), dtype=np.float32)
    for i, b in enumerate(sub):
        Xs[i * L:(i + 1) * L] = b[:L]
    cand = knn_prefix(Xs, np.arange(n, dtype=np.int64) * L, sub_an,
                      device="cuda", k=ds.CANDIDATES)
    ap_i = np.full(qs.size, -1, dtype=np.int64)
    ap_d = np.full(qs.size, np.nan, dtype=np.float64)
    for r, i in enumerate(qs.tolist()):
        ii, dd = ds.rerank(sub, i, cand[i], metric=metric)
        ap_i[r], ap_d[r] = int(ii[0]), float(dd[0])
    rd = ds.recovery_read(ap_i, ex_i, ap_d, ex_d, metric=metric,
                          scored_object={"dataset": "luna", "arm": "seg_dist",
                                         "group": group, "metric": metric,
                                         "n_bank": int(n)},
                          n_effective=int(qs.size))
    log("    " + rd.line())
    return rd.to_dict()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default=None, choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--metric", default=None, choices=ds.METRICS)
    p.add_argument("--k-mad", type=float, default=rc.K_MAD_PRIMARY)
    p.add_argument("--device", default="cuda")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.combine:
        from vieb.seg import combine_dist
        a.out = a.out or config.PATHS.result("seg_dist.json")
        return combine_dist.main(a, out_dir())
    if a.group and a.metric:
        if a.device != "cuda":
            raise SystemExit("--device cuda is asserted, never auto-detected")
        return run(a)
    raise SystemExit("pass --group and --metric, or --combine")


if __name__ == "__main__":
    raise SystemExit(main())
