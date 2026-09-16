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
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, splits                                   # noqa: E402
from recur.recurrence import bank as bk_                            # noqa: E402
from recur.util import log, peak_rss_gb, write_json                 # noqa: E402
from vieb.io import spine                                           # noqa: E402
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
        (B, an, _r, _f, nfr, spl), _o = bk_.sort_by_animal(
            B, b["animal"], b["rec"], b["frame"], b["n_frames"], split)
        del b
        keep = np.flatnonzero(spl == "report")
        B, an, nfr = B[keep], an[keep], nfr[keep]
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
            labels=lab, animal=an, n_frames=nfr,
            nn_dist=(nn_d[:, 0] / scale).astype(np.float32))
        out["arms"][arm] = {"summary": summary, "participation": part[:200],
                            "n_participation_rows": len(part)}
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
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
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
