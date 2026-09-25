"""The falsifier: does the corpus beat a structureless signal of the same shape?

    python3 scripts/surrogate.py --write-grid
    sbatch jobs/surrogate.slurm --kind phase
    sbatch jobs/surrogate.slurm --kind var5
    python3 scripts/surrogate.py --combine

Registered in `results/FALSIFIER_PREREGISTRATION.md`, committed before any
surrogate was generated.

## The pipeline is not re-implemented, it is re-pointed

`--generate` writes surrogate egocentric shards into a parallel tree and copies
the corpus's abstain masks beside them. Everything after that is
`scripts/quantize.py` and `scripts/ladder.py` **unmodified**, run with
`VIEB_EGO_DIR` and `VIEB_TOK_DIR` redirected. "The identical pipeline" is then
true by construction rather than by inspection, and there is no second
implementation of any stage that could drift from the one it is a control for.

## What is carried over rather than generated

The **abstain mask**. A phase-randomised signal has no tracking failures, and a
surrogate abstaining on 0% of frames against the corpus's 7.03% would differ
from it in two ways at once.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits            # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.null import microstate as ms                          # noqa: E402
from recur.util import log, peak_rss_gb, write_json               # noqa: E402
from vieb import seeds                                            # noqa: E402
from vieb.io import spine                                         # noqa: E402
from vieb.tok import config, surrogate as sr                      # noqa: E402

POSE_ARM, SCALE_ARM = "raw", "bodylen"
#: The two cells where rung 1 passes on the corpus, per the registration.
CELLS: tuple[int, ...] = (256, 512)
ARM = "plain"
SEED = 0

ROOT = os.environ.get("VIEB_SURROGATE_ROOT",
                      os.path.join(config.REPO, "work", "surrogate"))


def tree(kind: str, *parts: str) -> str:
    return os.path.join(ROOT, kind, *parts)


def fit_path() -> str:
    return os.path.join(ROOT, "microstate_fit.npz")


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def fit_microstate(args) -> int:
    """Fit the microstate partition and the pooled chain on the CORPUS's tune.

    Two partitions exist in this arm and they must not be confused. **This** one
    is the null's internal granularity, fitted on the corpus because the
    surrogate does not exist yet. The tokenizer's partition is fitted later, on
    the surrogate's own tune split, by `scripts/quantize.py`.

    The chain is pooled across tune recordings rather than fitted per recording,
    for the arithmetic reason `pooled_chain`'s docstring gives: at 500 states a
    single recording leaves the Laplace prior holding almost every row.
    """
    os.makedirs(ROOT, exist_ok=True)
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    with open(os.path.join(config.REPO, "work", "tok", "basis.json"),
              encoding="utf-8") as fh:
        basis = json.load(fh)
    # The SD the tokenizer itself uses: segment-balanced, corpus-pooled, fitted
    # on tune, with the SE(2) dead columns floored to 1.0. Reused rather than
    # refitted so the null lives in the same space the analysis does.
    sd = np.asarray(basis["sd_used"], dtype=np.float64)
    cols = np.arange(sd.shape[0], dtype=np.int64)

    tags = [t for t in sorted(animals_of(spine.recording_ids()))
            if of.get(t) == "tune"]
    xs, blocks = [], []
    for tag in tags:
        with np.load(os.path.join(config.REPO, "work", "ego",
                                  f"{POSE_ARM}__{SCALE_ARM}__{tag}.npz"),
                     allow_pickle=False) as z:
            xs.append(z["X"])
            b = z["bounds"].astype(np.int64)
        base = sum(x.shape[0] for x in xs[:-1])
        blocks += [(base + int(b[r]), base + int(b[r + 1]))
                   for r in range(b.shape[0] - 1)]
    X = np.concatenate(xs)
    del xs
    log(f"fitting on {X.shape[0]:,} tune frames over {len(blocks)} recordings")

    part = ms.fit_partition(X, sd, cols, np.ones(X.shape[0], dtype=bool),
                            n_states=sr.N_MICROSTATES, seed=SEED, logger=log)
    states = np.asarray(ms.assign_states(X, part["centroids"], sd, cols),
                        dtype=np.int64)
    # Visit sequences per recording -- never across a seam.
    seqs = [np.asarray(ms.visits(states[lo:hi]), dtype=np.int64)[:, 0]
            for lo, hi in blocks if hi - lo > 1]
    chain = ms.pooled_chain(seqs, sr.N_MICROSTATES, order=1)
    lens = np.concatenate([np.diff(np.asarray(ms.visits(states[lo:hi]),
                                              dtype=np.int64)[:, 1:], axis=1
                                   ).ravel()
                           for lo, hi in blocks if hi - lo > 1])
    np.savez_compressed(
        fit_path(), centroids=part["centroids"], sd=sd, cols=cols,
        P=chain["P"], order=np.array(chain["order"]),
        n_states=np.array(chain["n_states"]), alpha=np.array(chain["alpha"]),
        n_transitions=np.array(chain["n_transitions"]),
        prior_share=np.array(chain["prior_share"]),
        input_hash=np.array(part["input_hash"]),
        median_visit_frames=np.array(float(np.median(lens))),
        n_tune_frames=np.array(int(X.shape[0])),
        inherited_digest=np.array(spine.digest()))
    log(f"microstate fit: N={sr.N_MICROSTATES}, "
        f"{chain['n_transitions']:,} visit transitions, prior share "
        f"{chain['prior_share']:.4f}, median visit "
        f"{float(np.median(lens)):.1f} frames, hash {part['input_hash']}")
    return 0


def read_fit() -> dict:
    p = fit_path()
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- run `--fit-microstate` first")
    with np.load(p, allow_pickle=False) as z:
        return {"centroids": z["centroids"], "sd": z["sd"], "cols": z["cols"],
                "chain": {"P": z["P"], "order": int(z["order"]),
                          "n_states": int(z["n_states"]),
                          "alpha": float(z["alpha"])},
                "input_hash": str(z["input_hash"])}


def generate(args) -> int:
    kind = args.kind
    ego_out = tree(kind, "ego")
    ab_out = tree(kind, "tok", "abstain")
    os.makedirs(ego_out, exist_ok=True)
    os.makedirs(ab_out, exist_ok=True)
    src_ego = os.path.join(config.REPO, "work", "ego")
    src_ab = os.path.join(config.REPO, "work", "tok", "abstain")

    fit = read_fit() if kind in sr.VISIT_KINDS else None
    fps = spine.fps()
    tags = sorted(animals_of(spine.recording_ids()))
    n_fallback = n_rec = n_repeat = 0
    for tag in tags:
        name = f"{POSE_ARM}__{SCALE_ARM}__{tag}.npz"
        with np.load(os.path.join(src_ego, name), allow_pickle=False) as z:
            fields = {k: z[k] for k in z.files}
        rng = np.random.default_rng(seeds.stable_seed(SEED, kind, tag))
        out = sr.generate(fields["X"], fields["bounds"], kind, rng,
                          fps=fps, fit=fit)
        fields["X"] = out["x"]
        fields["surrogate"] = np.array(kind)
        fields["surrogate_seed"] = np.array(seeds.stable_seed(SEED, kind, tag))
        np.savez_compressed(os.path.join(ego_out, name), **fields)
        # The corpus's abstain mask, unchanged. Copied rather than regenerated:
        # it is a property of the tracking, which the surrogate does not have.
        shutil.copyfile(os.path.join(src_ab, f"{tag}.npz"),
                        os.path.join(ab_out, f"{tag}.npz"))
        n_fallback += int(out["n_fallback"])
        n_repeat += int(out.get("n_residual_repeats", 0))
        n_rec += int(out["n_recordings"])
    log(f"[{kind}] {len(tags)} animals, {n_rec} recordings, "
        f"{n_fallback} fallbacks, {n_repeat} residual repeats, "
        f"{out['n_dead_columns']} dead columns copied, "
        f"peak_rss={peak_rss_gb():.2f} GB")
    write_json({"kind": kind, "n_animals": len(tags), "n_recordings": n_rec,
                "n_fallback": n_fallback, "n_residual_repeats": n_repeat,
                "seed": SEED, "inherited_digest": spine.digest(),
                "microstate_fit": (fit["input_hash"] if fit else None),
                "n_microstates": (sr.N_MICROSTATES if fit else None),
                "abstain": "copied from the corpus, unchanged"},
               tree(kind, "generate.json"))
    return 0


def _per_animal(path: str, mode: str = "conditioned") -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    pa = d["abstain_arms"][mode]["per_animal"]
    return {"animals": list(pa["rung0"]["animals"]),
            "rung0": np.asarray(pa["rung0"]["nats_per_second"], dtype=np.float64),
            "rung1": np.asarray(pa["rung1"]["nats_per_second"], dtype=np.float64)}


def combine(args) -> int:
    rows, reads = [], {}
    for n in CELLS:
        corpus_p = os.path.join(config.REPO, "work", "tok", "ladder",
                                f"{ARM}_N{n}.json")
        if not os.path.exists(corpus_p):
            raise SystemExit(f"{corpus_p} not found -- the corpus ladder must "
                             f"have run first")
        c = _per_animal(corpus_p)
        d_corpus = c["rung0"] - c["rung1"]
        for kind in sr.KINDS:
            sp = tree(kind, "tok", "ladder", f"{ARM}_N{n}.json")
            if not os.path.exists(sp):
                log(f"  {kind} N={n}: no ladder shard, skipping")
                continue
            s = _per_animal(sp)
            if s["animals"] != c["animals"]:
                raise SystemExit(f"{kind} N={n} scored different animals than "
                                 f"the corpus; the comparison is not paired")
            d_surr = s["rung0"] - s["rung1"]
            delta = d_corpus - d_surr
            ci = boot.animal_interval(delta, c["animals"], how="mean",
                                      n_boot=2000, seed=SEED)
            fi = boot.frame_interval(delta, n_boot=2000, seed=SEED)
            obj = {"dataset": "luna", "arm": ARM, "n_states": n,
                   "surrogate": kind, "split": "report", "pose_arm": POSE_ARM,
                   "retired": True}
            rd = sr.falsifier_read(
                {"animal_interval": ci, "frame_interval": fi},
                {"mean_delta": float(d_corpus.mean())},
                {"mean_delta": float(d_surr.mean())},
                kind=kind, scored_object=obj, n_effective=len(c["animals"]))
            reads[f"{kind}_N{n}"] = rd.to_dict()
            rows.append({"n_states": n, "kind": kind,
                         "corpus_advantage": float(d_corpus.mean()),
                         "surrogate_advantage": float(d_surr.mean()),
                         "gap": float(delta.mean()),
                         "animal_interval": dict(ci), "frame_interval": dict(fi),
                         "n_better": int((delta > 0).sum()),
                         "n_animals": len(c["animals"]),
                         "verdict": rd.verdict})
            log(rd.line())
    if not rows:
        raise SystemExit("no surrogate ladder shards found")
    doc = {**provenance.header(anchors.LUNA, stage="tok_falsifier",
                            unverified="report split only; the anchor counts all"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "arm": ARM, "cells": list(CELLS),
           "kinds": list(sr.KINDS),
           "registration": "results/FALSIFIER_PREREGISTRATION.md",
           "retired_by_runlength": True,
           "comparisons": rows, "reads": reads,
           "peak_rss_gb": peak_rss_gb()}
    write_json(doc, args.out)
    log(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--fit-microstate", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--kind", choices=sr.KINDS, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.fit_microstate:
        return fit_microstate(a)
    if a.generate:
        if a.kind is None:
            raise SystemExit("--generate needs --kind")
        return generate(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("falsifier.json")
        return combine(a)
    raise SystemExit("pass --generate --kind K, or --combine")


if __name__ == "__main__":
    raise SystemExit(main())
