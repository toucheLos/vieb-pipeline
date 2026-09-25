"""Where do the island's boundaries sit on the speed axis?

    python3 scripts/island_boundaries.py

UNREGISTERED DIAGNOSTIC. Carries no verdict and changes no published number.
Recorded in `results/DEVIATIONS.md` D11.

## The question it answers

`NOISEFLOOR.md` finds the detector at its own noise floor in the slowest speed
quintiles, and `BEHAVIOUR.md` finds the island **3.7x slower** than its animals'
other segments. Read together that looks like the one design-validated unit in
the programme sitting in the regime where boundaries are indistinguishable from
jitter.

**But a segment's content and its edges are not in the same regime.** A long
still segment is delimited by the animal *entering* and *leaving* stillness, and
those are motion events. The island's interior can be slow while the frames its
boundaries fall on are fast. Nothing on disk had checked which.

## What is compared

Per-frame speed (`quantize.speed`, body lengths/s, omega excluded) at:

* the **boundary frames** of island segments -- their `start` and `stop`;
* the **interior frames** of the same segments;
* the boundary frames of every **other** selected segment, as the base rate;
* every scored frame.

Each is placed in the **tune-derived** speed quintiles from
`results/noise_floor.json`, so the answer is on the same axis as the floor. That
is a yardstick transfer across splits and is stated rather than hidden: the
island lives on `report` and the floor was measured on `tune`. The report-split
quintile positions are reported beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, splits                               # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.tok import config, quantize as qz                           # noqa: E402

SEED = 0
POSE_ARM = "raw"
CLUMP = 0


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def _edges(raw: list) -> np.ndarray:
    """`write_json` stores +/-inf as null; put them back."""
    return np.asarray([-np.inf if v is None else float(v) for v in raw[:-1]]
                      + [np.inf], dtype=np.float64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("island_boundaries.json")

    with open(config.PATHS.result("noise_floor.json"), encoding="utf-8") as fh:
        nf = json.load(fh)
    s_edges = _edges(nf["strata"]["speed_edges"])
    floor = nf["rates"]["jitter1.0"]["ci"]
    marg = nf["strata"]["marginals"][0]["bins"]

    shard = os.path.join(config.PATHS.tok_dir, "seg_vocab",
                         "shape__k3__corpus.npz")
    with np.load(shard, allow_pickle=False) as z:
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr = z["n_frames"].astype(np.int64)
        frame = z["frame"].astype(np.int64)

    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    keep = np.asarray([split_of.get(t) == "report" for t in animal])
    log(f"  {int(keep.sum()):,} report segments, "
        f"{int((labels[keep] == CLUMP).sum()):,} in clump {CLUMP}")

    pools: dict[str, list[np.ndarray]] = {
        "island_boundary": [], "island_interior": [],
        "other_boundary": [], "all_frames": []}
    tags: dict[str, list[str]] = {k: [] for k in pools}
    for tag in sorted({animal[i] for i in np.flatnonzero(keep)}):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            sp = qz.speed(np.asarray(z["X"], dtype=np.float64))
            n_t = sp.size
        pools["all_frames"].append(sp)
        tags["all_frames"].append(tag)
        mine = np.flatnonzero(keep & (animal == tag))
        for name, sel in (("island", mine[labels[mine] == CLUMP]),
                          ("other", mine[labels[mine] != CLUMP])):
            edge_idx: list[int] = []
            inner: list[np.ndarray] = []
            for i in sel:
                s_, e_ = int(frame[i]), int(frame[i]) + int(nfr[i])
                edge_idx.extend([s_, min(e_, n_t - 1)])
                if name == "island" and e_ - s_ > 2:
                    inner.append(sp[s_ + 1:e_ - 1])
            if edge_idx:
                key = "island_boundary" if name == "island" \
                    else "other_boundary"
                pools[key].append(sp[np.asarray(edge_idx)])
                tags[key].append(tag)
            if name == "island" and inner:
                pools["island_interior"].append(np.concatenate(inner))
                tags["island_interior"].append(tag)

    obj = {"dataset": "luna", "arm": "island_boundaries", "split": "report",
           "pose_arm": POSE_ARM, "group": "shape", "clump": CLUMP}
    got: dict = {}
    for name, blocks in pools.items():
        v = np.concatenate(blocks)
        v = v[np.isfinite(v)]
        b = np.clip(np.searchsorted(s_edges, v, "right") - 1, 0,
                    len(s_edges) - 2)
        share = [float(np.mean(b == k)) for k in range(len(s_edges) - 1)]
        # animal-level interval on the share in the two SLOWEST quintiles --
        # the strata NOISEFLOOR.md finds at or below the floor.
        per, who = [], []
        per_q: list[list[float]] = [[] for _ in range(len(s_edges) - 1)]
        for blk, t in zip(blocks, tags[name]):
            x = blk[np.isfinite(blk)]
            if x.size == 0:
                continue
            bb = np.clip(np.searchsorted(s_edges, x, "right") - 1, 0,
                         len(s_edges) - 2)
            per.append(float(np.mean(bb <= 1)))
            for k in range(len(s_edges) - 1):
                per_q[k].append(float(np.mean(bb == k)))
            who.append(t)
        ci = boot.animal_interval(per, who, how="mean", n_boot=2000, seed=SEED)
        # Pooled and animal-weighted, both, because they DISAGREE here: the
        # island is animal-concentrated (`VOCAB.md`: top animal supplies 16%),
        # so pooling lets the prolific animals set the shape. The animal-level
        # figure is the one this programme quotes.
        q_animal = [dict(boot.animal_interval(per_q[k], who, how="mean",
                                              n_boot=2000, seed=SEED))
                    for k in range(len(s_edges) - 1)]
        got[name] = {"n_frames": int(v.size), "n_animals": len(who),
                     "median_speed": float(np.median(v)),
                     "quintile_share_pooled": share,
                     "quintile_share_animal_mean": q_animal,
                     "share_in_two_slowest": dict(ci)}
        log(f"  {name:18s} n={v.size:>9,}  median {np.median(v):.4f} bl/s  "
            f"slowest-two share {ci['point']:.3f} "
            f"[{ci['lo']:.3f}, {ci['hi']:.3f}]")

    ib = got["island_boundary"]["share_in_two_slowest"]
    ii = got["island_interior"]["share_in_two_slowest"]
    ob = got["other_boundary"]["share_in_two_slowest"]
    read = Read(
        "NOT_A_RESULT", obj, n_effective=len(tags["island_boundary"]),
        detail={"pools": got, "floor_ci": dict(floor),
                "speed_marginal_rates": [b["rate_per_s"] for b in marg],
                "speed_edges_source": "results/noise_floor.json (tune)"},
        reason=(
            f"UNREGISTERED DIAGNOSTIC, no verdict: {ib['point']:.3f} "
            f"[{ib['lo']:.3f}, {ib['hi']:.3f}] of island BOUNDARY frames fall "
            f"in the two slowest speed quintiles -- the strata NOISEFLOOR.md "
            f"finds at or below the noise floor -- against {ii['point']:.3f} "
            f"[{ii['lo']:.3f}, {ii['hi']:.3f}] of the same segments' INTERIOR "
            f"frames and {ob['point']:.3f} [{ob['lo']:.3f}, {ob['hi']:.3f}] "
            f"of every other selected segment's boundaries. Island boundary "
            f"median speed {got['island_boundary']['median_speed']:.4f} "
            f"against interior {got['island_interior']['median_speed']:.4f} "
            f"body lengths/s. Speed quintiles are the TUNE-derived edges from "
            f"the floor run, applied to report frames so the axis matches; "
            f"that is a yardstick transfer across splits"))
    log("  " + read.line())

    write_json({**provenance.header(anchors.LUNA, stage="island_boundaries",
                                 unverified="an unregistered diagnostic"),
                "inherited_digest": spine.digest(),
                "registered": False,
                "deviation": "results/DEVIATIONS.md D11",
                "clump": CLUMP, "split": "report", "seed": SEED,
                "speed_edges": [None if not np.isfinite(v) else float(v)
                                for v in s_edges],
                "floor": dict(floor),
                "speed_marginal_rates": [b["rate_per_s"] for b in marg],
                "pools": got, "reads": {"island_boundaries": read.to_dict()}},
               out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
