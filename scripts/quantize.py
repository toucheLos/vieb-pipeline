"""Step 2. Fit the alphabet on tune, assign the corpus, report occupancy.

    python3 scripts/quantize.py --write-grid
    sbatch --array=0-29%15 jobs/quantize.slurm --abstain
    sbatch jobs/quantize.slurm --basis
    sbatch --array=0-7%4 jobs/quantize.slurm --cell
    python3 scripts/quantize.py --combine

Four entry points because three things must happen exactly once and one thing
must happen eight times, and collapsing them would repeat expensive work or make
the arms incomparable.

**`--abstain` is per animal and cell-independent.** The mask comes from four
sources in three repositories and costs one `np.load` per recording; computing it
inside each of the eight cells would do that work eight times for an identical
answer.

**`--basis` is corpus-level and runs once.** The standardising SD and the speed
quantile cuts are fitted on **tune** and then frozen into `work/tok/basis.json`.
Every cell reads that file. If each cell fitted its own SD the two arms would be
standardised differently, and the MDL comparison between them -- the thing this
whole sweep exists to do -- would be measuring the standardisation rather than
the tokenization.

**`--cell` is one `(arm, n_states)` point.** Eight of them, each fitting on tune
and assigning the whole corpus.

Nothing here chooses anything on `report`. The partition, the SD and the strata
are all fitted on the 60 tune animals, and `--combine` reads occupancy from the
whole corpus only to describe it.
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
from recur.util import describe, log, peak_rss_gb, write_json     # noqa: E402
from vieb.io import spine                                         # noqa: E402
from vieb.tok import config, ego, quantize as qz, rle             # noqa: E402

#: The Step 1R arm this stage consumes. Named here rather than passed, because a
#: tokenizer built on a different pose arm than the one the freeze carries is the
#: exact failure this stage was re-run to remove.
POSE_ARM = "raw"
SCALE_ARM = "bodylen"

#: The published Step 1 bone cell. `results/bones.json`'s 2.134% is this array,
#: and `scripts/concentration.py` unpacks the same one. A second implementation
#: could disagree with the published rate, which is the thing to avoid here.
EPS = 0.10
BONE_CELL = f"viol|unfiltered|raw|{EPS}|skull"

SEED = 0
#: Distance-block budget when assigning. `microstate.CHUNK` is sized for N=1000
#: at 4 GB; at N=2048 the same chunk would be 8 GB, so the chunk is derived from
#: N rather than left at a constant that was right for a different sweep.
BLOCK_BYTES = 2 << 30


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


def tune_tags() -> list:
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    return sorted(t for t in animals_of(spine.recording_ids()) if of.get(t) == "tune")


# --------------------------------------------------------------------------
# 1. Abstain
# --------------------------------------------------------------------------

def bone_mask(tag: str, rids: list) -> dict:
    """Step 1's skull mask per recording, unpacked from the animal shard."""
    with np.load(config.PATHS.bones_shard(tag), allow_pickle=False) as z:
        ids = [str(v) for v in z["recording_ids"]]
        total = int(z["bounds"][-1])
        flat = np.unpackbits(z[BONE_CELL])[:total].astype(bool)
        out = {}
        for rid in rids:
            r = ids.index(str(rid))
            out[rid] = flat[int(z["bounds"][r]):int(z["bounds"][r + 1])]
    return out


def swap_mask(rid: str) -> np.ndarray:
    """Sustained bilateral swaps, from recur's own QC pass."""
    path = os.path.join(spine.path_in("recur", "work", "swap"), f"{rid}.npz")
    with np.load(path, allow_pickle=False) as z:
        return (z["sustained_ear"].astype(bool) | z["sustained_hip"].astype(bool))


def abstain_shard(args, tag: str) -> int:
    os.makedirs(tok_dir("abstain"), exist_ok=True)
    with np.load(ego_shard(tag), allow_pickle=False) as z:
        rids = [str(v) for v in z["recording_ids"]]
        bounds = z["bounds"].astype(np.int64)
        valid = z["valid"].astype(bool)

    bones = bone_mask(tag, rids)
    parts = {k: np.zeros(int(bounds[-1]), dtype=bool)
             for k in ("missing", "interpolated", "bone", "swap")}
    for r, rid in enumerate(rids):
        lo, hi = int(bounds[r]), int(bounds[r + 1])
        d = spine.clean(rid)
        # ANY keypoint. A frame with one held landmark is a frame whose pose is
        # partly an interpolant, and the symbol it would get is partly one too.
        parts["missing"][lo:hi] = d["missing"].astype(bool).any(axis=1)
        parts["interpolated"][lo:hi] = d["interpolated"].astype(bool).any(axis=1)
        parts["bone"][lo:hi] = bones[rid]
        parts["swap"][lo:hi] = swap_mask(rid)

    ab = (~valid) | parts["missing"] | parts["interpolated"] | parts["bone"] \
        | parts["swap"]
    shares = {k: float(v.mean()) for k, v in parts.items()}
    shares["ego_invalid"] = float((~valid).mean())
    shares["union"] = float(ab.mean())
    np.savez_compressed(
        tok_dir("abstain", f"{tag}.npz"), abstain=ab, bounds=bounds,
        recording_ids=np.array(rids), shares_json=np.array(json.dumps(shares)),
        inherited_digest=np.array(spine.digest()))
    log(f"[{tag}] abstain {ab.mean():.4%} "
        f"(ego {shares['ego_invalid']:.3%} miss {shares['missing']:.3%} "
        f"interp {shares['interpolated']:.3%} bone {shares['bone']:.3%} "
        f"swap {shares['swap']:.3%})")
    return 0


# --------------------------------------------------------------------------
# 2. The basis: one SD and one set of strata cuts, fitted on tune
# --------------------------------------------------------------------------

def load_animal(tag: str) -> dict:
    with np.load(ego_shard(tag), allow_pickle=False) as z:
        out = {"X": z["X"], "valid": z["valid"].astype(bool),
               "bounds": z["bounds"].astype(np.int64),
               "recording_ids": [str(v) for v in z["recording_ids"]]}
    p = tok_dir("abstain", f"{tag}.npz")
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            out["abstain"] = z["abstain"].astype(bool)
    return out


def basis(args) -> int:
    os.makedirs(config.PATHS.tok_dir, exist_ok=True)
    tags = tune_tags()
    log(f"basis on {len(tags)} tune animals")
    xs, sps, bounds, keep = [], [], [0], []
    for tag in tags:
        a = load_animal(tag)
        xs.append(a["X"])
        sps.append(qz.speed(a["X"]))
        for r in range(a["bounds"].shape[0] - 1):
            bounds.append(bounds[-1] + int(a["bounds"][r + 1] - a["bounds"][r]))
        keep.append(a["valid"] & ~a.get("abstain", np.zeros(len(a["valid"]), bool)))
    X = np.concatenate(xs)
    sp = np.concatenate(sps)
    ok = np.concatenate(keep)
    del xs, sps, keep
    log(f"  {X.shape[0]:,} tune frames, {int(ok.sum()):,} usable "
        f"({ok.mean():.2%}), {len(bounds) - 1} recordings")

    sd, n_bouts = qz.corpus_sd(X, sp, spine.fps(), np.asarray(bounds), valid=ok)
    dead_idx = [int(i) for i in np.flatnonzero(sd <= 0)]
    dead = [ego.CHANNELS[i] for i in dead_idx]
    # `ego.dead_columns()` keys are ROLES (origin_x, origin_y) and its values are
    # column indices, so the comparison is on indices. Matching on the keys would
    # compare 'origin_x' against 's_center_x' and refuse every run.
    #
    # Two columns, not three. SE(2) removal costs three degrees of freedom but
    # only two of them are a single column each -- s at the origin keypoint. The
    # third is the aligned axis's y-component, which is a linear combination and
    # so has non-zero marginal SD. That is why `rank` reads 11 of 14 while only
    # two channels are identically zero.
    expected = set(ego.dead_columns().values())
    unexpected = [ego.CHANNELS[i] for i in dead_idx if i not in expected]
    if unexpected:
        raise SystemExit(f"channels with zero segment-balanced SD: {unexpected}")
    sd_safe = np.where(sd > 0, sd, 1.0)

    edges = qz.strata_edges(sp, ok)
    strat = qz.stratum_of(sp, edges)
    counts = [int((strat[ok] == i).sum()) for i in range(qz.N_STRATA)]
    doc = {
        # `unverified`, not `observed`: the anchor counts the whole corpus and
        # this stage is fitted on tune alone, so asserting its totals here would
        # be asserting something false. `anchors.header` requires exactly one of
        # the two, which is what forces the choice to be made explicitly.
        **provenance.header(anchors.LUNA, stage="tok_basis",
                         unverified="fitted on the tune split; the anchor "
                                    "counts the whole corpus"),
        "n_tune_animals": len(tags),
        "n_tune_frames": int(X.shape[0]),
        "fps": spine.fps(),
        "inherited_digest": spine.digest(),
        "preprocessing_freeze": "F3",
        "pose_arm": POSE_ARM, "scale_arm": SCALE_ARM,
        "split_fitted_on": "tune",
        "channels": list(ego.CHANNELS),
        "sd": sd.tolist(), "sd_used": sd_safe.tolist(),
        "dead_channels": dead,
        "dead_note": ("SE(2) removal costs three pose dimensions and they are "
                      "left visible rather than projected away; their SD is "
                      "zero by construction and is floored to 1.0 so the "
                      "standardisation is a no-op there rather than a division "
                      "by zero"),
        "n_bouts": int(n_bouts),
        "speed_edges": edges.tolist(),
        "stratum_counts": counts,
        "n_usable_tune_frames": int(ok.sum()),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, tok_dir("basis.json"))
    log(f"  sd: {np.round(sd, 4).tolist()}")
    log(f"  speed edges (body lengths/s): {np.round(edges, 4).tolist()}")
    log(f"  stratum counts: {counts}")
    log(f"wrote {tok_dir('basis.json')}")
    return 0


# --------------------------------------------------------------------------
# 3. One cell: fit on tune, assign the corpus
# --------------------------------------------------------------------------

def read_basis() -> dict:
    p = tok_dir("basis.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- run `--basis` first")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return {"sd": np.asarray(d["sd_used"], dtype=np.float64),
            "edges": np.asarray(d["speed_edges"], dtype=np.float64)}


def cell(args) -> int:
    arm, n_states = args.arm, int(args.n_states)
    out_dir = tok_dir(f"{arm}_N{n_states}")
    os.makedirs(out_dir, exist_ok=True)
    b = read_basis()
    cols = np.arange(ego.N_DIMS, dtype=np.int64)

    tags = tune_tags()
    xs, sps, keep = [], [], []
    for tag in tags:
        a = load_animal(tag)
        xs.append(a["X"])
        sps.append(qz.speed(a["X"]))
        keep.append(a["valid"] & ~a.get("abstain", np.zeros(len(a["valid"]), bool)))
    X = np.concatenate(xs)
    sp = np.concatenate(sps)
    ok = np.concatenate(keep)
    del xs, sps, keep
    log(f"[{arm} N={n_states}] fitting on {int(ok.sum()):,} tune frames")
    model = qz.fit(X, b["sd"], cols, ok, arm=arm, n_states=n_states, seed=SEED,
                   sp=sp, logger=log)
    if arm == "speed":
        # Frozen on tune by `--basis`, so every N shares one set of cuts and the
        # arms differ only in alphabet size.
        model["edges"] = b["edges"]
    del X, sp, ok
    log(f"[{arm} N={n_states}] fitted, inertia {model['inertia']:.2f}, "
        f"hash {model['input_hash']}")

    chunk = max(50_000, BLOCK_BYTES // (4 * max(1, n_states)))
    all_tags = sorted(animals_of(spine.recording_ids()))
    per_animal = []
    for tag in all_tags:
        a = load_animal(tag)
        ab = a.get("abstain")
        if ab is None:
            raise SystemExit(f"[{tag}] no abstain shard -- run `--abstain` first")
        y = qz.assign(a["X"], model, abstain=ab | ~a["valid"],
                      sp=qz.speed(a["X"]))
        occ = qz.occupancy(y, n_states)
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            labels=y, bounds=a["bounds"],
            recording_ids=np.array(a["recording_ids"]),
            counts=np.asarray(occ["counts"], dtype=np.int64),
            inherited_digest=np.array(spine.digest()))
        per_animal.append({"animal": tag, "abstain_frac": occ["abstain_frac"],
                           "n_frames": occ["n_frames"]})
    np.savez_compressed(
        os.path.join(out_dir, "_model.npz"),
        centroids=model["centroids"], sd=model["sd"], cols=model["cols"],
        offsets=model["offsets"], edges=np.asarray(model["edges"]),
        arm=np.array(arm), n_states=np.array(n_states), seed=np.array(SEED),
        input_hash=np.array(model["input_hash"]),
        inertia=np.array(model["inertia"]),
        n_fit_frames=np.array(model["n_fit_frames"]),
        inherited_digest=np.array(spine.digest()))
    log(f"[{arm} N={n_states}] assigned {len(all_tags)} animals, "
        f"chunk={chunk:,}, peak_rss={peak_rss_gb():.2f} GB")
    return 0


# --------------------------------------------------------------------------
# 4. Combine
# --------------------------------------------------------------------------

def combine(args) -> int:
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    cells, rows = [], []
    n_recordings = n_frames_corpus = 0
    for arm in qz.ARMS:
        for n in qz.STATE_GRID:
            d = tok_dir(f"{arm}_N{n}")
            shards = sorted(p for p in glob.glob(os.path.join(d, "*.npz"))
                            if not os.path.basename(p).startswith("_"))
            if not shards:
                log(f"  {arm} N={n}: no shards, skipping")
                continue
            counts = np.zeros(n, dtype=np.int64)
            n_frames = n_abstain = 0
            per_split: dict = {}
            codes, durs, recs, rec_base = [], [], [], 0
            for p in shards:
                tag = os.path.basename(p)[:-4]
                with np.load(p, allow_pickle=False) as z:
                    c = z["counts"].astype(np.int64)
                    y = z["labels"]
                    bounds = z["bounds"].astype(np.int64)
                    n_rec = len(z["recording_ids"])
                counts += c
                n_frames += int(y.shape[0])
                if not cells:            # count the corpus once, on the first cell
                    n_recordings += n_rec
                    n_frames_corpus += int(y.shape[0])
                n_abstain += int((y == qz.ABSTAIN).sum())
                s = of.get(tag, "?")
                cell_ = per_split.setdefault(s, {"frames": 0, "abstain": 0})
                cell_["frames"] += int(y.shape[0])
                cell_["abstain"] += int((y == qz.ABSTAIN).sum())
                # Runs are pooled with the recording index OFFSET per animal, so
                # the last run of one animal and the first of the next are never
                # adjacent. Without the offset `self_transitions` would count
                # every animal boundary as a manufactured repeat, and the guard
                # that exists to catch a spliced abstain gap would fire on
                # nothing at all.
                r = rle.encode(y, bounds)
                codes.append(r["code"])
                durs.append(r["duration"])
                recs.append(r["recording"].astype(np.int64) + rec_base)
                rec_base += n_rec
            runs = {"code": np.concatenate(codes),
                    "duration": np.concatenate(durs),
                    "recording": np.concatenate(recs)}
            occ = _occ_from_counts(counts, n_frames, n_abstain)
            obj = {"dataset": "luna", "arm": arm, "n_states": n,
                   "pose_arm": POSE_ARM, "scale_arm": SCALE_ARM, "split": "all"}
            n_eff = len(tune_tags())
            rd = qz.alphabet_read(occ, scored_object=obj, n_effective=n_eff)
            dist = rle.distribution(runs, spine.fps())
            mass = rle.frame_versus_run_mass(runs, n)
            rl = rle.runlen_read(dist, scored_object=obj, n_effective=n_eff)
            cells.append({"arm": arm, "n_states": n, "n_animals": len(shards),
                          "occupancy": {k: v for k, v in occ.items()
                                        if k != "counts"},
                          "abstain_by_split": per_split,
                          "runlength": {k: v for k, v in dist.items()},
                          "frame_versus_run_mass": {
                              k: v for k, v in mass.items()
                              if k not in ("frame_share", "run_share")},
                          "read": rd.to_dict(),
                          "runlength_read": rl.to_dict(),
                          "retired": rl.verdict == "FAIL"})
            rows.append((arm, n, occ, rd))
            log(f"{rd.line()}")
            log(f"{rl.line()}")

    if not cells:
        raise SystemExit("no cells found -- run `--cell` first")
    doc = {
        # Assignment covers the whole corpus, so the anchor's own totals are
        # checkable here and are checked rather than asserted in prose.
        **provenance.header(anchors.LUNA, stage="tok_alphabet", observed={
            "n_recordings": n_recordings, "n_frames": n_frames_corpus,
            "fps": spine.fps()}),
        "n_cells": len(cells),
        "inherited_digest": spine.digest(),
        "preprocessing_freeze": "F3",
        "pose_arm": POSE_ARM, "scale_arm": SCALE_ARM,
        "fitted_on": "tune", "arms": list(qz.ARMS),
        "state_grid": list(qz.STATE_GRID),
        "cells": cells,
        "abstain_note": ("abstain is carried, never dropped: a flagged frame "
                         "becomes -1 and stays in the stream so the run-length "
                         "encoder can see where the gaps are"),
        "retired_cells": [f"{c['arm']}_N{c['n_states']}" for c in cells
                          if c["retired"]],
        "stop_condition": ("a median run at or below "
                           f"{rle.MEDIAN_RUN_MIN} frames retires THAT alphabet "
                           "and not the sweep; only every N tripping it would "
                           "be a finding about the representation"),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(f"wrote {args.out}")
    return 0


def _occ_from_counts(counts, n_frames: int, n_abstain: int) -> dict:
    """Corpus occupancy from pooled per-animal counts.

    Recomputed from counts rather than by concatenating 22.4M labels, and it
    must agree with `quantize.occupancy` -- `tests/test_quantize.py` is what
    keeps the two definitions from drifting.
    """
    c = np.asarray(counts, dtype=np.float64)
    total = float(c.sum())
    base = {"n_states": int(c.shape[0]), "n_frames": int(n_frames),
            "n_labelled": int(total),
            "abstain_frac": (float(n_abstain) / n_frames) if n_frames else float("nan"),
            "counts": c.astype(np.int64).tolist()}
    if total <= 0:
        return {**base, "top_share": float("nan"), "top10_share": float("nan"),
                "dead_frac": 1.0, "gini": float("nan")}
    share = np.sort(c / total)[::-1]
    cum = np.cumsum(share)
    return {**base, "top_share": float(share[0]),
            "top10_share": float(cum[min(9, cum.size - 1)]),
            "dead_frac": float((c == 0).mean()),
            "gini": qz._gini(c)}


# --------------------------------------------------------------------------
# 5. Diagnosing the stop condition
# --------------------------------------------------------------------------

def flicker(args) -> int:
    """Why is the median run one frame? Attribute it, on tune, three ways.

    Run ONLY because the stop condition fired on every cell in the grid. It is a
    diagnosis, not a new arm: nothing it fits is carried forward, and no number
    it produces enters the MDL comparison. Three fits of the same plain
    alphabet, differing in exactly one thing each:

    * **full/raw** -- the sweep's own configuration, reproduced here on tune
      alone so the three rows are on one footing.
    * **pose/raw** -- the 14 egocentric coordinates WITHOUT the three twist
      channels. If the flicker is the velocity block's frame-to-frame noise,
      this is where it goes away.
    * **full/wiener** -- the identical configuration on the Wiener array, the
      arm F3 does not carry. If dwell appears here and nowhere else, the dwell
      is the filter's and not the corpus's, which is the third instance of a
      failure mode this project has already named twice.

    The Wiener row is the one that matters most and it is the one most easily
    misread. A LONGER run there is not evidence that Wiener is better. It is
    evidence that a low-pass filter makes adjacent frames more similar, which is
    what low-pass filters do, and that any dwell measured on it would be partly
    the filter's autocorrelation.
    """
    tags = tune_tags()
    cols_full = np.arange(ego.N_DIMS, dtype=np.int64)
    cols_pose = np.arange(ego.N_POSE, dtype=np.int64)
    n_states = int(args.n_states or 256)
    rows = []

    for label, pose_arm, cols in (("full/raw", "raw", cols_full),
                                  ("pose/raw", "raw", cols_pose),
                                  ("full/wiener", "wiener", cols_full)):
        xs, sps, keep, bounds, per_animal = [], [], [], [0], []
        for tag in tags:
            path = (ego_shard(tag) if pose_arm == "raw" else
                    os.path.join(config.PATHS.ego_dir, f"{SCALE_ARM}__{tag}.npz"))
            with np.load(path, allow_pickle=False) as z:
                X, valid, b = z["X"], z["valid"].astype(bool), z["bounds"]
            with np.load(tok_dir("abstain", f"{tag}.npz"), allow_pickle=False) as z:
                ab = z["abstain"].astype(bool)
            if X.shape[0] != ab.shape[0]:
                raise SystemExit(f"[{tag}] {pose_arm} has {X.shape[0]} frames "
                                 f"against the abstain mask's {ab.shape[0]}")
            xs.append(X)
            sps.append(qz.speed(X))
            keep.append(valid & ~ab)
            for r in range(len(b) - 1):
                bounds.append(bounds[-1] + int(b[r + 1] - b[r]))
            per_animal.append((tag, int(X.shape[0])))
        X = np.concatenate(xs)
        sp = np.concatenate(sps)
        ok = np.concatenate(keep)
        del xs, sps, keep

        # Each row gets its OWN SD, fitted on its own coordinates. Standardising
        # the Wiener array by the raw arm's SD would make the comparison a
        # statement about the mismatch rather than about the filter.
        sd, _ = qz.corpus_sd(X, sp, spine.fps(), np.asarray(bounds), valid=ok)
        sd = np.where(sd > 0, sd, 1.0)
        model = qz.fit(X, sd, cols, ok, arm="plain", n_states=n_states,
                       seed=SEED, logger=None)
        y = qz.assign(X, model, abstain=~ok)
        runs = rle.encode(y, np.asarray(bounds))
        dist = rle.distribution(runs, spine.fps())
        rows.append({"row": label, "pose_arm": pose_arm,
                     "n_channels": int(cols.shape[0]), "n_states": n_states,
                     "median_frames": dist["median_frames"],
                     "median_seconds": dist["median_seconds"],
                     "p75_frames": dist["frames"].get("p75"),
                     "p90_frames": dist["frames"].get("p90"),
                     "mean_frames": dist["frames"].get("mean"),
                     "n_runs": dist["n_runs"]})
        log(f"  {label:12s} cols={cols.shape[0]:2d}  median={dist['median_frames']:.1f} "
            f"frames ({dist['median_seconds']:.3f} s)  p75="
            f"{dist['frames'].get('p75'):.1f}  mean="
            f"{dist['frames'].get('mean'):.2f}  runs={dist['n_runs']:,}")
        del X, sp, ok, y, runs

    doc = {
        **provenance.header(anchors.LUNA, stage="tok_flicker",
                         unverified="tune split only; the anchor counts all"),
        "inherited_digest": spine.digest(),
        "preprocessing_freeze": "F3",
        "why": ("run only because the run-length stop condition fired on every "
                "cell in the grid; a diagnosis, not an arm"),
        "n_tune_animals": len(tags), "n_states": n_states,
        "scale_arm": SCALE_ARM, "rows": rows,
        "wiener_caveat": ("a longer run on the wiener row is NOT evidence that "
                          "wiener is better. A low-pass filter makes adjacent "
                          "frames more similar, so any dwell measured on it is "
                          "partly the filter's own autocorrelation"),
        "carried_forward": "nothing",
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, tok_dir("flicker.json"))
    log(f"wrote {tok_dir('flicker.json')}")
    return 0


def write_grid(args) -> int:
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("quantize")
    with open(path, "w", encoding="utf-8") as fh:
        for arm in qz.ARMS:
            for n in qz.STATE_GRID:
                fh.write(f"{arm} {n}\n")
    log(f"wrote {path}: {len(qz.ARMS) * len(qz.STATE_GRID)} cells")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--abstain", action="store_true")
    p.add_argument("--basis", action="store_true")
    p.add_argument("--cell", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--flicker", action="store_true",
                   help="diagnose the run-length stop condition (tune only)")
    p.add_argument("--arm", choices=qz.ARMS, default=None)
    p.add_argument("--n-states", type=int, default=None)
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    if a.write_grid:
        return write_grid(a)
    if a.basis:
        return basis(a)
    if a.flicker:
        return flicker(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("alphabet.json")
        return combine(a)
    if a.cell:
        if a.arm is None or a.n_states is None:
            if a.task is None:
                raise SystemExit("--cell needs --arm/--n-states or --task")
            with open(config.PATHS.grid("quantize"), encoding="utf-8") as fh:
                grid = [ln.split() for ln in fh if ln.strip()]
            a.arm, a.n_states = grid[a.task][0], int(grid[a.task][1])
        return cell(a)
    if a.abstain:
        if a.animal:
            tags = [a.animal]
        elif a.task is not None:
            tags = sorted(animals_of(spine.recording_ids()))[a.task::a.n_tasks]
        else:
            raise SystemExit("--abstain needs --animal or --task")
        for tag in tags:
            if not a.force and os.path.exists(tok_dir("abstain", f"{tag}.npz")):
                log(f"[{tag}] abstain shard exists, skipping")
                continue
            abstain_shard(a, tag)
        return 0
    raise SystemExit("pass --write-grid, --abstain, --basis, --cell or --combine")


if __name__ == "__main__":
    raise SystemExit(main())
