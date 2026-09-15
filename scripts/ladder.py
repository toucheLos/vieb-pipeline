"""Step 4. Rungs 0-2 on each alphabet, scored by held-out description length.

    python3 scripts/ladder.py --write-grid
    sbatch --array=0-7%4 jobs/ladder.slurm --cell
    python3 scripts/ladder.py --combine

Registered in `results/TOK_PREREGISTRATION.md`, which also records that the
run-length stop condition retired all eight alphabets and that the ladder is
running anyway by the project owner's decision. Every number this produces
inherits that label.

Hyperparameters on **tune** (the duration grid, and nothing else), fits on
**fit**, every reported number and every bootstrap on **report** at
`n_effective = 89`.

## What is scored, and why both abstain arms

`symbol` widens the alphabet by one and models transitions into and out of
tracking failure. `conditioned` drops abstain runs and censors the run before
each one. If MDL improves only under `symbol`, the model is predicting tracking
dropout rather than behaviour, and the two arms are what makes that visible.
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
from recur.audit import leak                                      # noqa: E402
from recur.util import log, peak_rss_gb, write_json               # noqa: E402
from vieb.io import spine                                         # noqa: E402
from vieb.tok import config, hazard as hz, ladder as ld           # noqa: E402
from vieb.tok import mdl, quantize as qz, rle                     # noqa: E402

SEED = 0
#: Windows per recording for the identity probe, matching `scripts/ego.py` so
#: the two numbers sit on the same footing.
N_WINDOWS = 8


def tok_dir(*parts: str) -> str:
    return os.path.join(config.PATHS.tok_dir, *parts)


def out_dir() -> str:
    return tok_dir("ladder")


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def load_split(arm: str, n_states: int, split: str, of: dict) -> dict:
    """Every run in one split, with the recording index offset per animal.

    Without the offset the last run of one animal and the first of the next
    would be adjacent, and a transition would be recorded between two different
    mice.
    """
    d = tok_dir(f"{arm}_N{n_states}")
    codes, durs, recs, animals, base = [], [], [], [], 0
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        tag = os.path.basename(p)[:-4]
        if tag.startswith("_") or of.get(tag) != split:
            continue
        with np.load(p, allow_pickle=False) as z:
            y, bounds = z["labels"], z["bounds"].astype(np.int64)
            n_rec = len(z["recording_ids"])
        r = rle.encode(y, bounds)
        codes.append(r["code"])
        durs.append(r["duration"])
        recs.append(r["recording"].astype(np.int64) + base)
        animals.append(np.full(r["code"].shape[0], tag))
        base += n_rec
    if not codes:
        raise SystemExit(f"no {split} shards for {arm}_N{n_states}")
    return {"code": np.concatenate(codes), "duration": np.concatenate(durs),
            "recording": np.concatenate(recs),
            "animal": np.concatenate(animals)}


def _probe_rows(nats, animal, recording):
    """Per-window mean and SD of the model's own code length.

    **A reduced probe, and strictly weaker than the registered one.** The
    pre-registration says to feed the next-state distribution to `leak_read`;
    at 2,048 symbols that is a 2,048-vector per run and tens of gigabytes per
    cell. What goes in instead is how expensive and how variable the model finds
    each window -- which is a property of its predictions, not of the labels,
    but carries far less of them.

    A PASS here therefore does **not** establish the absence of style leakage.
    It establishes that the coarsest summary of the model's predictive behaviour
    does not identify the animal. Recorded in `DEVIATIONS.md`.
    """
    feats, who = [], []
    for r in np.unique(recording):
        m = recording == r
        x = nats[m]
        if x.shape[0] < 4:
            continue
        edges = np.linspace(0, x.shape[0], N_WINDOWS + 1).astype(int)
        for w in range(N_WINDOWS):
            a, b = edges[w], edges[w + 1]
            if b - a < 2:
                continue
            feats.append([float(x[a:b].mean()), float(x[a:b].std())])
            who.append(str(animal[m][0]))
    return np.asarray(feats, dtype=np.float64), who


def cell(args) -> int:
    arm, n = args.arm, int(args.n_states)
    os.makedirs(out_dir(), exist_ok=True)
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    fps = spine.fps()

    tune = load_split(arm, n, "tune", of)
    edges = hz.duration_grid(tune["duration"][tune["code"] != rle.ABSTAIN])
    log(f"[{arm} N={n}] duration grid from tune: {edges[:-1].tolist()} .. open")
    del tune

    fit_runs = load_split(arm, n, "fit", of)
    rep_runs = load_split(arm, n, "report", of)
    log(f"[{arm} N={n}] fit {fit_runs['code'].shape[0]:,} runs, "
        f"report {rep_runs['code'].shape[0]:,} runs")

    arms: dict = {}
    for mode in ld.ABSTAIN_MODES:
        f = ld.prepare(fit_runs, n, abstain=mode)
        r = ld.prepare(rep_runs, n, abstain=mode)
        # `prepare` drops abstain runs in the conditioned arm, so the animal and
        # recording labels have to be filtered the same way or every per-animal
        # number would be computed against a misaligned key. The mask comes back
        # from `prepare` rather than being re-derived here.
        keep = r["keep"]
        r_animal = rep_runs["animal"][keep]
        r_rec = rep_runs["recording"][keep]

        models, scores, per_an = {}, {}, {}
        specs = [("rung0", 0, 0), ("rung1", 1, 0)] + \
                [(f"rung2_k{k}", 2, k) for k in ld.K_GRID]
        for name, rung, k in specs:
            m = ld.fit_rung(rung, f, edges=edges, k=k)
            nats = ld.score_rung(m, r, edges=edges)
            scores[name] = nats
            per_an[name] = mdl.per_animal(
                nats, r["duration"], r_animal, fps,
                n_params=int(m["n_params"]), n_runs_fit=int(m["n_runs_fit"]))
            cl = mdl.code_length(nats, r["duration"], fps,
                                 n_params=int(m["n_params"]),
                                 n_runs_fit=int(m["n_runs_fit"]))
            log(f"  [{mode}] {name:10s} params={m['n_params']:>12,} "
                f"data={cl['data_nats_per_second']:8.3f} "
                f"book={cl['codebook_nats'] / cl['t_seconds']:9.3f} "
                f"total={cl['nats_per_second']:9.3f} nats/s")
            models[name] = {"n_params": int(m["n_params"]),
                            "n_runs_fit": int(m["n_runs_fit"]),
                            "code_length": cl,
                            "k": int(m.get("k", 0)), "rung": int(m["rung"])}
            if rung == 2 and k == 0:
                # The readable output of the whole module, and the honest
                # replacement for the retracted dwell test: is the hazard of
                # leaving a state flat in elapsed time, or does it bend?
                models[name]["hazard_shape"] = hz.shape_by_state(
                    m["hazard"], np.arange(int(f["alphabet"])), top=8)
            del m

        obj = {"dataset": "luna", "arm": arm, "n_states": n, "abstain": mode,
               "split": "report", "pose_arm": qz_pose(), "retired": True}
        reads = {}
        for lo, hi, label in (("rung0", "rung1", "rung 1 over rung 0"),
                              ("rung1", "rung2_k0", "rung 2 over rung 1"),
                              ("rung2_k0", "rung2_k1", "k = 1 over k = 0"),
                              ("rung2_k1", "rung2_k2", "k = 2 over k = 1")):
            imp = mdl.improvement(per_an[lo], per_an[hi])
            reads[f"{hi}_over_{lo}"] = mdl.ladder_read(
                imp, name=label, scored_object=obj, seed=SEED).to_dict()

        # The gates.
        best = "rung2_k0"
        fit_counts = np.zeros((int(f["alphabet"]), int(f["alphabet"])))
        done_f = ~f["censored"]
        np.add.at(fit_counts, (f["code"][done_f], f["next_state"][done_f]), 1.0)
        rare = mdl.rare_transition_recall(scores[best], r["code"],
                                          r["next_state"], r["censored"],
                                          fit_counts)
        feats, who = _probe_rows(scores[best], r_animal, r_rec)
        probe = leak.leak_read(feats, who, kind="animal",
                               scored_object={**obj, "probe": "code_length"},
                               n_effective=len(set(who))).to_dict() \
            if feats.shape[0] > 10 else None

        arms[mode] = {"models": models, "reads": reads,
                      "rare_transition_recall": rare,
                      "identity_probe": probe,
                      "n_runs_report": int(r["n_runs"]),
                      "n_dropped_report": int(r["n_dropped"]),
                      "n_censored_report": int(r["censored"].sum()),
                      "per_animal": {k: {"animals": v["animals"],
                                         "nats_per_second": v["nats_per_second"].tolist()}
                                     for k, v in per_an.items()}}
        for k2, rd in arms[mode]["reads"].items():
            log(f"  [{mode}] {rd['verdict']:13s} {rd['reason'][:150]}")

    doc = {
        **anchors.header(anchors.LUNA, stage="tok_ladder",
                         unverified="scored on the report split; the anchor "
                                    "counts the whole corpus"),
        "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
        "pose_arm": qz_pose(), "arm": arm, "n_states": n,
        "retired_by_runlength": True,
        "retired_note": ("this alphabet was retired by the pre-registered "
                         "run-length condition; the ladder runs under the "
                         "override recorded in TOK_PREREGISTRATION.md SS1"),
        "duration_grid": edges[:-1].tolist(),
        "abstain_arms": arms,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, os.path.join(out_dir(), f"{arm}_N{n}.json"))
    log(f"[{arm} N={n}] wrote shard, peak_rss={peak_rss_gb():.2f} GB")
    return 0


def qz_pose() -> str:
    return "raw"


def combine(args) -> int:
    shards = sorted(glob.glob(os.path.join(out_dir(), "*.json")))
    if not shards:
        raise SystemExit(f"no ladder shards in {out_dir()}")
    cells = []
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            cells.append(json.load(fh))
        c = cells[-1]
        log(f"{c['arm']}_N{c['n_states']}:")
        for mode, a in c["abstain_arms"].items():
            for k, rd in a["reads"].items():
                log(f"  [{mode}] {k:22s} {rd['verdict']}")
    doc = {
        **anchors.header(anchors.LUNA, stage="tok_ladder_combined",
                         unverified="report split only"),
        "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
        "n_cells": len(cells), "cells": cells,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("ladder")
    with open(path, "w", encoding="utf-8") as fh:
        for arm in qz.ARMS:
            for n in qz.STATE_GRID:
                fh.write(f"{arm} {n}\n")
    log(f"wrote {path}: {len(qz.ARMS) * len(qz.STATE_GRID)} cells")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--cell", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--arm", choices=qz.ARMS, default=None)
    p.add_argument("--n-states", type=int, default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("ladder.json")
        return combine(a)
    if a.cell:
        if a.arm is None or a.n_states is None:
            if a.task is None:
                raise SystemExit("--cell needs --arm/--n-states or --task")
            with open(config.PATHS.grid("ladder"), encoding="utf-8") as fh:
                grid = [ln.split() for ln in fh if ln.strip()]
            a.arm, a.n_states = grid[a.task][0], int(grid[a.task][1])
        return cell(a)
    raise SystemExit("pass --write-grid, --cell or --combine")


if __name__ == "__main__":
    raise SystemExit(main())
