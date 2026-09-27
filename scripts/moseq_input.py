"""MOSEQ §1-2: the fit set, and the unsmoothed pose + surrogates as kpms input.

    python3 scripts/moseq_input.py --phase select            # work/moseq/sets.json
    python3 scripts/moseq_input.py --phase build --shard I --of N
    # writes work/moseq/input/{real,white,phase}/<rid>.npz  (coords, conf)

Fit set: 100 recordings drawn with random.Random(0) from the tune split's
recordings, the same for every arm. Apply set: every report recording.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                     # noqa: E402
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance, seeds                                   # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.moseq import inputs                                        # noqa: E402
from vieb.tok import config                                          # noqa: E402

WORK = os.path.join(config.REPO, "work", "moseq")
SETS = os.path.join(WORK, "sets.json")
N_FIT = 100
FIT_SEED = 0


def select(a) -> int:
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    rids = sorted(spine.recording_ids())
    tune = [r for r in rids if split_of.get(lab.animal_tag(r)) == "tune"]
    rep = [r for r in rids if split_of.get(lab.animal_tag(r)) == "report"]
    fit = sorted(random.Random(FIT_SEED).sample(tune, N_FIT))
    os.makedirs(WORK, exist_ok=True)
    write_json({**provenance.header(anchors.LUNA, stage="moseq_sets",
                                    unverified="a tune sample plus the report split"),
                "registration": "results/MOSEQ_PREREGISTRATION.md §2",
                "fit_seed": FIT_SEED, "n_tune_recordings": len(tune),
                "fit": fit, "report": rep,
                "n_fit_animals": len({lab.animal_tag(r) for r in fit}),
                "n_report_animals": len({lab.animal_tag(r) for r in rep})}, SETS)
    log(f"  fit {len(fit)} of {len(tune)} tune recordings; report {len(rep)}")
    return 0


def build(a) -> int:
    with open(SETS, encoding="utf-8") as fh:
        sets = json.load(fh)
    want = set(sets["fit"]) | set(sets["report"])
    tags = sorted({lab.animal_tag(r) for r in want})[a.shard::a.of]
    for k in inputs.KINDS:
        os.makedirs(os.path.join(WORK, "input", k), exist_ok=True)
    for tag in tags:
        disp = np.load(os.path.join(config.REPO, "work", "disposition", f"{tag}.npz"))
        rep = np.load(os.path.join(config.REPO, "work", "repaired", f"{tag}.npz"))
        b = disp["bounds"]
        rids = [str(r) for r in disp["recording_ids"]]
        assert rids == [str(r) for r in rep["recording_ids"]], tag
        for i, rid in enumerate(rids):
            if rid not in want:
                continue
            if all(os.path.exists(os.path.join(WORK, "input", k, f"{rid}.npz"))
                   for k in inputs.KINDS):
                continue
            lo, hi = int(b[i]), int(b[i + 1])
            c = spine.clean(rid)
            coords, conf = inputs.real(disp["pose"][lo:hi], c["missing"],
                                       rep["vetoed"][lo:hi], disp["abstain"][lo:hi],
                                       c["conf"])
            for k in inputs.KINDS:
                rng = np.random.default_rng(seeds.stable_seed("moseq", k, rid))
                xy, cf = inputs.surrogate(coords, conf, k, rng)
                np.savez_compressed(os.path.join(WORK, "input", k, f"{rid}.npz"),
                                    coords=xy.astype(np.float32),
                                    conf=cf.astype(np.float32))
        log(f"  {tag}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("select", "build"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    a = p.parse_args(argv)
    return {"select": select, "build": build}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
