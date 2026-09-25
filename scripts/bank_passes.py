"""Bank the three passes this arm stands on, before any of it runs.

    python3 scripts/bank_passes.py

Writes `results/q1_banked.json` and `results/roughness.json`, each carrying a
`Read`, and re-states the dwell pair from `results/falsifier.json` where it is
already banked. Nothing is re-scored: Q1 is read and hashed, the roughness
levels are read from the diagnostic the separability failure produced, and the
dwell verdicts are read from the file that owns them.

The reason this exists is in `vieb/seg/banked.py`'s docstring. The short form:
Q1's +1.639% is a **wiener** number, the same file's **unfiltered** cell is
negative with an interval spanning zero, and this arm runs on raw. A banked Q1
that did not say so would be a wrong-object result rather than a defence
against one.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab                           # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.util import log, write_json                             # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.seg import banked as bkd                                 # noqa: E402
from vieb.tok import config                                        # noqa: E402

RECUR = os.environ.get("VIEB_RECUR", "/home/tul26194/recur")
Q1_JSON = os.path.join(RECUR, "results", "q1.json")
Q1_VERDICT = os.path.join(RECUR, "results", "q1_verdict.json")
Q1_OCC = os.path.join(RECUR, "results", "q1_occupancy_equivalent.json")
ROUGH_JSON = os.path.join(config.REPO, "work", "tok", "seg_validate",
                          "_separability_diagnose.json")
FALSIFIER = os.path.join(config.PATHS.results_dir, "falsifier.json")
#: The null the roughness read is scored against. `phase` preserves the power
#: spectrum exactly, so it is the only one of the four for which "same total
#: power, fewer windows" is a statement and not an artifact of scale.
ROUGH_NULL = "phase"


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        out: dict = json.load(fh)
    return out


def _cell(doc: dict, filt: str, dim: int, window_s: float) -> dict:
    """The one cell matching (filt, dim, window_s), refusing ambiguity.

    `q1.json` holds 28 cells and several share a filter. Selecting by the first
    match would be how the wrong arm gets banked, so this refuses unless
    exactly one cell matches.
    """
    got = [c for c in doc["cells"]
           if c["filt"] == filt and int(c["dim"]) == dim
           and float(c.get("pca_read", {}).get("scored_object", {})
                     .get("window_s", -1)) == window_s]
    if len(got) != 1:
        raise SystemExit(
            f"{len(got)} cells match filt={filt} dim={dim} w={window_s} in "
            f"{Q1_JSON!r}; banking the wrong arm is the failure this refuses")
    return got[0]


def main(argv=None) -> int:
    for p in (Q1_JSON, Q1_VERDICT, Q1_OCC, ROUGH_JSON, FALSIFIER):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p!r}: nothing can be banked from it")

    q1, verdict, occ = _load(Q1_JSON), _load(Q1_VERDICT), _load(Q1_OCC)
    wiener = _cell(q1, "wiener", 192, 0.4)
    unfilt = _cell(q1, "unfiltered", 192, 0.4)
    n_animals = int(wiener["vs"]["ar"]["delta"]["n_animals"])

    rd_q1 = bkd.q1_read(wiener, verdict, occ, n_effective=n_animals)
    rd_un = bkd.q1_unfiltered_read(unfilt, n_effective=n_animals)
    log(rd_q1.line())
    log(rd_un.line())

    write_json({
        **provenance.header(anchors.LUNA, stage="bank_q1",
                         unverified=("Q1 was computed in the recur repository "
                                     "and is quoted here, not recomputed")),
        "inherited_digest": spine.digest(),
        "banked_from": {
            "q1.json": {"path": Q1_JSON, "sha256": bkd.file_sha256(Q1_JSON)},
            "q1_verdict.json": {"path": Q1_VERDICT,
                                "sha256": bkd.file_sha256(Q1_VERDICT)},
            "q1_occupancy_equivalent.json": {
                "path": Q1_OCC, "sha256": bkd.file_sha256(Q1_OCC)}},
        "not_in_the_freeze": (
            "recur/results/q1.json is deliberately NOT added to "
            "spine.CONSUMED_JSON. Adding it would change the inherited digest "
            "198eb14ff258c7f6 that every result JSON already written carries, "
            "and a digest that moves because a file was banked stops meaning "
            "what it said. The SHA-256 above pins the same fact without "
            "rewriting history"),
        "reads": {"q1_wiener": rd_q1.to_dict(),
                  "q1_unfiltered": rd_un.to_dict()},
        "cells": {"wiener": {k: v for k, v in wiener.items() if k != "vs"},
                  "unfiltered": {k: v for k, v in unfilt.items() if k != "vs"}},
        "verdict_doc": verdict, "occupancy_equivalent": occ,
    }, config.PATHS.result("q1_banked.json"))

    rough = _load(ROUGH_JSON)
    levels = rough["roughness_levels"]
    obj = {"dataset": "luna", "arm": "roughness", "null": ROUGH_NULL,
           "pose_arm": "raw", "window_s": rough.get("window_s"),
           "split": "all", "n_channels": len(levels["corpus"]["mean_log_msd"])}
    # The probe ran on every animal, not the report split: it is a check on
    # how a null was CONSTRUCTED rather than an effect estimate, so it is not
    # scored on a held-out split. Counted from the corpus, never hardcoded.
    n_rough = len({lab.animal_tag(r) for r in spine.recording_ids()})
    rd_r = bkd.roughness_read(levels, null=ROUGH_NULL, scored_object=obj,
                              n_effective=n_rough)
    log(rd_r.line())
    write_json({
        **provenance.header(anchors.LUNA, stage="bank_roughness",
                         unverified=("windows are subsampled per animal; the "
                                     "anchor counts every frame")),
        "inherited_digest": spine.digest(),
        "banked_from": {"path": ROUGH_JSON,
                        "sha256": bkd.file_sha256(ROUGH_JSON)},
        "null": ROUGH_NULL,
        "why_phase": ("phase randomisation preserves the power spectrum and "
                      "therefore the mean squared increment EXACTLY, so a gap "
                      "in the mean of the LOG is a statement about "
                      "concentration and not about scale. No other null in the "
                      "set has that property"),
        "reads": {"roughness": rd_r.to_dict()},
        "roughness_levels": levels,
    }, config.PATHS.result("roughness.json"))

    fals = _load(FALSIFIER)
    obj_d = {"dataset": "luna", "arm": "dwell", "n_states": 256,
             "pose_arm": "raw", "split": "report",
             "null": "microstate|microstate0"}
    rd_d = bkd.dwell_read(fals.get("reads", {}), scored_object=obj_d,
                          n_effective=89)
    log(rd_d.line())
    log(f"dwell is already banked in {bkd.DWELL_BANKED_IN}; not duplicated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
