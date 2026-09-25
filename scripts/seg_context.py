"""Step C. Does the island move with context? One clump, one contrast.

    python3 scripts/seg_context.py --group shape --clump 0

READ results/FREEZING_PREREGISTRATION.md FIRST. The MDE gate runs before the
contrast: if the smallest detectable effect exceeds the registered plausible one
(0.02 of occupancy), NO contrast is computed and the MDE is the result.

`BEHAVIOUR.md` found the widest clump 3.7x slower than the same animals' other
segments and named freezing as the hypothesis while saying plainly that it does
not test it. This is that test, and a FAIL takes the hypothesis off the page.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.journey import simplex as sx                               # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import breaks as bk, context as cx                      # noqa: E402
from vieb.tok import config                                           # noqa: E402

SEED = 0
#: Asserted, not assumed. `learning_curve.py` pins the same number with
#: `Read.assert_cardinality`, and a cohort that silently shrank would change
#: every interval below without changing anything a reader could see.
EXPECTED_COHORT_N = 89


def shard(group: str, k_mad: float) -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_vocab",
                        f"{group}__k{k_mad:g}__corpus.npz")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="shape", choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--clump", type=int, default=0)
    p.add_argument("--k-mad", type=float, default=3.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("seg_context.json")

    with np.load(shard(a.group, a.k_mad), allow_pickle=False) as z:
        if "rec" not in z.files:
            raise SystemExit(f"{shard(a.group, a.k_mad)} predates the locator "
                             f"fix; re-run seg_vocab first")
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr, rec = z["n_frames"], z["rec"]

    # Resolve each segment's recording, then its day and context. `rec` indexes
    # the animal's OWN recording list and is never a pooled position.
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    days, ctxs, keep = [], [], []
    rid_cache: dict = {}
    for i in range(labels.size):
        tag = animal[i]
        if split_of.get(tag) != "report":
            days.append(-1), ctxs.append("?"), keep.append(False)
            continue
        if tag not in rid_cache:
            with np.load(os.path.join(config.REPO, "work", "ego",
                                      f"raw__bodylen__{tag}.npz"),
                         allow_pickle=False) as e:
                rid_cache[tag] = [str(v) for v in e["recording_ids"]]
        rid = rid_cache[tag][int(rec[i])]
        # The canonical strict parser, which RAISES rather than degrading to a
        # "?" group -- and `context_letter`, so `Context_A,_No_Shock` is context
        # A with a tag rather than a fourth context.
        parsed = lab.parse(rid)
        days.append(int(parsed["day"]))
        ctxs.append(str(parsed["context_letter"]))
        keep.append(True)
    days_a = np.asarray(days, dtype=np.int64)
    ctxs_a = np.asarray(ctxs)
    log(f"  {int(np.asarray(keep).sum()):,} report segments, "
        f"{int((labels == a.clump).sum()):,} in clump {a.clump}")

    occ, den = cx.cell_occupancy(labels, nfr, animal, days_a, ctxs_a,
                                 clump=a.clump)
    pairs = cx.paired_deltas(occ, den)
    cohort = sorted({k.split("|")[0] for k in pairs["pair_key"]})
    obj = {"dataset": "luna", "arm": "seg_context", "group": a.group,
           "clump": a.clump, "pose_arm": "raw", "split": "report",
           "contrast": "CFD d3-7 context B minus A", "days": list(cx.DAYS)}
    cohort_read = Read("PASS", obj,
                       f"{len(cohort)} report animals contribute at least one "
                       f"(animal, day) cell holding both contexts on days "
                       f"{cx.DAYS[0]}-{cx.DAYS[-1]}; {pairs['n_pairs']} cells, "
                       f"{pairs['n_dropped_incomplete']} dropped as incomplete",
                       n_effective=len(cohort))
    cohort_read.assert_cardinality(len(cohort), EXPECTED_COHORT_N,
                                   what="animals")
    log("  " + cohort_read.line())

    # THE GATE. Computed from the spread of the differences, and it decides
    # whether the mean of those differences is ever looked at.
    diff = pairs["diff"]
    within_sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    mde_r = sx.mde_read(within_sd, int(diff.size),
                        plausible_effect=cx.PLAUSIBLE_EFFECT,
                        scored_object={**obj, "arm": "seg_context_mde"},
                        n_effective=len(cohort))
    log("  " + mde_r.line())

    reads = {"cohort": cohort_read.to_dict(), "mde": mde_r.to_dict()}
    contrast = None
    if mde_r.verdict == "PASS":
        flip = boot.pair_flip_null(diff, pairs["pair_key"], seed=SEED)
        ci = boot.animal_interval(diff, pairs["animal"], how="mean", seed=SEED)
        frame_ci = boot.frame_interval(diff, seed=SEED)
        rd = cx.context_read(ci, flip, pairs, clump=a.clump,
                             scored_object=obj, n_effective=int(ci["n_animals"]))
        reads["context"] = rd.to_dict()
        log("  " + rd.line())
        contrast = {
            "n_pairs": int(pairs["n_pairs"]),
            "mean_occupancy_A": pairs["mean_A"],
            "mean_occupancy_B": pairs["mean_B"],
            "mean_signed_diff": float(np.mean(diff)),
            "ci": ci, "flip": flip,
            "frame_interval_for_comparison_only": frame_ci,
            "width_ratio": (float((ci["hi"] - ci["lo"])
                                  / max(frame_ci["hi"] - frame_ci["lo"], 1e-12))
                            if np.isfinite(ci["hi"]) else float("nan")),
        }
    else:
        log("  contrast not read -- the MDE gate did not pass")

    write_json({**provenance.header(anchors.LUNA, stage="seg_context",
                                 unverified="one clump, one contrast"),
                "inherited_digest": spine.digest(),
                "registration": "results/FREEZING_PREREGISTRATION.md",
                "group": a.group, "clump": a.clump, "k_mad": a.k_mad,
                "days": list(cx.DAYS), "contexts": list(cx.CONTEXTS),
                "plausible_effect": cx.PLAUSIBLE_EFFECT,
                "within_sd": within_sd,
                "n_animals": len(cohort),
                "confound_note": (
                    "context and day are completely confounded in the "
                    "whole-corpus marginal grouping -- Context C occurs only on "
                    "Day 2 -- and that is why C and day 2 are excluded here. On "
                    "days 3-7 every animal sees both A and B, so the confounded "
                    "factor does not enter this contrast"),
                "reads": reads, "contrast": contrast},
               out)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
