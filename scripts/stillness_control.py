"""Does clump membership beat being slow and long? The segment-level control.

    python3 scripts/stillness_control.py

READ results/STILLNESS_CONTROL_PREREGISTRATION.md FIRST.

The third attempt at one question. `ISLAND_LOOK.md` asked it of human scorers
and disqualified the scorer. `CONTEXT_CONTROLS.md` asked it of a frame-threshold
arm and the arm was vacuous -- it selected 2.27% of the island's own frames,
recorded as `METHODS_FINDINGS.md` M11.

This asks it of **segments matched on joint (log duration, log mean speed)**,
drawn from the same animal, without replacement -- the level at which the island
is actually defined. The precondition is BALANCE on those variables, not frame
overlap: a matched-segment arm is disjoint from the island by construction.
"""
from __future__ import annotations

import argparse
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
from vieb.seg import breaks as bk, context as cx, controls as co      # noqa: E402
from vieb.tok import config, quantize as qz                           # noqa: E402

SEED = 0
EXPECTED_COHORT_N = 89
POSE_ARM = "raw"


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="shape", choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--clump", type=int, default=0)
    p.add_argument("--k-mad", type=float, default=3.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("stillness_control.json")

    shard = os.path.join(config.PATHS.tok_dir, "seg_vocab",
                         f"{a.group}__k{a.k_mad:g}__corpus.npz")
    with np.load(shard, allow_pickle=False) as z:
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr = z["n_frames"].astype(np.int64)
        rec = z["rec"].astype(np.int64)
        frame = z["frame"].astype(np.int64)

    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    rid_cache: dict[str, list[str]] = {}
    days, ctxs = [], []
    for i in range(labels.size):
        tag = animal[i]
        if split_of.get(tag) != "report":
            days.append(-1)
            ctxs.append("?")
            continue
        if tag not in rid_cache:
            with np.load(ego_path(tag), allow_pickle=False) as e:
                got = str(e["pose_arm"]) if "pose_arm" in e.files else POSE_ARM
                if got != POSE_ARM:
                    raise SystemExit(f"[{tag}] pose_arm={got!r}")
                rid_cache[tag] = [str(v) for v in e["recording_ids"]]
        parsed = lab.parse(rid_cache[tag][int(rec[i])])
        days.append(int(parsed["day"]))
        ctxs.append(str(parsed["context_letter"]))
    days_a = np.asarray(days, dtype=np.int64)
    ctxs_a = np.asarray(ctxs)
    in_cell = np.asarray([int(d) in cx.DAYS and str(c) in cx.CONTEXTS
                          for d, c in zip(days_a, ctxs_a)])

    # Per-segment MEAN speed -- the quantity BEHAVIOUR.md's 3.7x is a ratio of,
    # and the level at which the island is defined.
    speed_seg = np.full(labels.size, np.nan)
    for tag in sorted({animal[i] for i in np.flatnonzero(in_cell)}):
        with np.load(ego_path(tag), allow_pickle=False) as e:
            sp = qz.speed(np.asarray(e["X"], dtype=np.float64))
        for i in np.flatnonzero(in_cell & (animal == tag)):
            seg = sp[int(frame[i]):int(frame[i]) + int(nfr[i])]
            speed_seg[i] = float(np.nanmean(seg)) if seg.size else np.nan

    fps = spine.fps()
    feats = np.column_stack([np.log(np.maximum(nfr, 1) / fps),
                             np.log(np.maximum(speed_seg, 1e-9))])
    usable = in_cell & np.isfinite(feats).all(axis=1)
    is_target = usable & (labels == a.clump)
    log(f"  {int(usable.sum()):,} usable segments, "
        f"{int(is_target.sum()):,} in clump {a.clump}")

    # The eligible pool is named explicitly: the default admits rows
    # outside the scored cells whose features are undefined.
    partners = co.matched_partners(is_target, animal, feats,
                                   pool=usable)
    t_idx = np.flatnonzero(is_target)
    ok = partners >= 0
    log(f"  {int(ok.sum())} of {t_idx.size} island segments matched")

    obj = {"dataset": "luna", "arm": "stillness_control", "group": a.group,
           "clump": a.clump, "pose_arm": POSE_ARM, "split": "report",
           "contrast": "CFD d3-7 context B minus A", "days": list(cx.DAYS),
           "matched_on": ["log_duration_s", "log_mean_speed_bl_s"]}

    # THE PRECONDITION. Balance, not overlap -- §3.
    bal = co.balance_read(feats[t_idx[ok]], feats[partners[ok]],
                          names=("log_duration_s", "log_mean_speed_bl_s"),
                          scored_object={**obj, "arm": "balance"},
                          n_effective=int(ok.sum()))
    log("  " + bal.line())
    reads: dict = {"balance": bal.to_dict()}

    occ_isl, den = cx.cell_occupancy(labels, nfr, animal, days_a, ctxs_a,
                                     clump=a.clump)
    sel = np.zeros(labels.size, dtype=np.float64)
    sel[partners[ok]] = nfr[partners[ok]].astype(np.float64)
    occ_mat = co.occupancy(co.cell_sums(sel, animal, days_a, ctxs_a), den)
    al = co.aligned_deltas({"island": occ_isl, "matched": occ_mat}, den)
    cohort = sorted({k.split("|")[0] for k in al["pair_key"]})
    cr = Read("PASS", obj,
              f"{len(cohort)} report animals, {al['n_pairs']} (animal, day) "
              f"cells holding both contexts, identical across both arms",
              n_effective=len(cohort))
    cr.assert_cardinality(len(cohort), EXPECTED_COHORT_N, what="animals")
    reads["cohort"] = cr.to_dict()
    log("  " + cr.line())

    animals = al["animal"]
    for name in ("island", "matched"):
        d = al["arms"][name]["diff"]
        ci = boot.animal_interval(d, animals, how="mean", n_boot=co.N_BOOT,
                                  seed=SEED)
        fl = boot.pair_flip_null(d, al["pair_key"], n_perm=co.N_PERM,
                                 seed=SEED)
        reads[f"delta|{name}"] = Read(
            "NOT_A_RESULT", {**obj, "arm": f"delta_{name}"},
            f"descriptive, the raw arm delta: {name} occupancy B-A "
            f"{ci['point']:+.5f} [{ci['lo']:+.5f}, {ci['hi']:+.5f}], "
            f"pair-flip p = {fl['p_two_sided']:.4f}. The nested residual is "
            f"the verdict, not this", n_effective=len(cohort),
            detail={"ci": dict(ci), "flip": dict(fl),
                    "mean_A": al["arms"][name]["mean_A"],
                    "mean_B": al["arms"][name]["mean_B"]}).to_dict()
        log("  " + Read.from_dict(reads[f"delta|{name}"]).line())

    rc = 0
    if bal.verdict != "PASS":
        log("  THE ARM IS REFUSED. No residual is read: an unbalanced matched "
            "control cannot answer the question it was built for.")
        rc = 1
    else:
        beta, r = co.residual(al["arms"]["island"]["diff"],
                              al["arms"]["matched"]["diff"])
        within_sd = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
        mde = sx.mde_read(within_sd, int(r.size),
                          plausible_effect=cx.PLAUSIBLE_EFFECT,
                          scored_object={**obj, "arm": "mde"},
                          n_effective=len(cohort))
        reads["mde"] = mde.to_dict()
        log("  " + mde.line())
        if mde.verdict != "PASS":
            log("  GATE DID NOT PASS; no residual is read, the MDE is the "
                "result")
            rc = 1
        else:
            ci = boot.animal_interval(r, animals, how="mean",
                                      n_boot=co.N_BOOT, seed=SEED)
            fl = boot.pair_flip_null(r, al["pair_key"], n_perm=co.N_PERM,
                                     seed=SEED)
            rd = co.controls_read(ci, fl, arm="matched", beta=beta,
                                  scored_object={**obj, "control": "matched"},
                                  n_effective=len(cohort))
            reads["residual|matched"] = rd.to_dict()
            log("  " + rd.line())

    write_json({**provenance.header(anchors.LUNA, stage="stillness_control",
                                 unverified="a control on a published "
                                 "contrast"),
                "inherited_digest": spine.digest(),
                "registration":
                    "results/STILLNESS_CONTROL_PREREGISTRATION.md",
                "group": a.group, "clump": a.clump, "k_mad": a.k_mad,
                "pose_arm": POSE_ARM, "seed": SEED, "days": list(cx.DAYS),
                "n_targets": int(t_idx.size), "n_matched": int(ok.sum()),
                "n_pairs": al["n_pairs"], "n_animals": len(cohort),
                "balance_bound": co.BALANCE_SMD,
                "reads": reads, "pair_key": al["pair_key"],
                "deltas": {k: al["arms"][k]["diff"].tolist()
                           for k in al["arms"]}}, out)
    log(f"  wrote {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
