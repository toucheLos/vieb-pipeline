"""Stage 3: does the island split, and on which channel?

    python3 scripts/island_split.py

READ results/VALIDATION_PREREGISTRATION.md §3 FIRST.

`BEHAVIOUR.md`'s `chaining_clump0` FAILs -- typical pair 0.398 against
theta = 0.190, 2.1x -- so clump 0 is a single-linkage chain and must not be
named as one behaviour. A chain is the shape of something with sub-types, and
this asks whether those sub-types are visible in **configuration**
(`body_extension`: crouched freeze against stretch-attend) or in **dynamics**
(`pole_radius`).

Scored with `vocab.gmm1d_bic_gain` **against a duration-matched non-clump-0
null from the same animals**, not against a threshold: any large set of
segments has some bimodality, and the question is whether clump 0 has more.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                      # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.audit import triage                                        # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb.clean import arms as clean_arms                             # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import controls as co, descriptors as ds, vocab as vc   # noqa: E402
from vieb.tok import config                                           # noqa: E402

SEED = 0
POSE_ARM = "raw"
CLUMP = 0
GROUP = "shape"


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("island_split.json")
    fps = spine.fps()

    shard = os.path.join(config.PATHS.tok_dir, "seg_vocab",
                         f"{GROUP}__k3__corpus.npz")
    with np.load(shard, allow_pickle=False) as z:
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr = z["n_frames"].astype(np.int64)
        rec = z["rec"].astype(np.int64)
        frame = z["frame"].astype(np.int64)
    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    keep = np.asarray([split_of.get(t) == "report" for t in animal])
    log(f"  {int(keep.sum()):,} report segments, "
        f"{int((labels[keep] == CLUMP).sum()):,} in clump {CLUMP}")

    ext = np.full(labels.size, np.nan)
    rad = np.full(labels.size, np.nan)
    tags = sorted({animal[i] for i in np.flatnonzero(keep)})
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float64)
            bounds = z["bounds"].astype(np.int64)
            rids = [str(v) for v in z["recording_ids"]]
        e = np.full(X.shape[0], np.nan)
        for r, rid in enumerate(rids):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            e[lo:hi] = triage.body_extension(held)
        mine = np.flatnonzero(keep & (animal == tag))
        for i in mine:
            s, t = int(frame[i]), int(frame[i]) + int(nfr[i])
            seg_e = e[s:t]
            ext[i] = float(np.nanmean(seg_e)) if seg_e.size else np.nan
            blk = X[s:t][:, :14]
            if blk.shape[0] >= 16:
                pf = ds.pole_features(blk, fps=fps)
                with np.errstate(invalid="ignore"):
                    rad[i] = float(np.nanmedian(pf["radius"]))
        if n % 10 == 0 or n == len(tags):
            log(f"  {n}/{len(tags)} animals")

    is_isl = keep & (labels == CLUMP)
    usable = keep & np.isfinite(ext) & np.isfinite(rad)
    feats = np.column_stack([np.log(np.maximum(nfr, 1) / fps),
                             np.zeros(labels.size)])
    partners = co.matched_partners(usable & (labels == CLUMP), animal, feats,
                                   pool=usable & (labels != CLUMP))
    t_idx = np.flatnonzero(usable & (labels == CLUMP))
    ok = partners >= 0
    log(f"  {int(ok.sum())} of {t_idx.size} island segments duration-matched")

    obj = {"dataset": "luna", "arm": "island_split", "split": "report",
           "clump": CLUMP, "pose_arm": POSE_ARM, "group": GROUP}
    reads: dict = {}
    table: dict = {}
    for name, vals in (("body_extension", ext), ("pole_radius", rad)):
        isl = vals[t_idx[ok]]
        null = vals[partners[ok]]
        g_i = vc.gmm1d_bic_gain(isl, seed=SEED)
        g_n = vc.gmm1d_bic_gain(null, seed=SEED)
        gi = float(g_i.get("bic_gain", float("nan")))
        gn = float(g_n.get("bic_gain", float("nan")))
        table[name] = {"island": g_i, "null": g_n, "island_gain": gi,
                       "null_gain": gn, "excess": gi - gn,
                       "island_median": float(np.nanmedian(isl)),
                       "null_median": float(np.nanmedian(null)),
                       "n": int(ok.sum())}
        beats = np.isfinite(gi) and np.isfinite(gn) and gi > gn
        rd = Read(
            "PASS" if beats else "FAIL", {**obj, "channel": name},
            (f"clump 0 {'IS' if beats else 'is NOT'} more bimodal in "
             f"{name} than its duration-matched partners: BIC gain "
             f"{gi:+.1f} against the null's {gn:+.1f} over {int(ok.sum())} "
             f"pairs. A positive gain alone is not evidence -- any large set "
             f"of segments has some bimodality -- which is why the null is "
             f"the same statistic on the same animals' own segments"),
            n_effective=len(tags), detail=table[name])
        reads[name] = rd.to_dict()
        log("  " + rd.line())

    cfg = np.isfinite(table["body_extension"]["excess"]) and \
        table["body_extension"]["excess"] > 0
    dyn = np.isfinite(table["pole_radius"]["excess"]) and \
        table["pole_radius"]["excess"] > 0
    if cfg and not dyn:
        v, why = "PASS", ("THE ISLAND SPLITS ON CONFIGURATION, as registered: "
                          "body_extension is more bimodal than the null and "
                          "pole_radius is not. Crouched freeze against "
                          "stretch-attend is a posture")
    elif dyn and not cfg:
        v, why = "PASS", ("THE ISLAND SPLITS ON DYNAMICS, which is the "
                          "outcome registered as novel: pole_radius is more "
                          "bimodal than the null and body_extension is not")
    elif cfg and dyn:
        v, why = "INCONCLUSIVE", ("both channels beat their nulls, so the "
                                  "split is not attributable to one of them")
    else:
        v, why = "FAIL", ("the island does not split on either channel "
                          "against a duration-matched null. On the registered "
                          "reading it is a unitary slow freezing state")
    hd = Read(v, {**obj, "arm": "which_channel"},
              why + f" (body_extension excess "
              f"{table['body_extension']['excess']:+.1f}, pole_radius excess "
              f"{table['pole_radius']['excess']:+.1f})",
              n_effective=len(tags), detail=table)
    reads["which_channel"] = hd.to_dict()
    log("  " + hd.line())

    write_json({**provenance.header(anchors.LUNA, stage="island_split",
                                 unverified="a post-hoc split on a published "
                                 "clump"),
                "inherited_digest": spine.digest(),
                "registration": "results/VALIDATION_PREREGISTRATION.md",
                "split": "report", "clump": CLUMP, "seed": SEED,
                "n_animals": len(tags), "n_pairs": int(ok.sum()),
                "channels": table, "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
