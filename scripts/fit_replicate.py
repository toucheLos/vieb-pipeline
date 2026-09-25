"""Step 1: transfer the island to 149 held-out animals, and test the transfer.

    python3 scripts/fit_replicate.py --prep
    python3 scripts/fit_replicate.py --shard I --of N
    python3 scripts/fit_replicate.py --combine

READ results/FIT_REPLICATION_PREREGISTRATION.md FIRST.

Every context claim in this programme rests on 439 cells from 89 `report`
animals. This asks whether the island's context effect survives on the 149
`fit` animals, which have never been touched.

## The island is transferred, not re-derived

Re-running single linkage on a different population gives different components,
and nothing would guarantee a new "clump 0" is the same object. So the frozen
detector runs on `fit`, each segment is resampled to the same 40-point stack,
and it is assigned by distance to the nearest of the **361 report clump-0
members**, normalised by the **report** arm's ambient scale so `theta` means
what it meant.

**No PCA is refitted.** `seg_vocab`'s bank goes through a PCA fitted on `tune`
at `n_components = 40 x 14 = 560` -- the full rank -- so it is a rotation and
Euclidean distances are invariant. Verified against 7 stored neighbour pairs to
six decimal places before this was written.

## What the control set is, and why it is the TARGET that is matched

§3's rationale is that a fit segment landing inside `theta` of a report member
"could reflect a shared low-speed regime rather than the same state". Testing
that requires varying **which report segments are the target**, not which fit
segments are the source: the control is **361 non-island report segments matched
to the island on joint (log duration, log mean speed)**, and the question is
whether fit segments land near the island more often than near an equally slow,
equally long set. §3's wording, "length-and-speed-matched fit segments", is
ambiguous; its stated logic is not, and the target reading is the one that makes
it true. Recorded in `DEVIATIONS.md` D15.
"""
from __future__ import annotations

import argparse
import glob
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
from vieb.seg import breaks as bk, context as cx, controls as co      # noqa: E402
from vieb.seg import embed                                            # noqa: E402
from vieb.tok import config, quantize as qz                           # noqa: E402

SEED = 0
POSE_ARM = "raw"
GROUP = "shape"
CLUMP = 0
#: Inherited from `work/tok/seg_vocab/shape__k3.json`. Neither is re-derived.
THETA = 0.18998060778738327
SCALE = 25.319997787475586
#: Registered floor: below this the cell occupancies cannot carry a contrast.
MIN_TRANSFER = 200
EXPECTED_COHORT_N = 149
N_BOOT = 2000


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "fit_replicate")


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def _sr():
    """`scripts/seg_recur.py`, imported not copied -- the detector of record."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_recur.py")
    spec = importlib.util.spec_from_file_location("sr_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["seg_recur"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def _stacks(tag: str, frame, nfr, idx, sd, cache) -> np.ndarray:
    if tag not in cache:
        with np.load(ego_path(tag), allow_pickle=False) as z:
            cache[tag] = np.asarray(z["X"], dtype=np.float64) / sd[None, :]
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
    X = cache[tag][:, idx]
    return np.asarray([embed.resample_segment(X, int(f), int(f) + int(n),
                                              embed.SEG_GRID).ravel()
                       for f, n in zip(frame, nfr)], dtype=np.float32)


def prep(a) -> int:
    """Build the two target banks from `report`: the island, and its match."""
    os.makedirs(work_dir(), exist_ok=True)
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    fps = spine.fps()
    shard = os.path.join(config.PATHS.tok_dir, "seg_vocab",
                         f"{GROUP}__k3__corpus.npz")
    with np.load(shard, allow_pickle=False) as z:
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr = z["n_frames"].astype(np.int64)
        frame = z["frame"].astype(np.int64)

    cache: dict = {}
    speed = np.full(labels.size, np.nan)
    for tag in sorted(set(animal.tolist())):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            sp = qz.speed(np.asarray(z["X"], dtype=np.float64))
        for i in np.flatnonzero(animal == tag):
            seg = sp[int(frame[i]):int(frame[i]) + int(nfr[i])]
            speed[i] = float(np.nanmean(seg)) if seg.size else np.nan
    feats = np.column_stack([np.log(np.maximum(nfr, 1) / fps),
                             np.log(np.maximum(speed, 1e-9))])
    usable = np.isfinite(feats).all(axis=1)
    is_isl = usable & (labels == CLUMP)
    partners = co.matched_partners(is_isl, animal, feats, pool=usable
                                   & (labels != CLUMP))
    t_idx = np.flatnonzero(is_isl)
    ok = partners >= 0
    bal = co.balance_read(feats[t_idx[ok]], feats[partners[ok]],
                          names=("log_duration_s", "log_mean_speed_bl_s"),
                          scored_object={"dataset": "luna",
                                         "arm": "fit_replicate_balance"},
                          n_effective=int(ok.sum()))
    log("  " + bal.line())

    out: dict = {"balance": bal.to_dict(), "n_island": int(t_idx.size),
                 "n_matched": int(ok.sum())}
    banks = {}
    for name, sel in (("island", t_idx[ok]), ("control", partners[ok])):
        rows = []
        for tag in sorted({animal[i] for i in sel}):
            mine = [i for i in sel if animal[i] == tag]
            rows.append(_stacks(tag, frame[mine], nfr[mine], idx, sd, cache))
        banks[name] = np.concatenate(rows, axis=0)
        log(f"  {name} bank {banks[name].shape}")
    np.savez_compressed(os.path.join(work_dir(), "targets.npz"),
                        island=banks["island"], control=banks["control"])
    write_json(out, os.path.join(work_dir(), "prep.json"))
    log(f"  wrote {work_dir()}/targets.npz")
    return 0 if bal.verdict == "PASS" else 1


def shard(a) -> int:
    os.makedirs(work_dir(), exist_ok=True)
    tg = np.load(os.path.join(work_dir(), "targets.npz"), allow_pickle=False)
    banks = {k: np.asarray(tg[k], dtype=np.float32) for k in ("island",
                                                              "control")}
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    fps = spine.fps()
    sr = _sr()
    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    by: dict[str, list[str]] = {}
    for rid in spine.recording_ids():
        t = lab.animal_tag(rid)
        if split_of.get(t) == "fit":
            by.setdefault(t, []).append(rid)
    tags = sorted(by)[a.shard::a.of]
    log(f"  shard {a.shard}/{a.of}: {len(tags)} fit animals")

    rows: list[dict] = []
    cache: dict = {}
    for n, tag in enumerate(tags, 1):
        arm = sr.load_arm("corpus", tag, sd)
        segs = sr.segments_of(arm, idx, fps=fps, k_mad=bk.K_MAD)
        keep = embed.selectable(segs)
        sel = [r for r, k in zip(segs, keep) if k]
        if not sel:
            continue
        with np.load(ego_path(tag), allow_pickle=False) as z:
            sp = qz.speed(np.asarray(z["X"], dtype=np.float64))
        frame = np.asarray([int(r["start"]) for r in sel], dtype=np.int64)
        nfr = np.asarray([int(r["n_frames"]) for r in sel], dtype=np.int64)
        S = _stacks(tag, frame, nfr, idx, sd, cache)
        d = {}
        for k, B in banks.items():
            g = (S ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] \
                - 2.0 * (S @ B.T)
            d[k] = np.sqrt(np.maximum(g, 0.0)).min(axis=1) / SCALE
        for m, r in enumerate(sel):
            seg = sp[int(frame[m]):int(frame[m]) + int(nfr[m])]
            rows.append({"animal": tag, "rec": int(r["rec"]),
                         "frame": int(frame[m]), "n_frames": int(nfr[m]),
                         "mean_speed": float(np.nanmean(seg)) if seg.size
                         else float("nan"),
                         "d_island": float(d["island"][m]),
                         "d_control": float(d["control"][m])})
        log(f"  {n}/{len(tags)} {tag}: {len(sel)} selectable, "
            f"{int((d['island'] <= THETA).sum())} transfer")
    out = os.path.join(work_dir(), f"shard_{a.shard:03d}.json")
    write_json({"shard": a.shard, "of": a.of, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


def selfcheck(a) -> int:
    """DIAGNOSTIC, not a result: does the transfer RULE carry the contrast?

    A null on `fit` has two readings and the registration cannot separate them:
    the island is sample-specific, or the transfer rule does not carry the
    contrast that clump membership carried. So the identical rule is run back
    over the 89 `report` animals -- **leave-one-animal-out**, so no segment can
    match an island member from its own animal -- and the resulting occupancy
    contrast is compared with the published -0.00811 that clump LABELS gave on
    the same animals, the same days and the same cells.

    MEASURED, AND THE ANSWER IS A TAUTOLOGY THAT IS WORTH HAVING. The rule
    selects exactly the 361 labelled members: no member missed, no non-member
    admitted. That cannot be evidence, because `theta` IS the single-linkage
    merge height that defined clump 0 -- a non-member within `theta` of a member
    would have BEEN a member -- so agreement was guaranteed before it was
    computed and no unfaithful rule could have been detected here.

    What it does establish is sharper than corroboration: the transfer rule is
    not a lossy stand-in for clump membership, it is **the membership criterion
    itself**, applied one step. A fit segment transfers exactly when adding it
    to the report bank would have placed it in clump 0. So a null on `fit` is
    about the animals rather than about a degraded instrument.

    **One honest gap.** Single linkage on a combined report+fit bank could also
    admit a fit segment that is far from every report member but close to
    another fit segment that is close to one. This rule does not chain through
    new points, so it is the CONSERVATIVE version of membership and can only
    under-count. It cannot manufacture a null by admitting the wrong segments;
    it could hide members that only chaining would reach.

    NOT_A_RESULT by construction. It scores no hypothesis. Registration §7
    forbids re-reading the report contrast, and this does not re-read it -- the
    incumbent is quoted as the fixed value it already was.
    """
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    shard_p = os.path.join(config.PATHS.tok_dir, "seg_vocab",
                           f"{GROUP}__k3__corpus.npz")
    with np.load(shard_p, allow_pickle=False) as z:
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr = z["n_frames"].astype(np.int64)
        frame = z["frame"].astype(np.int64)
        rec = z["rec"].astype(np.int64) if "rec" in z else None
    if rec is None:
        raise SystemExit("corpus shard carries no `rec`; cannot resolve context")
    isl = labels == CLUMP
    cache: dict = {}
    stacks = np.zeros((labels.size, embed.SEG_GRID * len(idx)), dtype=np.float32)
    tags = sorted(set(animal.tolist()))
    for n, tag in enumerate(tags, 1):
        m = np.flatnonzero(animal == tag)
        stacks[m] = _stacks(tag, frame[m], nfr[m], idx, sd, cache)
        if n % 20 == 0:
            log(f"  stacked {n}/{len(tags)}")
    d = np.full(labels.size, np.inf)
    for tag in tags:
        src = np.flatnonzero(animal == tag)
        tgt = np.flatnonzero(isl & (animal != tag))      # leave-one-animal-out
        if not tgt.size:
            continue
        S, B = stacks[src], stacks[tgt]
        g = (S ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (S @ B.T)
        d[src] = np.sqrt(np.maximum(g, 0.0)).min(axis=1) / SCALE
    tr = d <= THETA
    log(f"  report: {int(isl.sum())} labelled island, {int(tr.sum())} "
        f"transferred by rule, {int((tr & isl).sum())} both")

    rid_cache: dict = {}
    days, ctxs = [], []
    for i in range(labels.size):
        t = animal[i]
        if t not in rid_cache:
            with np.load(ego_path(t), allow_pickle=False) as e:
                rid_cache[t] = [str(v) for v in e["recording_ids"]]
        p = lab.parse(rid_cache[t][int(rec[i])])
        days.append(int(p["day"]))
        ctxs.append(str(p["context_letter"]))
    days_a = np.asarray(days, dtype=np.int64)
    ctxs_a = np.asarray(ctxs)

    out_reads: dict = {}
    got: dict = {}
    for name, mask in (("rule", tr), ("labels", isl)):
        occ, den = cx.cell_occupancy(np.where(mask, CLUMP, CLUMP + 1), nfr,
                                     animal, days_a, ctxs_a, clump=CLUMP)
        al = co.aligned_deltas({"x": occ}, den)
        ci = boot.animal_interval(al["arms"]["x"]["diff"], al["animal"],
                                  how="mean", n_boot=N_BOOT, seed=SEED)
        got[name] = {"ci": dict(ci), "n_pairs": al["n_pairs"],
                     "mean_A": al["arms"]["x"]["mean_A"],
                     "mean_B": al["arms"]["x"]["mean_B"],
                     "n_selected": int(mask.sum())}
        log(f"  {name:6s} B-A {ci['point']:+.5f} [{ci['lo']:+.5f}, "
            f"{ci['hi']:+.5f}]  A {got[name]['mean_A']:.5f} "
            f"B {got[name]['mean_B']:.5f}  n={int(mask.sum())}")
    rr, ll = got["rule"]["ci"], got["labels"]["ci"]
    only_rule = int((tr & ~isl).sum())
    only_lab = int((isl & ~tr).sum())
    ident = only_rule == 0 and only_lab == 0
    rd = Read("NOT_A_RESULT",
              {"dataset": "luna", "arm": "transfer_rule_selfcheck",
               "split": "report", "group": GROUP, "clump": CLUMP,
               "theta": THETA},
              (f"DIAGNOSTIC, AND ITS AGREEMENT IS A TAUTOLOGY: on the 89 report "
               f"animals the rule selects {int(tr.sum())} segments against "
               f"{int(isl.sum())} labelled members, "
               f"{only_rule} admitted that the labels exclude and {only_lab} "
               f"members missed, so the two contrasts are "
               f"{'IDENTICAL' if ident else 'different'} "
               f"({rr['point']:+.5f} [{rr['lo']:+.5f}, {rr['hi']:+.5f}] against "
               f"{ll['point']:+.5f} [{ll['lo']:+.5f}, {ll['hi']:+.5f}]). This "
               f"could not have come out otherwise: theta IS the single-linkage "
               f"merge height that defined the clump, so no unfaithful rule was "
               f"detectable here. What it establishes is that the rule is the "
               f"membership criterion itself rather than a lossy stand-in -- "
               f"applied one step, without chaining through new points, so it "
               f"can only UNDER-count on fit and cannot manufacture a null. "
               f"Scores no hypothesis"),
              n_effective=len(tags),
              detail={**got, "identical_to_labels": ident,
                      "n_rule_only": only_rule, "n_label_only": only_lab,
                      "agreement_is_tautological": True,
                      "chaining": "not performed; conservative"})
    out_reads["selfcheck"] = rd.to_dict()
    log("  " + rd.line())
    out = a.out or config.PATHS.result("fit_replicate_selfcheck.json")
    write_json({**provenance.header(anchors.LUNA, stage="fit_replicate_selfcheck",
                                 unverified="a post-hoc instrument diagnostic"),
                "inherited_digest": spine.digest(),
                "diagnostic": "NOT a result; see DEVIATIONS.md D10",
                "theta": THETA, "scale": SCALE, "reads": out_reads}, out)
    log(f"  wrote {out}")
    return 0


def combine(a) -> int:
    paths = sorted(glob.glob(os.path.join(work_dir(), "shard_*.json")))
    if not paths:
        raise SystemExit("no shards")
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.load(fh)["rows"])
    animal = np.asarray([r["animal"] for r in rows])
    d_i = np.asarray([r["d_island"] for r in rows], dtype=np.float64)
    d_c = np.asarray([r["d_control"] for r in rows], dtype=np.float64)
    nfr = np.asarray([r["n_frames"] for r in rows], dtype=np.int64)
    rec = np.asarray([r["rec"] for r in rows], dtype=np.int64)
    frame = np.asarray([r["frame"] for r in rows], dtype=np.int64)
    tr_i, tr_c = d_i <= THETA, d_c <= THETA
    log(f"  {len(rows):,} fit segments; island {int(tr_i.sum()):,} transfer, "
        f"matched control {int(tr_c.sum()):,}")

    obj = {"dataset": "luna", "arm": "fit_replicate", "split": "fit",
           "pose_arm": POSE_ARM, "group": GROUP, "clump": CLUMP,
           "theta": THETA, "scale": SCALE}
    reads: dict = {}
    tags = sorted(set(animal.tolist()))
    per_i, per_c, who = [], [], []
    for t in tags:
        m = animal == t
        if not m.any():
            continue
        per_i.append(float(tr_i[m].mean()))
        per_c.append(float(tr_c[m].mean()))
        who.append(t)
    ci_i = boot.animal_interval(per_i, who, how="mean", n_boot=N_BOOT,
                                seed=SEED)
    ci_c = boot.animal_interval(per_c, who, how="mean", n_boot=N_BOOT,
                                seed=SEED)
    sep = float(ci_i["lo"]) > float(ci_c["hi"])
    ind = Read(
        "PASS" if sep and int(tr_i.sum()) >= MIN_TRANSFER else "FAIL", obj,
        (f"{'the transfer is SPECIFIC to the island' if sep else 'THE TRANSFER RULE CAPTURES SLOWNESS, NOT THE ISLAND'}: "
         f"{ci_i['point']:.4f} [{ci_i['lo']:.4f}, {ci_i['hi']:.4f}] of fit "
         f"segments land within theta of an island member against "
         f"{ci_c['point']:.4f} [{ci_c['lo']:.4f}, {ci_c['hi']:.4f}] for a "
         f"duration-and-speed-matched set of non-island report segments, "
         f"{'non-overlapping' if sep else 'OVERLAPPING'}. "
         f"{int(tr_i.sum()):,} transferred against a registered floor of "
         f"{MIN_TRANSFER}"),
        n_effective=len(who),
        detail={"island_rate": dict(ci_i), "control_rate": dict(ci_c),
                "n_transferred": int(tr_i.sum()),
                "n_control_transferred": int(tr_c.sum()),
                "n_segments": len(rows), "min_transfer": MIN_TRANSFER})
    reads["independence"] = ind.to_dict()
    log("  " + ind.line())

    out = a.out or config.PATHS.result("fit_replicate.json")
    if ind.verdict != "PASS":
        log("  THE CONTRAST IS NOT READ. §3 stops the stage when the rule "
            "cannot be shown to select the island rather than slowness.")
        write_json({**provenance.header(anchors.LUNA, stage="fit_replicate",
                                     unverified="a held-out replication"),
                    "inherited_digest": spine.digest(),
                    "registration":
                        "results/FIT_REPLICATION_PREREGISTRATION.md",
                    "theta": THETA, "scale": SCALE, "reads": reads}, out)
        log(f"  wrote {out}")
        return 1

    rid_cache: dict = {}
    days, ctxs = [], []
    for i in range(len(rows)):
        t = animal[i]
        if t not in rid_cache:
            with np.load(ego_path(t), allow_pickle=False) as e:
                rid_cache[t] = [str(v) for v in e["recording_ids"]]
        p = lab.parse(rid_cache[t][int(rec[i])])
        days.append(int(p["day"]))
        ctxs.append(str(p["context_letter"]))
    days_a = np.asarray(days, dtype=np.int64)
    ctxs_a = np.asarray(ctxs)

    occ, den = cx.cell_occupancy(np.where(tr_i, CLUMP, CLUMP + 1), nfr,
                                 animal, days_a, ctxs_a, clump=CLUMP)
    al = co.aligned_deltas({"island": occ}, den)
    cohort = sorted({k.split("|")[0] for k in al["pair_key"]})
    cr = Read("PASS", obj,
              f"{len(cohort)} fit animals contribute a complete (animal, day) "
              f"cell on days 3-7; {al['n_pairs']} cells",
              n_effective=len(cohort))
    cr.assert_cardinality(len(cohort), EXPECTED_COHORT_N, what="animals")
    reads["cohort"] = cr.to_dict()
    log("  " + cr.line())

    d = al["arms"]["island"]["diff"]
    within_sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    mde = sx.mde_read(within_sd, int(d.size),
                      plausible_effect=cx.PLAUSIBLE_EFFECT,
                      scored_object={**obj, "arm": "mde"},
                      n_effective=len(cohort))
    reads["mde"] = mde.to_dict()
    log("  " + mde.line())
    rc = 0
    if mde.verdict != "PASS":
        log("  GATE DID NOT PASS; the MDE is the result")
        rc = 1
    else:
        ci = boot.animal_interval(d, al["animal"], how="mean", n_boot=N_BOOT,
                                  seed=SEED)
        fl = boot.pair_flip_null(d, al["pair_key"], n_perm=N_BOOT, seed=SEED)
        pub = -0.008114690381327477
        same = (float(ci["point"]) < 0) == (pub < 0)
        excl = float(ci["hi"]) < 0 or float(ci["lo"]) > 0
        v = "PASS" if (same and excl) else "FAIL"
        rd = Read(v, {**obj, "arm": "context"},
                  (f"{'THE ISLAND REPLICATES on held-out animals' if v == 'PASS' else 'the island does NOT replicate'}: "
                   f"fit occupancy B-A {ci['point']:+.5f} [{ci['lo']:+.5f}, "
                   f"{ci['hi']:+.5f}] over {len(cohort)} animals and "
                   f"{al['n_pairs']} cells, pair-flip p = "
                   f"{fl['p_two_sided']:.4f}, against the published report "
                   f"value of {pub:+.5f} [-0.01520, -0.00243]. Sign "
                   f"{'matches' if same else 'DIFFERS'}; interval "
                   f"{'excludes' if excl else 'includes'} zero"),
                  n_effective=len(cohort),
                  detail={"ci": dict(ci), "flip": dict(fl),
                          "published_report": pub,
                          "mean_A": al["arms"]["island"]["mean_A"],
                          "mean_B": al["arms"]["island"]["mean_B"]})
        reads["context"] = rd.to_dict()
        log("  " + rd.line())
        rc = 0 if v == "PASS" else 1

    write_json({**provenance.header(anchors.LUNA, stage="fit_replicate",
                                 unverified="a held-out replication"),
                "inherited_digest": spine.digest(),
                "registration": "results/FIT_REPLICATION_PREREGISTRATION.md",
                "split": "fit", "theta": THETA, "scale": SCALE, "seed": SEED,
                "n_segments": len(rows), "n_transferred": int(tr_i.sum()),
                "n_animals": len(cohort), "n_pairs": al["n_pairs"],
                "reads": reads,
                "deltas": al["arms"]["island"]["diff"].tolist(),
                "pair_key": al["pair_key"]}, out)
    log(f"  wrote {out}")
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prep", action="store_true")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--combine", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.prep:
        return prep(a)
    if a.selfcheck:
        return selfcheck(a)
    if a.combine:
        return combine(a)
    if a.shard is None:
        raise SystemExit("pass --prep, --shard I --of N, or --combine")
    return shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
