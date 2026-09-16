"""Step 2. Do segments recur across animals? Q1's statistic, segments as unit.

    sbatch jobs/seg_recur.slurm --group both      # one job per channel group
    sbatch jobs/seg_recur.slurm --group shape
    sbatch jobs/seg_recur.slurm --group twist
    python3 scripts/seg_recur.py --combine

READ results/SEGRECUR_PREREGISTRATION.md FIRST. It was committed before any bank
or search existed and it fixes the gate: if the segment excess sits BELOW a
length-matched windowed control in the same space, the boundaries cut through
behaviours rather than between them and Step 1's criterion is wrong.

## Why one job per group and not one per animal

The bank is global -- a cross-animal nearest neighbour needs every animal's
units in the same space -- so the unit of work is a channel group, not an
animal. The ego arrays for 298 animals are about 1.5 GB per arm, the feature
matrices are built one arm at a time and freed, and the search is a tiled matmul
on the GPU.

## The six banks per group, and why the comparator is half of them

Three arms {corpus, microstate, microstate0} x two unit types {segment, window}.
`paired_excess(corpus_segments, null_segments)` alone cannot answer the
question: a bank of ANY units cut from a real mouse beats a surrogate. What is
being asked is whether the BOUNDARIES contributed, and that is the same excess
recomputed with length-matched random windows in place of segments.
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

from recur import anchors, labels as lab, splits                    # noqa: E402
from recur.recurrence import bank as bk_, search as se              # noqa: E402
from recur.util import frames, log, peak_rss_gb, write_json         # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seg import breaks as bk, embed, planted as pl, recur as rc  # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.tok import config                                         # noqa: E402

POSE_ARM, SCALE_ARM = "raw", "bodylen"
DEGREE, DERIV_SEC = 3, 0.133
ARMS: tuple[str, ...] = ("corpus",) + rc.NULLS
UNITS: tuple[str, ...] = ("segment", "window")
SEED = 0
#: Q1's, inherited rather than chosen here.
NULL_RATE = 0.01
#: **No reduction. Every arm is embedded at its full dimension.**
#:
#: Q1 reduced because its bank held 7.45M windows; these banks hold about
#: 200,000 units and the reduction buys nothing. It also cost something real: at
#: d = 192 the `both` group's microstate arms failed `pca_read` at Spearman rho
#: 0.977-0.988 against a 0.99 requirement, while the corpus passed -- and
#: raising only the failing arms would have embedded the corpus in 192
#: dimensions and its null in 384. The threshold comes from the NULL's own
#: quantile, so two arms in differently-shaped spaces make theta incomparable
#: for reasons unrelated to recurrence. A rotation to the full dimension
#: preserves every distance exactly and keeps all arms in one space.
#:
#: The ladder is kept, with the full dimension as its only rung, so the PCA
#: validity read still runs and still has to pass rather than being skipped.
DIM_LADDER: tuple[int, ...] = (10_000,)
PCA_SAMPLE = 200_000
RAND_PAIRS = 2_000_000
RAND_SAMPLE = 400_000
EXACT_BANK, EXACT_QUERIES = 6000, 300


def out_dir() -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_recur")


def ego_path(arm: str, tag: str) -> str:
    name = f"{POSE_ARM}__{SCALE_ARM}__{tag}.npz"
    if arm == "corpus":
        return os.path.join(config.REPO, "work", "ego", name)
    return os.path.join(config.REPO, "work", "surrogate", arm, "ego", name)


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def load_arm(arm: str, tag: str, sd: np.ndarray) -> dict:
    """One animal's ego array, standardised, with its pose arm asserted."""
    with np.load(ego_path(arm, tag), allow_pickle=False) as z:
        if arm == "corpus":
            got = str(z["pose_arm"]) if "pose_arm" in z.files else "<absent>"
            if got != POSE_ARM:
                raise SystemExit(
                    f"[{tag}] shard says pose_arm={got!r}, expected "
                    f"{POSE_ARM!r}. A low-pass filter manufactures the "
                    f"smoothness whose breaks this detects, so the arm is "
                    f"asserted rather than assumed")
        out = {"X": np.asarray(z["X"], dtype=np.float64) / sd[None, :],
               "bounds": z["bounds"].astype(np.int64),
               "valid": z["valid"].astype(bool),
               "recording_ids": [str(v) for v in z["recording_ids"]]}
    p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
    with np.load(p, allow_pickle=False) as z:
        out["abstain"] = z["abstain"].astype(bool) | ~out["valid"]
    return out


def segments_of(a: dict, idx, *, fps: float, k_mad: float) -> list:
    """Detected segments for one animal, with edge provenance attached."""
    h = max(2, frames(DERIV_SEC, fps))
    guard = bk.guard_frames(fps, deriv_sec=DERIV_SEC)
    min_gap = bk.min_segment_frames(DEGREE, guard)
    sub = a["X"][:, list(idx)]
    rows: list = []
    for r in range(a["bounds"].shape[0] - 1):
        lo, hi = int(a["bounds"][r]), int(a["bounds"][r + 1])
        d = bk.discontinuity(sub[lo:hi], h)
        pk = bk.boundaries(d, bk.mad_threshold(d, k_mad), min_gap=min_gap,
                           blocked=a["abstain"][lo:hi])
        tab = bk.segment_table(a["X"], pk + lo, lo=lo, hi=hi,
                               abstain=a["abstain"], guard=guard,
                               degree=DEGREE, fps=fps)
        for row in embed.mark_edges(tab, a["abstain"], lo=lo, hi=hi):
            rows.append({**row, "rec": r})
    return rows


def units_for(a: dict, idx, *, fps: float, k_mad: float, unit: str,
              rng: np.random.Generator) -> tuple[list, dict]:
    """Segments, or the length-matched windowed control built from them."""
    rows = segments_of(a, idx, fps=fps, k_mad=k_mad)
    keep = embed.selectable(rows)
    sel = [r for r, k in zip(rows, keep) if k]
    counts = {"n_segments": len(rows), "n_selected": len(sel),
              "n_abstain_adjacent": int(sum(1 for r in rows
                                            if r["abstain_adjacent"])),
              "n_seam_adjacent": int(sum(1 for r in sel if r["seam_adjacent"]))}
    if unit == "segment":
        return sel, counts
    lens = np.asarray([int(r["n_frames"]) for r in sel], dtype=np.int64)
    win = embed.matched_windows(rng, bounds=a["bounds"], abstain=a["abstain"],
                                durations=lens, n_target=len(sel))
    for w in win:
        w["rec"] = int(np.searchsorted(a["bounds"], w["start"], "right") - 1)
    counts["n_windows"] = len(win)
    return win, counts


#: The floor is planted into the NULL, never into the corpus.
#:
#: The first run planted into the corpus and produced a dose-response that ran
#: BACKWARDS -- +5.93% at 0.25% occupancy falling monotonically to +2.51% at
#: 10%. The reason is that the corpus background already scores +6.09% against
#: this null, so every planted cell measured the corpus's own recurrence while
#: the planting slowly overwrote it. A floor has to sit on a background with no
#: recurrence in it, which is what Q1 does: its ladder is
#: `ar-planted_wiener_occ<X>` scored against `ar_wiener`.
PLANT_BASE = rc.NULLS[0]
#: Animals the planted template is averaged over. ONE template for the whole
#: corpus, not one per animal.
#:
#: The first run built the template inside the per-animal loop, so every animal
#: received a DIFFERENT stereotype. The statistic is CROSS-ANIMAL recurrence, so
#: a per-animal template plants a signal the measurement cannot see, and the
#: ladder recovered nothing at any occupancy -- going negative with dose, because
#: planting only overwrote whatever real structure the null retained. Caught by
#: the control returning nothing, which is what the control is for.
PLANT_DONORS = 3
PLANT_TEMPLATE_S = 0.5


def plant_template(tags, idx, *, fps: float, sd) -> np.ndarray:
    """One stereotype for the whole corpus, averaged over a few donor animals.

    Built from **real** windows rather than invented, so its spectrum lands in
    the band the corpus occupies: ledger 23 records a control that planted its
    structure in the band the pipeline deletes and so measured the filter.
    """
    blocks, ok = [], []
    for tag in list(tags)[:PLANT_DONORS]:
        a = load_arm(PLANT_BASE, tag, sd)
        blocks.append(a["X"])
        ok.append(~a["abstain"])
    x = np.concatenate(blocks)
    w = frames(PLANT_TEMPLATE_S, fps)
    rng = np.random.default_rng(seeds.stable_seed(SEED, "plant_template"))
    return pl.ego_template(x, w, rng, valid=np.concatenate(ok))


def build_arm(arm: str, tags, idx, *, fps: float, sd, unit: str, k_mad: float,
              plant_occ: float | None = None,
              template: np.ndarray | None = None) -> dict:
    """Every animal's units for one arm, stacked into one synthetic array."""
    blocks, meta, offset = [], [], 0
    counts: dict = {}
    plant_meta: list = []
    for tag in tags:
        base = PLANT_BASE if plant_occ is not None else arm
        a = load_arm(base, tag, sd)
        if plant_occ is not None:
            if template is None:
                raise SystemExit("planting needs the corpus-wide template: a "
                                 "per-animal one plants a signal a CROSS-animal "
                                 "statistic cannot see")
            rng_p = np.random.default_rng(
                seeds.stable_seed(SEED, f"plant{plant_occ:g}", tag))
            a["X"], _mask, pm = pl.ego_plant(
                a["X"], template, rng_p, bounds=a["bounds"],
                occupancy=plant_occ, blocked=a["abstain"])
            plant_meta.append(pm)
        rng = np.random.default_rng(seeds.stable_seed(SEED, f"{arm}{unit}", tag))
        rows, c = units_for(a, idx, fps=fps, k_mad=k_mad, unit=unit, rng=rng)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        if rows:
            feat, _s = embed.stack_segments(a["X"], rows, cols=idx,
                                            n=embed.SEG_GRID)
            blocks.append(feat)
            for r in rows:
                meta.append((tag, int(r["rec"]), int(r["start"]),
                             int(r["n_frames"]), offset))
                offset += embed.SEG_GRID
        del a
    if not blocks:
        raise SystemExit(f"{arm}/{unit}: no units at all")
    Xs = np.concatenate(blocks)
    del blocks
    out = {"X": Xs,
           "start": np.asarray([m[4] for m in meta], dtype=np.int64),
           "animal": np.asarray([m[0] for m in meta]),
           "rec": np.asarray([m[1] for m in meta], dtype=np.int32),
           "frame": np.asarray([m[2] for m in meta], dtype=np.int64),
           "n_frames": np.asarray([m[3] for m in meta], dtype=np.int64),
           "counts": counts}
    if plant_meta:
        out["planted"] = {
            "requested_occupancy": plant_occ,
            "realized_occupancy": float(np.mean([p["realized_occupancy"]
                                                 for p in plant_meta])),
            "n_instances": int(sum(p["n_instances"] for p in plant_meta))}
    return out


def subsample(b: dict, n_units: int, tag: str) -> dict:
    """Cut an arm's bank down to `n_units`, uniformly at random.

    **A declared post-hoc control, not the registered primary.** The registered
    design handles ambient scale per arm but not bank SIZE, and the first run
    showed `microstate` producing 25% fewer segments than the corpus on the
    shape group. Nearest-neighbour distance falls as a bank grows through
    extreme-value effects alone, and because theta is set from the NULL's own
    quantile, a smaller null bank inflates the excess in the direction that
    flatters the corpus.

    Uniform rather than stratified per animal: the statistic is a mean over
    per-animal rates, which is unbiased under uniform thinning, and stratifying
    would introduce a second choice where a result is read.
    """
    m = int(b["start"].size)
    if n_units >= m:
        return b
    rng = np.random.default_rng(seeds.stable_seed(SEED, "matchbank", tag))
    take = np.sort(rng.choice(m, size=int(n_units), replace=False))
    keep = np.concatenate([b["start"][take][:, None]
                           + np.arange(embed.SEG_GRID)[None, :]]).ravel()
    out = {"X": b["X"][keep],
           "start": np.arange(take.size, dtype=np.int64) * embed.SEG_GRID}
    for k in ("animal", "rec", "frame", "n_frames"):
        out[k] = b[k][take]
    for k in ("counts", "planted"):
        if k in b:
            out[k] = b[k]
    return out


def search_arm(b: dict, idx, *, split_of, device: str, tag: str,
               exclusion: int) -> dict:
    """PCA on tune units, bank, exact leave-one-animal-out search."""
    cols = np.arange(b["X"].shape[1])
    split = np.asarray([split_of.get(a, "?") for a in b["animal"]])
    tune = np.flatnonzero(split == "tune")
    if tune.size < 100:
        raise SystemExit(f"{tag}: only {tune.size} tune units; the PCA basis "
                         f"would be fitted where a result is read")
    full = embed.SEG_GRID * len(idx)
    tried: list[dict] = []
    pca = pr = None
    dim = full
    for rung in DIM_LADDER:
        dim = int(min(rung, full))
        pca = bk_.fit_pca(b["X"], b["start"][tune], embed.SEG_GRID, cols,
                          n_components=dim, sample=PCA_SAMPLE, seed=SEED,
                          device=device)
        obj = {"dataset": "luna", "arm": tag, "dim": dim, "split": "report",
               "unit_grid": embed.SEG_GRID, "pose_arm": POSE_ARM}
        pr = bk_.pca_read(b["X"], b["start"][tune], embed.SEG_GRID, cols, pca,
                          scored_object=obj,
                          n_effective=len(set(b["animal"][tune].tolist())),
                          seed=SEED)
        tried.append({"dim": dim, "verdict": pr.verdict})
        log(f"    d={dim}: " + pr.line())
        if pr.verdict == "PASS" or dim >= full:
            break
    assert pca is not None and pr is not None
    if pr.verdict != "PASS":
        raise SystemExit(
            f"{tag}: the PCA reduction still distorts the distances at the "
            f"full dimension {dim}, which should be impossible. The statistic "
            f"is made of those distances, so nothing here is readable")
    B = bk_.build(b["X"], b["start"], embed.SEG_GRID, cols, pca=pca)
    (B, an, rec, frame, nfr, spl), _o = bk_.sort_by_animal(
        B, b["animal"], b["rec"], b["frame"], b["n_frames"], split)
    uniq = {a: i for i, a in enumerate(sorted(set(an.tolist())))}
    code = np.asarray([uniq[a] for a in an], dtype=np.int32)
    q = np.flatnonzero(spl == "report").astype(np.int64)
    log(f"    {B.shape[0]:,} units, {q.size:,} report queries, d={dim}")
    res = se.search(B, code, rec.astype(np.int32), frame, query_idx=q,
                    exclusion=int(exclusion), device=device, log=None)
    # The ambient scale: random cross-animal pairs, no NN feedback.
    rng = np.random.default_rng(SEED)
    take = np.sort(rng.choice(B.shape[0], size=min(RAND_SAMPLE, B.shape[0]),
                              replace=False))
    i = rng.integers(0, take.size, RAND_PAIRS)
    j = rng.integers(0, take.size, RAND_PAIRS)
    ok = an[take][i] != an[take][j]
    dr = np.linalg.norm(B[take][i[ok]] - B[take][j[ok]], axis=1)
    return {"d_cross": res["d_cross"].astype(np.float32),
            "d_within": res["d_within"].astype(np.float32),
            "i_cross": res["i_cross"],
            "q_animal": an[q], "q_len": nfr[q],
            "nn_len": np.where(res["i_cross"] >= 0, nfr[np.maximum(
                res["i_cross"], 0)], -1).astype(np.int64),
            "scale": float(np.median(dr)),
            "pca_read": pr.to_dict(), "dim": dim, "dim_ladder": tried,
            "n_bank": int(B.shape[0]), "n_queries": int(q.size),
            "exactness": exactness(B, code, rec, frame, q, device, exclusion)}


def exactness(B, code, rec, frame, q, device: str, exclusion: int) -> dict:
    """The GPU search against the float64 CPU reference, before any excess."""
    # A CONTIGUOUS slice, because the search requires the bank sorted by
    # animal and loops over contiguous animal blocks. It has to be placed where
    # the report queries are: the bank is animal-sorted, report animals do not
    # sit at the front, and slicing [:n] gave a sub-bank with no query in it --
    # an INCONCLUSIVE that looked like a property of the data.
    n = min(EXACT_BANK, B.shape[0])
    if q.size == 0:
        return {"verdict": "INCONCLUSIVE", "why": "no report queries"}
    s0 = int(np.clip(int(q[q.size // 2]) - n // 2, 0, B.shape[0] - n))
    qs = (q[(q >= s0) & (q < s0 + n)] - s0)[:EXACT_QUERIES]
    if qs.size < 10:
        return {"verdict": "INCONCLUSIVE",
                "why": f"only {qs.size} report queries inside the reference slice"}
    sl = slice(s0, s0 + n)
    kw = dict(query_idx=qs, exclusion=int(exclusion))
    g = se.search(B[sl], code[sl], rec[sl].astype(np.int32), frame[sl],
                  device=device, **kw)
    r = se.search_reference(B[sl], code[sl], rec[sl].astype(np.int32),
                            frame[sl], **kw)
    rd = se.exactness_read(g, r, scored_object={"dataset": "luna",
                                                "arm": "seg_recur",
                                                "n_bank": int(n),
                                                "slice_start": int(s0)},
                           n_effective=int(qs.size))
    log("    " + rd.line())
    return rd.to_dict()


def no_tf32() -> dict:
    """Refuse TF32. An A100 runs float32 matmuls at 10 mantissa bits by default.

    The exactness gate caught it: one arm's `d_within` disagreed with the
    float64 CPU reference by 1.87e-04 against a 1e-04 tolerance, and
    `exactness_read`'s own text names TF32 as the first suspect. The Gram form
    loses precision by cancellation exactly where the distance is small, which
    is where the entire statistic lives, so ten mantissa bits is not enough.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"allow_tf32": False,
             "matmul_precision": torch.get_float32_matmul_precision()}


def run_group(args) -> int:
    os.makedirs(out_dir(), exist_ok=True)
    tf32 = no_tf32()
    log(f"  TF32 disabled: {tf32}")
    fps = spine.fps()
    sd = basis_sd()
    idx = bk.CHANNEL_GROUPS[args.group]
    suffix = "__matched" if args.n_units else ""
    tags = sorted(animals_of(spine.recording_ids()))
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    guard = bk.guard_frames(fps, deriv_sec=DERIV_SEC)
    excl = bk.min_segment_frames(DEGREE, guard)
    meta_path = os.path.join(out_dir(),
                             f"{args.group}__k{args.k_mad:g}{suffix}.json")
    if args.planted_only:
        with open(meta_path, encoding="utf-8") as fh:
            out = json.load(fh)
        args.planted = True
    else:
        out = {"group": args.group, "k_mad": args.k_mad, "arms": {},
               "matched_bank": int(args.n_units) if args.n_units else None}
    for unit in ([] if args.planted_only else UNITS):
        for arm in ARMS:
            name = f"{arm}|{unit}"
            log(f"  [{args.group}] {name}")
            b = build_arm(arm, tags, idx, fps=fps, sd=sd, unit=unit,
                          k_mad=args.k_mad)
            counts = b.pop("counts")
            if args.n_units:
                b = subsample(b, int(args.n_units), name)
            r = search_arm(b, idx, split_of=split_of, device=args.device,
                           tag=name, exclusion=excl)
            del b
            np.savez_compressed(
                os.path.join(out_dir(),
                             f"{args.group}__k{args.k_mad:g}{suffix}"
                             f"__{arm}__{unit}.npz"),
                **{k: v for k, v in r.items()
                   if isinstance(v, np.ndarray)})
            out["arms"][name] = {**{k: v for k, v in r.items()
                                    if not isinstance(v, np.ndarray)},
                                 "counts": counts}
            log(f"    peak_rss={peak_rss_gb():.1f} GB")
    if args.planted:
        tpl = plant_template(tags, idx, fps=fps, sd=sd)
        out["planted"] = {"planted_into": PLANT_BASE,
                          "scored_against": f"unplanted {PLANT_BASE}",
                          "template_donors": PLANT_DONORS,
                          "template_s": PLANT_TEMPLATE_S,
                          "one_template_corpus_wide": True}
        for occ in pl.OCCUPANCIES:
            log(f"  [{args.group}] planted occ={occ:g}")
            b = build_arm("corpus", tags, idx, fps=fps, sd=sd, unit="segment",
                          k_mad=args.k_mad, plant_occ=occ, template=tpl)
            b.pop("counts")
            pmeta = b.pop("planted")
            r = search_arm(b, idx, split_of=split_of, device=args.device,
                           tag=f"planted{occ:g}", exclusion=excl)
            del b
            np.savez_compressed(
                os.path.join(out_dir(),
                             f"{args.group}__k{args.k_mad:g}__planted{occ:g}"
                             f"__segment.npz"),
                **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
            out["planted"][f"occ{occ:g}"] = {
                **pmeta, **{k: v for k, v in r.items()
                            if not isinstance(v, np.ndarray)}}
    write_json({**anchors.header(anchors.LUNA, stage="seg_recur",
                                 unverified="one channel group per job"),
                "inherited_digest": spine.digest(),
                "registration": "results/SEGRECUR_PREREGISTRATION.md",
                "pose_arm": POSE_ARM, "scale_arm": SCALE_ARM,
                "degree": DEGREE, "deriv_sec": DERIV_SEC,
                "exclusion_frames": int(excl), "null_rate": NULL_RATE,
                "float32_matmul": tf32,
                "peak_rss_gb": peak_rss_gb(), **out},
               meta_path)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default=None, choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--k-mad", type=float, default=rc.K_MAD_PRIMARY)
    p.add_argument("--device", default="cuda")
    p.add_argument("--planted", action="store_true")
    p.add_argument("--n-units", type=int, default=None,
                   help="subsample every arm's bank to this many units; the "
                        "declared post-hoc bank-size control")
    p.add_argument("--planted-only", action="store_true",
                   help="re-run only the planted ladder, merging into the "
                        "group JSON that already holds the six arms")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.k_mad <= rc.K_MAD_FLOOR:
        raise SystemExit(
            f"k_mad={a.k_mad} is at or below the registered floor "
            f"{rc.K_MAD_FLOOR}. Below it the detector saturates -- the closed "
            f"gate's four nulls agreed to 0.40% at 53% of the NMS ceiling -- "
            f"and a MAD-standardised threshold penalises a heavy-tailed D "
            f"there by construction. SEGRECUR_PREREGISTRATION.md excludes it")
    if a.combine:
        from vieb.seg import combine_recur
        a.out = a.out or config.PATHS.result("seg_recur.json")
        return combine_recur.main(a, out_dir())
    if a.group:
        if a.device != "cuda":
            raise SystemExit("--device cuda is asserted, never auto-detected: "
                             "there is no CPU fallback that finishes")
        return run_group(a)
    raise SystemExit("pass --group or --combine")


if __name__ == "__main__":
    raise SystemExit(main())
