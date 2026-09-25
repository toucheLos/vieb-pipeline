"""Is the island's context effect just a speed effect?

    python3 scripts/context_controls.py --group shape --clump 0

READ results/CONTEXT_CONTROLS_PREREGISTRATION.md FIRST. The MDE gate runs
before any residual: if the smallest detectable effect exceeds the registered
plausible one (0.02 of occupancy, inherited from FREEZING_PREREGISTRATION.md
§4), NO residual is read and the MDE is the result.

`CONTEXT.md` found clump-0 occupancy differs by context within animal and day.
`BEHAVIOUR.md` found the same clump 3.7x slower than its animals' other
segments. This asks whether the first survives the second.

Three arms on ONE denominator -- that cell's total selectable segment frames:

* `island`    clump-0 segment frames. The published arm; its delta is not
              recomputed to a different value, it is reproduced.
* `stillness` frames whose OWN speed is below a matched threshold. Never
              consults a boundary.
* `windows`   length- and count-matched random windows. Never consults speed
              or a clump.

The statistic is nested, not three deltas side by side, and the residual is
fitted through the origin so that it CAN fail. See `vieb/seg/controls.py`.
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
from vieb import seeds                                                # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import breaks as bk, context as cx, controls as co      # noqa: E402
from vieb.seg import embed                                            # noqa: E402
from vieb.tok import config, quantize as qz                           # noqa: E402

SEED = 0
EXPECTED_COHORT_N = 89
POSE_ARM = "raw"


def shard(group: str, k_mad: float) -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_vocab",
                        f"{group}__k{k_mad:g}__corpus.npz")


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
    out = a.out or config.PATHS.result("context_controls.json")

    with np.load(shard(a.group, a.k_mad), allow_pickle=False) as z:
        if "rec" not in z.files or "frame" not in z.files:
            raise SystemExit("segment shard predates the locator fix; "
                             "re-run seg_vocab first")
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr = z["n_frames"].astype(np.int64)
        rec = z["rec"].astype(np.int64)
        frame = z["frame"].astype(np.int64)

    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))

    # Resolve each segment's recording, then its day and context. `rec` indexes
    # the animal's OWN recording list, never a pooled position.
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
                    raise SystemExit(
                        f"[{tag}] shard says pose_arm={got!r}, expected "
                        f"{POSE_ARM!r}. A low-pass filter manufactures the "
                        f"smoothness whose breaks this rests on")
                rid_cache[tag] = [str(v) for v in e["recording_ids"]]
        parsed = lab.parse(rid_cache[tag][int(rec[i])])
        days.append(int(parsed["day"]))
        ctxs.append(str(parsed["context_letter"]))
    days_a = np.asarray(days, dtype=np.int64)
    ctxs_a = np.asarray(ctxs)

    # --- arm 1: the island, exactly as CONTEXT.md built it ------------------
    occ_isl, den = cx.cell_occupancy(labels, nfr, animal, days_a, ctxs_a,
                                     clump=a.clump)
    num_isl = co.cell_sums(np.where(labels == a.clump, nfr, 0),
                           animal, days_a, ctxs_a)
    in_cell = np.asarray([int(d) in cx.DAYS and str(c) in cx.CONTEXTS
                          for d, c in zip(days_a, ctxs_a)])
    log(f"  {int(in_cell.sum()):,} segments in scored cells, "
        f"{int((labels[in_cell] == a.clump).sum()):,} in clump {a.clump}")

    target = (sum(num_isl.values()) / sum(den.values())
              if sum(den.values()) > 0 else float("nan"))
    log(f"  corpus-wide clump-{a.clump} occupancy in these cells: "
        f"{target:.6f}")

    # --- arm 2: stillness. Per-frame speed, never a boundary ----------------
    # Pass one collects every scored frame's own speed so the threshold can be
    # MATCHED to the island's occupancy rather than chosen. Pass two counts.
    tags = sorted({animal[i] for i in range(labels.size) if in_cell[i]})
    speed_cache: dict[str, np.ndarray] = {}
    pool: list[np.ndarray] = []
    for tag in tags:
        with np.load(ego_path(tag), allow_pickle=False) as e:
            sp = qz.speed(np.asarray(e["X"], dtype=np.float64))
        speed_cache[tag] = sp
    for i in np.flatnonzero(in_cell):
        sp = speed_cache[animal[i]]
        pool.append(sp[int(frame[i]):int(frame[i]) + int(nfr[i])])
    allsp = np.concatenate(pool) if pool else np.zeros(0)
    theta_still = co.match_threshold(allsp, target)
    realised = float(np.mean(allsp < theta_still)) if allsp.size else float("nan")
    log(f"  theta_still = {theta_still:.6f} body lengths/s "
        f"(realised below-share {realised:.6f} against target {target:.6f})")

    still_n = np.zeros(labels.size, dtype=np.float64)
    zero_n = np.zeros(labels.size, dtype=np.float64)
    m_still: list[np.ndarray] = []
    m_isl: list[np.ndarray] = []
    for i in np.flatnonzero(in_cell):
        sp = speed_cache[animal[i]]
        seg = sp[int(frame[i]):int(frame[i]) + int(nfr[i])]
        still_n[i] = float(np.count_nonzero(seg < theta_still))
        # Exactly zero ego speed is the whole pose HELD -- a tracking dropout,
        # not a slow animal. Counted separately because it turns out to be a
        # large share of what the stillness arm selects.
        zero_n[i] = float(np.count_nonzero(seg == 0.0))
        m_still.append(seg < theta_still)
        m_isl.append(np.full(seg.size, bool(labels[i] == a.clump)))
    sel_mask = np.concatenate(m_still) if m_still else np.zeros(0, bool)
    isl_mask = np.concatenate(m_isl) if m_isl else np.zeros(0, bool)
    occ_still = co.occupancy(
        co.cell_sums(still_n, animal, days_a, ctxs_a), den)

    # --- arm 3: length-matched random windows. Never speed, never a clump ---
    # Drawn per animal from that animal's own clump-0 duration distribution,
    # count matched, avoiding abstain and never straddling a recording seam --
    # the function the repo already uses and already asserts about.
    win_num: dict[tuple[str, int, str], float] = {}
    n_win = 0
    for tag in tags:
        mine = np.flatnonzero((animal == tag) & in_cell)
        durs = nfr[mine[labels[mine] == a.clump]]
        if durs.size == 0:
            continue
        with np.load(ego_path(tag), allow_pickle=False) as e:
            bounds = e["bounds"].astype(np.int64)
            valid = e["valid"].astype(bool)
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as e:
            abstain = e["abstain"].astype(bool) | ~valid
        rng = np.random.default_rng(seeds.stable_seed(SEED, f"win|{tag}"))
        wins = embed.matched_windows(rng, bounds=bounds, abstain=abstain,
                                     durations=durs, n_target=int(durs.size))
        for w in wins:
            rec_i = int(np.searchsorted(bounds, int(w["start"]),
                                        "right") - 1)
            parsed = lab.parse(rid_cache[tag][rec_i])
            key = (tag, int(parsed["day"]), str(parsed["context_letter"]))
            if key[1] not in cx.DAYS or key[2] not in cx.CONTEXTS:
                continue
            win_num[key] = win_num.get(key, 0.0) + float(w["n_frames"])
            n_win += 1
    occ_win = co.occupancy(win_num, den)
    log(f"  {n_win:,} length-matched windows drawn")

    # --- the deltas, cell for cell across all three arms --------------------
    al = co.aligned_deltas({"island": occ_isl, "stillness": occ_still,
                            "windows": occ_win}, den)
    cohort = sorted({k.split("|")[0] for k in al["pair_key"]})
    obj = {"dataset": "luna", "arm": "context_controls", "group": a.group,
           "clump": a.clump, "pose_arm": POSE_ARM, "split": "report",
           "contrast": "CFD d3-7 context B minus A", "days": list(cx.DAYS)}
    cohort_read = Read("PASS", obj,
                       f"{len(cohort)} report animals contribute at least one "
                       f"(animal, day) cell holding both contexts; "
                       f"{al['n_pairs']} cells, identical across all three "
                       f"arms by construction and asserted",
                       n_effective=len(cohort))
    cohort_read.assert_cardinality(len(cohort), EXPECTED_COHORT_N,
                                   what="animals")
    log("  " + cohort_read.line())

    d_isl = al["arms"]["island"]["diff"]
    animals = al["animal"]
    reads: dict = {"cohort": cohort_read.to_dict()}
    for name in ("island", "stillness", "windows"):
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
            f"the verdict, not this",
            n_effective=len(cohort),
            detail={"ci": dict(ci), "flip": dict(fl),
                    "mean_A": al["arms"][name]["mean_A"],
                    "mean_B": al["arms"][name]["mean_B"]}).to_dict()
        log("  " + Read.from_dict(reads[f"delta|{name}"]).line())

    # --- does the control select anything like the island? ------------------
    # Added after seeing theta_still, and recorded in DEVIATIONS.md as such.
    # Without it a reader takes a residual that excludes zero at face value,
    # when the arms may simply be measuring different things.
    ov = co.overlap_read(sel_mask, isl_mask, arm="stillness",
                         theta=theta_still,
                         scored_object={**obj, "arm": "overlap_stillness"},
                         n_effective=len(cohort))
    reads["overlap|stillness"] = ov.to_dict()
    log("  " + ov.line())
    vacuous = "NOT A CONTROL" in ov.reason

    # A held pose is a tracking dropout, not a still animal. If dropout itself
    # moves with context then EVERY occupancy in this design partly measures
    # tracking, the published island delta included.
    occ_zero = co.occupancy(co.cell_sums(zero_n, animal, days_a, ctxs_a), den)
    alz = co.aligned_deltas({"island": occ_isl, "zero": occ_zero}, den)
    dz = alz["arms"]["zero"]["diff"]
    ciz = boot.animal_interval(dz, animals, how="mean", n_boot=co.N_BOOT,
                               seed=SEED)
    flz = boot.pair_flip_null(dz, alz["pair_key"], n_perm=co.N_PERM, seed=SEED)
    bz, rz = co.residual(d_isl, dz)
    cir = boot.animal_interval(rz, animals, how="mean", n_boot=co.N_BOOT,
                               seed=SEED)
    corr = float(np.corrcoef(d_isl, dz)[0, 1]) if d_isl.size > 1 else float("nan")
    held = Read(
        "NOT_A_RESULT", {**obj, "arm": "held_frames"},
        (f"UNREGISTERED DIAGNOSTIC, added after the fact: frames whose whole "
         f"pose is HELD -- exactly zero ego speed, a tracking dropout -- sit "
         f"at {alz['arms']['zero']['mean_A']:.5f} of segment frames in "
         f"context A against {alz['arms']['zero']['mean_B']:.5f} in B, a "
         f"difference of {ciz['point']:+.5f} [{ciz['lo']:+.5f}, "
         f"{ciz['hi']:+.5f}], pair-flip p = {flz['p_two_sided']:.4f}. "
         f"TRACKING QUALITY DIFFERS BY CONTEXT, in the same direction as the "
         f"island. Across cells the two deltas correlate only {corr:+.3f} and "
         f"the island residual on held frames is {cir['point']:+.5f} "
         f"[{cir['lo']:+.5f}, {cir['hi']:+.5f}], so dropout does not explain "
         f"the island cell for cell -- but no occupancy in this design is "
         f"free of it"),
        n_effective=len(cohort),
        detail={"ci": dict(ciz), "flip": dict(flz),
                "mean_A": alz["arms"]["zero"]["mean_A"],
                "mean_B": alz["arms"]["zero"]["mean_B"],
                "island_residual_on_held": dict(cir), "beta": bz,
                "corr_island_held": corr}).to_dict()
    reads["held_frames"] = held
    log("  " + Read.from_dict(held).line())

    # --- THE GATE, before any residual is read ------------------------------
    ok = True
    for name in ("stillness", "windows"):
        beta, r = co.residual(d_isl, al["arms"][name]["diff"])
        within_sd = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
        mde = sx.mde_read(within_sd, int(r.size),
                          plausible_effect=cx.PLAUSIBLE_EFFECT,
                          scored_object={**obj, "arm": f"mde_{name}"},
                          n_effective=len(cohort))
        reads[f"mde|{name}"] = mde.to_dict()
        log("  " + mde.line())
        if mde.verdict != "PASS":
            log(f"  GATE DID NOT PASS for {name}; no residual is read and the "
                f"MDE is the result")
            ok = False
            continue
        ci = boot.animal_interval(r, animals, how="mean", n_boot=co.N_BOOT,
                                  seed=SEED)
        fl = boot.pair_flip_null(r, al["pair_key"], n_perm=co.N_PERM,
                                 seed=SEED)
        rd = co.controls_read(ci, fl, arm=name, beta=beta,
                              scored_object={**obj, "control": name},
                              n_effective=len(cohort))
        reads[f"residual|{name}"] = rd.to_dict()
        log("  " + rd.line())
        if name == "stillness" and vacuous:
            log("  ^^ VACUOUS: the arm above selects near-none of the "
                "island's frames, so this verdict is not about the detector")

    write_json({**provenance.header(anchors.LUNA, stage="context_controls",
                                 unverified="a control on a published "
                                 "contrast; no corpus count is recomputed"),
                "inherited_digest": spine.digest(),
                "registration": "results/CONTEXT_CONTROLS_PREREGISTRATION.md",
                "group": a.group, "clump": a.clump, "k_mad": a.k_mad,
                "pose_arm": POSE_ARM, "seed": SEED,
                "days": list(cx.DAYS), "contexts": list(cx.CONTEXTS),
                "n_pairs": al["n_pairs"], "n_animals": len(cohort),
                "theta_still": theta_still,
                "theta_still_target_occupancy": target,
                "theta_still_realised_share": realised,
                "n_windows": n_win,
                "stillness_arm_vacuous": vacuous,
                "published_island_delta": {
                    "source": "results/CONTEXT.md",
                    "point": -0.008114690381327477,
                    "lo": -0.015197907015411752,
                    "hi": -0.0024324024793461172},
                "reads": reads,
                "pair_key": al["pair_key"],
                "deltas": {k: al["arms"][k]["diff"].tolist()
                           for k in al["arms"]}}, out)
    log(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
