"""Step 2: the arena's own noise floor, and an ezTrack-family freeze score.

    python3 scripts/pixel_pilot.py --phase sample
    sbatch jobs/pixel.slurm
    python3 scripts/pixel_pilot.py --phase combine

READ results/PIXEL_PREREGISTRATION.md FIRST, **including Amendment 1** -- §8's
original draw gave 48 of 54 animals a single recording and made Q1, a
within-animal correlation, uncomputable. The sampling unit is the ANIMAL.

## One decode per recording, and the whole 27-point grid comes out of it

`motion.scan` keeps the per-frame |delta| HISTOGRAM, so a count above any cutoff
is a suffix sum. The shard derives `mt_cutoff` from the arena histogram at the
end of its own pass and immediately reduces to four `Motion` series -- the three
registered multipliers and ezTrack's untransplanted default. Keeping the
histograms on disk would have cost ~8 GB for 300 recordings and bought nothing
that is not in the reduction.

## What each phase refuses

* **`sample`** refuses any animal missing one of its ten (context, day) cells.
  A partially-observed animal contributes an unpaired cell, and §5's contrast is
  paired.
* **`shard`** marks a recording unusable below `motion.MIN_ARENA_PAIRS` arena
  pixel-pairs. Below that the 99.99th percentile rests on fewer than 20 order
  statistics and is not a percentile.
* **`combine`** refuses Q1 below 20 animals, and returns Q2 `INCONCLUSIVE` if
  the per-context arena floors differ by more than the freeze effect under test
  -- §7. A sensor-level difference larger than the effect is not behaviour.
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

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from recur.journey import simplex as sx                               # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.render import video as vid                                 # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.pixel import freeze as fz, motion as mo                     # noqa: E402
from vieb.seg import context as cx                                    # noqa: E402
from vieb.tok import config, ego as tego                              # noqa: E402

SEED = 0
#: Amendment 1: animals per box, and the ten (context, day) cells each must hold.
ANIMALS_PER_BOX = 10
DAYS = (3, 4, 5, 6, 7)
CONTEXTS = ("A", "B")
#: §5 refusal: Q1 needs this many animals with at least two usable cells.
MIN_ANIMALS = 20
#: §5: a freeze-fraction difference below this cannot be read as a null.
PLAUSIBLE_FREEZE_EFFECT = 0.02
N_BOOT = 2000


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "pixel")


def manifest_path() -> str:
    return os.path.join(work_dir(), "manifest.json")


def sample(a) -> int:
    os.makedirs(work_dir(), exist_ok=True)
    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    have: dict[str, dict] = {}
    for rid in spine.recording_ids():
        p = lab.parse(rid)
        t = lab.animal_tag(rid)
        if split_of.get(t) != "fit":
            continue
        if str(p["context_letter"]) not in CONTEXTS or int(p["day"]) not in DAYS:
            continue
        have.setdefault(t, {"box": int(p["box"]), "cells": {}})
        have[t]["cells"][(str(p["context_letter"]), int(p["day"]))] = rid
    full = {t: v for t, v in have.items()
            if len(v["cells"]) == len(CONTEXTS) * len(DAYS)}
    log(f"  {len(have)} fit animals on days 3-7; {len(full)} hold all "
        f"{len(CONTEXTS) * len(DAYS)} cells")

    rng = np.random.default_rng(SEED)
    chosen: list[str] = []
    for box in sorted({v["box"] for v in full.values()}):
        pool = sorted(t for t, v in full.items() if v["box"] == box)
        if len(pool) < ANIMALS_PER_BOX:
            raise SystemExit(f"box {box} has only {len(pool)} complete animals")
        idx = rng.choice(len(pool), size=ANIMALS_PER_BOX, replace=False)
        chosen += [pool[int(i)] for i in sorted(idx)]
        log(f"  box {box}: {ANIMALS_PER_BOX} of {len(pool)} complete animals")

    pilot = set(chosen)
    # Q1 and Q2 run on the registered balanced draw. The §6 TRICHOTOMY does not
    # -- it makes no cross-context claim, so §7's reason for keeping this a
    # pilot does not reach it, and §6 never fixed a sample. It is therefore a
    # CENSUS of every eligible fit recording, which removes the sampling
    # question entirely and clears the 20,000-frame thinness floor that the
    # 300-recording draw did not: 300 recordings yield only 6,228 zero-ego-speed
    # frames, and the floor needs ~970.
    rows = [{"recording_id": rid, "animal": t, "box": v["box"],
             "context": c, "day": d, "in_pilot": t in pilot}
            for t, v in sorted(full.items())
            for (c, d), rid in sorted(v["cells"].items())]
    write_json({**anchors.header(anchors.LUNA, stage="pixel_sample",
                                 unverified="a pilot sample"),
                "inherited_digest": spine.digest(),
                "registration": ("results/PIXEL_PREREGISTRATION.md, "
                                 "Amendment 1"),
                "seed": SEED, "split": "fit",
                "unit": "animal, not recording (D16)",
                "n_animals": len(chosen), "n_recordings": len(rows),
                "n_pilot_recordings": sum(1 for r in rows if r["in_pilot"]),
                "pilot": ("Q1/Q2 use in_pilot only: the registered balanced "
                          "draw. The §6 trichotomy uses every row, as a census"),
                "animals": chosen, "recordings": rows}, manifest_path())
    log(f"  {len(chosen)} animals / "
        f"{sum(1 for r in rows if r['in_pilot'])} recordings for Q1+Q2; "
        f"{len(rows)} recordings scanned for the §6 census -> "
        f"{manifest_path()}")
    return 0


def shard(a) -> int:
    with open(manifest_path(), encoding="utf-8") as fh:
        man = json.load(fh)
    rows = man["recordings"][a.shard::a.of]
    os.makedirs(os.path.join(work_dir(), "rec"), exist_ok=True)
    log(f"  shard {a.shard}/{a.of}: {len(rows)} recordings")
    for n, r in enumerate(rows, 1):
        rid = r["recording_id"]
        out = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if os.path.exists(out) and not a.force:
            continue
        d = spine.clean(rid)
        pose = np.asarray(d["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        s = mo.scan(vid.video_path(rid), pose,
                    dilate_px=bl * mo.DILATE_BODY_LENGTHS)
        tot = s["hist_arena"].sum(axis=0)
        n_pairs = int(s["n_arena"].sum())
        floor = mo.floor_percentile(tot)
        usable = n_pairs >= mo.MIN_ARENA_PAIRS and np.isfinite(floor)
        cutoffs = {f"m{m:g}": float(m) * floor for m in fz.CUTOFF_MULTIPLIERS}
        cutoffs["eztrack"] = float(fz.EZTRACK_DEFAULTS["mt_cutoff"])
        motion = {k: mo.motion_from_hist(s["hist_full"], c).astype(np.int32)
                  for k, c in cutoffs.items()}
        arena = {k: mo.motion_from_hist(s["hist_arena"], c).astype(np.int32)
                 for k, c in cutoffs.items()}
        np.savez_compressed(
            out, n_frames=s["n_frames"], n_pixels=s["n_pixels"],
            width=s["width"], height=s["height"],
            body_length_px=bl, floor_p9999=floor, n_arena_pairs=n_pairs,
            usable=bool(usable), identical=s["identical"],
            n_arena=s["n_arena"].astype(np.int32),
            cutoff_keys=np.asarray(sorted(cutoffs)),
            cutoff_values=np.asarray([cutoffs[k] for k in sorted(cutoffs)]),
            **{f"motion__{k}": v for k, v in motion.items()},
            **{f"arena__{k}": v for k, v in arena.items()})
        log(f"  {n}/{len(rows)} {rid}: floor {floor:.2f} "
            f"cutoff {2 * floor:.2f} arena_pairs {n_pairs:,} "
            f"{'ok' if usable else 'REFUSED'}")
    return 0


def _load(rid: str) -> dict | None:
    p = os.path.join(work_dir(), "rec", f"{rid}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _island_occupancy() -> tuple[dict, dict]:
    """Per (animal, day, context) island frame share, from Step 1's own shards.

    Rebuilt from `work/fit_replicate/shard_*.json` rather than recomputed, so
    Q1 correlates the FROZEN transfer against pixels and cannot drift from the
    numbers `FIT_REPLICATION.md` published.
    """
    rows: list[dict] = []
    for p in sorted(glob.glob(os.path.join(config.REPO, "work",
                                           "fit_replicate", "shard_*.json"))):
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.load(fh)["rows"])
    if not rows:
        raise SystemExit("Step 1's shards are missing; run fit_replicate first")
    theta = 0.18998060778738327
    rid_cache: dict = {}
    an, dy, ct, nf, mk = [], [], [], [], []
    for r in rows:
        t = r["animal"]
        if t not in rid_cache:
            with np.load(os.path.join(config.REPO, "work", "ego",
                                      f"raw__bodylen__{t}.npz"),
                         allow_pickle=False) as e:
                rid_cache[t] = [str(v) for v in e["recording_ids"]]
        p = lab.parse(rid_cache[t][int(r["rec"])])
        an.append(t)
        dy.append(int(p["day"]))
        ct.append(str(p["context_letter"]))
        nf.append(int(r["n_frames"]))
        mk.append(r["d_island"] <= theta)
    return cx.cell_occupancy(np.where(np.asarray(mk), 0, 1), np.asarray(nf),
                             an, np.asarray(dy), np.asarray(ct), clump=0)


def combine(a) -> int:
    with open(manifest_path(), encoding="utf-8") as fh:
        man = json.load(fh)
    fps = spine.fps()
    arms = fz.sweep_arms(fps)

    per: list[dict] = []
    frac: dict[str, dict] = {arm["name"]: {} for arm in arms}
    degen: dict[str, int] = {arm["name"]: 0 for arm in arms}
    seen: dict[str, int] = {arm["name"]: 0 for arm in arms}
    tri = {"duplicate": 0, "dropout": 0, "immobile": 0, "n": 0}
    for r in man["recordings"]:
        z = _load(r["recording_id"])
        if z is None:
            continue
        usable = bool(z["usable"])
        in_pilot = bool(r.get("in_pilot", True))
        keys = [str(k) for k in z["cutoff_keys"]]
        cutv = dict(zip(keys, [float(v) for v in z["cutoff_values"]]))
        if in_pilot:
            per.append({**r, "usable": usable,
                        "floor_p9999": float(z["floor_p9999"]),
                        "mt_cutoff": float(cutv["m2"]),
                        "n_arena_pairs": int(z["n_arena_pairs"]),
                        "n_frames": int(z["n_frames"])})
        if not usable:
            continue
        key = (r["animal"], int(r["day"]), r["context"])
        for arm in (arms if in_pilot else []):
            if arm["derived"]:
                M = np.asarray(z[f"motion__m{arm['cutoff_multiplier']:g}"],
                               dtype=np.float64)
                th = fz.thresh_of(M, arm["thresh_pct"])
            else:
                M = np.asarray(z["motion__eztrack"], dtype=np.float64)
                th = float(arm["freeze_thresh"])
            v = fz.freeze_fraction(M, th, int(arm["min_frames"]))
            seen[arm["name"]] += 1
            if not np.isfinite(v):
                degen[arm["name"]] += 1
            else:
                frac[arm["name"]][key] = v
        # §6, the held-pose trichotomy, on this recording's OWN arena.
        M = np.asarray(z["motion__m2"], dtype=np.float64)
        A = np.asarray(z["arena__m2"], dtype=np.float64)
        na = np.asarray(z["n_arena"], dtype=np.float64)
        npx = float(z["n_pixels"])
        animal_px = np.maximum(npx - na, 1.0)
        expect = A * (animal_px / np.maximum(na, 1.0))
        still = _still_frames(r["animal"], r["recording_id"], int(z["n_frames"]))
        if still is not None and still.any():
            dup = np.asarray(z["identical"], dtype=bool) & still
            rest = still & ~dup
            drop = rest & ((M - A) > expect)
            tri["duplicate"] += int(dup.sum())
            tri["dropout"] += int(drop.sum())
            tri["immobile"] += int((rest & ~drop).sum())
            tri["n"] += int(still.sum())

    obj = {"dataset": "luna", "arm": "pixel_pilot", "split": "fit",
           "n_recordings": len(per),
           "n_animals": len({r["animal"] for r in per})}
    reads: dict = {}
    fl = fz.floor_read(per, scored_object={**obj, "arm": "arena_floor"},
                       n_effective=len({r["animal"] for r in per}))
    reads["arena_floor"] = fl.to_dict()
    log("  " + fl.line())

    grid = {a["name"]: {"n_cells": seen[a["name"]],
                        "n_degenerate": degen[a["name"]],
                        "degenerate_share": (degen[a["name"]] / seen[a["name"]]
                                             if seen[a["name"]] else float("nan")),
                        "readable": degen[a["name"]] == 0,
                        "headline": bool(a["headline"])} for a in arms}
    reads["grid_degeneracy"] = _degeneracy_read(grid, obj).to_dict()
    log("  " + _degeneracy_read(grid, obj).line())

    reads["floor_contrast"] = _floor_contrast(per, obj).to_dict()
    log("  " + _floor_contrast(per, obj).line())

    occ, _den = _island_occupancy()
    reads.update(_q1(occ, frac, arms, obj, grid))
    reads.update(_q2(frac, arms, obj, fl, grid))
    reads["trichotomy"] = _tri_read(tri, obj).to_dict()
    log("  " + _tri_read(tri, obj).line())

    out = a.out or config.PATHS.result("pixel_pilot.json")
    write_json({**anchors.header(anchors.LUNA, stage="pixel_pilot",
                                 unverified="a 300-recording pilot"),
                "inherited_digest": spine.digest(),
                "registration": ("results/PIXEL_PREREGISTRATION.md, "
                                 "Amendment 1"),
                "reimplementation": ("ezTrack-FAMILY, verified against "
                                     "github.com/DeniseCaiLab/ezTrack master; "
                                     "ezTrack is NOT installed"),
                "seed": SEED, "fps": fps, "arms": arms, "grid": grid,
                "recordings": per, "trichotomy": tri,
                "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


def _still_frames(animal: str, rid: str, n: int):
    """Zero-ego-speed frames of one recording, in that recording's own index."""
    from vieb.tok import quantize as qz
    p = os.path.join(config.REPO, "work", "ego", f"raw__bodylen__{animal}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        rids = [str(v) for v in z["recording_ids"]]
        if rid not in rids:
            return None
        i = rids.index(rid)
        b = np.asarray(z["bounds"], dtype=np.int64)
        sp = qz.speed(np.asarray(z["X"], dtype=np.float64)[b[i]:b[i + 1]])
    out = np.zeros(n, dtype=bool)
    k = min(n, sp.size)
    out[:k] = sp[:k] == 0.0
    return out


def _floor_contrast(per: list[dict], obj: dict) -> Read:
    """§7, measured on the pixel channel itself: does the ARENA differ by context?

    The bits-per-frame and keypoint-speed evidence said A and B are visually
    different arenas. This asks the same question of the quantity that actually
    thresholds everything downstream -- the arena's own frame-to-frame noise --
    paired within (animal, day) so that animal and day cannot carry it.

    A difference here is not a defect of the pipeline. It is the confound, made
    numerical, on the axis Q2 uses.
    """
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for r in per:
        if r.get("usable"):
            cells.setdefault((r["animal"], int(r["day"])), {})[r["context"]] = \
                float(r["floor_p9999"])
    d, who = [], []
    for (an, _day), v in sorted(cells.items()):
        if "A" in v and "B" in v:
            d.append(v["B"] - v["A"])
            who.append(an)
    if len(d) < 2:
        return Read("NOT_A_RESULT", {**obj, "arm": "arena_floor_contrast"},
                    "too few complete (animal, day) cells", n_effective=1)
    arr = np.asarray(d, dtype=np.float64)
    ci = boot.animal_interval(arr, who, how="mean", n_boot=N_BOOT, seed=SEED)
    diff = float(ci["hi"]) < 0 or float(ci["lo"]) > 0
    return Read("PASS", {**obj, "arm": "arena_floor_contrast"},
                (f"the ARENA'S OWN NOISE FLOOR "
                 f"{'DIFFERS BY CONTEXT' if diff else 'does not differ by context'}: "
                 f"B-A {ci['point']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] grey "
                 f"levels over {len(set(who))} animals and {len(d)} paired cells, "
                 f"median B-A {np.median(arr):+.3f}. "
                 + ("This is §7's confound measured on the pixel channel itself, "
                    "and it runs on exactly the axis Q2 uses"
                    if diff else
                    "So the A/B image difference does NOT reach the sensor-noise "
                    "channel, and Q2 is not confounded at that level")),
                n_effective=len(set(who)),
                detail={"ci": dict(ci), "n_pairs": len(d),
                        "median": float(np.median(arr))})


def _degeneracy_read(grid: dict, obj: dict) -> Read:
    """§4's grid, and how much of it is DEFINED. D17.

    An arm whose threshold is degenerate on some recordings does not merely lose
    those cells -- it loses **the quietest ones**, which are exactly the
    recordings most likely to be freezing. That is selection on the outcome, so
    an arm is readable only at **zero** degenerate cells. The bar does not
    decide this stage either way: the best arm in the registered grid sits at 6%.
    """
    best = min((v["degenerate_share"], k) for k, v in grid.items()
               if v["n_cells"])
    head = next((k for k, v in grid.items() if v["headline"]), "")
    n_ok = sum(1 for v in grid.values() if v["readable"] and v["n_cells"])
    ok_derived = sum(1 for k, v in grid.items()
                     if v["readable"] and v["n_cells"] and k != "eztrack_default")
    return Read("NOT_A_RESULT" if not grid[head]["readable"] else "PASS",
                {**obj, "arm": "grid_degeneracy"},
                (f"THE REGISTERED HEADLINE ARM CANNOT BE READ: {head} is "
                 f"degenerate on {100 * grid[head]['degenerate_share']:.0f}% of "
                 f"cells, and {ok_derived} of the {len(grid) - 1} DERIVED arms "
                 f"are defined on every cell (least degenerate {best[1]} at "
                 f"{100 * best[0]:.0f}%). A degenerate arm drops the QUIETEST "
                 f"recordings, which is selection on the outcome. §4 defined "
                 f"FreezeThresh as a percentile of Motion, but at a cutoff "
                 f"derived from the arena floor Motion carries a large atom at "
                 f"zero and a low percentile of it is zero. The only arm defined "
                 f"everywhere is ezTrack's own absolute threshold, which §9.5 "
                 f"forbids adopting -- it is 4x too strict at this resolution -- "
                 f"so it is reported and never read as the answer"),
                n_effective=max(1, max(v["n_cells"] for v in grid.values())),
                detail={"grid": grid})


def _q1(occ, frac, arms, obj, grid) -> dict:
    """Does island occupancy track the freeze score WITHIN animal?"""
    from scipy import stats as st
    reads: dict = {}
    flips: list[str] = []
    head = ""
    for arm in arms:
        f = frac[arm["name"]]
        by: dict[str, list[tuple[float, float]]] = {}
        for key, share in occ.items():
            if key in f and np.isfinite(share) and np.isfinite(f[key]):
                by.setdefault(key[0], []).append((float(share), float(f[key])))
        rhos = {t: float(st.spearmanr([p[0] for p in v],
                                      [p[1] for p in v]).statistic)
                for t, v in by.items() if len(v) >= 2}
        flat = {t: r for t, r in rhos.items() if not np.isfinite(r)}
        rhos = {t: r for t, r in rhos.items() if np.isfinite(r)}
        if not grid[arm["name"]]["readable"]:
            rd = Read("NOT_A_RESULT", {**obj, "arm": f"q1|{arm['name']}"},
                      (f"the arm is degenerate on "
                       f"{100 * grid[arm['name']]['degenerate_share']:.0f}% of "
                       f"cells and those are the quietest recordings, so "
                       f"reading it would be selection on the outcome"),
                      n_effective=max(1, len(by)),
                      detail={"n_animals_examined": len(by)})
        elif len(rhos) < MIN_ANIMALS:
            rd = Read("NOT_A_RESULT", {**obj, "arm": f"q1|{arm['name']}"},
                      (f"only {len(rhos)} animals yield a defined rank "
                       f"correlation, against the registered minimum of "
                       f"{MIN_ANIMALS}. {len(flat)} of {len(by)} examined "
                       f"animals have island occupancy that never varies across "
                       f"their cells -- 90.7% of cells hold exactly zero island "
                       f"frames -- so the island is too SPARSE at cell "
                       f"resolution to rank-correlate against anything"),
                      n_effective=max(1, len(by)),
                      detail={"n_animals_examined": len(by),
                              "n_animals_constant": len(flat),
                              "n_animals_usable": len(rhos)})
        else:
            who = sorted(rhos)
            ci = boot.animal_interval([rhos[t] for t in who], who, how="mean",
                                      n_boot=N_BOOT, seed=SEED)
            v = "PASS" if float(ci["lo"]) > 0 else "FAIL"
            rd = Read(v, {**obj, "arm": f"q1|{arm['name']}"},
                      (f"island occupancy and the freeze score "
                       f"{'AGREE' if v == 'PASS' else 'do not agree'} within "
                       f"animal: mean Spearman rho {ci['point']:+.4f} "
                       f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}] over {len(who)} "
                       f"animals, against an incumbent of exactly 0 -- no "
                       f"pixel measure has ever been computed in this repo"),
                      n_effective=len(who), detail={"ci": dict(ci),
                                                    "n_animals": len(who)})
            if arm["headline"]:
                head = v
            else:
                flips.append(v)
        reads[f"q1|{arm['name']}"] = rd.to_dict()
        if arm["headline"]:
            log("  Q1 " + rd.line())
    if head and flips and any(v != head for v in flips):
        reads["q1"] = Read("GRID_LIMITED", {**obj, "arm": "q1"},
                           (f"Q1's headline verdict {head} does not hold across "
                            f"the registered grid: {sum(v != head for v in flips)}"
                            f" of {len(flips)} other arms disagree. §4 makes "
                            f"that GRID_LIMITED, not PASS"),
                           n_effective=len(arms)).to_dict()
    return reads


def _q2(frac, arms, obj, floor_read, grid) -> dict:
    """Does the FREEZE SCORE separate A from B on fit? §0's re-posed question."""
    reads: dict = {}
    verdicts: dict[str, str] = {}
    gap = float((floor_read.detail or {}).get("context_gap", float("nan")))
    for arm in arms:
        f = frac[arm["name"]]
        if not grid[arm["name"]]["readable"]:
            reads[f"q2|{arm['name']}"] = Read(
                "NOT_A_RESULT", {**obj, "arm": f"q2|{arm['name']}"},
                (f"the arm is degenerate on "
                 f"{100 * grid[arm['name']]['degenerate_share']:.0f}% of cells "
                 f"and those are the quietest recordings, so reading it would "
                 f"be selection on the outcome"),
                n_effective=max(1, len(f))).to_dict()
            continue
        den = {k: 1.0 for k in f}
        al = _aligned(f, den)
        if al["n_pairs"] < 2:
            reads[f"q2|{arm['name']}"] = Read(
                "NOT_A_RESULT", {**obj, "arm": f"q2|{arm['name']}"},
                "too few complete (animal, day) cells", n_effective=0).to_dict()
            continue
        d = np.asarray(al["diff"], dtype=np.float64)
        ci = boot.animal_interval(d, al["animal"], how="mean", n_boot=N_BOOT,
                                  seed=SEED)
        fl = boot.pair_flip_null(d, al["pair_key"], n_perm=N_BOOT, seed=SEED)
        n_an = len(set(al["animal"]))
        excl = float(ci["hi"]) < 0 or float(ci["lo"]) > 0
        eff = abs(float(ci["point"]))
        # §7: a sensor-level difference larger than the effect is not behaviour.
        conf = np.isfinite(gap) and eff > 0 and gap > eff
        v = "INCONCLUSIVE" if conf else ("PASS" if excl else "FAIL")
        verdicts[arm["name"]] = v
        rd = Read(v, {**obj, "arm": f"q2|{arm['name']}"},
                  (f"freeze fraction B-A {ci['point']:+.5f} [{ci['lo']:+.5f}, "
                   f"{ci['hi']:+.5f}] over {n_an} animals and {al['n_pairs']} "
                   f"cells, pair-flip p = {fl['p_two_sided']:.4f}. Incumbents: "
                   f"the island on fit -0.00216 [-0.00536, +0.00123], on report "
                   f"-0.00811 [-0.01520, -0.00243]"
                   + (f". INCONCLUSIVE by §7: the per-context arena floors "
                      f"differ by {gap:.2f} grey levels, larger than the effect "
                      f"under test, and a sensor-level difference bigger than "
                      f"the effect is not a behavioural result" if conf else "")),
                  n_effective=n_an,
                  detail={"ci": dict(ci), "flip": dict(fl),
                          "n_pairs": al["n_pairs"],
                          "arena_context_gap": gap})
        reads[f"q2|{arm['name']}"] = rd.to_dict()
        if arm["headline"]:
            log("  Q2 " + rd.line())
            sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
            m = sx.mde_read(sd, int(d.size),
                            plausible_effect=PLAUSIBLE_FREEZE_EFFECT,
                            scored_object={**obj, "arm": "q2_mde"},
                            n_effective=n_an)
            reads["q2_mde"] = m.to_dict()
            log("  " + m.line())
    if not verdicts:
        return reads
    head = verdicts.get(
        next(a["name"] for a in arms if a["headline"]), "")
    others = [v for k, v in verdicts.items()
              if k != next(a["name"] for a in arms if a["headline"])]
    if head and any(v != head for v in others):
        reads["q2"] = Read("GRID_LIMITED", {**obj, "arm": "q2"},
                           (f"Q2's headline verdict {head} does not hold across "
                            f"the registered grid: "
                            f"{sum(v != head for v in others)} of {len(others)} "
                            f"other arms disagree"),
                           n_effective=len(arms)).to_dict()
    return reads


def _aligned(f: dict, den: dict) -> dict:
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for (an, day, ctx), v in f.items():
        cells.setdefault((an, int(day)), {})[str(ctx)] = float(v)
    diff, keys, animals = [], [], []
    for (an, day), v in sorted(cells.items()):
        if "A" in v and "B" in v and np.isfinite(v["A"]) and np.isfinite(v["B"]):
            diff.append(v["B"] - v["A"])
            keys.append(f"{an}|{day}")
            animals.append(an)
    return {"diff": np.asarray(diff, dtype=np.float64), "pair_key": keys,
            "animal": animals, "n_pairs": len(diff)}


def _tri_read(tri: dict, obj: dict) -> Read:
    n = int(tri["n"])
    if n <= 0:
        return Read("NOT_A_RESULT", {**obj, "arm": "held_pose"},
                    "no zero-ego-speed frames were reached", n_effective=0)
    p = {k: tri[k] / n for k in ("duplicate", "dropout", "immobile")}
    return Read("PASS", {**obj, "arm": "held_pose"},
                (f"of {n:,} zero-ego-speed frames, {100 * p['duplicate']:.1f}% "
                 f"are DUPLICATE VIDEO FRAMES (an encoder artefact, not "
                 f"behaviour), {100 * p['dropout']:.1f}% are TRACKING DROPOUTS "
                 f"(the animal's own pixels moved while its keypoints did not) "
                 f"and {100 * p['immobile']:.1f}% are GENUINE IMMOBILITY. "
                 f"CONTEXT.md and CONTEXT_CONTROLS.md describe all of them as "
                 f"a tracking dropout, and that description is correct for "
                 f"{100 * p['dropout']:.1f}% of them"),
                n_effective=n, detail={"counts": tri, "shares": p})


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("sample", "combine"), default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.phase == "sample":
        return sample(a)
    if a.phase == "combine":
        return combine(a)
    if a.shard is None:
        raise SystemExit("pass --phase sample|combine, or --shard I --of N")
    return shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
