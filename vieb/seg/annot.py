"""Boundary annotation: tolerance-matched F1, and the inter-rater ceiling.

READ results/ANNOTATION_PREREGISTRATION.md FIRST.

## Why one-to-one matching, and not the matcher already in the repo

`scripts/breaks.py:165 _agreement` compares two boundary sets inside a tolerance
window with `searchsorted`, and it is the right skeleton. But it counts
**unmatched hits**: if two of B's boundaries fall inside one tolerance window
around a single boundary of A, both score. That inflates recall exactly where a
detector is noisiest -- a criterion that fires three times around every real
change looks better than one that fires once.

`match` here is greedy and **nearest-first**, and refuses to reuse a boundary
that has already been matched. A detector that fires three times around one
human mark gets one true positive and two false positives, which is what it
earned.

## Why F1 and not kappa

`recur.render.partition.agreement` computes Cohen's kappa over a categorical
response, and kappa is the right statistic there. It is **not defined for event
sets**: there is no category, no confusion matrix, and no marginal to correct
for. Two raters marking boundaries agree to the extent their marks coincide,
which is F1, and F1 is symmetric so neither rater has to be designated the truth.

## Why a chance level is computed at all

An F1 of 0.4 means nothing without knowing what two raters scattering the same
number of marks at random would score. In a 10-second clip with 8 marks each and
a +/-10 frame tolerance, random agreement is not small. `chance_f1` is computed
per clip under the SAME matcher, and the registration fixed "raters agree
poorly" as the +/-5 interval including it -- before any observed F1 was read.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import boot                                              # noqa: E402
from recur.read import Read                                         # noqa: E402

__all__ = ["TOLERANCES", "N_BOOT", "N_CHANCE", "match", "prf",
           "chance_f1", "pair_rows", "ceiling_read", "coverage_read",
           "decompose", "decompose_read", "offsets", "offset_read",
           "source_read"]

I64 = NDArray[np.int64]

#: The tolerance bands, registered. Not a swept parameter: these three are
#: reported together and none may be dropped after the fact.
TOLERANCES: tuple[int, ...] = (2, 5, 10)
N_BOOT = 2000
#: Random restarts behind `chance_f1`. Enough that the chance level is stable to
#: the third decimal, which is finer than any interval it is compared against.
N_CHANCE = 200


def match(a: ArrayLike, b: ArrayLike, tol: int) -> list[tuple[int, int]]:
    """Greedy nearest-first one-to-one pairing of two boundary sets.

    Returns the matched `(i, j)` index pairs into `a` and `b`. Every candidate
    pair within `tol` is considered in order of increasing distance, and a
    boundary already spoken for is skipped -- so `len(match(...))` is the number
    of true positives under a matcher that cannot double-count.

    Ties broken by index, so the result is a function of the inputs alone and
    not of numpy's sort stability.
    """
    x = np.sort(np.asarray(a, dtype=np.int64))
    y = np.sort(np.asarray(b, dtype=np.int64))
    if x.size == 0 or y.size == 0:
        return []
    cand: list[tuple[int, int, int]] = []
    for i, v in enumerate(x):
        lo = int(np.searchsorted(y, v - tol, side="left"))
        hi = int(np.searchsorted(y, v + tol, side="right"))
        for j in range(lo, hi):
            cand.append((abs(int(y[j]) - int(v)), i, j))
    cand.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[tuple[int, int]] = []
    for _d, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


def prf(a: ArrayLike, b: ArrayLike, tol: int) -> dict[str, float]:
    """Precision, recall and F1 of `b` against `a`, at `tol` frames.

    `a` is the reference. Between two raters neither is the reference, but F1 is
    symmetric, so the pair statistic does not depend on which is passed first --
    only precision and recall swap.

    Two empty sets score F1 = 1.0: two raters who both say "nothing changed
    here" agree completely, and scoring that as zero would punish the exact
    observation the continuum question turns on.
    """
    x = np.asarray(a, dtype=np.int64)
    y = np.asarray(b, dtype=np.int64)
    if x.size == 0 and y.size == 0:
        return {"tp": 0.0, "n_ref": 0.0, "n_hyp": 0.0,
                "precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = float(len(match(x, y, tol)))
    prec = tp / y.size if y.size else 0.0
    rec = tp / x.size if x.size else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "n_ref": float(x.size), "n_hyp": float(y.size),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def chance_f1(n_a: int, n_b: int, n_frames: int, tol: int,
              rng: np.random.Generator, *, n_restarts: int = N_CHANCE) -> float:
    """F1 two raters would reach scattering the same counts uniformly at random.

    The counts and the clip length are held at what was actually observed, so
    this is the chance level *for this clip*, not a generic one. With 8 marks
    each in 300 frames at +/-10 this is not a small number, which is the whole
    reason it is computed.
    """
    if n_a == 0 and n_b == 0:
        return 1.0
    if n_frames <= 0 or (n_a == 0) != (n_b == 0):
        return 0.0
    got = np.empty(n_restarts, dtype=np.float64)
    for k in range(n_restarts):
        a = rng.choice(n_frames, size=min(n_a, n_frames), replace=False)
        b = rng.choice(n_frames, size=min(n_b, n_frames), replace=False)
        got[k] = prf(a, b, tol)["f1"]
    return float(got.mean())


def pair_rows(marks: Mapping[str, Mapping[str, Sequence[int]]],
              clips: Mapping[str, Mapping[str, Any]], *,
              tolerances: Iterable[int] = TOLERANCES,
              seed: int = 0) -> list[dict[str, Any]]:
    """One row per (rater pair, clip, tolerance).

    `marks` is `{rater: {clip_id: [frame, ...]}}` and `clips` carries at least
    `{clip_id: {"animal": ..., "n_frames": ...}}`. Only clips **both** raters
    actually rated are compared: a clip one rater never opened is missing data,
    not a disagreement, and scoring it as one would let an unfinished shard look
    like a poor rater.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for ra, rb in itertools.combinations(sorted(marks), 2):
        shared = sorted(set(marks[ra]) & set(marks[rb]))
        for cid in shared:
            meta = clips[cid]
            a = list(marks[ra][cid])
            b = list(marks[rb][cid])
            for tol in tolerances:
                got = prf(a, b, tol)
                out.append({
                    "rater_a": ra, "rater_b": rb, "clip": cid,
                    "animal": str(meta["animal"]), "tol": int(tol),
                    "n_frames": int(meta["n_frames"]),
                    "chance_f1": chance_f1(len(a), len(b),
                                           int(meta["n_frames"]), int(tol),
                                           rng),
                    **got})
    return out


def ceiling_read(rows: Sequence[Mapping[str, Any]], *, tol: int,
                 scored_object: dict[str, Any], n_effective: int,
                 seed: int = 0) -> Read:
    """The inter-rater ceiling at one tolerance, against its own chance level.

    Verdicts, from the registration's reading table:

    * fewer than two raters -> NOT_A_RESULT. One rater is no ceiling, and
      `score_ratings.py:327` already refuses this rather than reporting a
      detector score against nothing.
    * observed F1 interval includes the chance F1 -> **FAIL**: the question is
      ill-posed at this tracking quality. Report and stop; no detector is
      scored, because there is nothing to score it against.
    * otherwise -> PASS, and this number is the benchmark. **It is not 1.0**,
      and a detector matching it is at ceiling rather than failing.
    """
    sel = [r for r in rows if int(r["tol"]) == int(tol)]
    raters = {r["rater_a"] for r in sel} | {r["rater_b"] for r in sel}
    if len(raters) < 2 or not sel:
        return Read(
            verdict="NOT_A_RESULT", scored_object=scored_object,
            n_effective=max(n_effective, 0), degenerate=True,
            detail={"n_raters": len(raters), "n_rows": len(sel)},
            reason=(f"NO CEILING: {len(raters)} rater(s) supplied marks and a "
                    f"ceiling needs two. A detector scored against a single "
                    f"rater is scored against nothing."))
    animals = [str(r["animal"]) for r in sel]
    obs = boot.animal_interval([float(r["f1"]) for r in sel], animals,
                               how="mean", n_boot=N_BOOT, seed=seed)
    cha = boot.animal_interval([float(r["chance_f1"]) for r in sel], animals,
                               how="mean", n_boot=N_BOOT, seed=seed)
    prec = boot.animal_interval([float(r["precision"]) for r in sel], animals,
                                how="mean", n_boot=N_BOOT, seed=seed)
    rec = boot.animal_interval([float(r["recall"]) for r in sel], animals,
                               how="mean", n_boot=N_BOOT, seed=seed)
    detail = {"f1": obs, "chance_f1": cha, "precision": prec, "recall": rec,
              "tol": int(tol), "n_pairs": len(sel),
              "n_raters": len(raters)}
    span = "[%.4f, %.4f]" % (obs["lo"], obs["hi"])
    if not np.isfinite(obs["lo"]) or obs["lo"] <= cha["point"]:
        return Read(
            verdict="FAIL", scored_object=scored_object,
            n_effective=n_effective, detail=detail,
            reason=(f"RATERS DO NOT AGREE WITH EACH OTHER at ±{tol} frames: "
                    f"F1 {obs['point']:.4f} {span} against a chance F1 of "
                    f"{cha['point']:.4f} for the same mark counts scattered at "
                    f"random. The question is ill-posed at this tracking "
                    f"quality; no detector is scored against this."))
    return Read(
        verdict="PASS", scored_object=scored_object, n_effective=n_effective,
        detail=detail,
        reason=(f"the ceiling at ±{tol} frames is F1 {obs['point']:.4f} {span}, "
                f"against a chance F1 of {cha['point']:.4f}. THIS IS THE "
                f"BENCHMARK AND IT IS NOT 1.0 — a detector reaching "
                f"{obs['point']:.4f} here is at ceiling, not failing."))


def coverage_read(marks: Mapping[str, Mapping[str, Sequence[int]]],
                  clips: Mapping[str, Mapping[str, Any]], *,
                  scored_object: dict[str, Any], n_effective: int,
                  seed: int = 0) -> Read:
    """How often raters saw any change at all — the continuum question, directly.

    This is the one measurement in the programme that can speak to the 98.1%
    unassigned mass without a detector in the loop. If people watching the video
    mark almost nothing, the continuum reading is supported by human perception
    independently; if they mark plenty where the detector is silent, the
    unassigned mass is detector limitation.

    Reported as boundaries per second and as the share of clips marked empty.
    No verdict is attached: the registration fixed no threshold for "near-zero",
    and inventing one now would be choosing a cutoff after seeing the number.
    """
    per: list[float] = []
    animals: list[str] = []
    empty = 0
    total = 0
    for rater, by_clip in marks.items():
        for cid, ms in by_clip.items():
            meta = clips[cid]
            secs = float(meta["n_frames"]) / float(meta["fps"])
            per.append(len(ms) / secs if secs > 0 else float("nan"))
            animals.append(str(meta["animal"]))
            empty += int(len(ms) == 0)
            total += 1
    rate = boot.animal_interval(per, animals, how="mean", n_boot=N_BOOT,
                                seed=seed)
    share = (empty / total) if total else float("nan")
    return Read(
        verdict="NOT_A_RESULT", scored_object=scored_object,
        n_effective=n_effective,
        detail={"per_second": rate, "empty_share": share,
                "n_clip_ratings": total, "n_empty": empty},
        reason=(f"descriptive, no verdict registered: raters marked "
                f"{rate['point']:.3f} [{rate['lo']:.3f}, {rate['hi']:.3f}] "
                f"boundaries per second, and called {empty}/{total} clip "
                f"ratings empty. No threshold for 'near-zero' was registered "
                f"and choosing one now would be choosing it after the fact."))


def decompose(rows: Sequence[Mapping[str, Any]], *,
              tol: int) -> dict[str, Any]:
    """Split the pair rows at one tolerance by whether either rater marked.

    `prf` scores two empty sets as F1 = 1.0 deliberately -- two raters who both
    say "nothing changed" agree -- and `chance_f1` does the same. So a
    both-empty clip contributes 1.0 to the observed value **and** 1.0 to the
    chance level, and cannot separate them. Quoting a headline F1 without
    saying how much of it is that is overstating agreement.

    Three disjoint classes, and only the third carries information:

    * `both_empty` -- neither rater marked. F1 = 1.0 by construction, on both
      sides.
    * `one_empty` -- exactly one marked. F1 = 0.0 by construction.
    * `both_marked` -- both marked. The only clips where the matcher does work.
    """
    sel = [r for r in rows if int(r["tol"]) == int(tol)]
    out: dict[str, Any] = {"tol": int(tol), "n_rows": len(sel)}
    classes: dict[str, list[Mapping[str, Any]]] = {
        "both_empty": [], "one_empty": [], "both_marked": []}
    for r in sel:
        a, b = float(r["n_ref"]), float(r["n_hyp"])
        if a == 0.0 and b == 0.0:
            classes["both_empty"].append(r)
        elif a == 0.0 or b == 0.0:
            classes["one_empty"].append(r)
        else:
            classes["both_marked"].append(r)
    for name, got in classes.items():
        out[f"n_{name}"] = len(got)
    out["rows_both_marked"] = classes["both_marked"]
    return out


def decompose_read(rows: Sequence[Mapping[str, Any]], *, tol: int,
                   scored_object: dict[str, Any], n_effective: int,
                   seed: int = 0) -> Read:
    """The decomposition, and the ceiling recomputed on informative clips only.

    No verdict. The registered ceiling is the one `ceiling_read` returns over
    every clip, and it is not restated or replaced here -- §7 forbids changing
    what is scored after an agreement number has been seen. This reports what
    the registered number is **made of**, which is a different thing and is owed
    to anyone reading it.
    """
    got = decompose(rows, tol=tol)
    sub = got.pop("rows_both_marked")
    detail: dict[str, Any] = dict(got)
    if sub:
        f1 = boot.animal_interval([float(r["f1"]) for r in sub],
                                  [str(r["animal"]) for r in sub],
                                  how="mean", n_boot=N_BOOT, seed=seed)
        ch = float(np.mean([float(r["chance_f1"]) for r in sub]))
        detail["both_marked_f1"] = f1
        detail["both_marked_chance_f1"] = ch
        body = (f"on the {len(sub)} clips where BOTH raters marked something, "
                f"F1 is {f1['point']:.4f} [{f1['lo']:.4f}, {f1['hi']:.4f}] "
                f"against chance {ch:.4f}")
    else:
        detail["both_marked_f1"] = None
        detail["both_marked_chance_f1"] = float("nan")
        body = "no clip had marks from both raters"
    n_be = int(got["n_both_empty"])
    n_rows = int(got["n_rows"])
    share = (n_be / n_rows) if n_rows else float("nan")
    return Read(
        verdict="NOT_A_RESULT", scored_object=scored_object,
        n_effective=n_effective, detail=detail,
        reason=(f"descriptive, no verdict: at ±{tol} the registered F1 is "
                f"carried by {n_be} clips BOTH raters called empty, which "
                f"score 1.0 on the observed value and 1.0 on chance alike and "
                f"cannot separate them ({share:.3f} of rows). "
                f"{got['n_one_empty']} rows score 0.0 because exactly one "
                f"rater marked. {body}. This does not restate or replace the "
                f"registered ceiling."))


def offsets(marks: Mapping[str, Mapping[str, Sequence[int]]],
            clips: Mapping[str, Mapping[str, Any]], *,
            sources: Mapping[str, Mapping[str, str]] | None = None
            ) -> list[dict[str, Any]]:
    """Signed frame offset from each of rater A's marks to rater B's nearest.

    One distribution, not a tolerance sweep. Every "within N" share quoted
    anywhere is read off this, so no second matching pass at an unregistered
    tolerance ever happens -- §7 forbids scoring at a band chosen after the
    fact, and a nearest-neighbour distance is not a score.

    Positive means A's mark falls **later** than B's nearest. Defined only on
    clips where both raters marked; elsewhere there is no nearest mark.
    """
    raters = sorted(marks)
    out: list[dict[str, Any]] = []
    for ra, rb in itertools.combinations(raters, 2):
        for cid in sorted(set(marks[ra]) & set(marks[rb])):
            a = np.sort(np.asarray(list(marks[ra][cid]), dtype=np.int64))
            b = np.sort(np.asarray(list(marks[rb][cid]), dtype=np.int64))
            if a.size == 0 or b.size == 0:
                continue
            meta = clips[cid]
            src = ""
            if sources is not None:
                src = str(sources.get(ra, {}).get(cid, ""))
            for m in a.tolist():
                j = int(np.argmin(np.abs(b - m)))
                out.append({"rater_a": ra, "rater_b": rb, "clip": cid,
                            "animal": str(meta["animal"]),
                            "fps": float(meta["fps"]),
                            "mark": int(m), "offset": int(m - int(b[j])),
                            "frame_source_a": src})
    return out


def offset_read(rows: Sequence[Mapping[str, Any]], *,
                scored_object: dict[str, Any], n_effective: int,
                seed: int = 0) -> Read:
    """Coarse agreement against fine agreement, off one offset distribution.

    No verdict. This is the shape of the disagreement, not a re-score: raters
    can be looking at the same events and still be unable to place them, and
    the registered F1 cannot tell those apart from raters looking at different
    events.
    """
    if not rows:
        return Read(verdict="NOT_A_RESULT", scored_object=scored_object,
                    n_effective=n_effective, detail={"n_marks": 0},
                    reason=("descriptive, no verdict: no clip had marks from "
                            "both raters, so no offset is defined"))
    off = np.asarray([int(r["offset"]) for r in rows], dtype=np.float64)
    fps = float(rows[0]["fps"])
    animals = [str(r["animal"]) for r in rows]
    near_s = boot.animal_interval((np.abs(off) <= fps).astype(float), animals,
                                  how="mean", n_boot=N_BOOT, seed=seed)
    near_f = boot.animal_interval((np.abs(off) <= 2.0).astype(float), animals,
                                  how="mean", n_boot=N_BOOT, seed=seed)
    detail = {"n_marks": int(off.size), "fps": fps,
              "within_one_second": near_s, "within_two_frames": near_f,
              "median_abs_offset_frames": float(np.median(np.abs(off))),
              "iqr_abs_offset_frames": [float(np.percentile(np.abs(off), 25)),
                                        float(np.percentile(np.abs(off), 75))]}
    return Read(
        verdict="NOT_A_RESULT", scored_object=scored_object,
        n_effective=n_effective, detail=detail,
        reason=(f"descriptive, no verdict: {near_s['point']:.3f} "
                f"[{near_s['lo']:.3f}, {near_s['hi']:.3f}] of marks sit within "
                f"one second of the other rater's nearest, against "
                f"{near_f['point']:.3f} [{near_f['lo']:.3f}, "
                f"{near_f['hi']:.3f}] within two frames. The raters are "
                f"looking at the same events and cannot place them. Read off "
                f"one nearest-neighbour distribution, not a second matching "
                f"pass at an unregistered tolerance."))


def source_read(rows: Sequence[Mapping[str, Any]], *,
                scored_object: dict[str, Any], n_effective: int) -> Read:
    """The paused-versus-playing gap, which the registration did not close.

    The marker permitted marking while the video played and while it was
    paused, and specified neither. A mark placed while playing carries visual
    reaction time **and** a `currentTime` that is stale by up to a frame
    interval; a mark placed while paused carries neither. One rater did each,
    so the choice is confounded with rater identity and nothing here separates
    them. Reported because an unregistered free choice inside an instrument is
    a variable, not because it explains the ceiling.
    """
    by: dict[str, list[int]] = {}
    for r in rows:
        by.setdefault(str(r["frame_source_a"]) or "unknown", []).append(
            int(r["offset"]))
    detail = {src: {"n": len(v),
                    "median_offset_frames": float(np.median(v)),
                    "iqr_offset_frames": [float(np.percentile(v, 25)),
                                          float(np.percentile(v, 75))]}
              for src, v in sorted(by.items())}
    parts = [f"{src} n={d['n']} median {d['median_offset_frames']:+.1f}"
             for src, d in sorted(detail.items())]
    return Read(
        verdict="NOT_A_RESULT", scored_object=scored_object,
        n_effective=n_effective, detail=detail,
        reason=("descriptive, no verdict: offset by how the mark was placed, "
                + "; ".join(parts) + " frames. The samples are small and the "
                "choice is confounded with rater identity, so this is an "
                "instrument defect on the record, not an explanation of the "
                "ceiling -- correcting a median offset would not rescue ±2 "
                "agreement when the spread is what kills it."))
