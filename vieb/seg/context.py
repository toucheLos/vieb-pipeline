r"""Does the island move with context? The paired test `BEHAVIOUR.md` asked for.

Registered in `results/FREEZING_PREREGISTRATION.md`.

## The design, and where the confound is not

`CONCENTRATION.md` records context and day as completely confounded -- Context C
occurs only on Day 2 -- and that is true of the whole-corpus marginal grouping.
It is **not** true here: days 3-7 each carry both Context A and Context B, and
441 of 708 (animal, day) cells on the report split hold both. C and day 2 are
excluded, so the confounded factor never enters the contrast.

## Occupancy is a rate, and that is not a detail

Clump-0 frames over that cell's **total selectable segment frames**. A session
with more usable segments would otherwise contribute more clump-0 frames for
reasons that have nothing to do with context, and the contrast would measure how
much of each session survived QC. `vocab.session_composition` reports counts with
no denominator and is deliberately not what this uses.

## The gate runs before the contrast

`recur.journey.simplex.mde_read`, the pattern `journeys.py` and
`learning_curve.py` both use: if the smallest detectable effect exceeds the
registered plausible one, **no contrast is computed** -- no permutation, no
interval -- and the MDE is the result. A null from a design that could not have
seen the effect is not evidence of absence.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur import boot
from recur.read import Read

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]
#: (animal, day, context) -> value. Keyed by tuple, so not a `Detail`.
CellMap = dict[tuple[str, int, str], float]

__all__ = ["CONTEXTS", "DAYS", "PLAUSIBLE_EFFECT", "cell_occupancy",
           "context_read", "paired_deltas"]

#: CFD days carrying both contexts. Days 0-1 are Context A only and day 2 is
#: Context C only, which is the pairing CONCENTRATION.md records as confounded.
DAYS: tuple[int, ...] = (3, 4, 5, 6, 7)
CONTEXTS: tuple[str, str] = ("A", "B")
#: Two percentage points of occupancy, fixed in the registration. Clump-0
#: occupancy is on the order of 1.9% corpus-wide, so an effect smaller than this
#: could not be told from between-session variability.
PLAUSIBLE_EFFECT = 0.02


def cell_occupancy(labels: npt.ArrayLike, n_frames: npt.ArrayLike,
                   animals: Sequence[str], days: Sequence[int],
                   contexts: Sequence[str], *,
                   clump: int = 0) -> tuple[CellMap, CellMap]:
    """Clump-`clump` frame share per (animal, day, context) cell.

    Frames, not segment counts: a clump that holds a few very long segments and
    one that holds many short ones are different objects, and the behavioural
    quantity is how much of the session the animal spent in the state.
    """
    lab = np.asarray(labels, dtype=np.int64)
    nf = np.asarray(n_frames, dtype=np.float64)
    an = np.asarray(animals)
    dy = np.asarray(days, dtype=np.int64)
    cx = np.asarray(contexts)
    num: CellMap = {}
    den: CellMap = {}
    for i in range(lab.size):
        if int(dy[i]) not in DAYS or str(cx[i]) not in CONTEXTS:
            continue
        key = (str(an[i]), int(dy[i]), str(cx[i]))
        den[key] = den.get(key, 0.0) + float(nf[i])
        if int(lab[i]) == int(clump):
            num[key] = num.get(key, 0.0) + float(nf[i])
    return {k: (num.get(k, 0.0) / v if v > 0 else float("nan"))
            for k, v in den.items()}, den


def paired_deltas(occ: Mapping[tuple[str, int, str], float],
                  den: Mapping[tuple[str, int, str], float], *,
                  min_frames: float = 0.0) -> Detail:
    """`Δ = occupancy_B − occupancy_A`, one per (animal, day) cell holding both.

    Signed and paired, because the null flips signs: an unsigned or unpaired
    quantity has nothing to flip, which is the failure `journeys.py` records for
    the raw W2 contrast.
    """
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for key, share in occ.items():
        animal, day, ctx = key
        cells.setdefault((animal, day), {})[str(ctx)] = float(share)
    diff: list[float] = []
    keys: list[str] = []
    animals: list[str] = []
    dropped = 0
    for (animal, day), v in sorted(cells.items()):
        if "A" not in v or "B" not in v:
            dropped += 1
            continue
        if not (np.isfinite(v["A"]) and np.isfinite(v["B"])):
            dropped += 1
            continue
        if min_frames > 0 and min(den.get((animal, day, "A"), 0.0),
                                  den.get((animal, day, "B"), 0.0)) < min_frames:
            dropped += 1
            continue
        diff.append(float(v["B"] - v["A"]))
        keys.append(f"{animal}|{day}")
        animals.append(str(animal))
    return {"diff": np.asarray(diff, dtype=np.float64),
            "pair_key": keys, "animal": animals,
            "n_pairs": len(diff), "n_dropped_incomplete": dropped,
            "mean_A": float(np.mean([v["A"] for v in cells.values()
                                     if "A" in v and np.isfinite(v["A"])])),
            "mean_B": float(np.mean([v["B"] for v in cells.values()
                                     if "B" in v and np.isfinite(v["B"])]))}


def context_read(ci: Mapping[str, Any], flip: Mapping[str, Any],
                 pairs: Mapping[str, Any], *, clump: int,
                 scored_object: Detail, n_effective: int) -> Read:
    """Does clump occupancy differ by context, within animal and day?

    **The verdict follows the interval, not the p-value.**
    `LEARNING_CURVE.md` fixed that precedent: a stabilisation arm at p = 0.0475
    still read FAIL because its CI spanned zero.

    The p reported here is `p_two_sided`, read directly rather than through
    `simplex.journey_read`, which looks for a key `pair_flip_null` does not
    return and would therefore always take its FAIL branch.
    """
    lo, point, hi = float(ci["lo"]), float(ci["point"]), float(ci["hi"])
    p = float(flip.get("p_two_sided", float("nan")))
    detail: Detail = {"clump": clump, "ci": dict(ci), "flip": dict(flip),
                      "n_pairs": pairs.get("n_pairs"),
                      "mean_occupancy_A": pairs.get("mean_A"),
                      "mean_occupancy_B": pairs.get("mean_B")}
    tail = (f"occupancy B − A = {point:+.5f} [{lo:+.5f}, {hi:+.5f}] over "
            f"{int(ci['n_animals'])} animals and {int(pairs['n_pairs'])} "
            f"(animal, day) cells, pair-flip p = {p:.4f}")
    if lo > 0.0 or hi < 0.0:
        where = "B" if point > 0 else "A"
        return Read("PASS", scored_object,
                    f"THE ISLAND MOVES WITH CONTEXT: {tail}. Occupancy is "
                    f"higher in context {where}, within animal and day, on days "
                    f"3-7 where both contexts are present. This is the "
                    f"experimental design rather than appearance — and it still "
                    f"does not name the state, which needs the context-to-shock "
                    f"mapping this repository does not hold",
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"the island does not move with context: {tail} — the interval "
                f"spans zero. The state is real and is not context-dependent, "
                f"so the freezing hypothesis is not supported by the design and "
                f"comes off the page",
                n_effective=n_effective, detail=detail)
