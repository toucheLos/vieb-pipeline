r"""Do segments recur across animals? Q1's statistic, segments as the unit.

Registered in `results/SEGRECUR_PREREGISTRATION.md`.

## The statistic is Q1's and is not reimplemented

`recur/scripts/q1.py:paired_excess` is imported and called. Two properties of it
carry the whole design and are restated here because they decide how the numbers
below must be read:

**θ is set by the NULL.** The threshold is the null arm's quantile at
`NULL_RATE = 0.01` of its own normalised distances, so the observed arm is never
asked to clear a threshold chosen from itself.

**Both arms are divided by their own ambient scale first**, the median distance
between *random* cross-animal pairs. Without it the statistic measures spread:
at w = 0.4 s the median cross-animal NN distance is 3.83 on the corpus and 8.98
on an OU control, so scored raw the corpus "beats" OU by +78% while saying
nothing about recurrence. The normaliser has no nearest-neighbour feedback,
which dividing by the median NN distance would.

## What is new here is the comparator, not the statistic

Q1 compared observed windows against surrogate windows. This compares observed
**segments** against surrogate segments, and then compares *that* excess against
the excess of a **length-matched windowed arm in the same space**. The gate is
the second comparison, because the first one alone cannot say whether the
segmentation contributed anything: a bank of any units drawn from a real mouse
beats a surrogate.

## Why the duration match is reported and not gated

A null whose segments are systematically shorter has a larger bank of shorter
units, and some of any gap would be that. The separability precondition that
tried to gate this class of confound was **contradictory** (`DEVIATIONS.md` D7)
and is not reinstated here under another name. The reader is given the size of
the effect instead of a pass/fail on it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur import boot
from recur.read import Read

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]

__all__ = ["DURATION_QUANTILES", "GRID", "K_MAD_PRIMARY", "K_MAD_SWEEP",
           "K_MAD_FLOOR", "NULLS", "duration_match", "gate_read",
           "length_match_control", "excess_delta", "recur_read"]

#: The dwell-matched pair. NOT the closed gate's four families: `phase` and
#: `var5` diverge 2.1x in runs per second from the corpus, which is the confound
#: that made a per-second comparison against them uninterpretable.
NULLS: tuple[str, ...] = ("microstate", "microstate0")
#: Registered: primary, and the sensitivity points around it.
K_MAD_PRIMARY = 3.0
K_MAD_SWEEP: tuple[float, ...] = (2.5, 3.0, 4.0)
#: Nothing at or below this is swept, and the reason is registered rather than
#: discovered. At k = 1 the closed gate's four nulls agreed to 0.40% at 53% of
#: the NMS resolution ceiling, so a rate there is a property of the refractory
#: period; and a MAD-standardised threshold penalises a heavy-tailed `D` at the
#: bottom of a sweep by construction.
K_MAD_FLOOR = 2.0
GRID = 40
DURATION_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def duration_match(obs: npt.ArrayLike, null: npt.ArrayLike) -> Detail:
    """Two-sample comparison of log segment duration. Reported, never gating.

    KS on the logs plus the quantile differences, because a KS statistic alone
    says two distributions differ without saying by how much or in which
    direction, and the direction is what decides whether a gap flatters the
    corpus or the null.
    """
    a = np.log(np.asarray(obs, dtype=np.float64))
    b = np.log(np.asarray(null, dtype=np.float64))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"ks": float("nan"), "n_obs": int(a.size), "n_null": int(b.size),
                "why": "one arm has fewer than two finite durations"}
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    qa = np.quantile(a, DURATION_QUANTILES)
    qb = np.quantile(b, DURATION_QUANTILES)
    return {
        "ks": float(np.max(np.abs(fa - fb))),
        "n_obs": int(a.size), "n_null": int(b.size),
        "mean_log_obs": float(a.mean()), "mean_log_null": float(b.mean()),
        "median_ratio": float(np.exp(float(np.median(a) - np.median(b)))),
        "quantile_diff_log": {f"{q:g}": float(x - y)
                              for q, x, y in zip(DURATION_QUANTILES, qa, qb)},
        "note": ("reported beside the comparison, never gating it: the "
                 "precondition that tried to gate this class of confound was "
                 "contradictory and is recorded as DEVIATIONS.md D7"),
    }


def length_match_control(q_len: npt.ArrayLike, nn_len: npt.ArrayLike,
                         rng: np.random.Generator, *, n_draws: int = 20
                         ) -> Detail:
    """Are nearest neighbours more similar in LENGTH than chance?

    `partition.py:20-37` records a rater scoring **+0.183** -- about a third of
    the real effect -- purely by calling the six longest clips "same". A
    nearest-neighbour hit between two segments of similar duration may be a
    length match wearing a shape match's clothes.

    Chance is the same query lengths against **permuted** partner lengths, which
    holds both marginals fixed, so the comparison is about pairing and not about
    the length distribution.
    """
    a = np.log(np.asarray(q_len, dtype=np.float64))
    b = np.log(np.asarray(nn_len, dtype=np.float64))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return {"n": int(a.size), "why": "too few matched pairs"}
    obs = float(np.abs(a - b).mean())
    null = [float(np.abs(a - rng.permutation(b)).mean())
            for _ in range(int(n_draws))]
    return {"n": int(a.size),
            "mean_abs_log_length_gap": obs,
            "mean_abs_log_length_gap_shuffled": float(np.mean(null)),
            "sd_shuffled": float(np.std(null)),
            "ratio": float(obs / np.mean(null)) if np.mean(null) > 0 else
            float("nan"),
            "note": ("a ratio well below 1 means nearest neighbours share "
                     "duration more than chance, and part of the match may be "
                     "length rather than shape")}


def excess_delta(seg: Mapping[str, Any], win: Mapping[str, Any], *,
                 seed: int = 0) -> Detail:
    """Per-animal (segment excess - windowed excess), animal bootstrap.

    The gate's quantity. Paired within animal because both arms contain every
    animal by construction, and an unpaired interval would discard the pairing
    the design already has.

    The frame-level interval is computed alongside and is printed **solely** to
    show how much narrower the wrong denominator looks. It never licenses a
    claim; `n_effective` on every `Read` here is animals.
    """
    a, b = seg["delta_per_animal"], win["delta_per_animal"]
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return {"n_animals": len(shared),
                "why": "fewer than two animals in both arms"}
    d = np.asarray([float(a[k]) - float(b[k]) for k in shared],
                   dtype=np.float64)
    ci = boot.animal_interval(d, shared, how="mean", seed=seed)
    return {"animals": shared, "n_animals": len(shared),
            "delta": ci,
            "segment_excess": float(np.mean([float(a[k]) for k in shared])),
            "windowed_excess": float(np.mean([float(b[k]) for k in shared])),
            "per_animal": {k: float(a[k] - b[k]) for k in shared}}


def recur_read(pair: Mapping[str, Any], *, null: str, unit: str,
               scored_object: Detail, n_effective: int) -> Read:
    """Does this unit recur above its dwell-matched surrogate?"""
    ci = pair.get("delta")
    if not ci:
        return Read("INCONCLUSIVE", scored_object,
                    f"the {unit}-versus-{null} comparison did not produce an "
                    f"interval: {pair.get('why', 'no reason recorded')}",
                    n_effective=n_effective, detail=dict(pair))
    lo, point, hi = float(ci["lo"]), float(ci["point"]), float(ci["hi"])
    tail = (f"{point * 100:+.4f}% [{lo * 100:+.4f}%, {hi * 100:+.4f}%] "
            f"cross-animal recurrence excess over {null}, animal bootstrap at "
            f"n = {int(ci['n_animals'])}")
    detail: Detail = {k: v for k, v in pair.items() if k != "per_animal"}
    if lo > 0.0:
        weak = (" -- read WEAKLY: microstate preserves one-step visit dynamics, "
                "so much of what this compares is the null's construction"
                if null == "microstate" else "")
        return Read("PASS", scored_object,
                    f"{unit}s recur above their dwell-matched surrogate: "
                    f"{tail}{weak}",
                    n_effective=n_effective, detail=detail)
    if hi < 0.0:
        return Read("FAIL", scored_object,
                    f"{unit}s recur LESS than their dwell-matched surrogate: "
                    f"{tail}", n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"no detectable excess: {tail} -- the interval spans zero, so "
                f"this unit is not shown to recur above a surrogate that "
                f"matches its dwell",
                n_effective=n_effective, detail=detail)


def gate_read(delta: Mapping[str, Any], floor: Mapping[str, Any], *,
              group: str, scored_object: Detail, n_effective: int) -> Read:
    """The registered gate: did segmenting CHANGE the excess, and which way?

    Reading the segment-versus-surrogate excess alone would not answer the
    question the arm is for. A bank of any units cut from a real mouse beats a
    surrogate; what is being asked is whether **the boundaries** contributed,
    and that is the comparison against a length-matched windowed arm in the same
    space.

    `floor` is the planted dose-response. A `FAIL` that cannot say what it would
    have detected is not a result, so the smallest recovered occupancy travels
    inside the verdict.
    """
    ci = delta.get("delta")
    smallest = floor.get("smallest_recovered")
    fl = (f"; the planted floor recovers {float(smallest):.2%} occupancy"
          if smallest is not None else
          "; the planted floor did not run, so this cannot say what it would "
          "have detected")
    if not ci:
        return Read("INCONCLUSIVE", scored_object,
                    f"the {group} gate has no interval: "
                    f"{delta.get('why', 'no reason recorded')}{fl}",
                    n_effective=n_effective, detail=dict(delta))
    lo, point, hi = float(ci["lo"]), float(ci["point"]), float(ci["hi"])
    tail = (f"segments {float(delta['segment_excess']) * 100:+.4f}% against a "
            f"length-matched windowed control's "
            f"{float(delta['windowed_excess']) * 100:+.4f}%, a difference of "
            f"{point * 100:+.4f}% [{lo * 100:+.4f}%, {hi * 100:+.4f}%]")
    detail: Detail = {k: v for k, v in delta.items() if k != "per_animal"}
    detail["planted_floor"] = dict(floor)
    if hi < 0.0:
        return Read("FAIL", scored_object,
                    f"THE BOUNDARIES CUT THROUGH BEHAVIOURS on {group}: {tail}. "
                    f"Segmenting made cross-animal recurrence WORSE than "
                    f"length-matched random windows in the same space, so "
                    f"Step 1's criterion is placing its edges inside the units "
                    f"rather than between them{fl}",
                    n_effective=n_effective, detail=detail)
    if lo > 0.0:
        return Read("PASS", scored_object,
                    f"the boundaries contribute on {group}: {tail}. Segments "
                    f"recur across animals more than length-matched random "
                    f"windows cut from the same signal{fl}",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"segments are a valid unit that ADDS NOTHING on {group}: "
                f"{tail}, an interval spanning zero. The boundaries neither "
                f"help nor hurt, so the vocabulary question stays open and "
                f"nothing here argues the edges are behavioural{fl}",
                n_effective=n_effective, detail=detail)
