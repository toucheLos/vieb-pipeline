r"""Did the arm get closer to the truth, and what did it break getting there?

Two numbers, both in body lengths, both against a known true position:

* **repair** -- mean error on the keypoint-frames that were corrupted. Low is
  good. An arm that does nothing scores exactly the injected magnitude.
* **damage** -- mean error on the keypoint-frames that were **not** corrupted.
  Low is good, and zero is achievable: a perfect arm leaves correct data alone.

Damage is the axis nothing else in this repo has had. Violation rate, distortion,
retention and MDL are all computed without knowing where the keypoint was, so
none of them can charge an arm for confidently moving a *correct* point. The eye
cannot see it either -- a filter that shifts every landmark slightly towards the
body's centroid looks tidier, not worse.

## net is the number that answers the question

.. code-block::

    net = (total error against truth, this arm)
        - (total error against truth, raw)

summed over **every** keypoint-frame in the pool, corrupted or not. Negative
means the arm left the data closer to the truth than it found it.

This is the one figure an arm cannot win by flattening. Corrupted frames are a
few percent of the pool by construction, matched to the corpus's measured rates,
so damage spread over the other ninety-odd percent outweighs perfect repair of
the few. A smoother that halves the error on 4% of frames and adds a tenth of
that to 96% of them comes out behind, and should.

## What this cannot tell you

The pool is selected for cleanliness and is therefore biased towards slow
behaviour -- see `truth.py`. Damage here **understates** what a smoother does to
fast movement. Every read below carries the speed stratum it was computed on, and
`recovery_read` refuses a pooled verdict when the strata disagree in sign.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: Pre-registered. An arm below this on damage is "leaves correct data alone".
NEGLIGIBLE_DAMAGE_BL = 0.01


def errors(truth: npt.ArrayLike, cleaned: npt.ArrayLike, ell: float) -> F64:
    """``(T, K)`` distance from truth, in body lengths. NaN where truth is NaN."""
    a = np.asarray(truth, dtype=np.float64)
    b = np.asarray(cleaned, dtype=np.float64)
    if not np.isfinite(ell) or ell <= 0:
        return np.full(a.shape[:2], np.nan)
    d = np.linalg.norm(b - a, axis=-1) / float(ell)
    return np.asarray(np.where(np.isfinite(d), d, np.nan), dtype=np.float64)


def score(truth: npt.ArrayLike, cleaned: npt.ArrayLike,
          corrupted_mask: npt.ArrayLike, in_pool: npt.ArrayLike,
          ell: float) -> Detail:
    """`repair`, `damage` and the totals `net` is built from, for one recording.

    `in_pool` is the ``(T,)`` frame mask of clean segments. Everything is scored
    inside it and nowhere else: outside it there is no truth to compare against.
    """
    err = errors(truth, cleaned, ell)
    corrupt = np.asarray(corrupted_mask, dtype=bool)
    pool = np.asarray(in_pool, dtype=bool)[:, None] & np.ones_like(corrupt)
    hit = pool & corrupt & np.isfinite(err)
    miss = pool & ~corrupt & np.isfinite(err)
    return {
        "repair": float(err[hit].mean()) if hit.any() else float("nan"),
        "damage": float(err[miss].mean()) if miss.any() else float("nan"),
        "repair_sum": float(err[hit].sum()) if hit.any() else 0.0,
        "damage_sum": float(err[miss].sum()) if miss.any() else 0.0,
        "total_sum": float(err[hit].sum() + err[miss].sum()),
        "n_corrupted": int(hit.sum()), "n_clean": int(miss.sum()),
        # An arm that returns NaN -- declining to fill an injected dropout --
        # would otherwise vanish from both means and pay nothing for it. The
        # count is reported so declining is visible rather than free.
        "n_unscoreable": int((pool & ~np.isfinite(err)).sum()),
        "n_pool_keypoint_frames": int(pool.sum()),
    }


def by_kind(truth: npt.ArrayLike, cleaned: npt.ArrayLike,
            events: Sequence[Mapping[str, Any]], ell: float) -> Detail:
    """Mean post-arm error per corruption kind.

    The kind split is the point of the exercise, not a decoration: a teleport and
    a park are different errors and an arm that fixes one and not the other is
    correctly described only by two numbers.
    """
    err = errors(truth, cleaned, ell)
    out: Detail = {}
    for ev in events:
        vals = [err[t, k] for k in ev["keypoints"]
                for t in range(int(ev["a"]), int(ev["b"]))
                if np.isfinite(err[t, k])]
        if not vals:
            continue
        cell = out.setdefault(str(ev["kind"]), {"sum": 0.0, "n": 0,
                                                "injected_sum": 0.0})
        cell["sum"] += float(np.sum(vals))
        cell["n"] += len(vals)
        mag = float(ev.get("magnitude_bl", float("nan")))
        if np.isfinite(mag):
            cell["injected_sum"] += mag * len(vals)
    for kind, cell in out.items():
        n = max(1, int(cell["n"]))
        cell["error_bl"] = float(cell["sum"] / n)
        cell["injected_bl"] = float(cell["injected_sum"] / n)
        cell["fraction_of_error_removed"] = (
            float(1.0 - cell["error_bl"] / cell["injected_bl"])
            if cell["injected_bl"] > 0 else float("nan"))
    return out


def recovery_read(rows: Sequence[Mapping[str, Any]],
                  intervals: Mapping[str, Mapping[str, Any]],
                  strata: Mapping[str, Sequence[float]], *,
                  scored_object: Detail, n_effective: int,
                  baseline: str = "raw") -> Read:
    """Which arms left the data closer to the truth than they found it.

    Three guards, each one a lesson this repo paid for:

    * **The baseline must be exact.** `raw` does nothing, so its damage is
      identically zero. If it is not, the harness is measuring itself and no
      other row means anything.
    * **Non-overlapping intervals.** An arm has to clear the baseline's
      animal-bootstrap interval, not its point estimate.
    * **The strata must agree in sign.** The pool is biased towards slow
      behaviour, so a pooled `net` that is negative overall but positive in the
      fastest stratum is an arm that helps still frames and harms moving ones.
      On a fear-conditioning corpus that is a failure being reported as a
      success, and it is refused rather than averaged away.
    """
    by_arm = {str(r["arm"]): r for r in rows}
    if baseline not in by_arm:
        return Read("NOT_A_RESULT", scored_object,
                    f"the {baseline!r} baseline is missing, so there is nothing "
                    f"to measure an improvement against",
                    n_effective=n_effective, degenerate=True,
                    detail={"arms": sorted(by_arm)})

    base_damage = float(by_arm[baseline]["damage"])
    if not (abs(base_damage) < 1e-12):
        return Read(
            "NOT_A_RESULT", scored_object,
            f"the {baseline!r} arm moves nothing, so its damage must be exactly "
            f"zero and it is {base_damage:.3e}. The harness is measuring itself "
            f"and no other row in this table means anything",
            n_effective=n_effective, degenerate=True,
            detail={"baseline_damage": base_damage})

    helped: list[str] = []
    harmed: list[str] = []
    mixed: list[str] = []
    for arm, row in by_arm.items():
        if arm == baseline:
            continue
        iv = intervals.get(arm, {}).get("net")
        if iv is None:
            continue
        if float(iv["hi"]) >= 0.0:
            harmed.append(arm)
            continue
        signs = {float(v) < 0.0 for v in strata.get(arm, []) if np.isfinite(v)}
        (helped if len(signs) <= 1 else mixed).append(arm)

    detail: Detail = {"baseline": baseline, "helped": sorted(helped),
                      "harmed": sorted(harmed), "mixed_by_speed": sorted(mixed),
                      "strata": {k: list(v) for k, v in strata.items()}}
    if mixed:
        return Read(
            "GRID_LIMITED", scored_object,
            f"{', '.join(sorted(mixed))} lower the total error against truth "
            f"overall but not in every speed stratum -- they help still frames "
            f"and harm moving ones. The pool is selected for cleanliness and is "
            f"already biased towards slow behaviour, so a pooled average here "
            f"would report a failure as a success"
            + (f". {', '.join(sorted(helped))} helped in every stratum"
               if helped else ""),
            n_effective=n_effective, detail=detail)
    if not helped:
        return Read(
            "FAIL", scored_object,
            f"no arm lowers the total error against known truth below {baseline}. "
            f"Every one of {', '.join(sorted(harmed))} does more damage to "
            f"correct keypoint-frames than it repairs on corrupted ones, which "
            f"no violation rate, distortion or retention axis could have shown, "
            f"because none of them knows where the keypoint was",
            n_effective=n_effective, detail=detail)
    return Read(
        "PASS", scored_object,
        f"{', '.join(sorted(helped))} leave the data closer to the truth than "
        f"they found it, in every speed stratum, with the animal-bootstrap "
        f"interval on net error entirely below zero"
        + (f". {', '.join(sorted(harmed))} do not" if harmed else ""),
        n_effective=n_effective, detail=detail)
