"""The two controls `CONTEXT.md` never had: stillness, and random windows.

READ results/CONTEXT_CONTROLS_PREREGISTRATION.md FIRST.

## Why this exists

`CONTEXT.md` found clump-0 occupancy differs by context within animal and day --
1.275% in A against 0.470% in B, p = 0.0050 on 89 animals. `BEHAVIOUR.md` found
the same clump runs at 0.272 [0.223, 0.337] of its animals' other-segment speed,
**3.7x slower**. Nothing on disk rules out the possibility that a plain speed
threshold reproduces the whole effect, in which case the detector is a freeze
scorer and the one design-validated finding in this programme is a speed effect.

## The one decision that decides whether this test can fail

The residual is fitted **through the origin**:

    beta = sum(x*y) / sum(x*x)        r = y - beta*x

**not** by ordinary least squares with an intercept. With an intercept the
residuals have mean exactly zero by construction, the sign-flip null could never
reject, and the test `could not have failed` -- the precise failure `journeys.py`
records for the raw W2 contrast and that `METHODS_FINDINGS.md` M5 generalises.
The intercept is the quantity under test and must stay in the residual.

## Why one denominator

Every arm is a rate over the **same** denominator: that cell's total selectable
segment frames, exactly `context.cell_occupancy`'s `den`. Counting instead would
measure how much of each session survived QC, and letting the arms use different
denominators would make their deltas incommensurable at the point where one is
regressed on the other.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import boot                                              # noqa: E402
from recur.read import Read                                         # noqa: E402
from vieb.seg import context as cx                                  # noqa: E402

__all__ = ["cell_sums", "occupancy", "match_threshold", "residual",
           "aligned_deltas", "controls_read", "overlap_read",
           "N_BOOT", "N_PERM"]

CellMap = dict[tuple[str, int, str], float]
F64 = npt.NDArray[np.float64]

N_BOOT = 2000
N_PERM = 2000


def cell_sums(values: npt.ArrayLike, animals: Sequence[str],
              days: npt.ArrayLike, contexts: Sequence[str]) -> CellMap:
    """Sum a per-unit quantity into (animal, day, context) cells.

    Applies the same `DAYS`/`CONTEXTS` filter `cell_occupancy` does, so a
    numerator built here and a denominator built there cover the same cells.
    """
    v = np.asarray(values, dtype=np.float64)
    an = np.asarray(animals)
    dy = np.asarray(days, dtype=np.int64)
    cxs = np.asarray(contexts)
    out: CellMap = {}
    for i in range(v.size):
        if int(dy[i]) not in cx.DAYS or str(cxs[i]) not in cx.CONTEXTS:
            continue
        key = (str(an[i]), int(dy[i]), str(cxs[i]))
        out[key] = out.get(key, 0.0) + float(v[i])
    return out


def occupancy(num: Mapping[tuple[str, int, str], float],
              den: Mapping[tuple[str, int, str], float]) -> CellMap:
    """`num / den` over the denominator's cells, 0 where the arm put nothing.

    Keyed on `den`, never on `num`: an arm that placed nothing in a cell has an
    occupancy of zero there, not a missing cell. A missing cell would drop the
    pair and silently change which cells each arm is compared on.
    """
    return {k: (num.get(k, 0.0) / v if v > 0 else float("nan"))
            for k, v in den.items()}


def match_threshold(speed: npt.ArrayLike, target: float) -> float:
    """The speed whose below-share equals `target`. Matched, never tuned.

    A quantile, so the match is exact by construction rather than searched for.
    The arms are then matched on rate and differ only in HOW frames were
    selected, which is the question. Any other rule -- by eye, a round quantile,
    or whichever value moves a delta -- would be fitting.
    """
    s = np.asarray(speed, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0 or not (0.0 < target < 1.0):
        return float("nan")
    return float(np.quantile(s, target))


def residual(y: npt.ArrayLike, x: npt.ArrayLike) -> tuple[float, F64]:
    """`y` with its `x`-explained part removed, fitted THROUGH THE ORIGIN.

    See the module docstring: an intercept would zero the residual mean by
    construction and the null could never reject.
    """
    a = np.asarray(y, dtype=np.float64)
    b = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    ss = float((b[ok] ** 2).sum())
    beta = float((a[ok] * b[ok]).sum() / ss) if ss > 0 else 0.0
    return beta, a - beta * b


def aligned_deltas(arms: Mapping[str, Mapping[tuple[str, int, str], float]],
                   den: Mapping[tuple[str, int, str], float]) -> dict[str, Any]:
    """One signed delta per (animal, day) cell, per arm, on identical cells.

    `paired_deltas` sorts its cells, so every arm built on the same `den` comes
    back in the same order -- but that is asserted here rather than assumed,
    because a silent misalignment would regress one animal's delta on another's
    and look entirely reasonable doing it.
    """
    out: dict[str, Any] = {"arms": {}}
    keys: list[str] | None = None
    for name, occ in arms.items():
        got = cx.paired_deltas(occ, den)
        if keys is None:
            keys = list(got["pair_key"])
            out["pair_key"] = keys
            out["animal"] = list(got["animal"])
        elif list(got["pair_key"]) != keys:
            raise SystemExit(
                f"arm {name!r} produced a different cell set from the first "
                f"arm; the arms would not be paired cell-for-cell")
        out["arms"][name] = got
    out["n_pairs"] = len(keys or [])
    return out


def controls_read(res: Mapping[str, Any], flip: Mapping[str, Any], *,
                  arm: str, beta: float, scored_object: dict[str, Any],
                  n_effective: int) -> Read:
    """Does clump membership add anything beyond `arm`?

    `PASS` means it does -- the residual interval excludes zero, so boundary
    placement carries context information the control arm does not. `FAIL` means
    it does not, and for the stillness arm that is the registered expectation:
    the detector is a freeze scorer.

    The verdict follows the interval and not the p-value, the precedent
    `LEARNING_CURVE.md` fixed when a stabilisation arm at p = 0.0475 still read
    FAIL because its CI spanned zero.
    """
    lo, hi = float(res["lo"]), float(res["hi"])
    point = float(res["point"])
    p = float(flip.get("p_two_sided", float("nan")))
    excludes = (lo > 0.0) or (hi < 0.0)
    detail = {"residual": dict(res), "flip": dict(flip), "beta": float(beta),
              "control_arm": arm}
    if excludes:
        reason = (
            f"CLUMP MEMBERSHIP ADDS TO {arm.upper()}: residual "
            f"{point:+.5f} [{lo:+.5f}, {hi:+.5f}] over {res['n_animals']} "
            f"animals and {res['n_units']} (animal, day) cells, pair-flip "
            f"p = {p:.4f}, after removing {beta:.3f} x the {arm} delta fitted "
            f"through the origin. The context effect is not reducible to "
            f"{arm} alone")
        return Read("PASS", scored_object, reason, n_effective=n_effective,
                    detail=detail)
    reason = (
        f"clump membership adds NOTHING beyond {arm}: residual {point:+.5f} "
        f"[{lo:+.5f}, {hi:+.5f}] spans zero over {res['n_animals']} animals "
        f"and {res['n_units']} cells, pair-flip p = {p:.4f}, after removing "
        f"{beta:.3f} x the {arm} delta fitted through the origin. "
        + ("On the registered reading that makes the detector a FREEZE SCORER "
           "and the context result a speed effect -- a useful instrument, but "
           "not a tokenizer" if arm == "stillness" else
           "which is what a control arm is supposed to do"))
    return Read("FAIL", scored_object, reason, n_effective=n_effective,
                detail=detail)


def overlap_read(selected: npt.ArrayLike, island: npt.ArrayLike, *,
                 arm: str, theta: float, scored_object: dict[str, Any],
                 n_effective: int, floor: float = 0.25) -> Read:
    """Does the control arm select anything like the thing it controls for?

    **A control that selects a disjoint set cannot test the hypothesis it was
    registered to test.** If almost no island frame is also a control frame,
    then "clump membership adds to the control" is near-trivially true -- the
    two arms are measuring different things, and the residual says so rather
    than saying anything about the detector.

    `METHODS_FINDINGS.md` M5: a control that returns nothing is usually testing
    the control. This read exists so that failure is visible in the result
    rather than inferred by a reader who happens to check.

    `floor` is the share of island frames the control must also select for the
    comparison to mean anything. It is a reporting threshold, not a verdict
    boundary -- the read is always `NOT_A_RESULT`.
    """
    sel = np.asarray(selected, dtype=bool)
    isl = np.asarray(island, dtype=bool)
    n_isl = int(isl.sum())
    n_sel = int(sel.sum())
    both = int((sel & isl).sum())
    share_isl = (both / n_isl) if n_isl else float("nan")
    share_sel = (both / n_sel) if n_sel else float("nan")
    detail = {"control_arm": arm, "theta": float(theta),
              "n_island_frames": n_isl, "n_control_frames": n_sel,
              "n_both": both, "share_of_island_selected": share_isl,
              "share_of_control_in_island": share_sel,
              "reporting_floor": float(floor)}
    if not np.isfinite(share_isl) or share_isl < floor:
        return Read(
            "NOT_A_RESULT", scored_object,
            (f"THE {arm.upper()} ARM IS NOT A CONTROL FOR THIS CLUMP: it "
             f"selects only {share_isl:.4f} of the island's own frames at "
             f"theta = {theta:.6g}, against a reporting floor of {floor:.2f}. "
             f"The two arms select near-disjoint frame sets, so a residual "
             f"that excludes zero says they measure different things and says "
             f"nothing about whether the detector is a freeze scorer. Any "
             f"verdict computed against this arm is vacuous and is reported "
             f"as such"),
            n_effective=n_effective, detail=detail)
    return Read(
        "NOT_A_RESULT", scored_object,
        (f"descriptive: the {arm} arm selects {share_isl:.4f} of the island's "
         f"own frames at theta = {theta:.6g}, and {share_sel:.4f} of what it "
         f"selects is island. The arms overlap enough for the residual to be "
         f"about the detector"),
        n_effective=n_effective, detail=detail)
