"""Is the planted instance findable at all? The statistic, and its verdict.

READ results/PROBE_AUDIT_PREREGISTRATION.md FIRST.

Step B asked whether the detector isolates a planted instance and answered
12.8% against a 20% gate, with `merged` and `split` at 0.0% in every cell -- so
the refractory period and the threshold are both exonerated and the failure is
`no_edge`: no boundary at either edge of the thing the detector is meant to
find.

This module asks the prior question. **Not** "does the detector find the plant"
but "is there anything at the plant's edges for an acceleration criterion to
respond to". That is a property of the probe, and it is measurable without a
detector, without a threshold and without a human.

## Why a percentile and not the raw magnitude

`D(t) = ‖ẍ⁺ − ẍ⁻‖` carries the units of the ego space and its scale differs
between recordings -- a fast animal in a bright box has a larger `D` everywhere
than a still one. Pooling raw magnitudes would rank recordings, not frames. So
every frame is scored by its **percentile within its own recording's `D`
distribution**: unit-free, bounded, and with a null that is 0.50 by
construction rather than by estimation.

That null is the reason this design can fail loudly. `random` frames must land
at 0.50; if they do not, the scoring is wrong and `anchor_read` refuses
rather than reporting a number that happens to look plausible. This programme
has a ledger entry for a plausible-looking wrong result -- a double
scale-division that produced zero clumps and 100% unassigned in every arm, and
was caught only because a threshold was compared against the data it was applied
to.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import boot                                              # noqa: E402
from recur.read import Read                                         # noqa: E402

__all__ = ["ANCHOR_TOL", "N_BOOT", "percentile_of", "edge_frames",
           "populations", "anchor_read", "separation_read"]

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

#: How far the `random` anchor may sit from 0.50 before the stage refuses. A
#: count-matched uniform draw scored against its own distribution has a mean
#: percentile of 0.50 by construction; drift beyond this is a bug in the
#: scoring, not a finding about the plant.
ANCHOR_TOL = 0.02
N_BOOT = 2000


def percentile_of(d: ArrayLike, at: ArrayLike) -> F64:
    """Percentile rank of `d[at]` within the finite values of `d`.

    Ties are averaged rather than taken low, so a recording whose `D` is
    constant over a long still stretch does not hand every frame in it a rank of
    zero. `searchsorted` on both sides is the cheap way to get that without
    sorting twice.
    """
    v = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(v)
    ref = np.sort(v[ok])
    if ref.size == 0:
        return np.full(np.asarray(at).size, np.nan, dtype=np.float64)
    q = v[np.asarray(at, dtype=np.int64)]
    lo = np.searchsorted(ref, q, side="left").astype(np.float64)
    hi = np.searchsorted(ref, q, side="right").astype(np.float64)
    out = 0.5 * (lo + hi) / float(ref.size)
    return np.where(np.isfinite(q), out, np.nan)


def edge_frames(mask: ArrayLike, *, lo: int, hi: int, tol: int) -> I64:
    """Frames within `tol` of the start or stop of every planted run.

    The mask is the planted-instance mask over the whole corpus array; `lo`/`hi`
    bound one recording. Runs are read off the mask's own transitions rather
    than from the planting metadata, so this cannot drift from what was actually
    written into the signal.

    Returned in the SAME index space as the slice -- zero-based within
    `[lo, hi)` -- because `D` is computed per recording span and indexing it
    with corpus-global frames is the seam bug this repo asserts against.
    """
    m = np.asarray(mask, dtype=bool)[lo:hi]
    if not m.any():
        return np.zeros(0, dtype=np.int64)
    pad = np.concatenate(([False], m, [False]))
    step = np.flatnonzero(pad[1:] != pad[:-1])
    starts, stops = step[0::2], step[1::2]
    want: list[int] = []
    n = int(hi - lo)
    for e in np.concatenate([starts, stops]):
        want.extend(range(max(0, int(e) - tol), min(n, int(e) + tol + 1)))
    return np.unique(np.asarray(want, dtype=np.int64))


def populations(d: ArrayLike, planted: ArrayLike, fired: ArrayLike,
                rng: np.random.Generator, *, selectable: ArrayLike | None = None
                ) -> dict[str, float]:
    """Mean percentile of `D` at the planted edges, the firings, and random.

    `random` is drawn **count-matched to `planted`** from the selectable span,
    so the anchor is estimated at the same precision as the thing it anchors.
    Every index is slice-local.

    Returns NaNs rather than raising when a recording has no planted edge or no
    firing: a recording that contributes nothing is dropped by the bootstrap,
    which is different from one that contributes a zero.
    """
    v = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(v)
    if selectable is not None:
        ok = ok & np.asarray(selectable, dtype=bool)
    pool = np.flatnonzero(ok)
    p = np.asarray(planted, dtype=np.int64)
    f = np.asarray(fired, dtype=np.int64)
    out: dict[str, float] = {"n_planted": float(p.size),
                             "n_fired": float(f.size),
                             "n_pool": float(pool.size)}
    if p.size and pool.size:
        draw = rng.choice(pool, size=int(p.size), replace=pool.size < p.size)
        out["pct_planted"] = float(np.nanmean(percentile_of(v, p)))
        out["pct_random"] = float(np.nanmean(percentile_of(v, draw)))
    else:
        out["pct_planted"] = float("nan")
        out["pct_random"] = float("nan")
    out["pct_fired"] = (float(np.nanmean(percentile_of(v, f))) if f.size
                        else float("nan"))
    return out


def anchor_read(rows: Sequence[Mapping[str, Any]], *,
                scored_object: dict[str, Any], n_effective: int,
                tol: float = ANCHOR_TOL, seed: int = 0) -> Read:
    """The anchor must land at 0.50. If it does not, nothing else is readable.

    A uniform draw scored against its own distribution has mean percentile 0.50
    exactly. This is not a hypothesis being tested -- it is an arithmetic
    identity, so a departure is a defect in the scoring and the correct verdict
    is NOT_A_RESULT for the whole stage rather than FAIL for this line.
    """
    v = [float(r["pct_random"]) for r in rows]
    a = [str(r["animal"]) for r in rows]
    ci = boot.animal_interval(v, a, how="mean", n_boot=N_BOOT, seed=seed)
    off = abs(float(ci["point"]) - 0.5)
    if not np.isfinite(off) or off > tol:
        return Read(
            verdict="NOT_A_RESULT", scored_object=scored_object,
            n_effective=n_effective, detail={"anchor": ci, "tol": tol},
            reason=(f"THE ANCHOR MOVED. Count-matched random frames scored "
                    f"{ci['point']:.4f} against the 0.5000 that a uniform draw "
                    f"has by construction, off by {off:.4f} > {tol}. That is a "
                    f"defect in the scoring, not a finding about the plant, so "
                    f"no other number from this stage is readable."))
    return Read(
        verdict="PASS", scored_object=scored_object, n_effective=n_effective,
        detail={"anchor": ci, "tol": tol},
        reason=(f"the anchor holds: count-matched random frames score "
                f"{ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] against "
                f"the 0.5000 a uniform draw has by construction"))


def separation_read(rows: Sequence[Mapping[str, Any]], *,
                    scored_object: dict[str, Any], n_effective: int,
                    seed: int = 0) -> Read:
    """Does the plant's edge stand above random, and how far below a real firing?

    The three verdicts are the three rows of the registration's reading table
    and are not chosen here:

    * interval includes 0 -> the plant has no edge. Step B measured the probe.
    * excludes 0, but `pct_planted` below the LOWER BOUND of `pct_fired` ->
      a weak edge; the 12.8% is part probe and part criterion.
    * `pct_planted` at or above that lower bound -> the plant is findable and
      Step B stands.

    "Well below" was fixed as "below the lower bound of the firing interval"
    before any number existed, so no threshold is chosen after the fact.
    """
    a = [str(r["animal"]) for r in rows]
    delta = [float(r["pct_planted"]) - float(r["pct_random"]) for r in rows]
    ci = boot.animal_interval(delta, a, how="mean", n_boot=N_BOOT, seed=seed)
    fired = boot.animal_interval([float(r["pct_fired"]) for r in rows], a,
                                 how="mean", n_boot=N_BOOT, seed=seed)
    planted = boot.animal_interval([float(r["pct_planted"]) for r in rows], a,
                                   how="mean", n_boot=N_BOOT, seed=seed)
    # The frame-level interval, printed ONLY to show how much narrower the wrong
    # method looks. It licenses nothing.
    d = np.asarray(delta, dtype=np.float64)
    d = d[np.isfinite(d)]
    naive = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
             ) if d.size > 1 else (float("nan"), float("nan"))
    detail: dict[str, Any] = {"delta": ci, "pct_planted": planted, "pct_fired": fired,
              "frame_level_interval_do_not_quote": list(naive)}
    span = ("[%+.4f, %+.4f]" % (ci["lo"], ci["hi"]))
    if not np.isfinite(ci["lo"]) or ci["lo"] <= 0.0 <= ci["hi"]:
        return Read(
            verdict="NOT_A_RESULT", scored_object=scored_object,
            n_effective=n_effective, detail=detail,
            reason=(f"THE PLANT HAS NO EDGE TO FIND. Planted edges score "
                    f"{planted['point']:.4f} against random's "
                    f"{0.5:.4f}, a difference of {ci['point']:+.4f} {span} "
                    f"which spans zero, while the detector's own firings score "
                    f"{fired['point']:.4f}. Step B measured the probe, not the "
                    f"detector: its FAIL is withdrawn as uninterpretable, "
                    f"which is not a pass and says nothing about real "
                    f"boundaries."))
    if planted["point"] < fired["lo"]:
        return Read(
            verdict="INCONCLUSIVE", scored_object=scored_object,
            n_effective=n_effective, detail=detail,
            reason=(f"the plant has a WEAK edge: planted edges score "
                    f"{planted['point']:.4f}, above random by {ci['point']:+.4f} "
                    f"{span}, but below the {fired['lo']:.4f} lower bound of "
                    f"the detector's own firings at {fired['point']:.4f}. The "
                    f"12.8% is part probe and part criterion and this design "
                    f"cannot separate them."))
    return Read(
        verdict="PASS", scored_object=scored_object, n_effective=n_effective,
        detail=detail,
        reason=(f"the plant IS findable: planted edges score "
                f"{planted['point']:.4f}, above random by {ci['point']:+.4f} "
                f"{span} and inside the firing interval "
                f"[{fired['lo']:.4f}, {fired['hi']:.4f}]. The criterion really "
                f"is limited and Step B stands."))
