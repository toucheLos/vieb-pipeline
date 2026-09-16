r"""Segments as rows of a bank, so Q1's machinery runs on them unmodified.

## The one structural problem, and the trick that avoids new code

`recur.recurrence.bank.iter_chunks` gathers windows as
``s[:, None] + arange(w)[None, :]``, a rectangular index. Unequal-length rows
cannot be gathered that way, and `build` allocates a single ``(N, k)``. That is
the whole reason segments could not previously be fed to the recurrence search.

The fix needs no library change. **Time-normalise each segment to a fixed grid,
then lay the results end to end in a synthetic array** where segment *i* occupies
rows ``G·i … G·i+G-1``. With ``start = G·arange(n)`` and ``w = G`` the rows are
rectangular again, and `fit_pca`, `build`, `pca_read` and `sort_by_animal` run
**completely unmodified** -- PCA validity read included.

## What is excluded from the bank, and why it is an exclusion and not a flag

**Segments touching abstain, and segments beside it.** Abstain is 7.03% of
frames arriving in short runs, so it fragments the stream far more than it
covers it: on one animal the detector found 1,081 peaks while the table held
2,532 segments, the difference being abstain edges. `segment_table` already cuts
at every abstain run, which means a segment can contain no abstained frame and
still have had one of its two boundaries **manufactured by the dropout beside
it**. That is not a behavioural unit, and flagging it would leave it in the bank.

**Seam adjacency is measured but not excluded**, because a recording edge is a
real edge rather than a tracking failure. It is reported so the reader can see
how much of the bank has one undetected boundary.

## The windowed control lives here too, on purpose

The comparator for "did the excess rise or fall" is a fixed-window arm in **this
same space**, and the cheapest way to guarantee it is the same space is to build
it with the same function. Its lengths are **drawn from the arm's own clean
segment durations**, so bank size and length distribution are both controlled
and the only difference left is where the edges fall.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.render.meanskel import resample

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["SEG_GRID", "mark_edges", "matched_windows", "open_end_distance",
           "resample_segment", "selectable", "stack_segments"]

#: Points every segment is warped to. `meanskel.N_GRID` is 40 and is the grid
#: this project already time-normalises runs on; reusing the number keeps one
#: definition of "time-normalised" rather than two.
SEG_GRID = 40


def mark_edges(rows: Sequence[Mapping[str, Any]], abstain: npt.ArrayLike, *,
               lo: int, hi: int) -> list[Detail]:
    """Tag each segment with whether its edges are real boundaries.

    `abstain_adjacent` is true when the frame immediately before `start` or at
    `stop` is abstained -- i.e. this segment's edge is where a dropout began or
    ended, not where the acceleration broke. `seam_adjacent` is true at the
    recording's own ends.

    Both are computed from the abstain mask rather than inferred from the peak
    list, because `segment_table` cuts at abstain runs without recording that it
    did, and re-deriving it from peaks would need the peaks to be passed here
    only to answer a question the mask already answers.
    """
    ab = np.asarray(abstain, dtype=bool)
    out: list[Detail] = []
    for r in rows:
        s_, e_ = int(r["start"]), int(r["stop"])
        before = bool(ab[s_ - 1]) if s_ - 1 >= int(lo) else False
        after = bool(ab[e_]) if e_ < int(hi) else False
        out.append({**r,
                    "abstain_adjacent": bool(before or after),
                    "seam_adjacent": bool(s_ <= int(lo) or e_ >= int(hi))})
    return out


def selectable(rows: Sequence[Mapping[str, Any]]) -> BOOL:
    """The registered mask: fittable, no abstained frame, not beside one.

    Three conditions and not two. `abstain_frac == 0` alone leaves in every
    segment whose boundary the dropout created, which is the population the
    exclusion exists for.
    """
    return np.asarray(
        [bool(r.get("fittable", False)) and float(r["abstain_frac"]) == 0.0
         and not bool(r.get("abstain_adjacent", False)) for r in rows],
        dtype=bool)


def resample_segment(x: npt.ArrayLike, start: int, stop: int,
                     n: int = SEG_GRID) -> F64:
    """One segment, time-normalised to `n` points.

    Sliced on the **original** samples and warped once. `fit_quality` deliberately
    fits before any resampling -- resampling then fitting would let the fit
    explain an interpolant -- but the distance is a different object: here the
    warp is the registered primary form, and every arm including the nulls is
    warped identically, so whatever it manufactures applies to both sides.
    """
    a = np.asarray(x, dtype=np.float64)[int(start):int(stop)]
    if a.shape[0] < 1:
        raise ValueError(f"empty segment [{start}, {stop})")
    # `recur.render.meanskel.resample` is untyped; reused rather than
    # reimplemented so "time-normalised" has one definition in this
    # programme, not two that could drift apart.
    warped = resample(a, n)  # type: ignore[no-untyped-call]
    return np.asarray(warped, dtype=np.float64)


def stack_segments(x: npt.ArrayLike, rows: Sequence[Mapping[str, Any]],
                   *, cols: Sequence[int], n: int = SEG_GRID
                   ) -> tuple[F32, I64]:
    """`(G·m, C)` synthetic array plus `start = G·arange(m)`.

    The shape `bank.build` needs, from rows it could not otherwise gather.
    """
    a = np.asarray(x, dtype=np.float64)[:, list(cols)]
    m = len(rows)
    out = np.empty((m * int(n), a.shape[1]), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i * n:(i + 1) * n] = resample_segment(a, int(r["start"]),
                                                  int(r["stop"]), n)
    return out, (np.arange(m, dtype=np.int64) * int(n))


def matched_windows(rng: np.random.Generator, *, bounds: npt.ArrayLike,
                    abstain: npt.ArrayLike, durations: npt.ArrayLike,
                    n_target: int, max_tries: int = 40) -> list[Detail]:
    """Fixed windows whose LENGTHS are drawn from the segment durations.

    The registered comparator. Matching the length distribution and the count
    means the windowed arm differs from the segment arm in exactly one way --
    where its edges fall -- which is the thing under test. A comparator at a
    single arbitrary window length would differ in two ways at once, and the
    bank-size effect alone would move the answer: nearest-neighbour distance
    falls as a bank grows, through extreme-value effects that have nothing to do
    with recurrence.

    Windows containing any abstained frame are rejected rather than repaired, so
    the two arms carry the same abstain policy.
    """
    b = np.asarray(bounds, dtype=np.int64)
    ab = np.asarray(abstain, dtype=bool)
    lens = np.asarray(durations, dtype=np.int64)
    if lens.size == 0 or b.size < 2:
        return []
    # Cumulative abstain count makes "is this window clean" one subtraction
    # rather than a slice mean per try.
    cum = np.concatenate([[0], np.cumsum(ab.astype(np.int64))])
    spans = [(int(b[r]), int(b[r + 1])) for r in range(b.shape[0] - 1)]
    widths = np.asarray([hi - lo for lo, hi in spans], dtype=np.float64)
    p = widths / widths.sum() if widths.sum() > 0 else None
    out: list[Detail] = []
    for _ in range(int(n_target)):
        w = int(rng.choice(lens))
        placed = False
        for _try in range(int(max_tries)):
            r = int(rng.choice(len(spans), p=p))
            lo, hi = spans[r]
            if hi - lo <= w:
                continue
            s_ = int(rng.integers(lo, hi - w))
            if int(cum[s_ + w] - cum[s_]) != 0:
                continue
            out.append({"start": s_, "stop": s_ + w, "n_frames": w,
                        "abstain_frac": 0.0, "fittable": True,
                        "abstain_adjacent": False,
                        "seam_adjacent": bool(s_ <= lo or s_ + w >= hi)})
            placed = True
            break
        if not placed:
            continue
    return out


def open_end_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """The registered SECONDARY: no warp, compared over the shared extent.

    The shorter segment is matched against the longer one's leading extent and
    the distance is normalised by the compared length, so it is a per-frame
    quantity and not a length readout. Registered as secondary precisely because
    it assumes the opposite of the primary -- that duration is signal -- and the
    gate reads the primary.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = int(min(x.shape[0], y.shape[0]))
    if m < 1:
        return float("nan")
    d = x[:m] - y[:m]
    return float(np.sqrt(float((d ** 2).sum()) / m))
