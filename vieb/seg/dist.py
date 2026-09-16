r"""Two distances over variable-length segments, and the search they need.

Registered in `results/DISTANCE_PREREGISTRATION.md`.

## Why there are two

The distance Step 3's clumps were found with time-normalises every segment to 40
points, so **duration is free by construction**: two movements of the same shape
at different tempos score as identical, and matched partners differ by a mean
factor of 3.7x in duration. The clumps may therefore be a property of the ruler.

`open` -- compare over `m = min(T_a, T_b)` -- is the secondary
`SEGRECUR_PREREGISTRATION.md` registered and never ran, so it is the primary
here. But it compares only the **shared prefix**, so a 0.5 s segment still
matches the first 0.5 s of a 65 s one at near-zero distance. It tests whether the
**warp** matters. It does not test whether **duration** does.

`union` -- compare over `M = max(T_a, T_b)`, the shorter held at its last frame
-- charges the unmatched tail as mismatch. It is the only one of the two that can
drive the matched-partner duration ratio toward 1, and it is registered as the
decisive diagnostic rather than as the primary.

**Hold-last, not zero-pad.** `vieb/clean/arms.py:held_array` already holds the
last observation across gaps, so the convention is the repository's own. Zero
padding would charge the tail against an origin the egocentric space does not
have: these channels are shape and SE(2) *increments*, and zero is "no change",
not "no data".

## Why neither can use the exact search

Both have a **per-pair extent**, so neither is a fixed-dimension Euclidean
distance. `recur.recurrence.search.search` computes `q_sq + bank_sq - 2 Q B^T`,
which requires one inner-product space shared by every pair. Segment lengths span
16-4,537 frames; there is no such space.

So: retrieve exactly in a fixed 16-frame prefix space -- rectangular, so the
existing machinery runs unmodified -- then rerank the candidates under the true
metric. The retrieval is an approximation and `recovery_read` measures it rather
than apologising for it.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

__all__ = ["CANDIDATES", "METRICS", "PREFIX_L", "metric_fn", "open_end",
           "pairwise", "recovery_read", "rerank", "union_end"]

#: Fixed in the registration. The identifiability floor, and the length at which
#: 100% of segments survive truncation -- so the retrieval stage drops nothing.
PREFIX_L = 16
#: The candidate budget, fixed in the registration.
CANDIDATES = 64
METRICS: tuple[str, ...] = ("open", "union")


def open_end(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compare the shared extent, `m = min(T_a, T_b)`, normalised per frame.

    The registered primary. Per-frame normalisation is what keeps it from being
    a length readout -- without it a longer shared extent would score worse
    simply for being longer.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = int(min(x.shape[0], y.shape[0]))
    if m < 1:
        return float("nan")
    d = x[:m] - y[:m]
    return float(np.sqrt(float((d ** 2).sum()) / m))


def union_end(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compare the union extent, `M = max(T_a, T_b)`, shorter held at its last.

    The decisive diagnostic. Where `open_end` stops at the shorter segment,
    this carries it forward as a constant and charges every frame the longer one
    continues to move. A 0.5 s segment and a 65 s segment can be identical under
    `open_end` and far apart here, which is the whole point.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape[0] < 1 or y.shape[0] < 1:
        return float("nan")
    m = int(max(x.shape[0], y.shape[0]))
    xs = x if x.shape[0] == m else np.concatenate(
        [x, np.repeat(x[-1:], m - x.shape[0], axis=0)])
    ys = y if y.shape[0] == m else np.concatenate(
        [y, np.repeat(y[-1:], m - y.shape[0], axis=0)])
    d = xs - ys
    return float(np.sqrt(float((d ** 2).sum()) / m))


def metric_fn(name: str) -> Callable[[npt.ArrayLike, npt.ArrayLike], float]:
    if name == "open":
        return open_end
    if name == "union":
        return union_end
    raise ValueError(f"unknown metric {name!r}; one of {METRICS}")


def rerank(blocks: Sequence[npt.ArrayLike], query: int,
           candidates: npt.ArrayLike, *, metric: str,
           forbid: npt.ArrayLike | None = None) -> tuple[int, float]:
    """True nearest among the candidates, under the true metric.

    `forbid` marks candidates that must not win -- the query's own animal, for a
    cross-animal statistic. Masking here rather than at retrieval keeps the
    retrieval stage a pure geometry question.
    """
    fn = metric_fn(metric)
    cand = np.asarray(candidates, dtype=np.int64)
    bad = (np.zeros(cand.size, dtype=bool) if forbid is None
           else np.asarray(forbid, dtype=bool))
    best_i, best_d = -1, float("inf")
    for j, c in enumerate(cand.tolist()):
        if c < 0 or c == int(query) or bad[j]:
            continue
        d = fn(blocks[int(query)], blocks[int(c)])
        if np.isfinite(d) and d < best_d:
            best_i, best_d = int(c), float(d)
    return best_i, (best_d if best_i >= 0 else float("nan"))


def pairwise(blocks: Sequence[npt.ArrayLike], queries: npt.ArrayLike,
             animal: npt.ArrayLike, *, metric: str) -> tuple[I64, F64]:
    """Exact cross-animal nearest neighbour over ALL of `blocks`.

    The reference the retrieval stage is measured against. O(n_queries x n), so
    it is run on a reduced bank and never on the corpus -- which is the reason
    the two-stage search exists at all.
    """
    fn = metric_fn(metric)
    q = np.asarray(queries, dtype=np.int64)
    an = np.asarray(animal)
    idx = np.full(q.size, -1, dtype=np.int64)
    dst = np.full(q.size, np.inf, dtype=np.float64)
    for r, i in enumerate(q.tolist()):
        for j in range(len(blocks)):
            if j == i or an[j] == an[i]:
                continue
            d = fn(blocks[i], blocks[j])
            if np.isfinite(d) and d < dst[r]:
                idx[r], dst[r] = j, d
    return idx, dst


def recovery_read(approx_idx: npt.ArrayLike, exact_idx: npt.ArrayLike,
                  approx_d: npt.ArrayLike, exact_d: npt.ArrayLike, *,
                  metric: str, scored_object: Detail, n_effective: int,
                  limit: float = 0.90) -> Read:
    """How much of the exact answer the candidate stage recovered.

    Q1's search is exact and says so; this one is not, and the registration says
    so in advance. What it must not do is leave the size of the approximation
    unmeasured -- an index that loses the true neighbour on a tenth of queries
    would shift every distance quantile the statistic reads.

    Two numbers, because they fail differently. **Index agreement** is how often
    the same neighbour was found. **Distance inflation** is how much worse the
    candidate answer is when it is not the same -- a retrieval that misses the
    true neighbour but lands on one equally close has not damaged the statistic.
    """
    ai = np.asarray(approx_idx, dtype=np.int64)
    ei = np.asarray(exact_idx, dtype=np.int64)
    ad = np.asarray(approx_d, dtype=np.float64)
    ed = np.asarray(exact_d, dtype=np.float64)
    ok = (ei >= 0) & np.isfinite(ed)
    n = int(ok.sum())
    if n < 10:
        return Read("INCONCLUSIVE", scored_object,
                    f"only {n} queries had an exact cross-animal neighbour, "
                    f"which is too few to measure the retrieval against",
                    n_effective=n_effective, detail={"n": n})
    agree = float((ai[ok] == ei[ok]).mean())
    ratio = ad[ok] / np.maximum(ed[ok], 1e-12)
    inflation = float(np.median(ratio))
    worst = float(np.quantile(ratio, 0.99))
    detail: Detail = {"n": n, "index_agreement": agree,
                      "median_distance_ratio": inflation,
                      "p99_distance_ratio": worst, "metric": metric,
                      "candidates": CANDIDATES, "prefix_L": PREFIX_L}
    if agree < limit:
        return Read("FAIL", scored_object,
                    f"the {PREFIX_L}-frame prefix retrieval recovers the true "
                    f"{metric} nearest neighbour on only {agree:.1%} of "
                    f"{n:,} queries, below the {limit:.0%} limit; the candidate "
                    f"answer is {inflation:.4f}x the exact distance at the "
                    f"median. The index is losing the answer, so the excess "
                    f"read off it is not the excess",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"the {PREFIX_L}-frame prefix retrieval recovers the true "
                f"{metric} nearest neighbour on {agree:.1%} of {n:,} queries, "
                f"and where it does not the candidate sits {inflation:.4f}x the "
                f"exact distance at the median ({worst:.4f}x at p99). The "
                f"approximation is at the retrieval step only and its size is "
                f"measured rather than assumed",
                n_effective=n_effective, detail=detail)
