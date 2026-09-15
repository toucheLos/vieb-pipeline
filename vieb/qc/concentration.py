r"""Are tracking errors spread evenly, or concentrated on particular footage?

The question behind the ensemble decision. If violations were independent network
noise they would fall at roughly the corpus rate everywhere; if the **footage** is
the limit they concentrate — on particular recordings, particular arena positions,
particular sessions. The worst recording already reads 52.7% violating against a
2.134% corpus mean, a 25x spread, and this module asks whether that spread is
structure or the tail of a binomial.

## What this can and cannot decide

**It cannot separate the two hypotheses.** Hard footage, and a network that fails
on hard footage, look identical from outside; only across-network variance
separates them, and that needs an ensemble this project does not have.

What it **can** do is bound the ensemble's likely benefit. An ensemble averages
away errors its members make *independently*. Errors driven by the situation --
an occlusion, a rear against a wall -- are errors every member would make, and
averaging cannot remove them. So a high dispersion ratio says the ensemble has
less to work with than an even spread would imply. That is a bound, not a verdict,
and every read here says so in its own reason string rather than leaving it to
the surrounding prose.

## The null that makes the ratio mean something

A bare Gini is uninterpretable: any finite sample of a Bernoulli process has
some. The comparison is to the **binomial** variance the same per-recording frame
counts would produce at the pooled rate:

.. math::

    \text{dispersion} = \frac{\operatorname{Var}(\hat p_i)}
                             {\operatorname{E}[p(1-p)/n_i]}

One is exactly what independent frames give. Well above one is structure. The
statistic is the standard overdispersion ratio and it is used here because it has
a closed-form null rather than because it is fancy.

Runs are the obvious confound and are not corrected for: violations come in runs
(median 1 frame, p75 3 -- `runlen.json`), so frames are not independent even under
a pure-noise model, and some overdispersion is guaranteed. The ratio is therefore
an **upper** bound on structure, and `concentration_read` says so.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

from vieb.checks import TOL, assert_share

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: Above this the per-recording spread is far enough beyond binomial that the
#: errors are situation-driven rather than independent. Not tuned -- an order of
#: magnitude is the scale at which the distinction stops being arguable.
DISPERSION_STRUCTURED = 10.0

#: Shares reported on the Lorenz curve, against the same share of a uniform
#: process. The baseline is printed beside every one of them.
LORENZ_SHARES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)

#: Edge-ness deciles for the arena profile.
N_EDGE_BINS = 10


def dispersion(counts: npt.ArrayLike, sizes: npt.ArrayLike) -> Detail:
    """Overdispersion of per-recording violation rates against a binomial null.

    `counts` is violating frames per recording, `sizes` total frames. The null is
    that every frame violates independently at the pooled rate.
    """
    k = np.asarray(counts, dtype=np.float64)
    n = np.asarray(sizes, dtype=np.float64)
    ok = n > 0
    k, n = k[ok], n[ok]
    if k.size < 2:
        return {"dispersion": float("nan"), "n_recordings": int(k.size),
                "why": "fewer than two recordings"}
    p = float(k.sum() / n.sum())
    rates = assert_share(k / n, name="per-recording violation rate")
    observed = float(np.var(rates, ddof=1))
    expected = float(np.mean(p * (1.0 - p) / n))
    return {
        "pooled_rate": p,
        "observed_variance": observed,
        "binomial_variance": expected,
        "dispersion": float(observed / expected) if expected > 0 else float("nan"),
        "n_recordings": int(k.size),
        "rate_min": float(rates.min()), "rate_max": float(rates.max()),
        "rate_p50": float(np.median(rates)), "rate_p99": float(np.percentile(rates, 99)),
    }


def lorenz(counts: npt.ArrayLike, *,
           shares: Sequence[float] = LORENZ_SHARES) -> Detail:
    """What share of violating frames the worst recordings carry.

    Each share is reported beside the share a uniform process would give, so the
    number is not read bare -- the worst 10% of recordings carrying 10% of the
    violations is no concentration at all.
    """
    k = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    total = float(k.sum())
    n = int(k.size)
    if n == 0 or total <= 0:
        return {"gini": float("nan"), "shares": {}, "n_recordings": n}
    out: Detail = {}
    for s in shares:
        take = max(1, int(round(float(s) * n)))
        out[f"worst_{int(s * 100)}pct"] = {
            "share_of_violations": float(k[:take].sum() / total),
            "uniform_baseline": float(take / n),
        }
    # Cumulative shares of a sorted non-negative vector: each is a fraction and
    # they must increase with the fraction of recordings taken. Neither was
    # checked, and a Lorenz curve that decreases is a sorting bug.
    cum = [out[f"worst_{int(s * 100)}pct"]["share_of_violations"] for s in shares]
    assert_share(cum, name="Lorenz cumulative share")
    if any(b < a - TOL for a, b in zip(cum, cum[1:])):
        raise AssertionError(f"Lorenz shares are not monotone: {cum}")
    # Gini over the recording-level counts.
    asc = np.sort(k)
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = float((2.0 * (idx * asc).sum()) / (n * asc.sum()) - (n + 1.0) / n)
    return {"gini": gini, "shares": out, "n_recordings": n,
            "n_violating_frames": int(total)}


def edgeness(centre: npt.ArrayLike) -> F64:
    """``(T,)`` how far from the middle of its own arena each frame sits.

    No arena definition exists in shapeflow's artifacts, so the arena is derived
    from the animal's own centroid cloud: the median position is the middle and
    the IQR is the scale. Per recording, so different boxes and camera mounts are
    comparable -- an absolute pixel radius would confound arena size with
    position.

    Returned in IQR units, not normalised to [0, 1], because the tail is the
    interesting part and clipping it would hide exactly the frames in question.
    """
    p = np.asarray(centre, dtype=np.float64)
    ok = np.isfinite(p).all(axis=1)
    out = np.full(p.shape[0], np.nan, dtype=np.float64)
    if int(ok.sum()) < 4:
        return out
    mid = np.median(p[ok], axis=0)
    q75, q25 = np.percentile(p[ok], [75, 25], axis=0)
    scale = np.maximum((q75 - q25), 1e-9)
    out[ok] = np.linalg.norm((p[ok] - mid) / scale, axis=1)
    return out


def profile(values: npt.ArrayLike, mask: npt.ArrayLike, *,
            n_bins: int = N_EDGE_BINS) -> Detail:
    """Violation rate by decile of `values`. Quantile bins, so each has mass."""
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    ok = np.isfinite(v)
    if int(ok.sum()) < n_bins * 10:
        return {"bins": [], "why": "too few finite values to bin"}
    edges = np.unique(np.percentile(v[ok], np.linspace(0, 100, n_bins + 1)))
    if edges.size < 3:
        return {"bins": [], "why": "the value is degenerate; no spread to bin"}
    idx = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, edges.size - 2)
    bins = []
    for b in range(edges.size - 1):
        sel = ok & (idx == b)
        n = int(sel.sum())
        bins.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": n,
                     "rate": float(m[sel].mean()) if n else float("nan")})
    return {"bins": bins, "n_bins": int(edges.size - 1)}


def grouped(counts: Mapping[str, float], sizes: Mapping[str, float]) -> Detail:
    """Violation rate per group, and the spread across groups.

    Used for animal, day, context and box. The spread is reported as the ratio of
    the worst group to the best, which is scale-free and readable, rather than as
    a variance component -- the groups are unbalanced and a variance component
    would need a model this does not have.
    """
    keys = sorted(set(counts) & set(sizes))
    rows = [{"group": k, "rate": float(counts[k] / sizes[k]),
             "n_frames": int(sizes[k])}
            for k in keys if sizes[k] > 0]
    if not rows:
        return {"groups": [], "worst_over_best": float("nan")}
    rates = np.array([r["rate"] for r in rows])
    lo = float(rates[rates > 0].min()) if (rates > 0).any() else float("nan")
    return {"groups": sorted(rows, key=lambda r: -r["rate"]),
            "n_groups": len(rows),
            "rate_worst": float(rates.max()), "rate_best": float(rates.min()),
            "worst_over_best": float(rates.max() / lo) if lo > 0 else float("nan")}


#: The one sentence every read here must carry. Stated once, used verbatim, so it
#: cannot drift between verdicts.
BOTH_HYPOTHESES = (
    "Concentration cannot separate the two explanations: hard footage, and a "
    "network that fails on hard footage, look identical from outside. What it "
    "bounds is how much an ensemble could help -- averaging removes errors "
    "members make INDEPENDENTLY, and situation-driven errors are the ones every "
    "member would make. Only across-network variance decides it"
)


def concentration_read(disp: Mapping[str, Any], lor: Mapping[str, Any], *,
                       scored_object: Detail, n_effective: int,
                       limit: float = DISPERSION_STRUCTURED) -> Read:
    """Is the per-recording spread structure, or the tail of a binomial?"""
    d = float(disp.get("dispersion", float("nan")))
    detail: Detail = {**dict(disp), "lorenz": dict(lor), "limit": limit,
                      "caveat": BOTH_HYPOTHESES}
    if not np.isfinite(d):
        return Read("NOT_A_RESULT", scored_object,
                    f"the dispersion ratio could not be computed: "
                    f"{disp.get('why', 'no variance to compare')}",
                    n_effective=n_effective, degenerate=True, detail=detail)

    worst10 = lor.get("shares", {}).get("worst_10pct", {})
    share = float(worst10.get("share_of_violations", float("nan")))
    if d < limit:
        return Read(
            "FAIL", scored_object,
            f"per-recording violation rates are within {d:.1f}x the binomial "
            f"spread their frame counts imply, below the {limit:g}x threshold. "
            f"The errors are close to independent across recordings, the "
            f"worst-recording list is the tail of that rather than a separate "
            f"population, and an ensemble has correspondingly more to average "
            f"away. {BOTH_HYPOTHESES}",
            n_effective=n_effective, detail=detail)
    return Read(
        "PASS", scored_object,
        f"per-recording violation rates are **{d:.0f}x** more dispersed than "
        f"independent frames at the same rate would give, and the worst 10% of "
        f"recordings carry {share:.1%} of all violating frames against a 10% "
        f"uniform baseline. The errors are situation-driven rather than random. "
        f"Note this ratio is an UPPER bound on structure: violations come in "
        f"runs (median 1 frame, p75 3), so frames are not independent even under "
        f"a pure-noise model and some overdispersion is guaranteed. "
        f"{BOTH_HYPOTHESES}",
        n_effective=n_effective, detail=detail)
