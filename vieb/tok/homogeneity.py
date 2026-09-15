r"""Does the same symbol behave the same way? The operational resolution test.

Distortion is geometric and behaviour is defined by futures, not by position. A
symbol can have high distortion and still be perfectly coherent -- if the poses
it contains differ but predict the same thing, the coarseness is free. So the
sharper form of "how many distinct positions fit in one token" is:

    as many as share a future.

For each symbol, its runs are split by how far from the cell's centre they sit,
and the **near** and **far** subpopulations are compared on what they do next and
on how long they last. Same, and the symbol is behaviourally coherent at this
resolution. Different, and it is under-resolved: two behaviours wearing one
label, which a finer alphabet would separate.

## Total variation, and why the raw value is useless on its own

The statistic is :math:`\tfrac12 \sum_j |p_j - q_j|`. Nothing in the three repos
computes one -- `journey.simplex.w2_simplex` is the nearest and is a linear
program with :math:`O(n^2)` variables, four million of them at N = 2048, per
symbol.

Plug-in TV between two empirical distributions is **biased upward by roughly**
:math:`\sqrt{K/n}`: about 0.37 at N = 256 with ~1,800 runs per quartile over 255
possible successors, and about 0.01 at N = 8. So a raw TV is not comparable
across `N`, and `N` is the one axis this measurement exists to inform. Reporting
it bare would make every fine alphabet look heterogeneous and every coarse one
look coherent, which is the exact opposite of the truth.

The effect is therefore an **excess over a permutation null** that shuffles the
near/far label among that symbol's own runs. The null has the same `K`, the same
`n`, and the same marginal, so the plug-in bias cancels by construction rather
than by correction.

## Where the animal grouping enters

Quartiles are cut **within (symbol, animal)**, not pooled across animals. An
animal whose frames sit systematically far from every centroid -- a large mouse,
a dim recording -- would otherwise fill the far quartile of every symbol, and the
test would be measuring that animal against the others. Cutting within the animal
removes the confound at the design stage instead of asking a null to model it.

The interval is an **animal-level bootstrap**: animals are resampled and every
symbol's observed TV is recomputed from the resampled runs.
`boot.animal_interval` is not used because the statistic is a mass-weighted mean
over *symbols* of a quantity computed from *all* animals at once, not a mean of
per-animal values, and handing it per-symbol values would resample symbols.

**One approximation, stated rather than buried:** the null mean is computed once
on the full data and held fixed across bootstrap replicates. It is a function of
the group sizes, the support size and the marginal, all of which a bootstrap
resample of the same size barely moves. Recomputing 2,000 permutations inside
each of 2,000 replicates is four million permutation passes per symbol.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
from recur.read import Read
from recur.render.partition import bh_fdr

from vieb.checks import assert_share

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: The typed boundary. `recur.render.partition.bh_fdr` carries no annotations.
#: Imported rather than reimplemented: three copies of Benjamini-Hochberg
#: already exist across these repos and a fourth would be a fourth thing to
#: keep right.
_bh_fdr = cast("Callable[..., tuple[list[Any], float]]", bh_fdr)

__all__ = ["FDR_Q", "MIN_RUNS_PER_SYMBOL", "N_BOOT", "N_FIX", "N_PERM",
           "aggregate", "homogeneity_read", "jackknife", "quartiles_within",
           "symbol_table", "total_variation"]

#: Runs a symbol needs before it is tested. Below this the near/far pools hold
#: single digits and the plug-in TV is all bias.
MIN_RUNS_PER_SYMBOL = 200
#: Permutation replicates for the null. Matches the repo-wide count.
N_PERM = 2000
#: Animal-level bootstrap replicates for the aggregate effect.
N_BOOT = 2000
#: Benjamini-Hochberg level for the DESCRIPTIVE failing fraction. It never
#: gates anything -- see `homogeneity_read`.
FDR_Q = 0.05

#: Runs drawn from each of the near and far pools for **every** comparison, in
#: the full-data estimate and in every replicate alike.
#:
#: This is the fix for three failed intervals. Plug-in total variation is biased
#: upward by roughly `sqrt(K/n)`, and the excess over a null inherits that
#: dependence: at a smaller `n` the observed TV is more bias-dominated and the
#: excess ATTENUATES. So a statistic computed at one sample size cannot be given
#: an interval by a procedure that computes it at another -- which is what a
#: bootstrap (effectively 0.63n distinct) and a subsample (m = n/2) both do.
#:
#: Holding the comparison size fixed makes the bias a constant rather than a
#: function of the replicate, so the excess is comparable across replicates, and
#: the interval then measures what it is supposed to: which animals contributed.
N_FIX = 100
#: Draws averaged for the full-data point estimate, to damp the sampling noise
#: that fixing `N_FIX` introduces.
N_DRAW = 64


def total_variation(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    r"""`0.5 * sum |p - q|`, on two probability vectors over the same support."""
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(a - b).sum())


def _tv_from_counts(ca: npt.ArrayLike, cb: npt.ArrayLike) -> float:
    a = np.asarray(ca, dtype=np.float64)
    b = np.asarray(cb, dtype=np.float64)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(0.5 * np.abs(a / sa - b / sb).sum())


def _fixed_draw(cn: F64, cf: F64, rng: np.random.Generator, *, n_fix: int
                ) -> tuple[float, float]:
    """One `(observed, null)` TV pair at exactly `n_fix` runs per side.

    `cn` and `cf` are the near and far next-symbol counts. The observed pair is
    a hypergeometric draw of `n_fix` from each; the null pair pools the two
    draws and re-splits them, which is a permutation of the near/far labels
    written in counts. Both sides therefore sit at exactly `2 * n_fix` items and
    carry the identical plug-in bias.
    """
    a, b = cn.astype(np.int64), cf.astype(np.int64)
    if int(a.sum()) < n_fix or int(b.sum()) < n_fix:
        return float("nan"), float("nan")
    da = rng.multivariate_hypergeometric(a, int(n_fix))
    db = rng.multivariate_hypergeometric(b, int(n_fix))
    obs = _tv_from_counts(da, db)
    pool = da + db
    ca = rng.multivariate_hypergeometric(pool, int(n_fix))
    return obs, _tv_from_counts(ca, pool - ca)


def quartiles_within(value: npt.ArrayLike, group: npt.ArrayLike) -> I64:
    """Quartile of `value` **within each group**. 0 is nearest, 3 is furthest.

    Groups with fewer than four finite values get -1 throughout: a quartile of
    three points is not a quartile.
    """
    v = np.asarray(value, dtype=np.float64)
    g = np.asarray(group, dtype=np.int64)
    out = np.full(v.shape[0], -1, dtype=np.int64)
    for u in np.unique(g):
        m = (g == u) & np.isfinite(v)
        n = int(m.sum())
        if n < 4:
            continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(v[idx], kind="stable")]
        edges = np.linspace(0, n, 5).astype(int)
        for q in range(4):
            out[order[edges[q]:edges[q + 1]]] = q
    return out


def symbol_table(code: npt.ArrayLike, next_code: npt.ArrayLike,
                 censored: npt.ArrayLike, duration_bin: npt.ArrayLike,
                 quartile: npt.ArrayLike, duration: npt.ArrayLike, *,
                 n_states: int, n_dur_bins: int, seed: int = 0,
                 n_perm: int = N_PERM, n_fix: int = N_FIX,
                 min_runs: int = MIN_RUNS_PER_SYMBOL) -> list[Detail]:
    """Per symbol: observed TV, null mean, excess, and a permutation p.

    Two comparisons per symbol, both between the **nearest** and **furthest**
    quartile of its own runs: the next-symbol distribution (censored runs
    excluded -- they have no next symbol) and the duration distribution (all
    runs, since a censored run still has an observed length).
    """
    c = np.asarray(code, dtype=np.int64)
    nx = np.asarray(next_code, dtype=np.int64)
    cen = np.asarray(censored, dtype=bool)
    db = np.asarray(duration_bin, dtype=np.int64)
    q = np.asarray(quartile, dtype=np.int64)
    dur = np.asarray(duration, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    rows: list[Detail] = []
    for s in range(int(n_states)):
        m = (c == s) & ((q == 0) | (q == 3))
        n = int(m.sum())
        if n < int(min_runs):
            continue
        near = q[m] == 0
        row: Detail = {"symbol": int(s), "n_runs": n,
                       "n_near": int(near.sum()), "n_far": int((~near).sum()),
                       "frames": int(dur[m].sum())}
        for what, values, k, mask in (
                ("next", nx[m], int(n_states) + 1, ~cen[m]),
                ("duration", db[m], int(n_dur_bins), np.ones(n, dtype=bool))):
            v = np.asarray(values, dtype=np.int64)
            v = np.where(v < 0, k - 1, v)          # CENSORED into its own bin
            sel = np.asarray(mask, dtype=bool)
            if int(sel.sum()) < int(min_runs) // 2:
                row[f"tv_{what}"] = float("nan")
                row[f"excess_{what}"] = float("nan")
                row[f"p_{what}"] = float("nan")
                continue
            vv, nn = v[sel], near[sel]
            cn = np.bincount(vv[nn], minlength=k).astype(np.float64)
            cf = np.bincount(vv[~nn], minlength=k).astype(np.float64)
            if min(int(cn.sum()), int(cf.sum())) < n_fix:
                row[f"tv_{what}"] = float("nan")
                row[f"excess_{what}"] = float("nan")
                row[f"p_{what}"] = float("nan")
                continue
            # Every comparison is at exactly `n_fix` per side, here and in the
            # bootstrap alike, so the plug-in bias is a constant rather than a
            # function of how many runs happened to be available.
            obs_d = np.empty(int(n_perm), dtype=np.float64)
            null_d = np.empty(int(n_perm), dtype=np.float64)
            for b in range(int(n_perm)):
                obs_d[b], null_d[b] = _fixed_draw(cn, cf, rng, n_fix=n_fix)
            obs = float(np.nanmean(obs_d))
            null_mean = float(np.nanmean(null_d))
            row[f"tv_{what}"] = obs
            row[f"null_{what}"] = null_mean
            row[f"excess_{what}"] = float(obs - null_mean)
            row[f"p_{what}"] = float((1 + int(np.sum(null_d >= obs)))
                                     / (1 + int(n_perm)))
            row[f"n_fix_{what}"] = int(n_fix)
        rows.append(row)
    return rows


def aggregate(rows: Sequence[Mapping[str, Any]], *, what: str = "next") -> Detail:
    """Frame-mass-weighted mean excess TV across symbols, and the FDR fraction.

    Weighted by frame mass rather than by run count, because the question Task 3
    asks of this number is how much of the *corpus* sits under an under-resolved
    symbol.
    """
    ok = [r for r in rows if np.isfinite(r.get(f"excess_{what}", np.nan))]
    if not ok:
        return {"n_symbols": 0, "excess": float("nan"),
                "fdr_failing_fraction": float("nan"),
                "fdr_failing_frame_mass": float("nan")}
    ex = np.array([r[f"excess_{what}"] for r in ok], dtype=np.float64)
    w = np.array([r["frames"] for r in ok], dtype=np.float64)
    pvals = {int(r["symbol"]): float(r[f"p_{what}"]) for r in ok}
    rejected, thresh = _bh_fdr(pvals, q=FDR_Q)
    bad = set(int(x) for x in rejected)
    mass = float(sum(r["frames"] for r in ok if int(r["symbol"]) in bad))
    return {
        "n_symbols": len(ok),
        "excess": float(np.average(ex, weights=w)),
        "excess_unweighted": float(ex.mean()),
        "tv_observed": float(np.average([r[f"tv_{what}"] for r in ok], weights=w)),
        "tv_null": float(np.average([r[f"null_{what}"] for r in ok], weights=w)),
        "fdr_failing_fraction": float(len(bad) / len(ok)),
        "fdr_failing_frame_mass": float(mass / max(w.sum(), 1.0)),
        "fdr_threshold": float(thresh),
        "fdr_q": FDR_Q,
        "p_resolution": 1.0 / (1 + N_PERM),
        "fdr_note": ("DESCRIPTIVE ONLY. The failing fraction confounds "
                     "resolution with statistical power -- a symbol holding "
                     "200k runs reaches significance on a difference a symbol "
                     "holding 900 cannot -- so it is reported because it was "
                     "specified and it gates nothing"),
    }


def _counts_by_animal(values: I64, near: BOOL, a_idx: I64, *, k: int,
                      n_animals: int) -> tuple[F64, F64]:
    """`(near, far)` next-symbol counts, per animal, for one symbol.

    Shape `(n_animals, k)` each. Built once so the bootstrap can form any
    resample as a weighted sum of these rows instead of re-scanning the run
    table, which is what made the first version take longer than the whole rest
    of the stage.
    """
    out = []
    for sel in (near, ~near):
        flat = np.bincount((a_idx[sel] * k + values[sel]).astype(np.int64),
                           minlength=int(n_animals) * int(k))
        out.append(flat.reshape(int(n_animals), int(k)).astype(np.float64))
    return out[0], out[1]


def _tv_rows(ca: F64, cb: F64) -> F64:
    """Row-wise TV between two `(n, k)` count matrices. `nan` on an empty row."""
    sa, sb = ca.sum(axis=1, keepdims=True), cb.sum(axis=1, keepdims=True)
    ok = (sa[:, 0] > 0) & (sb[:, 0] > 0)
    out = np.full(ca.shape[0], np.nan, dtype=np.float64)
    if ok.any():
        pa = ca[ok] / sa[ok]
        pb = cb[ok] / sb[ok]
        out[ok] = 0.5 * np.abs(pa - pb).sum(axis=1)
    return out


def jackknife(rows: Sequence[Mapping[str, Any]], code: npt.ArrayLike,
              next_code: npt.ArrayLike, censored: npt.ArrayLike,
              quartile: npt.ArrayLike, duration: npt.ArrayLike,
              animal: Sequence[Any], *, n_states: int, point: float,
              n_fix: int = N_FIX, seed: int = 0, z: float = 1.96) -> Detail:
    """Animal-level interval on the aggregate excess TV, by leave-one-out.

    ## Four resampling schemes failed first, and the reason is one thing

    Plug-in total variation is biased upward by roughly `sqrt(K/n)`, and the
    excess over a null inherits that. Any interval has to be built from
    replicates whose observed and null statistics carry the **same** bias and
    the **same** clustering.

    1. **Bootstrap, null computed once on the full data.** Percentile interval
       `[+0.3198, +0.3350]` around a point estimate of `+0.3097` -- an interval
       that does not contain its own estimate. A resample holds ~63% distinct
       runs, so every replicate read high against a fixed null.
    2. **Reflected into a basic interval.** When the bias exceeds half the
       width the reflected interval simply sits on the other side.
    3. **Bootstrap, null redrawn per replicate.** The gap grew with `N`:
       `+0.1706` against `[+0.2553, +0.3016]` at N = 2048. A duplicated animal
       doubles its counts without adding independent runs, so a hypergeometric
       over those counts believes there are twice as many exchangeable items.
    4. **Subsample without replacement, m = n/2, plus a fixed comparison size.**
       Closer, and the offset changed sign, but still `+0.1574` against
       `[+0.1594, +0.1827]`. **A duplicated or omitted animal's runs all land on
       the same side of the near/far split**, which is clustering an item-level
       null cannot reproduce however the sizes are fixed.

    ## What is done instead

    Leave one animal out, 89 times. No animal is ever duplicated, so there is no
    clustering artifact; the sample changes by one part in 89, so the bias shift
    is negligible where `m = n/2` made it decisive; and it is animal-level by
    construction. The interval is `theta +/- z * SE_jack` with the standard
    `SE_jack = sqrt((n-1)/n * sum (theta_(-a) - mean)^2)`.

    Every comparison is still drawn at exactly `n_fix` runs per side, and each
    symbol's null mean is held at its full-data value -- which is sound here in
    a way it was not for the bootstrap, because a leave-one-out sample differs
    from the full data by 1.1% rather than by 37%.
    """
    c = np.asarray(code, dtype=np.int64)
    nx = np.asarray(next_code, dtype=np.int64)
    cen = np.asarray(censored, dtype=bool)
    q = np.asarray(quartile, dtype=np.int64)
    dur = np.asarray(duration, dtype=np.int64)
    a = np.asarray([str(x) for x in animal])
    tags, a_idx = np.unique(a, return_inverse=True)
    n_an = int(tags.shape[0])
    k = int(n_states) + 1

    live = [r for r in rows if np.isfinite(r.get("excess_next", np.nan))]
    if not live or n_an < 3:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_animals": n_an, "n_jack": 0, "method": "not enough animals"}

    rng = np.random.default_rng(int(seed))
    keep_all = np.ones(n_an, dtype=bool)
    num = np.zeros(n_an, dtype=np.float64)
    den = np.zeros(n_an, dtype=np.float64)
    for r in live:
        s_ = int(r["symbol"])
        m = (c == s_) & ((q == 0) | (q == 3)) & (~cen)
        if not m.any():
            continue
        near = q[m] == 0
        if not near.any() or near.all():
            continue
        v = np.where(nx[m] < 0, k - 1, nx[m]).astype(np.int64)
        cn, cf = _counts_by_animal(v, near, a_idx[m], k=k, n_animals=n_an)
        w_an = np.bincount(a_idx[m], weights=dur[m].astype(np.float64),
                           minlength=n_an)
        tot_n, tot_f, tot_w = cn.sum(0), cf.sum(0), float(w_an.sum())
        for drop in range(n_an):
            left_n, left_f = tot_n - cn[drop], tot_f - cf[drop]
            obs, _null = _fixed_draw(left_n, left_f, rng, n_fix=n_fix)
            if not np.isfinite(obs):
                continue
            w = tot_w - float(w_an[drop])
            num[drop] += w * (obs - float(r["null_next"]))
            den[drop] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        theta_i = np.where(den > 0, num / den, np.nan)
    good = theta_i[np.isfinite(theta_i)]
    if good.size < 3:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_animals": n_an, "n_jack": int(good.size),
                "method": "too few finite leave-one-out values"}
    n = float(good.size)
    se = float(np.sqrt((n - 1.0) / n * ((good - good.mean()) ** 2).sum()))
    theta = float(point)
    return {"point": theta, "lo": theta - z * se, "hi": theta + z * se,
            "se": se, "jackknife_mean": float(good.mean()),
            "jackknife_minus_full_data": float(good.mean() - theta),
            "n_fix": int(n_fix), "n_animals": n_an, "n_jack": int(good.size),
            "z": float(z),
            "method": ("leave-one-animal-out jackknife; no animal is ever "
                       "duplicated so there is no clustering artifact, and the "
                       "sample changes by one part in n so the plug-in bias "
                       "does not shift; every comparison drawn at exactly "
                       "n_fix runs per side; theta +/- z * SE_jack")}


def homogeneity_read(agg: Mapping[str, Any], ci: Mapping[str, Any], *,
                     scored_object: Detail, n_effective: int) -> Read:
    """Do this alphabet's symbols share a future?

    The verdict reads the **effect interval**, never the FDR fraction. The
    fraction is a count of significant tests and significance scales with how
    much data each symbol holds, so it is not comparable across `N` -- which is
    the only axis anyone would want to compare it across.
    """
    detail: Detail = {**{k: v for k, v in agg.items() if k != "fdr_note"},
                      "fdr_note": agg.get("fdr_note"),
                      "animal_interval": dict(ci)}
    if not np.isfinite(float(agg.get("excess", np.nan))):
        return Read("NOT_A_RESULT", scored_object,
                    f"no symbol reached {MIN_RUNS_PER_SYMBOL} runs with both a "
                    f"near and a far quartile, so there is nothing to compare",
                    n_effective=n_effective, degenerate=True, detail=detail)
    lo, hi = float(ci["lo"]), float(ci["hi"])
    point = float(agg["excess"])
    assert_share([float(agg["tv_observed"]), float(agg["tv_null"])],
                 name="aggregate TV")
    tail = (f"near and far runs of the same symbol differ in what they do next "
            f"by {float(agg['tv_observed']):.4f} total variation against a "
            f"permutation null of {float(agg['tv_null']):.4f} -- an excess of "
            f"**{point:+.4f}** [{lo:+.4f}, {hi:+.4f}], frame-mass weighted over "
            f"{int(agg['n_symbols'])} symbols, animal-level over "
            f"{int(ci['n_animals'])} animals")
    extra = (f" For the record and not for the verdict: "
             f"{float(agg['fdr_failing_fraction']):.1%} of symbols reject "
             f"homogeneity at FDR {agg['fdr_q']}, carrying "
             f"{float(agg['fdr_failing_frame_mass']):.1%} of the frame mass")
    if lo > 0:
        return Read("PASS", scored_object,
                    f"THESE SYMBOLS ARE UNDER-RESOLVED: {tail}, with the "
                    f"interval excluding zero. Position within the cell "
                    f"predicts the future, so the cell contains more than one "
                    f"behaviour and a finer alphabet would separate them."
                    + extra,
                    n_effective=n_effective, detail=detail)
    if hi < 0:
        return Read("INCONCLUSIVE", scored_object,
                    f"the excess is NEGATIVE: {tail}. Near and far runs differ "
                    f"less than the null expects, which is not a coherence "
                    f"result but a sign the null is mis-specified for this "
                    f"alphabet." + extra,
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"THESE SYMBOLS SHARE A FUTURE: {tail}, with the interval "
                f"containing zero. The positions inside a cell differ in pose "
                f"but not in what they predict, which is the operational answer "
                f"to how many distinct positions fit in one token -- as many as "
                f"share a future." + extra,
                n_effective=n_effective, detail=detail)
