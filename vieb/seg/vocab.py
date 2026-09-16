r"""Step 3. Is there a vocabulary, or a continuum? Both are results.

Registered in `results/VOCAB_PREREGISTRATION.md`.

## No assumed count, and no clustering that requires one

ExBias's stack returned `n_states = 0` and self-diagnoses why: MiniBatchKMeans
centroids tile at uniform density, which is the opposite of what a density-based
clusterer needs. So nothing here fits `k` anything.

Clumps are **connected components of a neighbour graph thresholded at the null's
own distance quantile** -- the same theta the recurrence statistic already uses,
so the linking distance is not a new free parameter. A segment linked to nothing
is **unassigned**, and the unassigned fraction is a headline number rather than
a footnote: any method that assigns 100% of its input is misreporting its
coverage.

## Clumpiness a null reproduces is not clumpiness

Every statistic here is computed identically on the corpus and on both
dwell-matched nulls. A neighbour graph thresholded at a fixed quantile produces
components in *any* point cloud, including a Gaussian one, so the corpus's
component structure is only informative beside the null's.

## Participation is what separates a vocabulary entry from an artifact

A clump present in 80 of 89 animals is a vocabulary entry. One present in 12 is
an artifact of whichever animals dominate the sample. Ranked on the **lower
bound of an animal-level interval**, never on p-values, which saturate at
1.28e-16 at n = 89 and would rank on sample size instead of on effect.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur import boot
from recur.read import Read

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

__all__ = ["KNN", "MIN_CLUMP", "bimodality", "clump_read", "components",
           "coverage_read", "gmm1d_bic_gain", "participation",
           "participation_read"]

#: Neighbours considered per segment when building the graph. Bounded so the
#: graph cannot blow up on a dense region; a segment with more than KNN
#: neighbours inside theta keeps its KNN nearest, which can only ever SPLIT a
#: clump, never merge two.
KNN = 10
#: A component smaller than this is not reported as a clump. It is not a size
#: prior on behaviour -- it is the point below which "present in N animals"
#: cannot be estimated at all.
MIN_CLUMP = 20


def bimodality(x: npt.ArrayLike) -> Detail:
    """Bimodality coefficient, skew and kurtosis of a distance distribution.

    `BC = (skew^2 + 1) / kurtosis`, above 5/9 for a uniform and rising as mass
    splits into two groups. Reported with its ingredients because BC alone
    cannot tell a two-peaked distribution from a heavy-tailed one, and the
    corpus's distances are heavy-tailed for reasons the roughness measurement
    already documents.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return {"n": int(n), "why": "fewer than four finite distances"}
    m, s = float(a.mean()), float(a.std(ddof=1))
    if s <= 0:
        return {"n": int(n), "why": "zero variance"}
    z = (a - m) / s
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    # EXCESS kurtosis in the denominator, with the small-sample correction the
    # coefficient is conventionally defined with. Using the raw fourth moment
    # inverts the statistic: a clean two-peak mixture then scores BELOW an
    # exponential, which is the opposite of what it is for.
    denom = (kurt - 3.0) + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return {"n": int(n), "mean": m, "sd": s, "skew": skew, "kurtosis": kurt,
            "excess_kurtosis": float(kurt - 3.0),
            "bimodality_coefficient": float((skew ** 2 + 1.0) / denom),
            "uniform_reference": 5.0 / 9.0,
            "caveat": ("BC rises on heavy tails as well as on two peaks -- a "
                       "standard exponential scores 0.58 -- so the excess "
                       "kurtosis beside it is what separates a flat-topped "
                       "two-peak distribution (negative) from a heavy-tailed "
                       "one (positive)")}


def gmm1d_bic_gain(x: npt.ArrayLike, *, seed: int = 0, iters: int = 200,
                   sample: int = 200_000) -> Detail:
    """BIC(1 component) - BIC(2), positive when two components are preferred.

    A two-component Gaussian mixture in one dimension, fitted by EM. Kept
    deliberately small: this is a *shape* statistic on a distance distribution,
    not a clustering, and the clumps themselves are found by the neighbour
    graph. Its value is that the same number is computed on the nulls.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size > sample:
        a = a[rng.choice(a.size, size=sample, replace=False)]
    n = a.size
    if n < 50:
        return {"n": int(n), "why": "fewer than fifty finite distances"}
    var = float(a.var())
    if var <= 0:
        return {"n": int(n), "why": "zero variance"}
    ll1 = float(-0.5 * n * (np.log(2 * np.pi * var) + 1.0))
    bic1 = -2 * ll1 + 2 * np.log(n)
    lo, hi = np.quantile(a, [0.25, 0.75])
    mu = np.array([lo, hi], dtype=np.float64)
    sg = np.full(2, np.sqrt(var), dtype=np.float64)
    w = np.full(2, 0.5, dtype=np.float64)
    ll2 = -np.inf
    for _ in range(int(iters)):
        comp = (w[None, :] / (np.sqrt(2 * np.pi) * sg[None, :])
                * np.exp(-0.5 * ((a[:, None] - mu[None, :]) / sg[None, :]) ** 2))
        tot = comp.sum(axis=1) + 1e-300
        new_ll = float(np.log(tot).sum())
        r = comp / tot[:, None]
        nk = r.sum(axis=0) + 1e-12
        mu = (r * a[:, None]).sum(axis=0) / nk
        sg = np.sqrt(np.maximum((r * (a[:, None] - mu[None, :]) ** 2).sum(axis=0)
                                / nk, 1e-12))
        w = nk / n
        if abs(new_ll - ll2) < 1e-7 * max(1.0, abs(new_ll)):
            ll2 = new_ll
            break
        ll2 = new_ll
    bic2 = -2 * ll2 + 5 * np.log(n)
    return {"n": int(n), "bic_1": float(bic1), "bic_2": float(bic2),
            "bic_gain": float(bic1 - bic2),
            "means": mu.tolist(), "sds": sg.tolist(), "weights": w.tolist(),
            "separation_in_sds": float(abs(mu[1] - mu[0])
                                       / max(float(sg.mean()), 1e-12))}


def components(knn_idx: npt.ArrayLike, knn_dist: npt.ArrayLike, *,
               theta: float, min_size: int = MIN_CLUMP) -> tuple[I64, Detail]:
    """Connected components of the graph linking pairs closer than `theta`.

    Union-find over at most `KNN` edges per node. `theta` is the **null's**
    distance quantile, already fixed by the recurrence statistic, so the linking
    distance is inherited rather than chosen here.

    Returns labels (-1 for unassigned) and a summary.
    """
    idx = np.asarray(knn_idx, dtype=np.int64)
    dist = np.asarray(knn_dist, dtype=np.float64)
    n = idx.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:          # path compression
            parent[i], i = root, int(parent[i])
        return root

    ok = (dist < float(theta)) & (idx >= 0)
    rows, cols = np.nonzero(ok)
    for r, c in zip(rows.tolist(), cols.tolist()):
        j = int(idx[r, c])
        if j == r:
            continue
        ri, rj = find(int(r)), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.asarray([find(int(i)) for i in range(n)], dtype=np.int64)
    uniq, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    big = cnt >= int(min_size)
    relabel = np.full(uniq.size, -1, dtype=np.int64)
    relabel[big] = np.arange(int(big.sum()), dtype=np.int64)
    labels = relabel[inv]
    sizes = cnt[big]
    if sizes.size:
        # Renumber largest first, so "clump 0" means the same thing in every
        # arm and the top of a table is comparable across corpus and null.
        order = np.argsort(-sizes)
        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size, dtype=np.int64)
        labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1)
        sizes = sizes[order]
    else:
        labels = np.full(n, -1, dtype=np.int64)
    summary: Detail = {
        "n_units": int(n), "theta": float(theta), "min_size": int(min_size),
        "n_clumps": int(big.sum()),
        "n_edges": int(ok.sum()),
        "unassigned_fraction": float((labels < 0).mean()),
        "largest_clump_fraction": (float(sizes.max() / n) if sizes.size
                                   else 0.0),
        "sizes_top": np.sort(sizes)[::-1][:20].tolist(),
    }
    return labels, summary


def participation(labels: npt.ArrayLike, animals: Sequence[str], *,
                  n_animals_total: int) -> list[Detail]:
    """Per clump: how many distinct animals contribute, and its concentration.

    `top_animal_share` travels with the count because a clump can sit in many
    animals while one animal supplies most of its members, and that is not the
    same object as a shared behaviour.
    """
    lab = np.asarray(labels, dtype=np.int64)
    an = np.asarray(animals)
    out: list[Detail] = []
    for c in range(int(lab.max()) + 1 if lab.size and lab.max() >= 0 else 0):
        m = lab == c
        mine = an[m]
        uniq, cnt = np.unique(mine, return_counts=True)
        out.append({"clump": int(c), "size": int(m.sum()),
                    "n_animals": int(uniq.size),
                    "animal_fraction": float(uniq.size / max(n_animals_total, 1)),
                    "top_animal_share": float(cnt.max() / cnt.sum()),
                    "animals": uniq.tolist()})
    return out


def participation_read(rows: Sequence[Mapping[str, Any]], *,
                       n_animals_total: int, scored_object: Detail,
                       n_effective: int, seed: int = 0,
                       floor: float = 0.5) -> Read:
    """Are there clumps most animals take part in?

    Ranked on the lower bound of an animal-level interval on the per-clump
    animal fraction, never on p-values: at n = 89 they saturate at 1.28e-16 and
    would rank on sample size rather than on effect.
    """
    if not rows:
        return Read("FAIL", scored_object,
                    "no clump reached the minimum size, so there is nothing "
                    "whose cross-animal participation could be estimated",
                    n_effective=n_effective, detail={"n_clumps": 0})
    frac = np.asarray([float(r["animal_fraction"]) for r in rows])
    ci = boot.animal_interval(frac, [str(r["clump"]) for r in rows],
                              how="mean", seed=seed)
    shared = [r for r in rows if float(r["animal_fraction"]) >= floor]
    detail: Detail = {"n_clumps": len(rows), "mean_animal_fraction": ci,
                      "n_clumps_above_floor": len(shared),
                      "floor": floor,
                      "top": sorted(rows, key=lambda r: -float(
                          r["animal_fraction"]))[:10]}
    if not shared:
        return Read("FAIL", scored_object,
                    f"{len(rows)} clumps and not one reaches {floor:.0%} of "
                    f"animals -- the best spans "
                    f"{max(float(r['animal_fraction']) for r in rows):.1%}. "
                    f"These are artifacts of whichever animals dominate the "
                    f"sample, not vocabulary entries",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"{len(shared)} of {len(rows)} clumps are present in at least "
                f"{floor:.0%} of animals, mean animal fraction "
                f"{float(ci['point']):.3f} [{float(ci['lo']):.3f}, "
                f"{float(ci['hi']):.3f}]",
                n_effective=n_effective, detail=detail)


def clump_read(obs: Mapping[str, Any], nulls: Mapping[str, Mapping[str, Any]],
               *, scored_object: Detail, n_effective: int) -> Read:
    """Clumps, or a continuum? Only informative beside the nulls.

    A neighbour graph thresholded at a fixed quantile produces components in any
    point cloud. So the verdict is on the corpus's structure **relative to** the
    same statistic computed identically on both dwell-matched nulls, and a
    continuum verdict is as much a result as a vocabulary one.
    """
    detail: Detail = {"corpus": dict(obs),
                      "nulls": {k: dict(v) for k, v in nulls.items()}}
    if not nulls:
        return Read("INCONCLUSIVE", scored_object,
                    "no null ran, and component counts mean nothing alone",
                    n_effective=n_effective, detail=detail)
    o_bic = float(obs.get("bic_gain", float("nan")))
    n_bic = [float(v.get("bic_gain", float("nan"))) for v in nulls.values()]
    o_un = float(obs["unassigned_fraction"])
    n_un = [float(v["unassigned_fraction"]) for v in nulls.values()]
    o_cl, n_cl = int(obs["n_clumps"]), [int(v["n_clumps"]) for v in nulls.values()]
    tail = (f"corpus: {o_cl} clumps, {o_un:.1%} unassigned, two-component BIC "
            f"gain {o_bic:+.0f}; nulls: {n_cl} clumps, "
            f"{['%.1f%%' % (100 * u) for u in n_un]} unassigned, gains "
            f"{['%+.0f' % g for g in n_bic]}")
    beats = o_cl > max(n_cl) and o_bic > max(n_bic)
    if beats:
        return Read("PASS", scored_object,
                    f"THERE IS CLUMP STRUCTURE the nulls do not reproduce -- "
                    f"{tail}. Found by observation rather than by assuming a "
                    f"count", n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"A CONTINUUM, not a vocabulary: {tail}. The corpus's component "
                f"structure is not beyond what the dwell-matched nulls produce "
                f"under the identical graph, so there are local neighbourhoods "
                f"-- enough for averaging without a global partition -- and no "
                f"evidence of discrete entries",
                n_effective=n_effective, detail=detail)


def coverage_read(summary: Mapping[str, Any], *, scored_object: Detail,
                  n_effective: int, max_unassigned: float = 0.90) -> Read:
    """The unassigned bin, as a headline rather than a footnote."""
    un = float(summary["unassigned_fraction"])
    detail = dict(summary)
    if un > max_unassigned:
        return Read("NOT_A_RESULT", scored_object,
                    f"{un:.1%} of segments are unassigned, above the "
                    f"{max_unassigned:.0%} limit: whatever the clumps are, they "
                    f"do not cover the corpus and a vocabulary built on them "
                    f"would describe a fraction of behaviour while appearing to "
                    f"describe all of it",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"{1 - un:.1%} of segments fall in a clump of at least "
                f"{int(summary['min_size'])} members; {un:.1%} are unassigned "
                f"and are reported as such",
                n_effective=n_effective, detail=detail)
