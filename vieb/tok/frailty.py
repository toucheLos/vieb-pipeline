r"""Is the 148x fall in hazard duration dependence, or is it mixing?

`LADDER.md` reports that the hazard of leaving a state falls from 0.489 at the
first frame to 0.0033 beyond 259 -- a 148x fall, and nothing like the flat
profile a memoryless dwell would give. It also refuses to read that as evidence
that an individual bout becomes harder to leave the longer it lasts, because a
**falling pooled hazard is exactly what unmodelled heterogeneity produces**.

The mechanism, stated plainly: if runs are a mixture of shorter- and longer-lived
kinds, then the ones still alive at large elapsed time are increasingly the
long-lived kind. The population's hazard falls because its composition changes,
not because any member's hazard does. This is frailty, and it is the same class
of error that retracted the dwell result -- a shape produced by the instrument.

It cannot be settled completely without knowing the true mixture. It can be
**bounded**, by recomputing the hazard inside slices that are more homogeneous
than the pool and seeing how much of the fall survives.

## The slice that is actually powered

Three slices were specified. Their arithmetic at N = 256 on the report split,
1.83M runs over 89 animals, differs by three orders of magnitude:

============================  ==============================================
slice                         exits available
============================  ==============================================
speed quintile (over states)  ~366k per quintile -- **the primary**
state x speed quintile        ~1.1 runs per cell -- a spread statistic at best
state x animal, top 10        ~6 at-risk frames per cell -- refusable
============================  ==============================================

The hypothesis also lives in the first one. Slow and freezing runs are long,
fast runs are short; if the fall is frailty, that is the mixture producing it,
and conditioning on speed is what should flatten it.

## Speed is a run-level covariate

The mean of `quantize.speed` over the run's own frames, computed once per run.
Per frame it would reintroduce exactly the frame-mass bias run-length encoding
exists to remove: a long freeze would contribute hundreds of slow frames and a
brief dart a handful of fast ones, and the quintile edges would be a statement
about how the animal spends its time rather than about what it is doing.

## The statistic, and why it is a log

The **fall ratio** is the hazard in the first reportable bin over the hazard in
the last. Ratios are multiplicative and badly behaved under a mean, so the
bootstrap runs on `log(fall)` and the interval is exponentiated back.

`retained = log(fall_within_slice) / log(fall_pooled)` is what the verdict reads:
the share of the fall, on a log scale, that survives conditioning. One means the
slice explained none of it; zero means it explained all of it.

A flat curve computed on twelve runs is not a flat hazard, so every cell carries
its realised at-risk and exit counts and `MIN_EXITS_PER_CELL` marks a cell
unreportable rather than plotting it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur import boot
from recur.read import Read

from vieb.tok import hazard as hz

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["MIN_EXITS_PER_CELL", "N_QUINTILES", "RETAINED_FLOOR",
           "common_span", "curve", "fall_ratio", "frailty_read",
           "quintile_edges", "quintile_of", "reportable",
           "retained_by_animal"]

#: Exits a cell needs before its hazard is reportable. Round, registered, and
#: not chosen against an outcome.
MIN_EXITS_PER_CELL = 50

#: Speed quintiles, matching `quantize.N_STRATA` so the two decompositions of
#: speed on this project use the same number of bins.
N_QUINTILES = 5

#: Share of the log fall that must survive conditioning for the fall to be
#: called real. Below this, the slice explained most of it and the pooled fall
#: is frailty. A half is the point at which the slice explains more than it
#: leaves, and it is fixed here rather than after seeing the curves.
RETAINED_FLOOR = 0.5


def quintile_edges(mean_speed: npt.ArrayLike, mask: npt.ArrayLike, *,
                   n: int = N_QUINTILES) -> F64:
    """Interior quantile cuts of RUN-level mean speed, from `mask` runs only.

    `mask` must select the tune split. A slice boundary is a choice, and choices
    are made where every other choice on this branch is made.
    """
    s = np.asarray(mean_speed, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite run speeds to cut quintiles on")
    return np.asarray(np.quantile(s, np.linspace(0, 1, int(n) + 1)[1:-1]),
                      dtype=np.float64)


def quintile_of(mean_speed: npt.ArrayLike, edges: npt.ArrayLike) -> I64:
    """Which speed quintile each run falls in. Non-finite gives -1."""
    s = np.asarray(mean_speed, dtype=np.float64)
    out = np.searchsorted(np.asarray(edges, dtype=np.float64), s,
                          side="right").astype(np.int64)
    out[~np.isfinite(s)] = -1
    return out


def curve(duration: npt.ArrayLike, censored: npt.ArrayLike,
          edges: npt.ArrayLike, *, slice_id: npt.ArrayLike | None = None,
          n_slices: int = 1, n_states: int = 2) -> Detail:
    """Total exit hazard per elapsed bin, per slice.

    Delegates the at-risk accounting to `hazard.fit` rather than repeating it:
    the bin-overlap arithmetic and the censoring rule are the parts most easily
    got wrong, and they are already written and tested there. `next_state` is
    irrelevant to a total hazard, so a constant is passed and only `at_risk` and
    `exits_per_cell` are read back.
    """
    d = np.asarray(duration, dtype=np.int64)
    cen = np.asarray(censored, dtype=bool)
    ctx = (np.zeros(d.shape[0], dtype=np.int64) if slice_id is None
           else np.asarray(slice_id, dtype=np.int64))
    keep = ctx >= 0
    m = hz.fit(ctx[keep], d[keep], np.zeros(int(keep.sum()), dtype=np.int64),
               cen[keep], edges=edges, n_context=int(n_slices),
               n_states=int(n_states))
    at_risk = np.asarray(m["at_risk"], dtype=np.float64)
    exits = np.asarray(m["exits_per_cell"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        haz = np.where(exits >= MIN_EXITS_PER_CELL,
                       exits / np.maximum(at_risk, 1e-12), np.nan)
    return {"hazard": haz, "at_risk": at_risk, "exits": exits,
            "n_bins": int(m["n_bins"]), "n_slices": int(n_slices),
            "n_runs": int(keep.sum()), "n_dropped": int((~keep).sum()),
            "min_exits": MIN_EXITS_PER_CELL}


def reportable(hazard_row: npt.ArrayLike) -> I64:
    """Bin indices whose hazard is finite and positive."""
    h = np.asarray(hazard_row, dtype=np.float64)
    return np.flatnonzero(np.isfinite(h) & (h > 0)).astype(np.int64)


def common_span(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[int, int]:
    """The elapsed-bin window reportable in BOTH curves, as `(lo, hi)`.

    `(-1, -1)` when they do not overlap in at least two bins.
    """
    ra, rb = reportable(a), reportable(b)
    if ra.size < 2 or rb.size < 2:
        return (-1, -1)
    lo, hi = max(int(ra[0]), int(rb[0])), min(int(ra[-1]), int(rb[-1]))
    if hi <= lo:
        return (-1, -1)
    h_a, h_b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    for x in (h_a, h_b):
        if not (np.isfinite(x[lo]) and x[lo] > 0
                and np.isfinite(x[hi]) and x[hi] > 0):
            return (-1, -1)
    return (lo, hi)


def fall_ratio(hazard_row: npt.ArrayLike, *,
               span: tuple[int, int] | None = None) -> Detail:
    """Hazard at the first reportable bin over the last, and which bins those are.

    With `span` given, the ratio is taken between exactly those two bins.

    **Why `span` exists, and why nothing may be compared without it.** A fast
    quintile's runs never last long enough to populate the late elapsed bins, so
    its curve is only reportable over a short prefix -- bins 0-5 against the
    pool's 0-12 on this corpus. Comparing its fall to the pooled fall then
    compares a 6-bin fall against a 13-bin fall, and the slice looks flatter
    because its range is shorter rather than because its hazard is.

    Measured here: within the SLOWEST quintile, which spans the same 0-12 as the
    pool, the fall is 143.7x against the pooled 175.7x -- barely reduced. The
    unrestricted comparison read 16.2x and would have been reported as a
    collapse. Every comparison in this module is therefore taken over
    `common_span`.
    """
    h = np.asarray(hazard_row, dtype=np.float64)
    if span is not None:
        a, b = int(span[0]), int(span[1])
        if a < 0 or b < 0 or not (np.isfinite(h[a]) and np.isfinite(h[b])
                                  and h[a] > 0 and h[b] > 0):
            return {"fall": float("nan"), "log_fall": float("nan"),
                    "first_bin": -1, "last_bin": -1, "n_live_bins": 0}
    else:
        live = reportable(h)
        if live.size < 2:
            return {"fall": float("nan"), "log_fall": float("nan"),
                    "first_bin": -1, "last_bin": -1,
                    "n_live_bins": int(live.size)}
        a, b = int(live[0]), int(live[-1])
    ratio = float(h[a] / h[b])
    return {"fall": ratio, "log_fall": float(np.log(ratio)),
            "first_bin": a, "last_bin": b,
            "n_live_bins": int(reportable(h).size),
            "first_hazard": float(h[a]), "last_hazard": float(h[b])}


def retained_by_animal(duration: npt.ArrayLike, censored: npt.ArrayLike,
                       animal: Sequence[Any], slice_id: npt.ArrayLike,
                       edges: npt.ArrayLike, *, n_slices: int) -> Detail:
    """Per animal: how much of the log fall survives conditioning, like for like.

    For each animal and each of its slices, the fall is taken over the
    **common span** of that slice's curve and the animal's own pooled curve, and
    the pooled fall is recomputed over the same two bins. Without that the
    comparison is between a short-range fall and a long-range one, and a slice
    whose runs simply do not last looks flat.

    The animal's within-slice figure is the **exit-weighted mean** of its
    per-slice log falls, so an animal whose runs sit mostly in one quintile is
    described by that quintile rather than by an unweighted average over
    quintiles it barely occupies.

    Animals too thin to yield a common span in any slice come back `nan` and are
    counted, never silently dropped.
    """
    d = np.asarray(duration, dtype=np.int64)
    cen = np.asarray(censored, dtype=bool)
    a = np.asarray([str(x) for x in animal])
    sid = np.asarray(slice_id, dtype=np.int64)
    tags = np.unique(a)
    pooled, within, spans = [], [], []
    for t in tags:
        m = a == t
        h_pool = curve(d[m], cen[m], edges)["hazard"][0]
        c = curve(d[m], cen[m], edges, slice_id=sid[m], n_slices=n_slices)
        lp, lw, w, used = [], [], [], []
        for s_ in range(n_slices):
            span = common_span(c["hazard"][s_], h_pool)
            if span[0] < 0:
                continue
            f_s = fall_ratio(c["hazard"][s_], span=span)
            f_p = fall_ratio(h_pool, span=span)
            if not (np.isfinite(f_s["log_fall"]) and np.isfinite(f_p["log_fall"])
                    and abs(f_p["log_fall"]) > 1e-9):
                continue
            lp.append(f_p["log_fall"])
            lw.append(f_s["log_fall"])
            w.append(float(c["exits"][s_].sum()))
            used.append(span[1] - span[0] + 1)
        if not lw:
            pooled.append(np.nan)
            within.append(np.nan)
            spans.append(0.0)
            continue
        wa = np.asarray(w, dtype=np.float64)
        pooled.append(float(np.average(lp, weights=wa)))
        within.append(float(np.average(lw, weights=wa)))
        spans.append(float(np.average(used, weights=wa)))
    lp_a = np.asarray(pooled, dtype=np.float64)
    lw_a = np.asarray(within, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(np.abs(lp_a) > 1e-9, lw_a / lp_a, np.nan)
    good = np.isfinite(ret)
    return {"animals": [str(t) for t in tags],
            "log_fall_pooled": lp_a, "log_fall_within": lw_a, "retained": ret,
            "mean_span_bins": np.asarray(spans, dtype=np.float64),
            "n_animals": int(tags.shape[0]), "n_usable": int(good.sum()),
            "n_refused": int((~good).sum())}


def frailty_read(per_animal: Mapping[str, Any], pooled: Mapping[str, Any],
                 within: Mapping[str, Any], *, slice_name: str,
                 scored_object: Detail, n_effective: int,
                 floor: float = RETAINED_FLOOR, n_boot: int = 2000,
                 seed: int = 0) -> Read:
    """Frailty, real duration dependence, or too thin to say."""
    ret = np.asarray(per_animal["retained"], dtype=np.float64)
    animals = list(per_animal["animals"])
    ok = np.isfinite(ret)
    detail: Detail = {
        "slice": slice_name,
        "pooled_fall": pooled.get("fall"),
        "within_fall": within.get("fall"),
        "pooled_first_bin": pooled.get("first_bin"),
        "pooled_last_bin": pooled.get("last_bin"),
        "n_animals_usable": int(ok.sum()),
        "n_animals_refused": int((~ok).sum()),
        "min_exits_per_cell": MIN_EXITS_PER_CELL,
        "retained_floor": float(floor),
        "mean_span_bins": (float(np.nanmean(per_animal["mean_span_bins"]))
                           if "mean_span_bins" in per_animal else None),
        "span_note": ("every fall is taken over the elapsed-bin window "
                      "reportable in BOTH the slice and the pool; an "
                      "unrestricted comparison would read a slice whose runs "
                      "do not last as flat"),
    }
    if int(ok.sum()) < 2:
        return Read("GRID_LIMITED", scored_object,
                    f"the {slice_name} slice is too thin to resolve: only "
                    f"{int(ok.sum())} of {len(animals)} animals yield two bins "
                    f"with at least {MIN_EXITS_PER_CELL} exits. Reported as "
                    f"unresolvable rather than as a flat hazard -- a flat curve "
                    f"on a handful of runs is not a flat hazard",
                    n_effective=int(ok.sum()), degenerate=True, detail=detail)

    vals = ret[ok]
    tags = [animals[i] for i in np.flatnonzero(ok)]
    ci = boot.animal_interval(vals, tags, how="mean", n_boot=n_boot, seed=seed)
    detail["animal_interval"] = dict(ci)
    detail["frame_interval"] = dict(boot.frame_interval(vals, n_boot=n_boot,
                                                        seed=seed))
    detail["frame_interval_note"] = (
        "printed to show how much narrower the wrong method looks; the verdict "
        "does not read it")
    lo, hi = float(ci["lo"]), float(ci["hi"])
    point = float(ci["point"])
    # The pooled figure quoted here is the one recomputed over the SAME spans
    # as the slices, never the full-range 0-12 fall. Quoting the full-range
    # pooled fall against a common-span within fall compares a 13-bin fall to a
    # 6-bin one and is the confound this module exists to avoid; the full-range
    # value is carried in the detail for context and labelled as such.
    same = within.get("pooled_fall_on_same_spans")
    ref = float(same) if same is not None and np.isfinite(float(same)) \
        else float(pooled.get("fall", float("nan")))
    detail["pooled_fall_full_range"] = pooled.get("fall")
    detail["pooled_fall_on_same_spans"] = same
    detail["n_reportable_slices"] = within.get("n_reportable_slices")
    detail["n_slices"] = within.get("n_slices")
    tail = (f"conditioning on {slice_name} leaves {point:.3f} "
            f"[{lo:.3f}, {hi:.3f}] of the log fall, animal-level over "
            f"{int(ok.sum())} animals; over the same elapsed spans the fall is "
            f"{float(within.get('fall', float('nan'))):.1f}x within slices "
            f"against {ref:.1f}x pooled")

    if hi < floor:
        return Read("PASS", scored_object,
                    f"THE FALL IS SUBSTANTIALLY FRAILTY: {tail}, with the "
                    f"interval entirely below the {floor:.2f} floor. The pooled "
                    f"falling hazard is largely a property of mixing rather "
                    f"than of any individual bout, and no per-bout memory may "
                    f"be read from it",
                    n_effective=int(ok.sum()), detail=detail)
    if lo > floor:
        return Read("PASS", scored_object,
                    f"THE FALL SURVIVES CONDITIONING: {tail}, with the interval "
                    f"entirely above the {floor:.2f} floor. Duration dependence "
                    f"is not explained by this mixture. It remains a shared "
                    f"property across states and is still not evidence about "
                    f"any single bout, since an unobserved mixture would look "
                    f"the same",
                    n_effective=int(ok.sum()), detail=detail)
    return Read("INCONCLUSIVE", scored_object,
                f"{tail}, and the interval straddles the {floor:.2f} floor. The "
                f"slice explains some of the fall and the data cannot say "
                f"whether it explains most of it",
                n_effective=int(ok.sum()), detail=detail)
