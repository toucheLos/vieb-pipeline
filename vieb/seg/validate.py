r"""Are the boundaries real? The gate, registered in
`SEGMENTATION_PREREGISTRATION.md`.

A changepoint detector fires on noise. `BREAKS.md` reports 0.26–0.64 boundaries
per second and calls none of it evidence, because a detector run on a smooth,
structureless signal also produces boundaries at some rate with some duration
distribution. The question is whether it fires **more** on the corpus, across a
threshold sweep, in places not explained by something else.

## Why the comparison at a common `k_mad` is fair

The threshold is `median + k · 1.4826 · MAD` of each recording's own `D`, so it
is **scale-adaptive**: a common `k` means "this far above this signal's own noise
floor" for the corpus and for every surrogate alike. Comparing at a fixed
absolute threshold would instead compare how large the two signals' second
derivatives happen to be, which is a statement about amplitude rather than about
boundaries.

## Why a sweep and not a point

A single threshold is a parameter chosen against an outcome. ExBias's
`k_mad = 3.0` is adaptive to the recording but carries no stated false-positive
rate, and a gap that exists at one `k` and nowhere else is a gap in the
thresholding.

## The length-matched control, and the number it has to beat

Fit quality alone proves nothing: a cubic fits almost any short interval of an
autocorrelated signal. The honest test is against **length-matched random
intervals from the same recording**, which ExBias measured at **65.9% ± 1.5%**
(955 segments, 10 recordings, mean ΔR² +0.048) and described as *"real and highly
significant, but far from the >90% a cleanly piecewise-smooth process would
give."* That is the standard, not 50%.

## Polarity of the separability probe

**High separability is failure.** A null a probe can pick out differs from the
corpus in ways unrelated to boundaries, so a boundary-rate gap against it means
nothing. Stage A measured the corpus sitting 24× further from every surrogate
than from itself while matching ambient scale to 2% — a null can look identically
scaled and still be trivially separable in the basis the analysis uses.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

from vieb.seg import breaks as bk

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

__all__ = ["K_SWEEP", "NEGATIVE_CONTROL", "PRIMARY_NULL", "EXBIAS_WIN_RATE",
           "gate_read", "length_matched_control", "manifold_read", "rate_curve"]

#: Fixed in the registration. A single threshold is a parameter chosen against
#: an outcome.
K_SWEEP: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
#: The tightest null: same power spectrum, hence the same quantization-scale
#: roughness, with no stereotyped sequence.
PRIMARY_NULL = "phase"
#: Beating this is not a result. Every smooth signal beats i.i.d. noise.
NEGATIVE_CONTROL = "white"
#: ExBias's measured win rate against length-matched random intervals, and the
#: standard this control is judged against rather than 50%.
EXBIAS_WIN_RATE = 0.659


def rate_curve(x: npt.ArrayLike, bounds: npt.ArrayLike,
               abstain: npt.ArrayLike, *, h: int, min_gap: int,
               fps: float, k_values: Sequence[float] = K_SWEEP) -> Detail:
    """Boundaries per second at each `k_mad`, computed per recording.

    `D` is computed once per recording and thresholded at every `k`, so the
    sweep costs one pass rather than one pass per threshold, and every point on
    the curve is the same statistic read at a different height.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.int64)
    ab = np.asarray(abstain, dtype=bool)
    counts = {float(k): 0 for k in k_values}
    for r in range(b.shape[0] - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        d = bk.discontinuity(a[lo:hi], int(h))
        for k in k_values:
            pk = bk.boundaries(d, bk.mad_threshold(d, float(k)),
                               min_gap=int(min_gap), blocked=ab[lo:hi])
            counts[float(k)] += int(pk.size)
    secs = float(a.shape[0]) / float(fps)
    return {"n_peaks": counts, "seconds": secs,
            "rate": {k: (v / secs if secs > 0 else float("nan"))
                     for k, v in counts.items()}}


def length_matched_control(x: npt.ArrayLike, rows: Sequence[Mapping[str, Any]],
                           lo: int, hi: int, rng: np.random.Generator, *,
                           degree: int = bk.DEGREE,
                           n_draws: int = 1) -> Detail:
    """Detected segments against random intervals of the **same length**.

    Same length and same recording, because segment length is itself the
    strongest predictor of fit quality on this corpus — ExBias measured
    ρ(duration, R²) = −0.538 — so an unmatched control would compare lengths
    rather than smoothness.

    Only the guard-inset interval is fitted, for both sides, so the comparison is
    like for like.
    """
    a = np.asarray(x, dtype=np.float64)
    wins, deltas = 0, []
    for row in rows:
        if not row.get("fittable"):
            continue
        fs, fe = int(row["fit_start"]), int(row["fit_stop"])
        n = fe - fs
        if n <= degree + 1 or (hi - lo) <= n:
            continue
        got = float(row["fit_r2_adj"])
        if not np.isfinite(got):
            continue
        for _ in range(int(n_draws)):
            s = int(rng.integers(int(lo), int(hi) - n))
            try:
                _r2, adj = bk.fit_quality(a[s:s + n], degree)
            except ValueError:
                continue
            if not np.isfinite(adj):
                continue
            deltas.append(got - adj)
            wins += int(got > adj)
    n = len(deltas)
    return {"n_compared": n,
            "win_rate": float(wins / n) if n else float("nan"),
            "mean_delta_r2_adj": float(np.mean(deltas)) if n else float("nan"),
            "exbias_reference": EXBIAS_WIN_RATE}


def manifold_read(res: Mapping[str, Any], *, kind: str, scored_object: Detail,
                  n_effective: int, limit: float = 0.60) -> Read:
    """Can a probe tell corpus windows from surrogate windows?

    **It must not be able to**, and that polarity is the opposite of every other
    probe in this repo. A null a classifier can pick out differs from the corpus
    in ways unrelated to boundaries, so any boundary-rate gap against it is
    uninterpretable — the surrogate would be failing to be a surrogate, not
    failing to have boundaries.

    Written separately from `separable_read` rather than reused: that one's
    reason string is about a corrector depositing mass on a constraint surface,
    and printing it here would describe the wrong experiment entirely.
    """
    acc = res.get("balanced_accuracy")
    detail: Detail = {"kind": kind, **{k: v for k, v in res.items()}}
    if acc is None:
        return Read("INCONCLUSIVE", scored_object,
                    f"the {kind} probe could not run: "
                    f"{res.get('why', 'no reason recorded')}",
                    n_effective=n_effective, detail=detail)
    if res.get("converged") is False:
        return Read("NOT_A_RESULT", scored_object,
                    f"the {kind} probe did not converge, so its "
                    f"{float(acc):.3f} balanced accuracy is a property of the "
                    f"solver rather than of the surrogate",
                    n_effective=n_effective, degenerate=True, detail=detail)
    tail = (f"a linear probe on stacked windows separates corpus from {kind} at "
            f"{float(acc):.3f} balanced accuracy (AUC "
            f"{float(res.get('auc', float('nan'))):.3f}) over "
            f"{int(res.get('n_test', 0)):,} held-out rows, grouped by recording "
            f"block")
    if float(acc) > limit:
        return Read("FAIL", scored_object,
                    f"{kind} IS SEPARABLE from the corpus: {tail}, above the "
                    f"{limit:.2f} limit. It differs from the corpus in ways "
                    f"unrelated to boundaries, so a boundary-rate gap against "
                    f"it is uninterpretable and is not reported as evidence",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"{kind} is not separable from the corpus: {tail}, at or below "
                f"the {limit:.2f} limit. It is a usable null for a "
                f"boundary-rate comparison",
                n_effective=n_effective, detail=detail)


def gate_read(deltas: Mapping[str, Any], *, scored_object: Detail,
              n_effective: int,
              negative_control: str = NEGATIVE_CONTROL) -> Read:
    """The gate. Registered in `SEGMENTATION_PREREGISTRATION.md` §5.

    `deltas` maps `"{kind}|{k}"` to an interval on the per-animal difference
    `rate_corpus − rate_surrogate`. A PASS needs the corpus above **every**
    structured null at **every** threshold; beating the negative control alone is
    not a result, because every smooth signal beats i.i.d. noise.
    """
    rows: list[Detail] = []
    for key, ci in deltas.items():
        kind, k = key.split("|")
        rows.append({"kind": str(kind), "k_mad": float(k),
                     "point": float(ci.get("point", np.nan)),
                     "lo": float(ci.get("lo", np.nan)),
                     "hi": float(ci.get("hi", np.nan))})
    structured = [r for r in rows if r["kind"] != negative_control]
    detail: Detail = {"n_cells": len(rows), "cells": rows,
                      "negative_control": negative_control}
    if not structured:
        return Read("NOT_A_RESULT", scored_object,
                    "no structured null was compared, so there is no gate to "
                    "read", n_effective=n_effective, degenerate=True,
                    detail=detail)

    def _lo(r: Detail) -> float:
        return float(r["lo"])

    unpowered = [r for r in structured if not np.isfinite(_lo(r))]
    beaten = [r for r in structured if np.isfinite(_lo(r)) and _lo(r) > 0]
    failed = [r for r in structured if np.isfinite(_lo(r)) and _lo(r) <= 0]
    detail.update({"n_structured": len(structured), "n_beaten": len(beaten),
                   "n_failed": len(failed), "n_unpowered": len(unpowered)})
    worst = min(structured,
                key=lambda r: float("inf") if not np.isfinite(_lo(r)) else _lo(r))
    detail["worst_cell"] = worst
    w_kind = str(worst["kind"])
    w_k, w_pt = float(worst["k_mad"]), float(worst["point"])
    w_lo, w_hi = float(worst["lo"]), float(worst["hi"])

    if unpowered and not failed:
        return Read("GRID_LIMITED", scored_object,
                    f"{len(unpowered)} of {len(structured)} corpus-versus-null "
                    f"cells could not be resolved, so the sweep does not "
                    f"support a verdict across its whole range",
                    n_effective=n_effective, detail=detail)
    if failed:
        return Read("FAIL", scored_object,
                    f"THE SEGMENTATION ROUTE CLOSES: the corpus boundary rate "
                    f"sits inside the surrogate range in {len(failed)} of "
                    f"{len(structured)} cells — worst is {w_kind} at "
                    f"k = {w_k:g}, {w_pt:+.4f} "
                    f"[{w_lo:+.4f}, {w_hi:+.4f}] boundaries/s. A "
                    f"detector that does not fire more on the corpus than on a "
                    f"structureless signal is not finding behaviour, and that "
                    f"is the finding",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"the corpus beats every structured null at every threshold: "
                f"{len(beaten)} of {len(structured)} cells with the interval "
                f"excluding zero, tightest {w_kind} at "
                f"k = {w_k:g}, {w_pt:+.4f} "
                f"[{w_lo:+.4f}, {w_hi:+.4f}] boundaries/s. This "
                f"says the boundaries are not explained by a structureless "
                f"signal; it does NOT say they are behaviours, which is Step "
                f"3's question",
                n_effective=n_effective, detail=detail)
