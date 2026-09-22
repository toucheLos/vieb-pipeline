"""Configuration and dynamics as two streams, combined without collapsing them.

READ results/TWOSTREAM_PREREGISTRATION.md FIRST.

## The two streams, and why they get different bands

**Configuration** is the frozen detector -- an acceleration-mismatch threshold
on the `shape` channels -- plus `triage.body_extension`, the stretch-attend
descriptor. It localises to **+/-2 frames**, as `PLANT.md` characterised it.

**Dynamics** is a two-window contrast on `pole_radius` alone. `DYNAMICS.md`
licenses that and nothing else: against measured-colour noise, radius separates
corpus from null on **88.7%** of windows against the 5% chance gives, while
frequency separates on **2.1%** -- below chance -- and no band above 2 Hz
clears. It localises to **+/-15 frames**, derived from its own probe.

**The bands differ because the instruments differ**, and averaging them to a
common band would throw away the more precise one or flatter the less precise
one. The asymmetry is reported.

## Why the combination is neither union nor intersection

A union hides which stream found what; an intersection assumes they should
agree, and `shape` and `twist` already share only **23%** of boundaries at +/-2,
so agreement is not the expected state. Every boundary is labelled
`config_only`, `dyn_only` or `both`, and **the three-way split is the output**.
If `dyn_only` is negligible the dynamics stream adds nothing and the idea is
refuted on this corpus -- which is registered as prediction 1.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import boot                                             # noqa: E402
from recur.read import Read                                        # noqa: E402
from vieb.seg import annot as an                                   # noqa: E402
from vieb.seg import dynplant as dp                                # noqa: E402
from vieb.seg import floor as fl                                   # noqa: E402

__all__ = ["CONFIG_TOL", "DYN_TOL", "DYN_ONLY_MIN", "N_BOOT", "config_peaks",
           "dyn_peaks", "label_boundaries", "split_read", "separation_read",
           "location_read"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: Inherited, and characterised by `PLANT.md`.
CONFIG_TOL = 2
#: Derived from Stage 0's measured localisation: at the one probe cell where
#: the dynamics detector works, offsets run p05 -15.2 to p95 +8.0. Registered
#: before use.
DYN_TOL = 15
#: Registered prediction 1: below this share, the dynamics stream fires only
#: where configuration already did and adds nothing.
DYN_ONLY_MIN = 0.10
N_BOOT = 2000


def config_peaks(x: npt.ArrayLike, *, idx: Sequence[int], fps: float,
                 k_mad: float, blocked: npt.ArrayLike | None = None) -> I64:
    """The frozen detector, unchanged. `idx` may include the extra channel."""
    return fl.peaks_of(x, idx=idx, fps=fps, k_mad=k_mad, blocked=blocked)


def dyn_peaks(x: npt.ArrayLike, *, fps: float, win: int,
              alpha: float = dp.ALPHA,
              blocked: npt.ArrayLike | None = None,
              stride: int = 2) -> I64:
    """The persistence contrast. Radius only -- frequency is below chance."""
    return dp.peaks_of(x, fps=fps, win=win, alpha=alpha, blocked=blocked,
                       stride=stride, mode="radius")


def label_boundaries(config: npt.ArrayLike, dyn: npt.ArrayLike, *,
                     tol: int = DYN_TOL) -> dict[str, Any]:
    """Split the two boundary sets into `config_only`, `dyn_only`, `both`.

    Matched with `annot.match` -- greedy, nearest-first, one-to-one -- so one
    configuration boundary cannot absorb three dynamics boundaries and make the
    streams look more agreed than they are. `scripts/breaks.py:165 _agreement`
    counts unmatched hits and is deliberately not used.
    """
    c = np.sort(np.asarray(config, dtype=np.int64))
    d = np.sort(np.asarray(dyn, dtype=np.int64))
    pairs = an.match(c.tolist(), d.tolist(), tol)
    ci = {i for i, _ in pairs}
    di = {j for _, j in pairs}
    return {"n_config": int(c.size), "n_dyn": int(d.size),
            "n_both": len(pairs),
            "n_config_only": int(c.size) - len(pairs),
            "n_dyn_only": int(d.size) - len(pairs),
            "config_only": c[[i for i in range(c.size) if i not in ci]],
            "dyn_only": d[[j for j in range(d.size) if j not in di]],
            "both": c[[i for i, _ in pairs]]}


def split_read(rows: Sequence[Mapping[str, Any]], *,
               scored_object: dict[str, Any], n_effective: int,
               seed: int = 0) -> Read:
    """Prediction 1: does the dynamics stream find anything of its own?"""
    tags = [str(r["animal"]) for r in rows]
    tot = np.asarray([float(r["n_config"]) + float(r["n_dyn_only"])
                      for r in rows])
    share = np.asarray([float(r["n_dyn_only"]) / t if t > 0 else np.nan
                        for r, t in zip(rows, tot)])
    ok = np.isfinite(share)
    if int(ok.sum()) < 2:
        return Read("NOT_A_RESULT", scored_object,
                    "fewer than two animals produced any boundary",
                    n_effective=n_effective, detail={"n_animals": int(ok.sum())})
    ci = boot.animal_interval(share[ok], list(np.asarray(tags)[ok]),
                              how="mean", n_boot=N_BOOT, seed=seed)
    n_c = int(sum(r["n_config"] for r in rows))
    n_d = int(sum(r["n_dyn"] for r in rows))
    n_b = int(sum(r["n_both"] for r in rows))
    detail = {"dyn_only_share": dict(ci), "n_config": n_c, "n_dyn": n_d,
              "n_both": n_b, "n_dyn_only": n_d - n_b,
              "n_config_only": n_c - n_b, "threshold": DYN_ONLY_MIN,
              "tol": DYN_TOL}
    if float(ci["lo"]) > DYN_ONLY_MIN:
        return Read(
            "PASS", scored_object,
            (f"THE DYNAMICS STREAM FINDS BOUNDARIES OF ITS OWN: "
             f"{ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] of all "
             f"boundaries are `dyn_only`, above the registered "
             f"{DYN_ONLY_MIN:.0%}. {n_b:,} of {n_d:,} dynamics boundaries "
             f"match a configuration boundary within ±{DYN_TOL}; the rest do "
             f"not. Configuration and dynamics are not the same cut"),
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        (f"the dynamics stream adds nothing: only {ci['point']:.3f} "
         f"[{ci['lo']:.3f}, {ci['hi']:.3f}] of boundaries are `dyn_only`, at "
         f"or below the registered {DYN_ONLY_MIN:.0%}. It fires where the "
         f"configuration stream already fired, and the two-stream idea is "
         f"refuted on this corpus"),
        n_effective=n_effective, detail=detail)


def separation_read(corpus: Mapping[str, Any], floor: Mapping[str, Any], *,
                    incumbent: float, scored_object: dict[str, Any],
                    n_effective: int) -> Read:
    """Prediction 2, stated against the incumbent's value -- M12.

    The frozen detector's own jitter share is `incumbent`. A bar that the
    thing being replaced misses is not a bar, which is what M12 cost.
    """
    c, f = dict(corpus), dict(floor)
    share = float(f["point"]) / max(float(c["point"]), 1e-12)
    detail = {"corpus": c, "floor": f, "jitter_share": share,
              "incumbent_share": float(incumbent)}
    if float(c["lo"]) > float(f["hi"]) and share < incumbent:
        return Read(
            "PASS", scored_object,
            (f"the dynamics stream is above its own coloured-noise floor and "
             f"beats the incumbent on the comparable quantity: "
             f"{c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]/s against a "
             f"floor of {f['point']:.4f} [{f['lo']:.4f}, {f['hi']:.4f}]/s, a "
             f"jitter share of {share:.4f} against the frozen detector's "
             f"{incumbent:.4f}"),
            n_effective=n_effective, detail=detail)
    if float(c["lo"]) <= float(f["hi"]):
        return Read(
            "FAIL", scored_object,
            (f"the dynamics stream is NOT distinguishable from its own "
             f"coloured-noise floor: {c['point']:.4f} [{c['lo']:.4f}, "
             f"{c['hi']:.4f}]/s against {f['point']:.4f} [{f['lo']:.4f}, "
             f"{f['hi']:.4f}]/s, intervals overlapping"),
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        (f"the dynamics stream clears its floor but does NOT beat the "
         f"incumbent: jitter share {share:.4f} against the frozen detector's "
         f"{incumbent:.4f}. Clearing a floor is not the test; the test is "
         f"being better than what it replaces"),
        n_effective=n_effective, detail=detail)


def location_read(by_tercile: Mapping[str, Mapping[str, float]], *,
                  scored_object: dict[str, Any], n_effective: int) -> Read:
    """Prediction 4: does each stream track the wall, and how strongly?

    Descriptive. Tracking failure rises 3.7x monotone from arena centre to wall
    (`CONCENTRATION.md`), so a stream whose rate rises steeply with edgeness is
    partly reading the tracker. The comparison between the two streams is the
    point: a scale-free descriptor should care less than an acceleration
    threshold does.
    """
    detail = {k: dict(v) for k, v in by_tercile.items()}
    out = []
    for name, v in sorted(by_tercile.items()):
        lo, hi = float(v.get("t0", np.nan)), float(v.get("t2", np.nan))
        r = hi / lo if lo > 0 else float("nan")
        detail[name] = {**dict(v), "wall_over_centre": r}
        out.append(f"{name} {r:.2f}x")
    return Read(
        "NOT_A_RESULT", scored_object,
        ("descriptive, no verdict: boundary rate from arena centre to wall, "
         + "; ".join(out) + ". Tracking failure itself rises 3.7x over the "
         "same range, so a stream near that number is partly reading the "
         "tracker rather than the animal"),
        n_effective=n_effective, detail=detail)
