"""The boundary rate on pure tracking noise, and where the corpus sits on it.

READ results/NOISEFLOOR_PREREGISTRATION.md FIRST.

## What a floor is for

The detector of record cuts 0.443 boundaries per second (`BREAKS.md`, `shape`,
0.133 s, degree 3) on a corpus that is mostly a still mouse seen from above
through seven keypoints. `ROUGHNESS.md` establishes the signal is
piecewise-smooth. **That is not the same claim**: a piecewise-smooth signal can
still be sampled by an instrument whose own noise dominates in the still regime,
and nothing on disk separated those. This measures the floor instead of arguing
about it.

## The floor is not expected to be zero, and that is the point

`mad_threshold` recomputes from each stream's OWN `D`, so a jitter-only stream
gets its own adaptive threshold exactly as the corpus does. A `median + 3 MAD`
rule with a 16-frame refractory period fires at some rate on any noise process.
**That rate is the quantity of interest**, and it is reported against the NMS
resolution ceiling `fps / min_gap` as well as absolutely --
`SEGMENTATION_PREREGISTRATION.md` already established that rates near that
ceiling are a property of the refractory period rather than of the signal, and
the same reasoning applies to a floor.

## Why one axis would mislead

Low DLC confidence concentrates at the wall, where the mouse rears, and rearing
is a real behaviour that ALSO breaks tracking. So a marginal showing boundaries
piling up at low confidence is ambiguous between "the detector is cutting noise"
and "the detector is finding rearing onsets through bad tracking". Only the
joint cell -- still AND centre-arena AND low-confidence -- separates them.
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
from vieb.seg import breaks as bk                                   # noqa: E402

__all__ = ["ARMS", "MIN_CELL_FRAMES", "N_BOOT", "static_pose", "peaks_of",
           "rate_of", "nms_ceiling", "cell_index", "cell_rates",
           "static_read", "floor_read", "dose_read", "separation_read"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: The injection ladder. `static` is the mechanical check; the three `jitter`
#: arms are the floor and its dose-response.
ARMS: tuple[str, ...] = ("static", "jitter0.5", "jitter1.0", "jitter2.0")
SCALES: dict[str, float] = {"static": 0.0, "jitter0.5": 0.5,
                            "jitter1.0": 1.0, "jitter2.0": 2.0}
#: Registered refusal floor. A cell thinner than this is refused, never
#: reported with a caveat.
MIN_CELL_FRAMES = 20_000
N_BOOT = 2000


def static_pose(pose: npt.ArrayLike, usable: npt.ArrayLike) -> F64:
    """The animal's own median pose, held for the whole recording.

    A signal with no variance. Every boundary found in it, once noise is added,
    is noise -- there is nothing else left for the detector to respond to.
    """
    p = np.asarray(pose, dtype=np.float64)
    ok = np.asarray(usable, dtype=bool)
    if int(ok.sum()) == 0:
        return np.zeros_like(p)
    med = np.median(p[ok], axis=0)
    return np.repeat(med[None, ...], p.shape[0], axis=0)


def peaks_of(x: npt.ArrayLike, *, idx: Sequence[int], fps: float,
             k_mad: float, blocked: npt.ArrayLike | None = None) -> I64:
    """The frozen detector's boundaries for ONE recording, slice-local.

    The identical five calls every other stage makes, at the identical frozen
    settings. Nothing here is swept: a changed window, degree or threshold is a
    different detector and needs its own registration.
    """
    a = np.asarray(x, dtype=np.float64)[:, list(idx)]
    h = max(2, int(round(bk.DERIV_SEC * fps)))
    guard = bk.guard_frames(fps, deriv_sec=bk.DERIV_SEC)
    min_gap = bk.min_segment_frames(bk.DEGREE, guard)
    d = bk.discontinuity(a, h)
    return bk.boundaries(d, bk.mad_threshold(d, k_mad), min_gap=min_gap,
                         blocked=blocked)


def nms_ceiling(fps: float) -> float:
    """The most boundaries per second the refractory period can emit."""
    guard = bk.guard_frames(fps, deriv_sec=bk.DERIV_SEC)
    return float(fps) / max(2, bk.min_segment_frames(bk.DEGREE, guard))


def rate_of(x: npt.ArrayLike, bounds: npt.ArrayLike,
            abstain: npt.ArrayLike, *, idx: Sequence[int], fps: float,
            k_mad: float) -> dict[str, Any]:
    """Boundaries per second over one animal, never across a seam.

    The denominator is *eligible* frames: `discontinuity` zeroes the first and
    last `h` of every recording and blocked frames cannot be boundaries, so
    counting them would understate the rate by a fixed fraction that differs
    between a long recording and a short one.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.int64)
    ab = np.asarray(abstain, dtype=bool)
    h = max(2, int(round(bk.DERIV_SEC * fps)))
    n_pk = 0
    n_elig = 0
    for r in range(b.size - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        if hi - lo < 2 * h + 1:
            continue
        pk = peaks_of(a[lo:hi], idx=idx, fps=fps, k_mad=k_mad,
                      blocked=ab[lo:hi])
        n_pk += int(pk.size)
        n_elig += int((~ab[lo + h:hi - h]).sum())
    secs = n_elig / float(fps)
    return {"n_peaks": n_pk, "n_eligible_frames": n_elig,
            "eligible_seconds": secs,
            "rate_per_s": (n_pk / secs) if secs > 0 else float("nan")}


def cell_index(speed: npt.ArrayLike, conf: npt.ArrayLike,
               edge: npt.ArrayLike, *, speed_edges: npt.ArrayLike,
               conf_edges: npt.ArrayLike,
               edge_edges: npt.ArrayLike) -> I64:
    """Joint cell per frame: speed x confidence x arena. -1 where undefined.

    Joint and not three marginals. A marginal on confidence alone cannot tell
    a noisy still mouse from a rearing one, because rearing both raises
    edgeness and lowers confidence.
    """
    s = np.asarray(speed, dtype=np.float64)
    c = np.asarray(conf, dtype=np.float64)
    e = np.asarray(edge, dtype=np.float64)
    ns = len(np.asarray(speed_edges)) - 1
    nc = len(np.asarray(conf_edges)) - 1
    ne = len(np.asarray(edge_edges)) - 1
    ok = np.isfinite(s) & np.isfinite(c) & np.isfinite(e)
    bs = np.clip(np.searchsorted(speed_edges, s, "right") - 1, 0, ns - 1)
    bc = np.clip(np.searchsorted(conf_edges, c, "right") - 1, 0, nc - 1)
    be = np.clip(np.searchsorted(edge_edges, e, "right") - 1, 0, ne - 1)
    out = (bs * nc + bc) * ne + be
    return np.where(ok, out, -1).astype(np.int64)


def cell_rates(cells: npt.ArrayLike, is_peak: npt.ArrayLike,
               animals: Sequence[str], *, n_cells: int, fps: float,
               min_frames: int = MIN_CELL_FRAMES) -> list[dict[str, Any]]:
    """Boundary rate per joint cell, with an animal-level interval.

    A cell below `min_frames` is **refused**, not reported with a caveat. The
    refusal is a row, so the count of refused cells is visible rather than
    being an absence a reader has to notice.
    """
    cl = np.asarray(cells, dtype=np.int64)
    pk = np.asarray(is_peak, dtype=bool)
    an = np.asarray(animals)
    rows: list[dict[str, Any]] = []
    for c in range(n_cells):
        sel = cl == c
        n = int(sel.sum())
        row: dict[str, Any] = {"cell": c, "n_frames": n}
        if n < min_frames:
            row.update({"refused": True, "reason": "below 20,000 frames",
                        "rate_per_s": float("nan")})
            rows.append(row)
            continue
        tags = an[sel]
        per: list[float] = []
        who: list[str] = []
        for t in sorted(set(tags.tolist())):
            m = sel & (an == t)
            k = int(m.sum())
            if k == 0:
                continue
            per.append(float(pk[m].sum()) / (k / float(fps)))
            who.append(str(t))
        ci = boot.animal_interval(per, who, how="mean", n_boot=N_BOOT, seed=0)
        row.update({"refused": False, "n_peaks": int(pk[sel].sum()),
                    "rate_per_s": float(pk[sel].sum()) / (n / float(fps)),
                    "ci": dict(ci), "n_animals": len(who)})
        rows.append(row)
    return rows


def static_read(rate: Mapping[str, Any], *, scored_object: dict[str, Any],
                n_effective: int) -> Read:
    """Prediction 1: a signal with no variance produces no boundaries.

    Mechanical, and registered so the harness is checkable rather than
    trusted. If this fails nothing else in the stage means anything.
    """
    n = int(rate["n_peaks"])
    detail = dict(rate)
    if n == 0:
        return Read("PASS", scored_object,
                    (f"the constant-pose arm produced ZERO boundaries over "
                     f"{rate['eligible_seconds']:,.0f} eligible seconds, so "
                     f"the harness responds to variance and not to itself"),
                    n_effective=n_effective, detail=detail)
    return Read("NOT_A_RESULT", scored_object,
                (f"THE HARNESS IS WRONG: the constant-pose arm produced {n} "
                 f"boundaries in a signal with no variance. Every other "
                 f"number in this stage is void until that is explained"),
                n_effective=n_effective, detail=detail)


def floor_read(rate: Mapping[str, Any], *, fps: float,
               scored_object: dict[str, Any], n_effective: int) -> Read:
    """Prediction 2: does the frozen detector fire on pure tracking noise?

    `PASS` means it does -- there is a floor, and every rate in this programme
    has to be read against it. `FAIL` means it does not, the noise hypothesis
    is dead, and that is a complete and useful finding.
    """
    ci = rate["ci"]
    ceil = nms_ceiling(fps)
    frac = float(ci["point"]) / ceil if ceil > 0 else float("nan")
    detail = {**{k: v for k, v in rate.items() if k != "ci"}, "ci": dict(ci),
              "nms_ceiling_per_s": ceil, "fraction_of_ceiling": frac}
    if float(ci["lo"]) > 0.0:
        return Read(
            "PASS", scored_object,
            (f"THE DETECTOR FIRES ON PURE TRACKING NOISE at "
             f"{ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] "
             f"boundaries/s -- {frac:.1%} of the {ceil:.3f}/s NMS resolution "
             f"ceiling -- on a constant pose carrying nothing but measured "
             f"DLC jitter. There is a floor and every boundary rate in this "
             f"programme has to be read against it"),
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        (f"the detector does NOT fire on tracking noise at the measured "
         f"amplitude: {ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] "
         f"boundaries/s, interval including zero. The noise hypothesis is "
         f"dead and the boundary rate is not explained by jitter"),
        n_effective=n_effective, detail=detail)


def dose_read(rates: Mapping[str, Mapping[str, Any]], *,
              scored_object: dict[str, Any], n_effective: int) -> Read:
    """Prediction 3: rate rises with jitter amplitude.

    A flat or non-monotone response would mean the rate is set by the
    threshold rule and the refractory period rather than by the noise -- still
    reportable, but a statement about the detector's plumbing rather than
    about the tracker, and it must not be read as the first.
    """
    got = [(SCALES[k], float(rates[k]["rate_per_s"]))
           for k in ("jitter0.5", "jitter1.0", "jitter2.0") if k in rates]
    vals = [v for _, v in got]
    mono = all(b >= a for a, b in zip(vals, vals[1:])) and len(vals) > 1
    detail = {"points": [{"scale": s, "rate_per_s": v} for s, v in got],
              "monotone": bool(mono)}
    shown = ", ".join(f"x{s:g} -> {v:.4f}/s" for s, v in got)
    if mono:
        return Read("PASS", scored_object,
                    (f"boundary rate rises with injected jitter amplitude "
                     f"({shown}), so the floor tracks the tracker rather "
                     f"than the threshold rule"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"boundary rate is NOT monotone in jitter amplitude "
                 f"({shown}). The floor is a property of the threshold rule "
                 f"and the refractory period rather than of the tracker, and "
                 f"may not be read as a measurement of tracking noise"),
                n_effective=n_effective, detail=detail)


def separation_read(cell: Mapping[str, Any], floor: Mapping[str, Any], *,
                    label: str, scored_object: dict[str, Any],
                    n_effective: int) -> Read:
    """Prediction 5: is the corpus above the floor where it should be worst?

    `FAIL` -- overlapping intervals -- is the registered noise diagnosis: in
    the stillest, most central, least confident frames the detector is not
    distinguishable from its own noise floor. `PASS` says it is above the
    floor even there.

    Dominance is **non-overlapping intervals**, not a point comparison: a point
    comparison crowned a cleaning arm on 0.029 px once already.
    """
    if cell.get("refused"):
        return Read("NOT_A_RESULT", scored_object,
                    (f"the {label} cell holds {cell['n_frames']:,} frames, "
                     f"below the registered {MIN_CELL_FRAMES:,} floor. "
                     f"Refused rather than reported with a caveat"),
                    n_effective=n_effective, detail=dict(cell))
    c, f = cell["ci"], floor["ci"]
    detail = {"cell": {k: v for k, v in cell.items() if k != "ci"},
              "cell_ci": dict(c), "floor_ci": dict(f), "label": label}
    if float(c["lo"]) > float(f["hi"]):
        return Read(
            "PASS", scored_object,
            (f"in the {label} cell the corpus cuts at {c['point']:.4f} "
             f"[{c['lo']:.4f}, {c['hi']:.4f}] boundaries/s against a floor of "
             f"{f['point']:.4f} [{f['lo']:.4f}, {f['hi']:.4f}] -- "
             f"non-overlapping, so the detector is above its own noise floor "
             f"there. The floor is {float(f['point']) / float(c['point']):.1%} "
             f"of the rate"),
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        (f"IN THE {label.upper()} CELL THE DETECTOR IS NOT DISTINGUISHABLE "
         f"FROM ITS OWN NOISE FLOOR: corpus {c['point']:.4f} [{c['lo']:.4f}, "
         f"{c['hi']:.4f}] against floor {f['point']:.4f} [{f['lo']:.4f}, "
         f"{f['hi']:.4f}] boundaries/s, intervals overlapping. Boundaries "
         f"found there are not evidence of behaviour"),
        n_effective=n_effective, detail=detail)
