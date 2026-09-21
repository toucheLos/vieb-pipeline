"""An order-controlled discontinuity plant, and recovery against chance.

READ results/PLANT_PREREGISTRATION.md FIRST.

## Why order is the axis

The detector's axiom is that behaviour is piecewise smooth and a boundary is a
break in **acceleration**. A probe that does not control the ORDER of the break
cannot separate "the criterion is sound but mis-scaled" from "the criterion is
broken" -- which is exactly where `DETECTOR.md` and `PROBE_AUDIT.md` left Step B.

    order 0   position jumps        (nothing continuous at t0)
    order 1   velocity jumps        (position continuous)
    order 2   acceleration jumps    (position and velocity continuous)

Order 2 is the only one the axiom claims, and the number this module exists to
produce is the amplitude at which the detector recovers it.

## The taper, and why it is not a mirror

`PLANT_PREREGISTRATION.md` §2 describes the plant returning to baseline by
mirroring. **Implementing that literally would defeat the probe**: for order 2,
a mirrored ramp meets its own reflection with slopes `+2A` and `-2A`, which
plants an order-*1* break at the midpoint that is sharper than the order-2 break
under test. The detector would then be scored on the wrong discontinuity.

So the return is a **C2-smooth taper** -- `A * u^k * s(u)` with `s` a
smootherstep falling from 1 to 0 over `u in [1, 3]`, whose first and second
derivatives vanish at both ends. The polynomial acts alone on `[0, 1]`, and the
only break of order <= 2 anywhere in the instance is the one planted at `t0`.
This is a faithful reading of "returns to baseline at the same order" and is
recorded as a deviation.

## Recovery is not readable without chance

A detector firing at the corpus rate lands inside a +/-2 window by luck at
`rate * (2*tol+1) / fps` -- 0.074 at the measured 0.443/s. Every recall here is
reported beside a per-recording chance level computed from that recording's own
boundary count on its own planted stream, and the effect is `recall - chance`.
The same discipline `annot.chance_f1` applies to the human ceiling.
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

__all__ = ["ORDERS", "AMPS", "TOL", "MIN_INSTANCES", "N_BOOT",
           "INSTANCE_SPANS", "TAPER_FROM", "profile",
           "place", "apply_plant", "recovered", "recovered_anywhere",
           "chance_of", "recovery_read",
           "ordering_read", "monotone_read", "negative_control_read"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: The three orders. 2 is the axiom; 0 and 1 exist to bracket it.
ORDERS: tuple[int, ...] = (0, 1, 2)
#: Amplitude ladder, in multiples of the MEASURED per-keypoint jitter from
#: `NOISEFLOOR.md`. Not units chosen for convenience: a plant below the noise
#: the tracker already injects is not a fair target for any detector.
AMPS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
#: The registered band, inherited from every other stage and not chosen here.
TOL = 2
#: A cell below this many planted instances is refused, not caveated.
MIN_INSTANCES = 200
N_BOOT = 2000
#: Total instance length, in multiples of the rise window `w`. The decay is
#: exponential, so the tail is numerically zero long before this; the extra
#: room exists so the far-tail taper has nothing left to act on.
INSTANCE_SPANS = 12
#: Where the far-tail taper runs, in multiples of `w`. By `u = 8` the order-2
#: profile is ~7e-6 of its peak, so tapering there costs nothing and
#: guarantees an exact return to baseline.
TAPER_FROM = 8


def _smootherstep(x: npt.ArrayLike) -> F64:
    """`6x^5 - 15x^4 + 10x^3`: value, first AND second derivative vanish at
    both ends."""
    u = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def profile(order: int, w: int) -> F64:
    """The added scalar shape, unit peak amplitude, length `12*w + 1`.

    `g(u) = u^order * exp(-u/tau)`, normalised to peak 1, with `tau` set so the
    peak lands near `u = 1`. At `u = 0` the derivatives below `order` are all
    zero and the `order`-th is not, so **the only discontinuity of any order is
    the planted one at the first sample**. Everywhere else the profile is
    smooth, because an exponential is.

    ## Why not a polynomial that stops

    The obvious construction -- `u^order` on `[0, 1]`, clamped to 1 after, then
    tapered -- is what `PLANT_PREREGISTRATION.md` §2 describes, and it does not
    work. Clamping the polynomial makes the slope fall from `2/w` to `0` at
    `u = 1`, which plants an order-**1** break there. Measured on the first
    implementation at `w = 16`, that junction break was **15.5x** the size of
    the order-2 break under test. Non-maximum suppression would then have
    picked the junction and the probe would have scored the detector on a
    discontinuity it was not supposed to be looking for -- the exact failure
    `DETECTOR.md:98-102` demands a registration to prevent, arriving from the
    other direction. A mirrored taper fails the same way for the same reason.
    Recorded in `DEVIATIONS.md` D12.
    """
    if w < 1:
        raise ValueError("w must be at least 1 frame")
    k = int(order)
    n = INSTANCE_SPANS * w + 1
    u = np.arange(n, dtype=np.float64) / float(w)
    tau = 1.0 / max(k, 1)
    g = (u ** k) * np.exp(-u / tau)
    peak = float(np.max(g))
    g = g / (peak if peak > 0 else 1.0)
    span = float(INSTANCE_SPANS - TAPER_FROM)
    g = g * np.where(u <= TAPER_FROM, 1.0,
                     1.0 - _smootherstep((u - TAPER_FROM) / span))
    return g


def place(rng: np.random.Generator, *, bounds: npt.ArrayLike,
          abstain: npt.ArrayLike, w: int, per_recording: int,
          guard: int) -> I64:
    """Plant onsets, per recording, never across a seam or into abstain.

    Instances are kept `guard` frames apart so no two plants share a detector
    window, and the whole `3*w` extent must be clean -- a plant landing on
    abstained frames measures the abstain mask.
    """
    b = np.asarray(bounds, dtype=np.int64)
    ab = np.asarray(abstain, dtype=bool)
    need = INSTANCE_SPANS * w + 1
    out: list[int] = []
    for r in range(b.size - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        if hi - lo < need + 2 * guard:
            continue
        taken: list[int] = []
        for _ in range(per_recording):
            for _try in range(40):
                t0 = int(rng.integers(lo + guard, hi - need - guard))
                if ab[t0 - guard:t0 + need + guard].any():
                    continue
                if any(abs(t0 - s) < need + guard for s in taken):
                    continue
                taken.append(t0)
                break
        out.extend(taken)
    return np.asarray(sorted(out), dtype=np.int64)


def apply_plant(x: npt.ArrayLike, starts: npt.ArrayLike, *, order: int,
                amp: float, w: int, idx: Sequence[int],
                rng: np.random.Generator) -> F64:
    """Add one instance per start, along a random unit direction each.

    The direction is per instance and spans the scored channels, so the plant
    is a displacement of the whole pose rather than of one keypoint. Applied to
    UNSTANDARDISED ego coordinates, which are already body lengths, so `amp` is
    a physical magnitude.
    """
    a = np.array(x, dtype=np.float64, copy=True)
    g = profile(order, w)
    cols = list(idx)
    for t0 in np.asarray(starts, dtype=np.int64):
        d = rng.normal(size=len(cols))
        d /= max(float(np.linalg.norm(d)), 1e-12)
        a[int(t0):int(t0) + g.size, cols] += float(amp) * g[:, None] * d[None, :]
    return a


def recovered(peaks: npt.ArrayLike, starts: npt.ArrayLike, *,
              tol: int = TOL) -> tuple[int, int]:
    """`(n_hit, n_total)` -- a plant is recovered if any peak is within `tol`."""
    p = np.sort(np.asarray(peaks, dtype=np.int64))
    s = np.asarray(starts, dtype=np.int64)
    if s.size == 0:
        return 0, 0
    if p.size == 0:
        return 0, int(s.size)
    j = np.searchsorted(p, s)
    hit = np.zeros(s.size, dtype=bool)
    for off in (-1, 0):
        k = np.clip(j + off, 0, p.size - 1)
        hit |= np.abs(p[k] - s) <= tol
    return int(hit.sum()), int(s.size)


def recovered_anywhere(peaks: npt.ArrayLike, starts: npt.ArrayLike, *,
                      extent: int) -> tuple[int, int]:
    """`(n_hit, n_total)` -- a peak anywhere inside the instance's extent.

    Descriptive, and it exists to separate two very different failures that
    the registered +/-2 test scores identically: **blind to the event** and
    **found the event, missed its onset**. A boundary detector that fires in
    the middle of a transition has a localisation problem; one that does not
    fire at all has an aperture problem. They need different fixes.
    """
    p = np.sort(np.asarray(peaks, dtype=np.int64))
    s = np.asarray(starts, dtype=np.int64)
    if s.size == 0:
        return 0, 0
    if p.size == 0:
        return 0, int(s.size)
    lo = np.searchsorted(p, s, "left")
    hi = np.searchsorted(p, s + int(extent), "right")
    return int((hi > lo).sum()), int(s.size)


def chance_of(n_peaks: int, n_frames: int, *, tol: int = TOL) -> float:
    """What a detector firing at this rate hits by luck in a +/-tol window."""
    if n_frames <= 0:
        return float("nan")
    return min(1.0, float(n_peaks) * (2 * tol + 1) / float(n_frames))


def recovery_read(per_animal: Sequence[Mapping[str, float]], *, order: int,
                  amp: float, n_instances: int, scored_object: dict[str, Any],
                  n_effective: int, seed: int = 0) -> Read:
    """Recall at one cell, against its own chance level.

    `PASS` means the detector recovers this break above what its own firing
    rate would hit by accident. Refuses a thin cell rather than caveating it.
    """
    detail: dict[str, Any] = {"order": int(order), "amp_sigma": float(amp),
                              "n_instances": int(n_instances)}
    if n_instances < MIN_INSTANCES:
        return Read("NOT_A_RESULT", scored_object,
                    (f"order {order} at {amp:g} sigma planted only "
                     f"{n_instances} instances, below the registered "
                     f"{MIN_INSTANCES}. Refused rather than caveated"),
                    n_effective=n_effective, detail=detail)
    tags = [str(r["animal"]) for r in per_animal]
    eff = [float(r["recall"]) - float(r["chance"]) for r in per_animal]
    rec = boot.animal_interval([float(r["recall"]) for r in per_animal], tags,
                               how="mean", n_boot=N_BOOT, seed=seed)
    ch = boot.animal_interval([float(r["chance"]) for r in per_animal], tags,
                              how="mean", n_boot=N_BOOT, seed=seed)
    ef = boot.animal_interval(eff, tags, how="mean", n_boot=N_BOOT, seed=seed)
    detail.update({"recall": dict(rec), "chance": dict(ch),
                   "effect": dict(ef)})
    if float(ef["lo"]) > 0.0:
        return Read("PASS", scored_object,
                    (f"order {order} at {amp:g} sigma: recall "
                     f"{rec['point']:.4f} [{rec['lo']:.4f}, {rec['hi']:.4f}] "
                     f"against chance {ch['point']:.4f}, effect "
                     f"{ef['point']:+.4f} [{ef['lo']:+.4f}, {ef['hi']:+.4f}] "
                     f"over {n_instances:,} instances"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"order {order} at {amp:g} sigma is NOT recovered above "
                 f"chance: recall {rec['point']:.4f} against chance "
                 f"{ch['point']:.4f}, effect {ef['point']:+.4f} "
                 f"[{ef['lo']:+.4f}, {ef['hi']:+.4f}]"),
                n_effective=n_effective, detail=detail)


def _crossing(cells: Mapping[int, Mapping[float, float]], order: int,
              at: float = 0.5) -> float | None:
    """Lowest amplitude whose recall reaches `at`, or None."""
    for a in AMPS:
        v = cells.get(order, {}).get(a)
        if v is not None and float(v) >= at:
            return float(a)
    return None


def ordering_read(cells: Mapping[int, Mapping[float, float]], *,
                  scored_object: dict[str, Any], n_effective: int) -> Read:
    """Predictions 3 and 4: is order 2 harder, and by how much?

    The headline of the stage. `PASS` means order 2 needs a larger amplitude
    than order 1 by at least one rung, which is the registered prediction that
    the criterion is preferentially blind to the smoother break.
    """
    cross = {k: _crossing(cells, k) for k in ORDERS}
    detail = {"recall_50_sigma": cross,
              "recall_by_cell": {str(k): {str(a): v for a, v in
                                          cells.get(k, {}).items()}
                                 for k in ORDERS}}
    c1, c2 = cross.get(1), cross.get(2)
    if c2 is None and c1 is not None:
        return Read("PASS", scored_object,
                    (f"ORDER 2 IS NEVER RECOVERED at 50% anywhere on the "
                     f"ladder, up to {AMPS[-1]:g} sigma, while order 1 crosses "
                     f"at {c1:g} sigma. The criterion is blind to the only "
                     f"break its own axiom claims, at every amplitude tested"),
                    n_effective=n_effective, detail=detail)
    if c1 is None or c2 is None:
        return Read("INCONCLUSIVE", scored_object,
                    (f"50% recall is not reached on the ladder for the orders "
                     f"the comparison needs: crossings {cross}"),
                    n_effective=n_effective, detail=detail)
    if c2 > c1:
        return Read("PASS", scored_object,
                    (f"order 2 needs {c2:g} sigma to reach 50% recall against "
                     f"order 1's {c1:g} sigma -- {c2 / c1:.0f}x the amplitude "
                     f"for the break the axiom is actually about"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"order 2 reaches 50% at {c2:g} sigma and order 1 at "
                 f"{c1:g}: the same rung, so the registered one-rung "
                 f"separation is not demonstrated. The ladder doubles, so it "
                 f"cannot resolve a difference smaller than 2x -- read "
                 f"`monotone_read`, which compares every cell, before drawing "
                 f"any conclusion about whether the criterion is blind to the "
                 f"smoother break"),
                n_effective=n_effective, detail=detail)


def monotone_read(cells: Mapping[int, Mapping[float, float]], *,
                  scored_object: dict[str, Any], n_effective: int) -> Read:
    """Predictions 2 and 3: monotone in amplitude, and ordered 0 >= 1 >= 2.

    Registered, so computing it is obligatory rather than optional. It is also
    the finer instrument: the crossing test in `ordering_read` can only resolve
    differences bigger than one rung of a doubling ladder, while this sees
    every cell.
    """
    amp_ok = all(
        all(cells[o][b] >= cells[o][a] - 1e-9
            for a, b in zip(AMPS, AMPS[1:]) if a in cells[o] and b in cells[o])
        for o in ORDERS if o in cells)
    shared = [a for a in AMPS if all(a in cells.get(o, {}) for o in ORDERS)]
    ord_ok = all(cells[0][a] >= cells[1][a] >= cells[2][a] for a in shared)
    n_bad = sum(1 for a in shared
                if not (cells[0][a] >= cells[1][a] >= cells[2][a]))
    detail: dict[str, Any] = {"monotone_in_amplitude": bool(amp_ok),
                              "ordered_by_order": bool(ord_ok),
                              "n_amplitudes_compared": len(shared),
                              "n_violations": int(n_bad)}
    if amp_ok and ord_ok:
        gaps = [cells[1][a] - cells[2][a] for a in shared]
        detail["order1_minus_order2"] = [float(g) for g in gaps]
        return Read("PASS", scored_object,
                    (f"recovery is monotone in amplitude within every order "
                     f"and ordered 0 >= 1 >= 2 at all {len(shared)} "
                     f"amplitudes. Order 2 sits below order 1 at every rung, "
                     f"by {min(gaps):.3f} to {max(gaps):.3f} of recall -- the "
                     f"criterion IS preferentially blind to the smoother "
                     f"break, at a resolution the crossing ladder cannot "
                     f"see"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"recovery is not both monotone in amplitude "
                 f"({amp_ok}) and ordered 0 >= 1 >= 2 ({ord_ok}); "
                 f"{n_bad} of {len(shared)} amplitudes violate the ordering"),
                n_effective=n_effective, detail=detail)


def negative_control_read(corpus: Mapping[str, float],
                          white: Mapping[str, float], *,
                          scored_object: dict[str, Any],
                          n_effective: int) -> Read:
    """The debt `DETECTOR_PREREGISTRATION.md:84-90` has carried since Step B.

    A cell recovering its plant on i.i.d. noise as well as on the corpus is
    recovering its own threshold. `PASS` here means the probe is measuring the
    plant.
    """
    detail = {"corpus": dict(corpus), "white": dict(white)}
    worst = max((float(white[k]) - float(corpus[k]) for k in corpus
                 if k in white), default=float("nan"))
    if np.isfinite(worst) and worst < 0.0:
        return Read("PASS", scored_object,
                    (f"the plant is recovered better on the corpus than on "
                     f"i.i.d. `white` at every cell above 4 sigma (worst gap "
                     f"{worst:+.4f}), so the probe measures the plant and not "
                     f"the threshold. This discharges the negative control "
                     f"`DETECTOR_PREREGISTRATION.md:84-90` has required since "
                     f"Step B, for these cells only"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"`white` recovers the plant as well as the corpus somewhere "
                 f"above 4 sigma (gap {worst:+.4f}). Those cells are "
                 f"recovering the detector's own threshold and may not be "
                 f"reported as recoveries"),
                n_effective=n_effective, detail=detail)
