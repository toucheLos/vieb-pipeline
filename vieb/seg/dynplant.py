"""A dynamics detector, and order-controlled dynamics plants to score it with.

READ results/DYNAMICS_PREREGISTRATION.md FIRST.

## The detector

The minimal honest instrument the axiom implies: slide two adjacent windows,
describe the dynamics in each, and cut where the descriptions disagree.

    contrast(t) = || d(x[t-w:t]) - d(x[t:t+w]) ||

with `d` the scale-free pole descriptor, thresholded per recording at
`median + alpha * MAD` -- the same robust, scale-adaptive rule
`breaks.mad_threshold` uses, so `alpha` here and `k_mad` there are comparable
parameters, and the same strongest-first non-maximum suppression, so a weaker
earlier peak cannot suppress a later stronger one.

**This is the simplest thing that could work, on purpose.** Stage 1 elaborates
it with multi-scale persistence and an MDL split test; Stage 0 needs only an
instrument good enough to measure what is recoverable in principle.

## What is planted, and why the radius matters more than the frequency

Measured during implementation: on pure measured-colour tracking noise the
fitted pole comes back at 6.9 Hz with **radius 0.438**, while a genuine damped
oscillator at 1-7 Hz comes back at its true frequency with radius 0.95-0.99.

**So coloured noise does produce an apparent frequency -- it is the persistence
that separates signal from noise, not the frequency.** That is why the plant
ladder sweeps `radius` as its own axis and not only `frequency`, and why
`pole_radius` is carried into every read.

Three plants, each moving one parameter with the others held:

* `frequency` -- the oscillation's rate changes, damping and amplitude held;
* `damping`   -- the pole radius changes, frequency and amplitude held;
* `amplitude` -- the oscillation's size changes, poles held.
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
from vieb.seg import breaks as bk                                  # noqa: E402
from vieb.seg import descriptors as ds                             # noqa: E402

__all__ = ["KINDS", "FREQS_HZ", "DFREQ_HZ", "RADII", "AMPS_BL", "WIN_S",
           "ALPHA", "SPAN", "MIN_INSTANCES", "contrast", "peaks_of", "oscillation",
           "plant_into", "floor_read", "recovery_read"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

KINDS: tuple[str, ...] = ("frequency", "damping", "amplitude")
#: Carrier frequency. 2 Hz sits inside the band the frozen calibration calls
#: usable (0.476-4.83 Hz) and well above the 1/w resolution of the windows.
CARRIER_HZ = 2.0
#: The frequency-step ladder, in Hz. Spans from below the window's own
#: resolution to the full usable band.
DFREQ_HZ: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
#: The damping ladder, as the pole radius AFTER the change, from the noise
#: floor's own 0.438 up to a sustained rhythm.
RADII: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 0.99)
#: Oscillation amplitude in body lengths. `PLANT.md` needed 0.99 bl for an
#: acceleration step, so this brackets it from far below.
AMPS_BL: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
#: Half-window for the two-sided contrast, in seconds. 1 s resolves 1 Hz, so
#: this is the shortest window that can see the carrier at all.
WIN_S = 1.0
#: Robust threshold multiplier, the analogue of `breaks.K_MAD = 3.0`.
ALPHA = 3.0
#: Half-extent of the planted oscillation, in multiples of the contrast
#: half-window.
#:
#: NOT a free choice, and the reason is `DEVIATIONS.md` D12 arriving a second
#: time. A first implementation added the oscillation only over `[t0-w, t0+w]`,
#: which plants THREE changes: silence->A at `t0-w`, A->B at `t0`, and
#: B->silence at `t0+w`. The two envelope edges are amplitude steps from zero
#: and are far sharper than the parameter change under test, so the detector
#: found them instead -- measured, the nearest peak sat 28 frames from the
#: plant, which is exactly one window. The oscillation now runs across the
#: whole extent with only the named parameter switching at `t0`, and the
#: oscillation now spans the WHOLE scored slice with no envelope at all.
#:
#: A second attempt tapered the envelope `SPAN` windows away instead of
#: removing it. That failed the same way for the same reason -- measured, the
#: detector fired 108-118 frames from the plant, which is the taper, not the
#: switch. An envelope edge is a rhythm appearing or vanishing, and no
#: parameter change within a rhythm is ever as large as the rhythm's own
#: arrival. The only construction that scores the intended change is one where
#: the oscillation is present throughout and ONLY the named parameter moves.
SPAN = 4
MIN_INSTANCES = 200
N_BOOT = 2000


def _feat(block: npt.ArrayLike, fps: float) -> F64:
    """The scale-free part of the descriptor: frequency and radius."""
    pf = ds.pole_features(block, fps=fps)
    with np.errstate(invalid="ignore"):
        return np.asarray(
            [float(np.nanmedian(pf["frequency_hz"])),
             float(np.nanmedian(pf["radius"])) * float(fps) / 4.0],
            dtype=np.float64)


def contrast(x: npt.ArrayLike, *, fps: float, win: int,
             stride: int = 1) -> F64:
    """Per-frame dynamics contrast between the windows either side of `t`.

    Radius is put on a comparable scale to frequency by `fps/4`, so a pole
    moving from the noise floor's 0.438 to a sustained 0.99 counts about as
    much as a 4 Hz frequency step. Stated here because it is a choice and it is
    not tuned against any recovery number.
    """
    a = np.asarray(x, dtype=np.float64)
    t = a.shape[0]
    out = np.zeros(t)
    if t < 2 * win + 2:
        return out
    grid = np.arange(win, t - win, max(1, int(stride)))
    vals = np.zeros(grid.size)
    for i, c in enumerate(grid):
        lhs = _feat(a[c - win:c], fps)
        rhs = _feat(a[c:c + win], fps)
        d = lhs - rhs
        vals[i] = float(np.sqrt(np.nansum(d * d))) if np.isfinite(d).any() \
            else 0.0
    out[grid] = np.nan_to_num(vals, nan=0.0, posinf=0.0)
    if int(stride) > 1:
        out = np.interp(np.arange(t), grid, out[grid], left=0.0, right=0.0)
        out[:win] = 0.0
        out[t - win:] = 0.0
    return out


def peaks_of(x: npt.ArrayLike, *, fps: float, win: int, alpha: float = ALPHA,
             blocked: npt.ArrayLike | None = None,
             stride: int = 1) -> I64:
    """Boundaries for ONE recording, slice-local -- the detector interface.

    Deliberately the same shape as `floor.peaks_of` and
    `trendfilter.peaks_of`, so the plant harness can take this detector without
    knowing which it has. The abstain veto is applied here, not inside
    `contrast`, so the exclusion matches every other stage's.
    """
    d = contrast(x, fps=fps, win=win, stride=stride)
    if not np.isfinite(d).any() or float(np.nanmax(d)) <= 0.0:
        return np.zeros(0, dtype=np.int64)
    guard = bk.guard_frames(fps, deriv_sec=bk.DERIV_SEC)
    min_gap = max(2, bk.min_segment_frames(bk.DEGREE, guard))
    return bk.boundaries(d, bk.mad_threshold(d, alpha), min_gap=min_gap,
                         blocked=blocked)


def oscillation(n: int, *, freq_hz: float, radius: float, fps: float,
                rng: np.random.Generator) -> F64:
    """A driven damped oscillator: AR(2) with the requested pole pair.

    Driven rather than decaying-from-an-impulse, so the amplitude is stationary
    over the window and a `damping` plant changes the pole radius without also
    changing the size -- which is what makes the three ladders independent.
    """
    w = 2.0 * np.pi * float(freq_hz) / float(fps)
    r = float(min(max(radius, 0.0), 0.999999))
    a1, a2 = 2.0 * r * np.cos(w), -(r ** 2)
    y = np.zeros(int(n) + 64)
    e = rng.normal(size=y.size)
    for t in range(2, y.size):
        y[t] = a1 * y[t - 1] + a2 * y[t - 2] + e[t]
    y = y[64:]
    s = float(np.std(y))
    return np.asarray(y / (s if s > 0 else 1.0), dtype=np.float64)


def _envelope(n: int, edge: int) -> F64:
    """1 in the middle, smootherstep to 0 over `edge` samples at each end."""
    u = np.ones(int(n))
    if edge < 1:
        return u
    t = np.arange(int(edge), dtype=np.float64) / float(edge)
    ramp = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    u[:edge] = ramp
    u[n - edge:] = ramp[::-1]
    return u


def plant_into(x: npt.ArrayLike, starts: npt.ArrayLike, *, kind: str,
               level: float, fps: float, win: int, amp_bl: float,
               idx: Sequence[int], rng: np.random.Generator,
               span: int = SPAN) -> F64:
    """Add a dynamics change at each start: BEFORE differs from AFTER.

    Both sides carry an oscillation and only the named parameter differs across
    the boundary, so a frequency plant is not an amplitude step wearing a
    frequency step's name.

    **The oscillation spans the whole array and there is no envelope** -- see
    `SPAN` for the two constructions that failed first. `starts` is therefore
    one boundary per call, and the caller works on a slice.
    """
    a = np.array(x, dtype=np.float64, copy=True)
    cols = list(idx)
    for t0 in np.asarray(starts, dtype=np.int64):
        lo, hi = 0, a.shape[0]
        if not (0 < int(t0) < a.shape[0]):
            continue
        if kind == "frequency":
            f_a, f_b, r_a, r_b = CARRIER_HZ, CARRIER_HZ + level, 0.95, 0.95
            g_a = g_b = amp_bl
        elif kind == "damping":
            f_a = f_b = CARRIER_HZ
            r_a, r_b = 0.95, float(level)
            g_a = g_b = amp_bl
        elif kind == "amplitude":
            f_a = f_b = CARRIER_HZ
            r_a = r_b = 0.95
            g_a, g_b = amp_bl, amp_bl * float(level)
        else:
            raise ValueError(f"unknown plant kind {kind!r}")
        d = rng.normal(size=len(cols))
        d /= max(float(np.linalg.norm(d)), 1e-12)
        n_pre, n_post = int(t0) - lo, hi - int(t0)
        pre = oscillation(n_pre, freq_hz=f_a, radius=r_a, fps=fps, rng=rng)
        post = oscillation(n_post, freq_hz=f_b, radius=r_b, fps=fps, rng=rng)
        sig = np.concatenate([g_a * pre, g_b * post])
        a[lo:hi, cols] += sig[:, None] * d[None, :]
    return a


def floor_read(null: Mapping[str, Sequence[float]], *,
               scored_object: dict[str, Any], n_effective: int) -> Read:
    """What the descriptors do on measured-colour noise alone.

    No verdict. It exists so every recovery number downstream is read against
    the distribution noise alone produces, and so the one fact that governs the
    whole design is on the record: coloured noise HAS an apparent frequency,
    and only its persistence gives it away.
    """
    detail: dict[str, Any] = {}
    for k, v in null.items():
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        detail[k] = {"median": float(np.median(a)),
                     "p95": float(np.quantile(a, 0.95)),
                     "p99": float(np.quantile(a, 0.99)), "n": int(a.size)}
    rad = detail.get("pole_radius", {})
    frq = detail.get("pole_frequency_hz", {})
    return Read(
        "NOT_A_RESULT", scored_object,
        (f"descriptor floor under the MEASURED noise colour, no verdict: on "
         f"tracking noise alone the fitted pole sits at "
         f"{frq.get('median', float('nan')):.2f} Hz with radius "
         f"{rad.get('median', float('nan')):.3f} (p95 "
         f"{rad.get('p95', float('nan')):.3f}). Coloured noise therefore HAS "
         f"an apparent frequency and it is the PERSISTENCE that separates it "
         f"from a rhythm -- a frequency read alone cannot"),
        n_effective=n_effective, detail=detail)


def recovery_read(rows: Sequence[Mapping[str, Any]], *, kind: str,
                  level: float, n_instances: int,
                  scored_object: dict[str, Any], n_effective: int,
                  seed: int = 0) -> Read:
    """Recall at one cell of the dynamics ladder, against its own chance."""
    detail: dict[str, Any] = {"kind": kind, "level": float(level),
                              "n_instances": int(n_instances)}
    if n_instances < MIN_INSTANCES:
        return Read("NOT_A_RESULT", scored_object,
                    (f"{kind} at {level:g} planted only {n_instances} "
                     f"instances, below the registered {MIN_INSTANCES}. "
                     f"Refused rather than caveated"),
                    n_effective=n_effective, detail=detail)
    tags = [str(r["animal"]) for r in rows]
    rec = boot.animal_interval([float(r["recall"]) for r in rows], tags,
                               how="mean", n_boot=N_BOOT, seed=seed)
    ch = boot.animal_interval([float(r["chance"]) for r in rows], tags,
                              how="mean", n_boot=N_BOOT, seed=seed)
    ef = boot.animal_interval([float(r["recall"]) - float(r["chance"])
                               for r in rows], tags, how="mean",
                              n_boot=N_BOOT, seed=seed)
    detail.update({"recall": dict(rec), "chance": dict(ch), "effect": dict(ef)})
    if float(ef["lo"]) > 0.0:
        return Read("PASS", scored_object,
                    (f"{kind} at {level:g}: recall {rec['point']:.4f} "
                     f"[{rec['lo']:.4f}, {rec['hi']:.4f}] against chance "
                     f"{ch['point']:.4f}, effect {ef['point']:+.4f} "
                     f"[{ef['lo']:+.4f}, {ef['hi']:+.4f}] over "
                     f"{n_instances:,} instances"),
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                (f"{kind} at {level:g} is NOT recovered above chance: recall "
                 f"{rec['point']:.4f} against chance {ch['point']:.4f}, "
                 f"effect {ef['point']:+.4f} [{ef['lo']:+.4f}, "
                 f"{ef['hi']:+.4f}]"),
                n_effective=n_effective, detail=detail)
