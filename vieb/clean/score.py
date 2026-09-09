r"""Scoring a cleaning arm on three axes that can disagree.

No single number decides this. A method wins only if it reduces the violation
rate **without** inflating distortion or collapsing the high-frequency content,
and where the three disagree that disagreement is the result.

``violation``
    Skull-triangle bone violations at fixed eps, the Step 1 diagnostic. The
    reference length is refitted **per arm**, deliberately: an arm that shrinks
    every distance uniformly would otherwise post a lower rate for having made
    the animal smaller. Refitting makes the axis about relative geometry, which
    is the thing a violation is.

``distortion``
    Displacement from the arm's own input, in px and body lengths. A method that
    fixes violations by moving everything has not cleaned anything, and this is
    the axis that says so.

``retention``
    Fraction of spectral power above the crossover frequency that survives.
    shapeflow measured f_c = 4.833 Hz as the point where coherent movement falls
    to tracking-noise power, and only 1.059% of coherent power sits above it --
    so an arm deleting far more than that up there is deleting movement, not
    noise. Catches over-smoothing the first two axes miss.

Retention is deliberately **not** a "higher is better" axis on its own. `raw`
retains 100% by definition and is the worst arm on violations. The three are
read together or not at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]

#: Above this frequency, movement power has fallen to tracking-noise power.
#: Read from shapeflow's calibration rather than hardcoded; this is the fallback
#: if the artifact cannot be reached.
DEFAULT_F_C_HZ = 4.832960531486109

#: An arm that deletes more than this share of the >f_c band is removing more
#: than the 1.059% of coherent power that lives up there.
MAX_HF_DELETED = 0.9


def high_frequency_retention(before: npt.ArrayLike, after: npt.ArrayLike,
                             fps: float, *, f_c: float = DEFAULT_F_C_HZ) -> Detail:
    r"""Share of above-`f_c` spectral power that survives the arm.

    Per keypoint per coordinate, linearly detrended, summed over the band. A
    single ratio over the whole corpus band rather than a per-bin curve, because
    the question here is "how much of the fast content is left", not "what shape
    is the response" -- shapeflow's `clean_gain.svg` already draws the shape.
    """
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    t = a.shape[0]
    if t < 64:
        return {"retained": float("nan"), "why": "recording too short"}
    flat_a = a.reshape(t, -1)
    flat_b = b.reshape(t, -1)
    idx = np.arange(t, dtype=np.float64)
    design = np.stack([np.ones(t), idx / t], axis=1)
    coef_a, *_ = np.linalg.lstsq(design, flat_a, rcond=None)
    coef_b, *_ = np.linalg.lstsq(design, flat_b, rcond=None)
    ra = flat_a - design @ coef_a
    rb = flat_b - design @ coef_b
    freq = np.fft.rfftfreq(t, d=1.0 / float(fps))
    hi = freq >= float(f_c)
    if not hi.any():
        return {"retained": float("nan"), "why": "f_c above Nyquist"}
    pa = float((np.abs(np.fft.rfft(ra, axis=0))[hi] ** 2).sum())
    pb = float((np.abs(np.fft.rfft(rb, axis=0))[hi] ** 2).sum())
    lo_a = float((np.abs(np.fft.rfft(ra, axis=0))[~hi] ** 2).sum())
    lo_b = float((np.abs(np.fft.rfft(rb, axis=0))[~hi] ** 2).sum())
    return {
        "retained": pb / pa if pa > 0 else float("nan"),
        "retained_below_f_c": lo_b / lo_a if lo_a > 0 else float("nan"),
        "f_c_hz": float(f_c), "n_bins_above": int(hi.sum()),
    }


def rank(arms: Mapping[str, Mapping[str, float]], *,
         baseline: str = "raw") -> list[Detail]:
    """Order arms by violation reduction, carrying the other two axes along.

    Sorted, never reduced to a score. Combining three axes into one number would
    hide exactly the case this bakeoff exists to find -- an arm that wins on
    violations by moving everything.
    """
    base = arms.get(baseline, {})
    v0 = float(base.get("violation_rate", float("nan")))
    rows = []
    for name, m in arms.items():
        v = float(m.get("violation_rate", float("nan")))
        rows.append({
            "arm": name,
            "violation_rate": v,
            "violation_reduction": (1.0 - v / v0) if np.isfinite(v0) and v0 > 0
            else float("nan"),
            "distortion_px": float(m.get("distortion_px", float("nan"))),
            "distortion_mean_px": float(m.get("distortion_mean_px", float("nan"))),
            "distortion_body_lengths": float(
                m.get("distortion_body_lengths", float("nan"))),
            "hf_retained": float(m.get("hf_retained", float("nan"))),
        })
    rows.sort(key=lambda r: (-r["violation_reduction"]
                             if np.isfinite(r["violation_reduction"]) else 0.0))
    return rows


def bakeoff_read(rows: Sequence[Mapping[str, Any]], *, scored_object: Detail,
                 n_effective: int, incumbent: str = "wiener",
                 max_hf_deleted: float = MAX_HF_DELETED) -> Read:
    """Did anything beat the incumbent on all three axes at once?

    The verdict is about whether a *decision* is available, not about which arm
    is prettiest. `NOT_A_RESULT` when the axes disagree is the honest outcome:
    the bakeoff ran, and it does not license a change.
    """
    by = {r["arm"]: r for r in rows}
    inc = by.get(incumbent)
    detail: Detail = {"rows": list(rows), "incumbent": incumbent,
                      "max_hf_deleted": max_hf_deleted}
    if inc is None:
        return Read("INCONCLUSIVE", scored_object,
                    f"the incumbent arm {incumbent!r} did not score",
                    n_effective=n_effective, detail=detail)

    better = [
        r for r in rows
        if r["arm"] not in (incumbent, "raw")
        and np.isfinite(r["violation_rate"]) and np.isfinite(r["distortion_px"])
        and r["violation_rate"] < inc["violation_rate"]
        # Compared on the MEAN, not the median: every targeted arm has a median
        # displacement of exactly zero, and `0 <= 0` would let all of them
        # "dominate" on an axis that never separated them.
        and r["distortion_mean_px"] <= inc["distortion_mean_px"]
        and (not np.isfinite(r["hf_retained"])
             or r["hf_retained"] >= 1.0 - max_hf_deleted)
    ]
    detail["dominating"] = [r["arm"] for r in better]

    if better:
        best = better[0]
        return Read(
            "PASS", scored_object,
            f"{best['arm']} beats the incumbent {incumbent} on all three axes at "
            f"once: violations {best['violation_rate']:.3%} against "
            f"{inc['violation_rate']:.3%}, displacement "
            f"{best['distortion_mean_px']:.3f} px mean against "
            f"{inc['distortion_mean_px']:.3f}, "
            f"and it keeps {best['hf_retained']:.1%} of the power above f_c "
            f"against the incumbent's {inc['hf_retained']:.1%}. "
            f"{len(better)} arm(s) dominate; a change of cleaning method is "
            f"licensed by this comparison",
            n_effective=n_effective, detail=detail)
    return Read(
        "NOT_A_RESULT", scored_object,
        f"no arm beats {incumbent} on all three axes at once. Every arm that "
        f"reduces violations below its {inc['violation_rate']:.3%} does so by "
        f"moving the data further than its {inc['distortion_mean_px']:.3f} px "
        f"mean, or by "
        f"deleting more than {max_hf_deleted:.0%} of the power above f_c. The "
        f"bakeoff ran and it does not license a change -- which is a result "
        f"about the axes disagreeing, not a failure to measure",
        n_effective=n_effective, detail=detail)
