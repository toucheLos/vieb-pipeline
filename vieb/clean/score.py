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
    # Violation reduction bought per pixel of mean displacement. Not an axis --
    # it is a ratio of two of them -- but it is the quantity that separates a
    # de-glitcher from a smoother, and reading the two columns side by side does
    # not make it obvious.
    for r in rows:
        dm = r["distortion_mean_px"]
        r["reduction_per_px"] = (r["violation_reduction"] / dm
                                 if np.isfinite(dm) and dm > 1e-9 else float("nan"))
    rows.sort(key=lambda r: (-r["violation_reduction"]
                             if np.isfinite(r["violation_reduction"]) else 0.0))
    return rows


#: Lower is better on this axis; higher is better on the rest.
_LOWER_IS_BETTER = {"violation_rate", "distortion_mean_px"}


def separates(a: Mapping[str, float], b: Mapping[str, float],
              *, lower_is_better: bool) -> int:
    """Does `a` beat `b` with non-overlapping animal-bootstrap intervals?

    Returns +1 if `a` is better and the intervals do not overlap, -1 if `a` is
    worse and they do not overlap, and 0 if they overlap at all.

    **Overlap means 0, not a tie broken on the point estimate.** Two arms whose
    intervals overlap are not distinguishable by this comparison, and calling one
    of them the winner is the single most repeated failure in this project's
    history -- a point estimate reported as though it had the precision its
    decimal places imply.
    """
    lo_a, hi_a = float(a["lo"]), float(a["hi"])
    lo_b, hi_b = float(b["lo"]), float(b["hi"])
    if not all(np.isfinite(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return 0
    if not (hi_a < lo_b or hi_b < lo_a):
        return 0
    a_below = hi_a < lo_b
    return (1 if a_below else -1) if lower_is_better else (-1 if a_below else 1)


def bakeoff_read(rows: Sequence[Mapping[str, Any]],
                 intervals: Mapping[str, Mapping[str, Mapping[str, float]]], *,
                 scored_object: Detail, n_effective: int,
                 incumbent: str = "wiener") -> Read:
    """Did anything beat the incumbent, with intervals that actually separate?

    ## The correction this function carries

    The first version compared **point estimates**: an arm "dominated" if its
    violation rate, its displacement and its retention were each better than the
    incumbent's to however many decimal places numpy printed. On this corpus that
    crowned `viterbi+median_0.50` over `median_0.50` on a **0.029 px** difference
    in mean displacement, and declared both to beat `wiener` on violations --
    while the animal-bootstrap intervals for violations ran 1.44-1.78% against
    1.49-1.87% and overlapped almost entirely.

    Every interval in this repo is an animal bootstrap for a reason, and a gate
    that then ignores them is worse than no gate: it launders a point estimate
    into a verdict. Dominance now requires **non-overlapping intervals**.

    An arm dominates when it is significantly better on at least one axis and
    significantly worse on none. Where nothing separates, the honest verdict is
    that the bakeoff ran and does not license a change.
    """
    by = {r["arm"]: r for r in rows}
    inc = by.get(incumbent)
    detail: Detail = {"rows": list(rows), "incumbent": incumbent,
                      "test": "non-overlapping animal-bootstrap intervals"}
    if inc is None or incumbent not in intervals:
        return Read("INCONCLUSIVE", scored_object,
                    f"the incumbent arm {incumbent!r} did not score",
                    n_effective=n_effective, detail=detail)

    axes = ("violation_rate", "distortion_mean_px", "hf_retained")
    verdicts: Detail = {}
    dominating = []
    for name, ci in intervals.items():
        if name in (incumbent, "raw"):
            continue
        got = {ax: separates(ci[ax], intervals[incumbent][ax],
                             lower_is_better=ax in _LOWER_IS_BETTER)
               for ax in axes if ax in ci}
        verdicts[name] = got
        if any(v > 0 for v in got.values()) and not any(v < 0 for v in got.values()):
            dominating.append(name)
    detail["per_axis"] = verdicts
    detail["dominating"] = dominating

    # Axes that separate for SOME arm, and axes that separate for NONE. The
    # union across arms is not a property of any one of them -- an earlier
    # version printed it as though it were and produced a reason that listed
    # displacement as both separating and not separating in the same sentence.
    ever = sorted({ax for n in dominating for ax, v in verdicts[n].items() if v > 0})
    never = [ax for ax in axes
             if not any(verdicts[n].get(ax, 0) > 0 for n in dominating)]
    detail["axes_that_separate_for_some_arm"] = ever
    detail["axes_that_separate_for_no_arm"] = never

    if dominating:
        red = {r["arm"]: r.get("violation_reduction", 0.0) for r in rows}
        best = max(dominating,
                   key=lambda n: (sum(v > 0 for v in verdicts[n].values()),
                                  red.get(n, 0.0) if np.isfinite(
                                      red.get(n, 0.0)) else 0.0))
        mine = sorted(ax for ax, v in verdicts[best].items() if v > 0)
        b, i = intervals[best], intervals[incumbent]
        tail = (f" No arm separates on {', '.join(never)}, so the ordering on "
                f"{'that axis' if len(never) == 1 else 'those axes'} is a "
                f"ranking of point estimates and not a finding." if never else "")
        return Read(
            "PASS", scored_object,
            f"{len(dominating)} arm(s) beat the incumbent {incumbent} with "
            f"non-overlapping animal-bootstrap intervals and none is "
            f"significantly worse on any axis. {best} separates on "
            f"{', '.join(mine)}: it retains {b['hf_retained']['point']:.1%} "
            f"[{b['hf_retained']['lo']:.1%}, {b['hf_retained']['hi']:.1%}] of the "
            f"power above f_c against the incumbent's "
            f"{i['hf_retained']['point']:.1%} "
            f"[{i['hf_retained']['lo']:.1%}, {i['hf_retained']['hi']:.1%}]."
            + tail,
            n_effective=n_effective, detail=detail)
    return Read(
        "NOT_A_RESULT", scored_object,
        f"no arm separates from {incumbent} on any axis once the animal-bootstrap "
        f"intervals are taken into account. The point estimates order the arms "
        f"and the intervals overlap, so the ordering is not a finding. The "
        f"bakeoff ran and does not license a change",
        n_effective=n_effective, detail=detail)
