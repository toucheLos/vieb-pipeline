r"""What the cleaning actually does to the data, layer by layer.

"How much does the bone check change the data" has three answers and **the bone
check is not one of them**. It is a flag, not an edit. Separate the layers or the
number means nothing:

======  ==========================================  ==========================
layer   what moves                                  driven by
======  ==========================================  ==========================
L1      gaps <= 0.1 s linearly interpolated;        `bone_flagged` and the
        longer runs left exactly as measured        confidence gate (inert here)
L2      every frame, by a per-keypoint frequency-   the calibration spectrum
        graded multiplicative shrinkage
L3      nothing -- the eps-violation diagnostic     this repo's Step 1
        only flags
======  ==========================================  ==========================

So the bone check's own cost is L1 restricted to the frames its flags caused to
be interpolated (1.34% of keypoint-frames) plus the mass its flags caused to be
abandoned (3.385%). Small mass; whether it is small *effect* is what this
measures.

## The baseline is `raw_pose.npz`, not `pose_unfiltered`

`pose_unfiltered` is the gap policy's **output**, not its input -- roughly 1.34%
of its keypoint-frames are already linear interpolants. Measuring L1 against it
returns exactly zero, which is the one answer that cannot be right. The genuinely
raw array is `~/shapeflow/work/raw_pose.npz`, reachable as `spine.raw_pose(rid)`.

## Why concentration is the question, not magnitude

A repair should move the frames it repairs and leave the rest alone. A transform
that moves 86% of keypoint-frames is not repairing 4.7% of bad ones; it is
reshaping the corpus, and whatever else that may be, calling it cleaning
overstates it. `concentration_read` is the gate: displacement on flagged frames
against displacement on clean frames. The magnitude travels beside it, because a
diffuse transform that moves everything by 0.1 px and a diffuse transform that
moves everything by 5 px are different objects.

Nothing here is a claim about whether the filter is *right*. Shrinkage that
touches every frame is exactly what a Wiener filter is supposed to do. The
finding is that the size of it has never been written down, while every number
downstream -- Q1 included -- is computed on the result.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read
from recur.util import describe

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: The layers, in the order the pipeline applies them.
LAYERS: tuple[str, ...] = ("gap_policy", "wiener", "butterworth")

#: `(before, after)` keys. `raw` comes from `spine.raw_pose`, the rest from
#: `spine.clean`. butterworth is a comparison arm and branches off the same
#: input as wiener, so the two are directly comparable.
LAYER_ARRAYS: dict[str, tuple[str, str]] = {
    "gap_policy": ("raw", "pose_unfiltered"),
    "wiener": ("pose_unfiltered", "pose"),
    "butterworth": ("pose_unfiltered", "pose_butterworth"),
}

#: Displacement thresholds reported as "fraction of keypoint-frames moved more
#: than". Sub-pixel first, because the question is whether a transform touches
#: everything a little or a few things a lot.
THRESHOLDS_PX: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 5.0, 20.0)

#: Below this ratio of flagged-frame to clean-frame displacement, the transform
#: is not targeting the frames it is nominally repairing.
MIN_CONCENTRATION = 2.0


def displacement(before: npt.ArrayLike, after: npt.ArrayLike) -> F64:
    """``(T, K)`` Euclidean distance per keypoint-frame, in the input's units."""
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    return np.asarray(np.linalg.norm(b - a, axis=-1), dtype=np.float64)


def summarize(d: npt.ArrayLike, *, mask: npt.ArrayLike | None = None,
              ell: float | None = None) -> Detail:
    """Distribution of a displacement field, optionally restricted to `mask`.

    `ell` is the animal's body length; when given, the same distribution is
    reported a second time in body lengths, which is the only unit in which two
    animals of different size are comparable.
    """
    v = np.asarray(d, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        v = v[m if m.shape == v.shape else np.broadcast_to(m[:, None], v.shape)]
    v = v[np.isfinite(v)]
    out: Detail = {"px": describe(v), "n_keypoint_frames": int(v.size)}
    out["frac_moved_over"] = {str(t): float((v > t).mean()) if v.size else float("nan")
                              for t in THRESHOLDS_PX}
    if ell and np.isfinite(ell) and ell > 0:
        out["body_lengths"] = describe(v / float(ell))
    return out


def by_keypoint(d: npt.ArrayLike, names: Sequence[str]) -> list[Detail]:
    """Per-landmark displacement.

    The Wiener gain at Nyquist runs 0.583 at `center` down to 0.000 at
    `right_hip` and `tail_base` -- the filter is doing very different things to
    different landmarks, and this is where that becomes visible rather than
    inferable from the calibration table.
    """
    v = np.asarray(d, dtype=np.float64)
    return [{"keypoint": str(n), "px": describe(v[:, k])}
            for k, n in enumerate(names)]


def by_flag(d: npt.ArrayLike, flags: Mapping[str, npt.ArrayLike]) -> Detail:
    """Displacement cross-tabulated by frame quality.

    Each flag gives two rows -- inside and outside -- so the ratio is readable
    directly. A per-frame flag is broadcast across keypoints; a per-keypoint
    flag is used as it is.
    """
    v = np.asarray(d, dtype=np.float64)
    out: Detail = {}
    for name, raw in flags.items():
        m = np.asarray(raw, dtype=bool)
        if m.ndim == 1:
            m = np.broadcast_to(m[:, None], v.shape)
        inside, outside = v[m], v[~m]
        out[name] = {
            "n_inside": int(inside.size), "n_outside": int(outside.size),
            "median_inside": float(np.median(inside)) if inside.size else float("nan"),
            "median_outside": float(np.median(outside)) if outside.size else float("nan"),
            "mean_inside": float(inside.mean()) if inside.size else float("nan"),
            "mean_outside": float(outside.mean()) if outside.size else float("nan"),
        }
        # Concentration on the median where the median is informative, and on
        # the mean where it is not. A layer that touches ~1% of keypoint-frames
        # -- which is exactly what the gap policy does -- has a median of zero
        # BOTH inside and outside the flag, and 0/0 is not "no concentration",
        # it is the wrong statistic. Which one was used travels in the record.
        mo, mi = out[name]["median_outside"], out[name]["median_inside"]
        if np.isfinite(mo) and mo > 1e-12:
            out[name]["concentration"] = mi / mo
            out[name]["concentration_on"] = "median"
        else:
            ao, ai = out[name]["mean_outside"], out[name]["mean_inside"]
            if np.isfinite(ao) and ao > 1e-12:
                out[name]["concentration"] = ai / ao
            elif np.isfinite(ai) and ai > 1e-12:
                # Nothing outside the flag moved at all. That is not a missing
                # value -- it is a layer that touches only what it flagged, which
                # is what the gap policy does by construction.
                out[name]["concentration"] = float("inf")
            else:
                out[name]["concentration"] = float("nan")
            out[name]["concentration_on"] = "mean"
        # See scripts/effect.py: an infinite concentration is a fact, and
        # `write_json` would erase it to null. Carry it as a boolean too.
        out[name]["touches_only_flagged"] = bool(
            np.isinf(out[name]["concentration"]))
    return out


def kinematic_shift(speed_before: npt.ArrayLike, speed_after: npt.ArrayLike,
                    turn_before: npt.ArrayLike, turn_after: npt.ArrayLike) -> Detail:
    r"""Does the transform change behaviour, or only position?

    A filter that preserves position while deleting the fast tail of the speed
    distribution has changed what a behaviour model can see, and no displacement
    statistic would show it. Reported as quantile ratios, because the tail is the
    part at risk: `coherent_power_above_f_c` is 1.059%, so the prediction is
    "very little" -- worth confirming rather than assuming.
    """
    out: Detail = {}
    for name, a, b in (("speed", speed_before, speed_after),
                       ("turn", turn_before, turn_after)):
        x = np.asarray(a, dtype=np.float64)
        y = np.asarray(b, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = np.abs(x[ok]), np.abs(y[ok])
        if x.size < 100:
            out[name] = {"n": int(x.size), "why": "too few finite frames"}
            continue
        qs = (0.5, 0.9, 0.99, 0.999)
        qx = np.quantile(x, qs)
        qy = np.quantile(y, qs)
        out[name] = {
            "n": int(x.size),
            "quantiles": [float(q) for q in qs],
            "before": qx.tolist(), "after": qy.tolist(),
            "ratio": (qy / np.where(qx > 1e-12, qx, np.nan)).tolist(),
        }
    return out


def concentration_read(layer: str, flagged: Detail, *, scored_object: Detail,
                       n_effective: int,
                       min_concentration: float = MIN_CONCENTRATION) -> Read:
    """Is this layer repairing the bad frames, or transforming the corpus?

    `flagged` is one entry from `by_flag`. A repair moves what it repairs; a
    ratio near 1 means the transform is indifferent to whether a frame was ever
    suspect, which makes "cleaning" the wrong word for it.
    """
    ratio = float(flagged.get("concentration", float("nan")))
    on = str(flagged.get("concentration_on", "median"))
    inside = float(flagged.get(f"{on}_inside", float("nan")))
    outside = float(flagged.get(f"{on}_outside", float("nan")))
    detail: Detail = {"layer": layer, **flagged,
                      "min_concentration": min_concentration}
    if np.isinf(ratio):
        return Read(
            "PASS", scored_object,
            f"{layer} moves nothing outside the flags at all ({on} "
            f"{inside:.3f} px inside against exactly 0 outside), so it touches "
            f"only the frames it marked. Perfectly targeted -- by construction "
            f"rather than by tuning, which is why it is worth stating: the "
            f"layer's entire effect is confined to the {flagged.get('n_inside', 0):,} "
            f"keypoint-frames its flags selected",
            n_effective=n_effective, detail=detail)
    if not np.isfinite(ratio):
        return Read("INCONCLUSIVE", scored_object,
                    f"{layer}: nothing moved either inside or outside the flags",
                    n_effective=n_effective, detail=detail)
    if ratio >= min_concentration:
        return Read(
            "PASS", scored_object,
            f"{layer} moves flagged frames {ratio:.1f}x further than clean ones "
            f"({on} {inside:.3f} px against {outside:.3f} px), so it is "
            f"targeting the frames it is nominally repairing rather than "
            f"transforming the corpus",
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        f"{layer} moves flagged frames only {ratio:.2f}x as far as clean ones "
        f"({on} {inside:.3f} px against {outside:.3f} px), below the "
        f"{min_concentration}x a targeted repair would leave. It is indifferent "
        f"to whether a frame was ever suspect, so it is a transform applied to "
        f"the whole corpus and 'cleaning' overstates what it does. That is not "
        f"a defect in a shrinkage filter -- it is what shrinkage is -- but every "
        f"number downstream is computed on the result and the size of it has "
        f"not been recorded anywhere until now",
        n_effective=n_effective, detail=detail)
