r"""Bone-length violations: excess length is error, short length is not.

**Blocking.** A tracking artifact is a short, highly stereotyped excursion with
predictable duration, which is exactly what a cause-specific hazard model
rewards. You can score excellently off tracking failure, and nothing downstream
of here is interpretable until the rate is known.

## The asymmetry, and the condition it needs

A 2-D projection of a 3-D animal can only ever *shorten* the image of a rigid
segment: :math:`\ell_{\text{img}} = \ell_{\text{true}}\cos\psi \le
\ell_{\text{true}}`. So excess length is unambiguous error and short length is
uninformative -- which is why this test is one-sided where shapeflow's is not.

That argument holds **at a fixed scale factor**, and this is an uncalibrated
top-down camera. A mouse rearing, or simply nearer the lens, multiplies every
distance by a common factor; shapeflow measured that factor's SD at 0.155 in log
units against 0.178 for the relative geometry itself, so it is not a small
correction. At :math:`\varepsilon = 0.10` or 0.20 a raw-pixel sweep fires on
rearing.

So the sweep runs on **two metrics**, and the pair is the deliverable:

``raw``
    Pixel lengths, exactly as specified. Primary.
``scalefree``
    The per-frame common scale removed first, by subtracting the median log
    length over the animal's own rigid pairs. A uniform scale change cancels
    exactly; a single displaced keypoint does not.

Where the elbow sits in each, and how far it moves between them, is the
measurement. A sharp elbow means two populations and says where to cut; a smooth
decay means no threshold here is principled.

## Skull first, trunk separately, never pooled

The skull triangle -- ``nose-left_ear``, ``nose-right_ear``, ``left_ear-right_ear``
-- is as close to rigid as this animal gets, and it is the primary read. Trunk
bones flex: a violation there is real posture at least as often as it is
tracking, and pooling the two produces a headline stronger than the evidence.

## What this is NOT

shapeflow's Stage 1 already flags 4.725% of frames as ``bone_flagged``. That is a
different object on every axis -- **symmetric** in the deviation, **scale-free by
construction**, **learned** (the below-median-MAD half of all 21 pairs) rather
than anatomical, and requiring **three** pairs to break at once. It is a
rejection mask, and it offers no curve, no attribution and no asymmetry.

It is not a competitor here, it is a second opinion. `overlap` reports the full
2x2 against it per (eps, metric), because two masks built on incompatible
principles disagreeing is more informative than either rate alone -- see that
function's docstring for what each cell claims.

## Nothing here crosses a seam

Every function takes ONE recording's pose, or one animal's already-concatenated
lengths with the reference fitted over that animal. `shuffled_ceiling` draws its
random frames from within the recording it was handed, for the same reason.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from recur.read import Read

from vieb.checks import assert_pmf, assert_share

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]

Bone = tuple[int, int]
#: What every diagnostic in this module returns beside its numbers. Deliberately
#: loose in its values -- a cell holds counts, rates, nested 2x2s and the eps
#: curve -- and deliberately `str` in its keys, because these dicts are written
#: straight to JSON and a non-string key would not survive the round trip.
Detail = dict[str, Any]

#: The skull triangle, in LUNA keypoint order (see `recur.qc.swap`):
#: 0 left_ear, 1 right_ear, 2 nose, 3 center, 4 left_hip, 5 right_hip,
#: 6 tail_base. PRIMARY -- this is the group the branch is read on.
SKULL: tuple[Bone, ...] = ((0, 1), (0, 2), (1, 2))

#: Trunk bones. Reported SEPARATELY and never pooled into the headline: these
#: flex, so a violation is real posture at least as often as it is tracking.
TRUNK: tuple[Bone, ...] = ((2, 3), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6))

#: The sweep. The curve is the deliverable, not any single cell of it.
EPS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.50)

#: The two metrics, run identically. See the module docstring.
METRICS: tuple[str, ...] = ("raw", "scalefree")

#: Percentile the reference iteration lands on. 99 rather than 100 because a
#: single mistracked frame must not be able to define the animal's anatomy.
REF_PCT = 99.0

#: Iterations of the reference fixed point. Three, as specified; the sequence is
#: monotone non-increasing and `reference_length` reports whether it converged.
REF_ITERS = 3

#: Branch thresholds on the SKULL rate. See `curve_read`.
EXCLUDE_BELOW = 0.02
CORRECT_BELOW = 0.15

#: An elbow has to be a drop that stands out, not merely a large one. A run of
#: increasing drops is a single population thinning out; two populations leave
#: one eps where the rate falls away much faster than anywhere else.
MIN_ELBOW_DROP = 0.5
MIN_ELBOW_PROMINENCE = 1.5

#: Below this |rho| the violation rate carries no information about segment
#: reconstruction quality, and the premise that derivatives are bounded at
#: 30 fps is what is in question rather than the tracking.
MIN_ABS_RHO = 0.1


# --------------------------------------------------------------------------
# Lengths, and the two metrics
# --------------------------------------------------------------------------

def bone_lengths(pose: npt.ArrayLike, bones: Sequence[Bone]) -> F64:
    """``(T, n_bones)`` Euclidean distances for ONE recording."""
    p = np.asarray(pose, dtype=np.float64)
    out = np.empty((p.shape[0], len(bones)), dtype=np.float64)
    for m, (i, j) in enumerate(bones):
        out[:, m] = np.linalg.norm(p[:, i] - p[:, j], axis=-1)
    return out


def pair_indices(n_keypoints: int) -> list[Bone]:
    """Every unordered keypoint pair.

    Ported from shapeflow's `clean.pair_indices` rather than imported: recur
    reads shapeflow's artifacts and never its code. All 21 pairs, because
    `rigid_pairs` selects from the data which of them constrain the body -- a
    named skeleton would be a constant that does not transfer to a 5-keypoint
    rat.
    """
    return [(i, j) for i in range(n_keypoints) for j in range(i + 1, n_keypoints)]


def log_lengths(pose: npt.ArrayLike, pairs: Sequence[Bone]) -> F64:
    """``(T, n_pairs)`` log distances. Zero distances become nan, not -inf."""
    d = bone_lengths(pose, pairs)
    with np.errstate(divide="ignore"):
        return np.where(d > 0.0, np.log(d), np.nan)


def _mad(x: npt.ArrayLike) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def rigid_pairs(logs: npt.ArrayLike, pairs: Sequence[Bone]) -> list[int]:
    """Indices of the pairs whose log length is most constant.

    The below-median-MAD half. A mouse is not rigid, so ear-to-tail varies with
    posture and says nothing about the camera's scale; selecting on constancy
    derives the set from the corpus rather than from an anatomical constant.
    Fitted per animal, over that animal's own pooled frames.
    """
    lg = np.asarray(logs, dtype=np.float64)
    mad = np.array([_mad(lg[:, m]) for m in range(lg.shape[1])])
    finite = mad[np.isfinite(mad)]
    if finite.size == 0:
        return []
    cut = float(np.median(finite))
    return [m for m in range(len(pairs)) if np.isfinite(mad[m]) and mad[m] <= cut]


def common_log_scale(pose: npt.ArrayLike, pairs: Sequence[Bone],
                     keep: Sequence[int]) -> F64:
    r"""``(T,)`` per-frame common log scale: the median over the rigid pairs.

    Rearing and camera distance are a uniform multiplication of every distance,
    which in log space is a common *offset*. The median over rigid pairs
    estimates it and is unmoved by one displaced keypoint, which perturbs at most
    :math:`K-1` of the 21 pairs.
    """
    lg = np.asarray(log_lengths(pose, pairs), dtype=np.float64)
    if not len(keep):
        return np.zeros(lg.shape[0], dtype=np.float64)
    return np.nanmedian(lg[:, list(keep)], axis=1)


def metric_lengths(pose: npt.ArrayLike, bones: Sequence[Bone], metric: str, *,
                   pairs: Sequence[Bone] | None = None,
                   keep: Sequence[int] | None = None) -> F64:
    """``(T, n_bones)`` lengths in one metric.

    ``raw`` is pixels. ``scalefree`` divides out the per-frame common scale, so
    the result is a dimensionless relative length and a uniform scale change
    cancels exactly. Both then go through the identical reference fit and
    threshold, which is what makes the two curves comparable.
    """
    if metric not in METRICS:
        raise ValueError(f"{metric!r} is not one of {METRICS}")
    lengths = bone_lengths(pose, bones)
    if metric == "raw":
        return lengths
    p = np.asarray(pose, dtype=np.float64)
    pairs = pair_indices(p.shape[1]) if pairs is None else pairs
    if keep is None:
        keep = rigid_pairs(log_lengths(p, pairs), pairs)
    scale = np.exp(common_log_scale(p, pairs, keep))
    with np.errstate(divide="ignore", invalid="ignore"):
        return lengths / np.where(scale > 1e-12, scale, np.nan)[:, None]


# --------------------------------------------------------------------------
# The reference length, and the violation
# --------------------------------------------------------------------------

def reference_length(lengths: npt.ArrayLike, eps: float, *,
                     n_iter: int = REF_ITERS, pct: float = REF_PCT) -> Detail:
    r"""Estimate the true length of ONE bone for ONE animal.

    .. math:: \hat\ell \leftarrow P_{99}\{\ell_t : \ell_t \le \hat\ell(1+\varepsilon)\}

    seeded at the plain 99th percentile and iterated `n_iter` times. Each pass
    can only remove points from the set, so the sequence is monotone
    non-increasing and cannot oscillate; it is not run to convergence because the
    brief fixes three passes, and whether it *had* converged is reported rather
    than assumed.

    Fitted **per animal per bone**, on that animal's own frames. A corpus-wide
    reference would flag small mice everywhere -- a systematic exclusion
    correlated with the animal, which is the shape of a confound rather than of
    a quality gate.
    """
    v = np.asarray(lengths, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 10:
        return {"l_hat": float("nan"), "n_frames": int(v.size),
                "converged": False, "trace": [],
                "why": "fewer than 10 finite lengths"}

    l_hat = float(np.percentile(v, pct))
    trace = [l_hat]
    for _ in range(int(n_iter)):
        sub = v[v <= l_hat * (1.0 + float(eps))]
        if sub.size == 0:
            break
        l_hat = float(np.percentile(sub, pct))
        trace.append(l_hat)
    converged = bool(len(trace) >= 2
                     and abs(trace[-1] - trace[-2]) <= 1e-9 * max(abs(trace[-1]), 1.0))
    return {"l_hat": l_hat, "n_frames": int(v.size), "converged": converged,
            "trace": [float(t) for t in trace],
            "relative_drift": float((trace[0] - trace[-1]) / trace[0])
            if trace[0] > 0 else float("nan")}


def violations(lengths: npt.ArrayLike, l_hat: npt.ArrayLike,
               eps: float) -> BOOL:
    r"""``(T, n_bones)`` bool: :math:`\ell_t/\hat\ell > 1+\varepsilon`.

    `l_hat` is **one reference per bone** and a bare scalar is refused when there
    is more than one. Bones have different true lengths -- the skull triangle's
    ear-ear and ear-nose differ by ~15% on the synthetic fixture alone -- so a
    scalar broadcast compares every bone against one of them and flags whichever
    are longer on *every* frame. It produced a 100%-violation column the first
    time this was written, and it produced it silently, which is the shape of
    error this project keeps paying for.

    A non-finite length is **not** a violation. It is a frame where the length
    could not be measured, and calling that an excess would put missingness into
    a statistic about geometry.
    """
    v = np.asarray(lengths, dtype=np.float64)
    ref = np.asarray(l_hat, dtype=np.float64)
    if ref.ndim == 0:
        if v.shape[1] != 1:
            raise ValueError(
                f"l_hat is a scalar but there are {v.shape[1]} bones. Every "
                f"bone has its own true length, so one reference across all of "
                f"them flags the longer bones on every frame rather than "
                f"measuring anything. Pass one reference per bone")
        ref = ref.reshape(1)
    if ref.shape != (v.shape[1],):
        raise ValueError(
            f"l_hat has shape {ref.shape}, expected ({v.shape[1]},) -- one "
            f"reference per bone, in the same order as `bones`")
    with np.errstate(invalid="ignore"):
        ratio = v / np.where(np.isfinite(ref) & (ref > 0), ref, np.nan)
    return np.asarray(np.nan_to_num(ratio, nan=0.0) > 1.0 + float(eps), dtype=np.bool_)


def frame_mask(viol: npt.ArrayLike) -> BOOL:
    """``(T,)``: any bone in the group violated on this frame.

    ONE bone, not three. shapeflow's gate needs three pairs to break at once
    because it is looking for the signature of a swap across all 21 pairs; this
    one asks whether a specific rigid segment is impossibly long, and one is
    enough for that. The two thresholds are not comparable and the difference is
    deliberate.
    """
    return np.asarray(np.asarray(viol, dtype=bool).any(axis=1), dtype=np.bool_)


# --------------------------------------------------------------------------
# Attribution, and the ceiling
# --------------------------------------------------------------------------

def attribution(viol: npt.ArrayLike, bones: Sequence[Bone],
                n_keypoints: int) -> Detail:
    """Which landmark participates in the violating bones.

    Per keypoint: the fraction of frames on which a bone touching it is
    violated, and that keypoint's share of all violating bone-frames. A single
    mistracked landmark shows up as one keypoint carrying most of the share; a
    real posture excursion spreads it.
    """
    v = np.asarray(viol, dtype=bool)
    per_frame = np.zeros((v.shape[0], n_keypoints), dtype=bool)
    counts = np.zeros(n_keypoints, dtype=np.int64)
    for m, (i, j) in enumerate(bones):
        per_frame[:, i] |= v[:, m]
        per_frame[:, j] |= v[:, m]
        counts[i] += int(v[:, m].sum())
        counts[j] += int(v[:, m].sum())
    total = int(counts.sum())
    # With no violations there is nothing to apportion. The zeros are recorded
    # as such rather than replaced by a uniform 1/n, because a fabricated
    # uniform would read as "blame is spread evenly" when the truth is "there
    # is no blame". `share_defined` says which of the two a reader is looking
    # at, and the assertion only applies to the defined case.
    share = (assert_pmf(counts / total, name="per-keypoint blame share").tolist()
             if total else [0.0] * n_keypoints)
    return {
        "frame_rate": assert_share(per_frame.mean(axis=0),
                                   name="per-keypoint frame rate").tolist(),
        "share": share,
        "share_defined": bool(total),
        "n_violating_bone_frames": int(v.sum()),
    }


def shuffled_ceiling(pose: npt.ArrayLike, bones: Sequence[Bone],
                     l_hat: npt.ArrayLike, eps: float,
                     rng: np.random.Generator, *, metric: str = "raw",
                     pairs: Sequence[Bone] | None = None,
                     keep: Sequence[int] | None = None) -> float:
    """Violation rate when each keypoint comes from a random frame.

    **Within one recording**, so the arena, the animal and the camera are held
    fixed and only the temporal association between landmarks is destroyed. This
    is the rate a detector would report on a body that is not a body -- the
    number every observed rate has to be read against, because a diagnostic that
    fires as often on shuffled keypoints as on real ones is measuring its own
    threshold.
    """
    p = np.asarray(pose, dtype=np.float64)
    t, k = p.shape[0], p.shape[1]
    if t < 2:
        return float("nan")
    idx = rng.integers(0, t, size=(t, k))
    shuffled = p[idx, np.arange(k)[None, :], :]
    lengths = metric_lengths(shuffled, bones, metric, pairs=pairs, keep=keep)
    return float(frame_mask(violations(lengths, l_hat, eps)).mean())


# --------------------------------------------------------------------------
# The second opinion
# --------------------------------------------------------------------------

def overlap(new_mask: npt.ArrayLike, other_mask: npt.ArrayLike, *,
            other_name: str = "bone_flagged") -> Detail:
    r"""The full 2x2 between this diagnostic and another per-frame mask.

    Reported instead of two marginal rates, because the masks are built on
    incompatible principles and the *disagreement* is what carries information.
    Each cell is a different claim:

    ``both``
        Agreed tracking failure. Two unrelated criteria firing on one frame is
        the strongest evidence available here that the frame is broken.
    ``new_only`` *(raw metric)*
        Excess length invisible to a scale-free gate. A uniform scale-up cancels
        exactly under a common-scale subtraction, so this is where rearing and
        camera distance live.
    ``new_only`` *(scalefree metric)*
        Excess length that **survives** common-scale removal. The cell closest to
        genuine tracking error, and the one the branch is read on if the two
        metrics disagree.
    ``other_only``
        Relative geometry broken with no excess length -- bilateral swaps and
        shortenings, which a one-sided test is blind to by construction. Expected
        to be large. It is not a failure of either mask.

    The reading: if ``new_only`` collapses from ``raw`` to ``scalefree`` while
    ``both`` holds roughly constant, the raw elbow is a rearing threshold rather
    than a tracking threshold. If it survives, it is error the existing gate
    misses and 4.725% is an undercount. Either is a finding; neither is visible
    from the two rates side by side.
    """
    a = np.asarray(new_mask, dtype=bool).ravel()
    b = np.asarray(other_mask, dtype=bool).ravel()
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    n = int(a.size)
    both = int((a & b).sum())
    new_only = int((a & ~b).sum())
    other_only = int((~a & b).sum())
    neither = n - both - new_only - other_only
    union = both + new_only + other_only
    return {
        "other_name": other_name,
        "n_frames": n,
        "both": both, "new_only": new_only,
        "other_only": other_only, "neither": neither,
        "rate_both": both / n if n else float("nan"),
        "rate_new_only": new_only / n if n else float("nan"),
        "rate_other_only": other_only / n if n else float("nan"),
        "rate_new": int(a.sum()) / n if n else float("nan"),
        "rate_other": int(b.sum()) / n if n else float("nan"),
        "jaccard": both / union if union else float("nan"),
        "p_other_given_new": both / int(a.sum()) if a.any() else float("nan"),
        "p_new_given_other": both / int(b.sum()) if b.any() else float("nan"),
    }


# --------------------------------------------------------------------------
# The join that decides the branch
# --------------------------------------------------------------------------

def segment_rates(mask: npt.ArrayLike, bounds: npt.ArrayLike) -> F64:
    """Per-segment violation rate, for ONE recording.

    `bounds` is ExBias's ``(N, 2)`` half-open ``[start, stop)`` in **that
    recording's own frame index**. Only the recording id and the frame range
    cross the repo boundary -- never a keypoint index, because ExBias sorts
    bodyparts alphabetically (`center, left_ear, left_hip, nose, ...`) where
    shapeflow uses file order, and the two orderings are not the same map.
    """
    m = np.asarray(mask, dtype=bool)
    b = np.asarray(bounds, dtype=np.int64)
    out = np.full(b.shape[0], np.nan, dtype=np.float64)
    for i in range(b.shape[0]):
        lo, hi = int(b[i, 0]), int(b[i, 1])
        lo, hi = max(lo, 0), min(hi, m.shape[0])
        if hi > lo:
            out[i] = float(m[lo:hi].mean())
    return out


def r2_deciles(rate: npt.ArrayLike, r2: npt.ArrayLike, *,
               durations: npt.ArrayLike | None = None,
               n_bins: int = 10) -> Detail:
    """Violation rate by decile of segment reconstruction R^2.

    **The curve, not the coefficient.** A single rho hides a threshold effect,
    and a threshold effect is exactly what the branch has to see: tracking
    failure concentrated in the worst-fitting tenth of segments reads very
    differently from a rate that slopes gently across all ten.

    Two statistics per bin, because on this corpus they disagree and the
    disagreement is the finding:

    ``mean_violation_rate``
        The mean over segments of (violating frames / segment length). A ratio
        whose denominator is the segment's own duration, so it is **coarse on
        short segments** -- one violating frame in a 6-frame segment is 17%.
    ``frac_segments_with_any``
        The fraction of segments carrying at least one violating frame.
        Duration-free in its numerator, though longer segments have more chances
        to contain one, which is why `mean_duration_s` travels in the same row.

    A rate that peaks mid-R^2 while the fraction declines monotonically is the
    signature of duration varying across the bins, not of a threshold. The
    duration column is what lets a reader tell those apart rather than having to
    trust either statistic alone.
    """
    v = np.asarray(rate, dtype=np.float64)
    q = np.asarray(r2, dtype=np.float64)
    dur = (np.full(v.shape, np.nan) if durations is None
           else np.asarray(durations, dtype=np.float64))
    ok = np.isfinite(v) & np.isfinite(q)
    v, q, dur = v[ok], q[ok], dur[ok]
    if v.size < n_bins * 10:
        return {"n_segments": int(v.size), "bins": [],
                "why": f"fewer than {n_bins * 10} joinable segments"}
    edges = np.quantile(q, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, n_bins - 1)
    bins = []
    for k in range(n_bins):
        sel = idx == k
        # Guarded rather than wrapped in errstate: an all-nan slice makes
        # `nanmean` raise a RuntimeWarning, which errstate does not catch, and
        # `durations=None` makes every slice all-nan.
        finite_dur = dur[sel][np.isfinite(dur[sel])] if sel.any() else dur[:0]
        mean_dur = float(finite_dur.mean()) if finite_dur.size else float("nan")
        bins.append({
            "decile": k + 1,
            "r2_lo": float(edges[k]), "r2_hi": float(edges[k + 1]),
            "n_segments": int(sel.sum()),
            "mean_violation_rate": float(v[sel].mean()) if sel.any() else float("nan"),
            "frac_segments_with_any": float((v[sel] > 0).mean()) if sel.any() else float("nan"),
            "mean_duration_s": mean_dur,
        })
    shape = [b["mean_violation_rate"] for b in bins]
    frac = [b["frac_segments_with_any"] for b in bins]
    return {"n_segments": int(v.size), "n_bins": n_bins, "bins": bins,
            "rate_is_monotone": _monotone(shape),
            "frac_is_monotone": _monotone(frac)}


def _monotone(values: Sequence[float]) -> bool:
    v = [x for x in values if np.isfinite(x)]
    if len(v) < 3:
        return False
    return (all(v[i] >= v[i + 1] for i in range(len(v) - 1))
            or all(v[i] <= v[i + 1] for i in range(len(v) - 1)))


def per_animal_spearman(rate: npt.ArrayLike, r2: npt.ArrayLike,
                        animals: Sequence[str], *,
                        min_segments: int = 30) -> tuple[F64, list[str]]:
    """One Spearman rho per animal, and the animals it was defined for.

    Per animal rather than pooled: if animal 1 has both a high violation rate and
    low R^2 while animal 2 has neither, a pooled correlation is between-animal
    heterogeneity wearing a within-animal coefficient's clothes. That is the same
    failure that cost ~90% of the dwell read.
    """
    from scipy import stats

    v = np.asarray(rate, dtype=np.float64)
    q = np.asarray(r2, dtype=np.float64)
    a = np.asarray([str(x) for x in animals])
    rhos: list[float] = []
    kept: list[str] = []
    for name in sorted(set(a.tolist())):
        sel = (a == name) & np.isfinite(v) & np.isfinite(q)
        if int(sel.sum()) < min_segments:
            continue
        vv, qq = v[sel], q[sel]
        if np.unique(vv).size < 2 or np.unique(qq).size < 2:
            continue
        rho = float(stats.spearmanr(vv, qq).statistic)
        if np.isfinite(rho):
            rhos.append(rho)
            kept.append(name)
    return np.asarray(rhos, dtype=np.float64), kept


def per_animal_partial_spearman(rate: npt.ArrayLike, r2: npt.ArrayLike,
                                control: npt.ArrayLike,
                                animals: Sequence[str], *,
                                min_segments: int = 30) -> tuple[F64, list[str]]:
    r"""Spearman of (rate, R^2) with `control` held fixed, per animal.

    .. math::
        \rho_{xy\cdot z} = \frac{\rho_{xy} - \rho_{xz}\rho_{yz}}
                             {\sqrt{(1-\rho_{xz}^2)(1-\rho_{yz}^2)}}

    **The control is segment duration, and it is not optional here.** ExBias
    fits a cubic over each segment, so a short segment fits well almost by
    construction: measured on this corpus, mean duration falls from 4.00 s in the
    worst R^2 decile to 0.53 s in the best, and rho(duration, R^2) = -0.54.
    Duration also drives the violation *rate*, whose denominator is the segment's
    own length. So the raw rho between violation rate and R^2 is substantially a
    restatement of "long segments fit cubics badly", and reading it as evidence
    that the R^2 floor is tracking failure would be attributing to the tracker
    what the segmenter's own geometry supplies.

    Returns nan for an animal where a control correlation reaches +-1, rather
    than dividing by zero and reporting the result.
    """
    from scipy import stats

    x = np.asarray(rate, dtype=np.float64)
    y = np.asarray(r2, dtype=np.float64)
    z = np.asarray(control, dtype=np.float64)
    a = np.asarray([str(v) for v in animals])
    out: list[float] = []
    kept: list[str] = []
    for name in sorted(set(a.tolist())):
        sel = (a == name) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if int(sel.sum()) < min_segments:
            continue
        xx, yy, zz = x[sel], y[sel], z[sel]
        if min(np.unique(xx).size, np.unique(yy).size, np.unique(zz).size) < 2:
            continue
        rxy = float(stats.spearmanr(xx, yy).statistic)
        rxz = float(stats.spearmanr(xx, zz).statistic)
        ryz = float(stats.spearmanr(yy, zz).statistic)
        denom = np.sqrt(max((1.0 - rxz ** 2) * (1.0 - ryz ** 2), 0.0))
        if not np.isfinite(denom) or denom < 1e-9:
            continue
        val = (rxy - rxz * ryz) / denom
        if np.isfinite(val):
            out.append(float(val))
            kept.append(name)
    return np.asarray(out, dtype=np.float64), kept


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def ceiling_read(observed: float, ceiling: float, *, scored_object: Detail,
                 n_effective: int) -> Read:
    """Is the observed rate distinguishable from shuffled keypoints?

    A diagnostic that fires as often on keypoints drawn from random frames as on
    real ones is reporting its own threshold. This refuses rather than reporting
    the rate.
    """
    detail = {"rate_observed": float(observed), "rate_shuffled": float(ceiling)}
    if not (np.isfinite(observed) and np.isfinite(ceiling)):
        return Read("INCONCLUSIVE", scored_object,
                    "the shuffled ceiling could not be computed",
                    n_effective=n_effective, detail=detail)
    if ceiling <= observed:
        return Read(
            "NOT_A_RESULT", scored_object,
            f"the shuffled-keypoint ceiling is {ceiling:.4%}, at or below the "
            f"observed {observed:.4%}. Keypoints drawn from random frames within "
            f"the same recording violate this bound as often as the real body "
            f"does, so the threshold is what is being measured and not the "
            f"tracking", n_effective=n_effective, degenerate=True, detail=detail)
    return Read(
        "PASS", scored_object,
        f"the observed rate is {observed:.4%} against a shuffled-keypoint "
        f"ceiling of {ceiling:.4%}, a margin of {ceiling / max(observed, 1e-12):.1f}x. "
        f"Destroying the temporal association between landmarks -- within the "
        f"same recording, so arena, animal and camera are held fixed -- makes "
        f"the violation far more common, which is what licenses reading the "
        f"observed rate as geometry", n_effective=n_effective, detail=detail)


def _ctrl(partial_ci: Mapping[str, float] | None) -> str:
    return " (segment duration held fixed)" if partial_ci is not None else ""


def join_read(rho_ci: Mapping[str, float], deciles: Mapping[str, Any], *,
              scored_object: Detail, n_effective: int,
              min_abs_rho: float = MIN_ABS_RHO,
              partial_ci: Mapping[str, float] | None = None) -> Read:
    """Does the violation rate track segment reconstruction quality?

    The join that decides the branch. Correlated means the R^2 floor is partly
    tracking failure and the diagnostic is measuring something real. Flat means
    derivatives genuinely are not bounded at 30 fps and the premise behind this
    whole gate is wrong -- which is a finding, not a pass.

    **The verdict is read off the DURATION-CONTROLLED rho when one is given.**
    ExBias's R^2 is substantially a readout of segment length, so the raw
    coefficient answers a question about the segmenter rather than about the
    tracker. The marginal rho is still reported beside it, because the gap
    between the two is what says how much of the raw association was duration.
    """
    if partial_ci is not None:
        rho_ci = partial_ci
    point = float(rho_ci.get("point", float("nan")))
    lo, hi = float(rho_ci.get("lo", float("nan"))), float(rho_ci.get("hi", float("nan")))
    detail: Detail = {"rho": dict(rho_ci), "deciles": deciles,
                      "min_abs_rho": min_abs_rho,
                      "controlled_for": "segment duration" if partial_ci is not None else None}
    if not np.isfinite(point):
        return Read("INCONCLUSIVE", scored_object,
                    "no animal had enough joinable segments to correlate",
                    n_effective=n_effective, detail=detail)
    excludes_zero = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    if excludes_zero and abs(point) >= min_abs_rho:
        return Read(
            "PASS", scored_object,
            f"per-animal Spearman rho between a segment's bone-violation rate "
            f"and its ExBias reconstruction R^2 is {point:+.3f} [{lo:+.3f}, "
            f"{hi:+.3f}]{_ctrl(partial_ci)}, animal bootstrap, interval excluding "
            f"zero and |rho| above {min_abs_rho}. The R^2 floor is partly "
            f"tracking, so the violation rate is measuring something the "
            f"segmenter also sees",
            n_effective=n_effective, detail=detail)
    if excludes_zero:
        return Read(
            "GRID_LIMITED", scored_object,
            f"rho = {point:+.3f} [{lo:+.3f}, {hi:+.3f}]{_ctrl(partial_ci)} "
            f"excludes zero but sits below |rho| = {min_abs_rho}. The "
            f"association is real and too small to attribute the R^2 floor to "
            f"tracking; this bounds the effect rather than estimating it", n_effective=n_effective,
            saturation="below the minimum interpretable |rho|", detail=detail)
    return Read(
        "FAIL", scored_object,
        f"rho = {point:+.3f} [{lo:+.3f}, {hi:+.3f}]{_ctrl(partial_ci)}, an "
        f"interval including zero. Segments that reconstruct badly are no more "
        f"likely to carry a bone violation than segments that reconstruct well, "
        f"so the R^2 floor is not tracking failure -- which puts the premise in "
        f"question rather than the corpus: derivatives may simply not be bounded "
        f"at 30 fps",
        n_effective=n_effective, detail=detail)


def curve_read(rows: Iterable[Mapping[str, Any]], *, group: str,
               metric: str, scored_object: Detail, n_effective: int,
               join: Read | None = None,
               exclude_below: float = EXCLUDE_BELOW,
               correct_below: float = CORRECT_BELOW) -> Read:
    """The eps curve for one (group, metric), and the branch it implies.

    `rows` are ``{eps, rate, n_recordings_above_1pct, shuffled_ceiling}`` across
    the sweep. The verdict names the branch; the curve travels in `detail`,
    because a single rate cannot say whether the elbow was sharp.
    """
    data = [dict(r) for r in rows]
    data.sort(key=lambda r: float(r["eps"]))
    eps = [float(r["eps"]) for r in data]
    rate = [float(r["rate"]) for r in data]
    detail: Detail = {"group": group, "metric": metric, "eps": eps, "rate": rate,
                    "rows": data, "exclude_below": exclude_below,
                    "correct_below": correct_below}
    if not data:
        return Read("INCONCLUSIVE", scored_object, "the sweep produced no cells",
                    n_effective=n_effective, detail=detail)

    # The elbow, and what does NOT count as one.
    #
    # An elbow is a drop that STANDS OUT from its neighbours -- one eps at which
    # the rate falls away much faster than at any other, which is the signature
    # of two populations with a gap between them. A large drop on its own is not
    # that: on this corpus the relative drops run 33%, 37%, 44%, 54% across the
    # sweep, monotonically increasing, which is a smooth accelerating decay of a
    # single population and means no threshold here is principled. The first
    # version of this rule fired on "largest drop >= 50%" alone and wrote "two
    # populations are separable and this is where to cut" into the generated
    # document, which was false.
    #
    # So the test is relative to the other drops, and a monotone run of drops is
    # refused outright however large the last one is.
    drops = [(rate[i] - rate[i + 1]) / rate[i] if rate[i] > 0 else 0.0
             for i in range(len(rate) - 1)]
    elbow_i = int(np.argmax(drops)) if drops else 0
    others = [d for k, d in enumerate(drops) if k != elbow_i]
    prominence = (float(drops[elbow_i] / max(float(np.median(others)), 1e-9))
                  if others else float("nan"))
    monotone = bool(len(drops) >= 3
                    and all(drops[k] <= drops[k + 1] + 1e-12
                            for k in range(len(drops) - 1)))
    is_elbow = bool(drops and drops[elbow_i] >= MIN_ELBOW_DROP
                    and prominence >= MIN_ELBOW_PROMINENCE and not monotone)
    detail["relative_drops"] = drops
    detail["elbow_eps"] = eps[elbow_i] if is_elbow else None
    detail["elbow_relative_drop"] = float(drops[elbow_i]) if drops else None
    detail["elbow_prominence"] = prominence
    detail["drops_are_monotone"] = monotone
    detail["curve_shape"] = "elbow" if is_elbow else "smooth decay"

    headline = rate[eps.index(0.10)] if 0.10 in eps else rate[len(rate) // 2]
    detail["headline_eps"] = 0.10 if 0.10 in eps else eps[len(rate) // 2]
    detail["headline_rate"] = headline
    if not drops:
        shape_note = " The sweep has a single cell, so it has no shape."
    elif detail["curve_shape"] == "elbow":
        shape_note = (
            f" The curve has an ELBOW at eps = {eps[elbow_i]:.2f}: the rate "
            f"falls {drops[elbow_i]:.0%} there against a median "
            f"{float(np.median(others)):.0%} elsewhere ({prominence:.1f}x). Two "
            f"populations are separable and that is where to cut.")
    else:
        why = ("the relative drops increase monotonically across the sweep "
               f"({', '.join(format(x, '.0%') for x in drops)}), which is one "
               f"population thinning out"
               if monotone else
               f"the largest drop ({drops[elbow_i]:.0%} at eps = "
               f"{eps[elbow_i]:.2f}) is only {prominence:.1f}x the median of the "
               f"others, below the {MIN_ELBOW_PROMINENCE}x a separable "
               f"population would leave")
        shape_note = (
            f" The curve DECAYS SMOOTHLY -- {why} -- so no threshold on this "
            f"grid is principled and the rate has to be read as a curve rather "
            f"than at a chosen eps.")

    if group != "skull":
        return Read(
            "NOT_A_RESULT", scored_object,
            f"trunk bones flex, so a length violation there is real posture at "
            f"least as often as it is tracking. The rate is {headline:.3%} at "
            f"eps = {detail['headline_eps']} and is reported for comparison "
            f"only; the branch is read on the skull triangle." + shape_note,
            n_effective=n_effective, detail=detail)

    if join is not None and join.verdict == "FAIL":
        return Read(
            "FAIL", scored_object,
            f"skull violations are {headline:.3%} at eps = "
            f"{detail['headline_eps']}, but they do not track segment "
            f"reconstruction quality. Uncorrelated with R^2 means the feature "
            f"space is suspect including Q1's, and re-tracking becomes blocking."
            + shape_note, n_effective=n_effective, detail=detail)

    if headline < exclude_below:
        return Read(
            "PASS", scored_object,
            f"skull violations affect {headline:.3%} of frames at eps = "
            f"{detail['headline_eps']}, below the {exclude_below:.0%} exclusion "
            f"threshold. Exclude the carrying recordings and proceed." + shape_note,
            n_effective=n_effective, detail=detail)
    if headline < correct_below:
        return Read(
            "GRID_LIMITED", scored_object,
            f"skull violations affect {headline:.3%} of frames at eps = "
            f"{detail['headline_eps']}, inside the {exclude_below:.0%}-"
            f"{correct_below:.0%} band. Too many to exclude and few enough to "
            f"correct: proceed with correction, and cost out an ensemble-DLC "
            f"path before anything is published on this feature space."
            + shape_note, n_effective=n_effective,
            saturation="between the exclude and correct thresholds", detail=detail)
    return Read(
        "FAIL", scored_object,
        f"skull violations affect {headline:.3%} of frames at eps = "
        f"{detail['headline_eps']}, above the {correct_below:.0%} limit. The "
        f"skull triangle is as close to rigid as this animal gets, so a rate "
        f"this high is not posture. The feature space is suspect including Q1's "
        f"and re-tracking becomes blocking." + shape_note,
        n_effective=n_effective, detail=detail)
