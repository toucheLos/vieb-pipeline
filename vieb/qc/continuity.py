r"""Is a frame continuous with the frames either side of it?

Nothing in this repo measured that until now. `qc/effect.py` measures how far one
array moved from another; `qc/bones.py` measures whether a length is plausible.
Neither can see the error that actually reads as *inconsistency* when a skeleton
is watched: a keypoint that slides **along** a bone, or a skull triangle that
drifts coherently. Those change no length at all.

## The measure

For keypoint `k` at frame `t`, fit the similarity :math:`x \mapsto sRx + c` that
carries the **six other keypoints** from a neighbouring frame onto their
positions in this one, apply it to that neighbour's `k`, and take what is left:

.. math::

    r^-(t,k) = \lVert p_{t,k} - T_{t-1 \to t}\, p_{t-1,k} \rVert / \ell_a
    r^+(t,k) = \lVert p_{t,k} - T_{t+1 \to t}\, p_{t+1,k} \rVert / \ell_a

**Why not a second difference.** A plain acceleration
:math:`\lVert p_t - (p_{t-1}+p_{t+1})/2 \rVert` is dominated by the animal
genuinely moving, and by the apparent size change a single overhead camera
produces when the animal rears or the body foreshortens. The similarity absorbs
translation, rotation and scale, so what survives is **this keypoint moving
relative to the rest of the body** -- which is the hypothesis that only one or two
landmarks ever deviate, turned into a number.

## Spike and step are different errors and must not be summed

.. code-block::

    spike(t,k) = min(r^-, r^+)     both neighbours disagree -> a one-frame excursion
    step(t,k)  = |r^- - r^+|       one side only            -> a persistent shift

A teleport leaves and comes back, so **both** residuals are large and `spike` is
large. A real fast movement -- and equally a landmark parked off the body and held
there -- is a *step*: at the transition :math:`r^-` is large while :math:`r^+` is
small, so `spike` stays small and `step` is large.

That distinction is the one this programme turns on. `CLEANING.md` already
concludes that what a temporal filter leaves behind is temporally smooth and
needs an anatomical prior instead; **step** is the first direct measurement of
that residue, and a filter that lowers `spike` while leaving `step` alone is
doing exactly what that paragraph predicts.

## More continuity is not better

A filter can make a recording perfectly continuous by deleting all the movement,
and two arms here are capable of it. So nothing in this module reports a lower
residual as an improvement on its own: `continuity_read` requires the
high-frequency retention it is handed alongside, and refuses to crown an arm that
bought its continuity by flattening the signal.

## One bad keypoint smears into the others, and by how much

The fit for keypoint `k` uses the other six, so a keypoint that is badly wrong
contaminates the prediction of every other keypoint in that frame. Measured on a
synthetic 45 px nose displacement: the nose reads 1.41 body lengths and the
innocent keypoints read up to 0.57 -- a margin of about 2.5x, not a clean zero.

Two consequences, both worth stating rather than discovering later. **Attribution
by `argmax` over keypoints is sound** -- the guilty landmark is the largest by a
clear margin, which is what the "only one or two keypoints deviate" hypothesis
needs. **The pooled residual is inflated** by the leak, equally for every arm, so
comparisons between arms hold and the absolute level should not be read as the
size of the error itself.

## What is circular here

Phase D's corrector **is** this predictor, from the nearer neighbour. Scoring its
output on that side would repeat the skull-violations-on-the-constrained-group
error one phase later, so the disposition arm is scored on the neighbour
`disposition.donor_frame` did *not* pick. `held_out_side` computes that mask and
the reason string says so.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read
from recur.util import describe

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

#: A residual above this many body lengths is a discontinuity. Pre-registered
#: from the prototype's p99 of 0.119 on one recording -- a round number just
#: below it, chosen before the corpus ran and not tuned against any outcome.
SPIKE_BODY_LENGTHS = 0.10

#: Below this many finite correspondences the similarity is not determined and
#: the residual is NaN rather than a number computed from too little.
MIN_CORRESPONDENCES = 3


def similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[F64, F64, F64]:
    r"""``(scale, theta, offset)`` carrying ``a`` onto ``b``, batched over frames.

    Both are ``(T, M, 2)``: `M` correspondences per frame, `T` frames, each frame
    fitted independently. Returns ``scale (T,)``, ``theta (T,)`` and
    ``offset (T, 2)``.

    **Closed form, not an SVD.** In two dimensions the least-squares similarity
    has an exact solution (Horn 1987): with `a` and `b` mean-centred,

    .. code-block::

        num_cos = sum(a . b)        num_sin = sum(a x b)
        theta   = atan2(num_sin, num_cos)
        scale   = hypot(num_cos, num_sin) / sum(|a|^2)

    `disposition.predict` does the same fit per frame through `np.linalg.svd`,
    which costs an SVD per (frame, keypoint) -- minutes for one recording, and far
    past the corpus budget at 22.4M frames times 7 keypoints times 2 directions.
    This is the same answer in one pass of arithmetic. A test pins the two
    together; that module is **not** refactored to call this one, because its
    results are committed and it stays as it ran.

    The closed form also cannot return a reflection, which the SVD version has to
    guard against explicitly -- there is no determinant to check.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mu_x = x.mean(axis=1, keepdims=True)
    mu_y = y.mean(axis=1, keepdims=True)
    cx, cy = x - mu_x, y - mu_y
    num_cos = np.einsum("tmd,tmd->t", cx, cy)
    num_sin = np.einsum("tm,tm->t", cx[..., 0], cy[..., 1]) - \
        np.einsum("tm,tm->t", cx[..., 1], cy[..., 0])
    denom = np.einsum("tmd,tmd->t", cx, cx)
    theta = np.arctan2(num_sin, num_cos)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.hypot(num_cos, num_sin) / denom
    scale = np.where(denom > 0, scale, np.nan)
    # offset is recovered per frame from the centroids once s and theta are known.
    c, s = np.cos(theta), np.sin(theta)
    rot_mu = np.stack([c * mu_x[:, 0, 0] - s * mu_x[:, 0, 1],
                       s * mu_x[:, 0, 0] + c * mu_x[:, 0, 1]], axis=1)
    offset = mu_y[:, 0, :] - scale[:, None] * rot_mu
    return (np.asarray(scale, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
            np.asarray(offset, dtype=np.float64))


def _apply(scale: F64, theta: F64, offset: F64, pts: F64) -> F64:
    """``(T, 2)`` points through ``(T,)`` similarities, one per frame."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([c * pts[:, 0] - s * pts[:, 1],
                    s * pts[:, 0] + c * pts[:, 1]], axis=1)
    return np.asarray(scale[:, None] * rot + offset, dtype=np.float64)


def _residual_one_direction(pose: F64, ell: float, shift: int) -> F64:
    """``(T, K)`` residual predicting each frame from its neighbour at ``t+shift``.

    Frames with no neighbour inside this array are NaN. This takes ONE recording
    and indexes only inside it, so nothing is ever predicted across a seam.
    """
    t_n, k_n = pose.shape[0], pose.shape[1]
    out = np.full((t_n, k_n), np.nan, dtype=np.float64)
    if t_n < 2:
        return out
    lo = max(0, -shift)
    hi = min(t_n, t_n - shift)
    if hi <= lo:
        return out
    idx = np.arange(lo, hi)
    nbr = idx + shift
    for k in range(k_n):
        others = [j for j in range(k_n) if j != k]
        a = pose[nbr][:, others, :]
        b = pose[idx][:, others, :]
        ok = (np.isfinite(a).all(axis=(1, 2)) & np.isfinite(b).all(axis=(1, 2))
              & np.isfinite(pose[nbr][:, k, :]).all(axis=1)
              & np.isfinite(pose[idx][:, k, :]).all(axis=1))
        if len(others) < MIN_CORRESPONDENCES or not ok.any():
            continue
        scale, theta, offset = similarity(a[ok], b[ok])
        pred = _apply(scale, theta, offset, pose[nbr][ok][:, k, :])
        res = np.linalg.norm(pose[idx][ok][:, k, :] - pred, axis=-1) / ell
        out[idx[ok], k] = res
    return out


def residuals(pose: npt.ArrayLike, ell: float) -> tuple[F64, F64]:
    """``(r_minus, r_plus)``, each ``(T, K)`` in body lengths, for ONE recording.

    `r_minus[0]` and `r_plus[-1]` are NaN: those frames have no neighbour on that
    side inside this recording, and reaching for one would cross a seam.
    """
    p = np.asarray(pose, dtype=np.float64)
    if not np.isfinite(ell) or ell <= 0:
        return (np.full(p.shape[:2], np.nan), np.full(p.shape[:2], np.nan))
    return (_residual_one_direction(p, float(ell), -1),
            _residual_one_direction(p, float(ell), +1))


def decompose(r_minus: npt.ArrayLike, r_plus: npt.ArrayLike) -> tuple[F64, F64]:
    """``(spike, step)``. See the module docstring -- they are not the same error.

    Where one side is NaN both outputs are NaN: a one-sided frame cannot tell a
    one-frame excursion from a persistent shift, and guessing which it is would
    put the recording's first and last frames into whichever bucket is convenient.
    """
    a = np.asarray(r_minus, dtype=np.float64)
    b = np.asarray(r_plus, dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    spike = np.where(both, np.minimum(a, b), np.nan)
    step = np.where(both, np.abs(a - b), np.nan)
    return (np.asarray(spike, dtype=np.float64),
            np.asarray(step, dtype=np.float64))


def held_out_side(donor: npt.ArrayLike, r_minus: npt.ArrayLike,
                  r_plus: npt.ArrayLike) -> F64:
    r"""``(T, K)`` the residual on the neighbour the corrector did NOT use.

    `donor` is `disposition.donor_frame`'s choice per frame, or -1 where none was
    taken. Phase D moves the suspect keypoint *to* the prediction from that side,
    so its residual there is near zero by construction and scoring it would be
    measuring the corrector's own premise. The other side is untouched by the
    correction and is the only side that can say whether the keypoint moved
    towards the animal.

    Frames the corrector never touched keep the two-sided `spike`, so an arm is
    not silently scored on a different statistic almost everywhere.
    """
    d = np.asarray(donor, dtype=np.int64)
    a = np.asarray(r_minus, dtype=np.float64)
    b = np.asarray(r_plus, dtype=np.float64)
    spike, _ = decompose(a, b)
    t = np.arange(d.shape[0])
    # `d < 0` means no donor was taken. Without that guard the sentinel -1
    # equals `t - 1` at t = 0 and frame 0 of every recording is scored on the
    # wrong side.
    taken = d >= 0
    used_back = taken & (d == (t - 1))
    used_fwd = taken & (d == (t + 1))
    out = spike.copy()
    out[used_back] = b[used_back]       # donor was t-1, so score t+1
    out[used_fwd] = a[used_fwd]         # donor was t+1, so score t-1
    return np.asarray(out, dtype=np.float64)


def summarize(x: npt.ArrayLike, *, names: Sequence[str]) -> Detail:
    """Pooled and per-keypoint description of a ``(T, K)`` residual field."""
    v = np.asarray(x, dtype=np.float64)
    flat = v[np.isfinite(v)]
    per = []
    for k, n in enumerate(names):
        col = v[:, k]
        col = col[np.isfinite(col)]
        per.append({"keypoint": n, "n": int(col.size),
                    **({} if col.size == 0 else describe(col.tolist()))})
    return {"pooled": describe(flat.tolist()) if flat.size else {},
            "per_keypoint": per,
            "frac_above_threshold": (float(np.mean(flat > SPIKE_BODY_LENGTHS))
                                     if flat.size else float("nan")),
            "threshold_body_lengths": SPIKE_BODY_LENGTHS}


def overlap_2x2(spike_mask: npt.ArrayLike, bone_mask: npt.ArrayLike) -> Detail:
    """The full 2x2, not two rates side by side.

    Two marginal rates cannot distinguish "the same 1.7% of frames, found twice"
    from "two disjoint nets". The cells are the only thing that can, and the
    off-diagonals are where the interesting frames are: `spike_only` is what the
    bone check structurally cannot see, because a keypoint sliding along a bone
    changes no length.
    """
    a = np.asarray(spike_mask, dtype=bool)
    b = np.asarray(bone_mask, dtype=bool)
    n = int(a.size)
    both = int((a & b).sum())
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    # Jaccard over the union, which is the honest agreement statistic when both
    # classes are rare -- plain accuracy would read 0.95 for two disjoint nets.
    union = both + a_only + b_only
    return {
        "n_frames": n,
        "spike_and_bone": both, "spike_only": a_only,
        "bone_only": b_only, "neither": neither,
        "frac_spike": float(a.mean()) if n else float("nan"),
        "frac_bone": float(b.mean()) if n else float("nan"),
        "jaccard": float(both / union) if union else float("nan"),
        "frac_of_spike_that_is_bone_flagged": (float(both / (both + a_only))
                                               if both + a_only else float("nan")),
        "threshold_body_lengths": SPIKE_BODY_LENGTHS,
    }


def continuity_read(rows: Sequence[Mapping[str, Any]],
                    intervals: Mapping[str, Mapping[str, Any]], *,
                    scored_object: Detail, n_effective: int,
                    incumbent: str = "wiener") -> Read:
    """Which arm lowers `spike` without buying it by flattening the signal.

    Two guards, both learned the hard way in this repo:

    * **Non-overlapping intervals.** `separates` crowned an arm on a 0.029 px
      point-estimate difference while the bootstrap intervals overlapped almost
      entirely. An arm has to clear the incumbent's interval, not its point.
    * **Retention beside it.** A lower residual is not an improvement if the
      high-frequency power went with it. An arm that lowers `spike` while also
      collapsing `hf_retained` is named as over-smoothing, not as a winner.
    """
    def lo(arm: str, key: str) -> float:
        return float(intervals[arm][key]["lo"])

    def hi(arm: str, key: str) -> float:
        return float(intervals[arm][key]["hi"])

    arms = [str(r["arm"]) for r in rows]
    if incumbent not in arms:
        return Read("NOT_A_RESULT", scored_object,
                    f"the incumbent {incumbent!r} is not among the scored arms, "
                    f"so there is nothing to compare against",
                    n_effective=n_effective, degenerate=True,
                    detail={"arms": arms})

    better, oversmoothed = [], []
    for arm in arms:
        if arm == incumbent:
            continue
        lowers = hi(arm, "spike") < lo(incumbent, "spike")
        if not lowers:
            continue
        if hi(arm, "hf_retained") < lo(incumbent, "hf_retained"):
            oversmoothed.append(arm)
        else:
            better.append(arm)

    detail: Detail = {"arms": arms, "incumbent": incumbent,
                      "separates_and_keeps_power": better,
                      "separates_by_flattening": oversmoothed}
    if not better and not oversmoothed:
        return Read(
            "INCONCLUSIVE", scored_object,
            f"no arm lowers the spike residual below {incumbent}'s "
            f"animal-bootstrap interval. The point estimates differ; the "
            f"intervals do not separate, and this repo has crowned an arm on a "
            f"0.029 px point-estimate difference once already",
            n_effective=n_effective, detail=detail)
    if not better:
        return Read(
            "INCONCLUSIVE", scored_object,
            f"{', '.join(oversmoothed)} lower the spike residual below "
            f"{incumbent}, and each also loses high-frequency power against it. "
            f"A filter can make a recording perfectly continuous by deleting the "
            f"movement, and on this evidence that is what these did",
            n_effective=n_effective, detail=detail)
    return Read(
        "PASS", scored_object,
        f"{', '.join(better)} lower the spike residual below {incumbent}'s "
        f"animal-bootstrap interval without losing high-frequency power against "
        f"it. Continuity is a fourth view and not a verdict: the banked bakeoff "
        f"in cleaning.json chose on three axes and is not restated here",
        n_effective=n_effective, detail=detail)
