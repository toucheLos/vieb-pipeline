r"""What to do with a suspect keypoint: correct it, abstain on it, or down-weight it.

Not a corrector. **Three dispositions and one gate that can actually fail.**

## The measured facts this is built on

* **96-99% of violating frames are explained by removing ONE keypoint** -- 4.2%
  (worst recordings) / 0.8% (ordinary) need two, and none need three. No landmark
  dominates; it is whichever one slips.
* **DLC confidence predicts a violation at AUC 0.60-0.67.** shapeflow's
  confidence *gate* refused on this corpus for want of a bimodal mode, and the
  project concluded confidence was useless. "No threshold exists" is not "no
  signal exists" -- and a tie-break needs monotonicity, not calibration.
* Violating runs are **short by count and long by mass**. Measured corpus-wide on
  this array in `results/runlen.json`: median run **1 frame**, p75 **3**, and runs
  of 3 frames or fewer carry only **38.4%** of violating keypoint-frames. So a
  3-frame envelope reaches most runs and under two-fifths of the error.

## Why the run length decides the disposition

The 61.6% of violating mass outside a 3-frame envelope is the reason abstention
is needed at all. A run past that envelope is not a detection glitch; it is the
tracker sitting on a wrong mode -- a keypoint on a different body part, or a swap that
persists until something breaks it. **Minimal projection is the wrong operator
for that.** The point is not perturbed from its true position, it is on another
object, and pulling it to the nearest feasible location yields something
anatomically legal and *still wrong* -- which no longer carries the flag that
would have made it abstainable downstream. That is laundering the error, and it
is strictly worse than leaving it violating.

So correction and abstention are not competing arms. **Abstention is the outer
envelope and projection operates inside it.**

`recur/qc/swap.py` already solved this exact shape for bilateral swaps:
`MAX_RUN_S = 0.25`, short negative runs relabelled, long ones flagged
`sustained` and left alone, the threshold read off a measured run-length
distribution. `swap.runs_of` is reused here rather than reimplemented.

## Why the constraint is an inequality

Projection can shorten a bone in the image plane but never lengthen it, so a
suspect is pulled *in* when its bones are too long and never pushed *out* when
they are short. This is the Phase 0 asymmetry, and it halves the feasible set.

## Why the target is the interior, not the boundary

A minimum-norm projection puts every corrected frame exactly on the constraint
boundary -- a codimension-1 surface. Depositing ~2% of frames onto it writes an
atom into an otherwise continuous feature distribution, k-means finds it, and it
reads as a state. This project manufactured a dwell signature through a
persistence prior once already; a corrector that deposits mass on a low-
dimensional surface is the same failure one step earlier.

So the target is the position the *unaffected* bones predict, clipped into the
feasible set -- which lands in the interior whenever the prediction is feasible.

## Nothing here crosses a seam

Every function takes ONE recording. `runs_of` has no seam argument and is
seam-safe only by that contract, which this module honours.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from recur.qc.swap import runs_of

from . import bones

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

#: Runs no longer than this are eligible for projection. Empirical: the median
#: violating run is 1-2 frames and p75 is 3. Above it, the tracker is on a wrong
#: mode and the frame is abstained rather than corrected.
MAX_CORRECT_FRAMES = 3

#: Below this many rigid pairs left after excluding the suspect's, the per-frame
#: scale estimate is not trustworthy and the frame is abstained instead.
#: `common_log_scale` returns 1.0 for an empty `keep`, silently reverting the arm
#: to raw pixels -- this guard is what stops that being invisible.
MIN_SCALE_PAIRS = 4

#: Suspect runs closer than this are merged before abstaining. Each abstain run
#: destroys ~(n-1) n-gram positions and a fragment shorter than n vanishes
#: entirely, so many short runs cost far more than the same mass in a few long
#: blocks.
MERGE_GAP_FRAMES = 5


def pairs_touching(keypoint: int, pairs: Sequence[tuple[int, int]]) -> list[int]:
    """Indices into `pairs` of every pair containing `keypoint`."""
    return [m for m, (i, j) in enumerate(pairs) if keypoint in (i, j)]


def suspects(viol: npt.ArrayLike, group: Sequence[tuple[int, int]],
             conf: npt.ArrayLike | None = None) -> I64:
    r"""``(T,)`` the suspect keypoint per frame, or -1 where there is none.

    The suspect is the keypoint present in **every** violating bone of the frame.
    Where more than one qualifies -- which happens whenever a single bone is
    violated, since both its endpoints qualify -- the tie is broken by lowest DLC
    confidence.

    Confidence is used only to *order* two candidates. Its calibration is
    irrelevant and known to be poor here; what matters is that it ranks the wrong
    keypoint below the right one more often than not, which AUC 0.60-0.67 says it
    does.
    """
    v = np.asarray(viol, dtype=bool)
    c = None if conf is None else np.asarray(conf, dtype=np.float64)
    out = np.full(v.shape[0], -1, dtype=np.int64)
    for t in np.flatnonzero(v.any(axis=1)):
        bad = [group[int(m)] for m in np.flatnonzero(v[t])]
        common = set(bad[0])
        for b in bad[1:]:
            common &= set(b)
        if not common:
            continue                      # needs >= 2 keypoints; left to abstain
        cands = sorted(common)
        out[t] = (cands[int(np.argmin(c[t, cands]))] if c is not None
                  else cands[0])
    return out


def envelope(mask: npt.ArrayLike, *, max_correct: int = MAX_CORRECT_FRAMES,
             merge_gap: int = MERGE_GAP_FRAMES) -> tuple[BOOL, BOOL]:
    """``(correctable, abstain)`` from a per-frame violation mask.

    Short runs are eligible for projection; longer ones are abstained. Abstain
    runs separated by less than `merge_gap` are merged, because the cost of
    abstention is per-RUN rather than per-frame: each one splits the symbol
    sequence and destroys ~(n-1) n-gram positions, so a few long blocks are
    cheaper than many short ones carrying the same mass.
    """
    m = np.asarray(mask, dtype=bool)
    correctable = np.zeros_like(m)
    abstain = np.zeros_like(m)
    for a, b in runs_of(m):
        if b - a <= max_correct:
            correctable[a:b] = True
        else:
            abstain[a:b] = True
    if abstain.any():
        runs = runs_of(abstain)
        for k in range(len(runs) - 1):
            if runs[k + 1][0] - runs[k][1] < merge_gap:
                abstain[runs[k][1]:runs[k + 1][0]] = True
    return correctable, abstain


def donor_frame(mask: npt.ArrayLike, frame: int) -> int:
    """The nearest non-violating frame of the same recording, or -1.

    The envelope guarantees a correctable run is at most `MAX_CORRECT_FRAMES`
    long, so the frames either side of it are non-violating by construction --
    unless the run touches a recording boundary, where there is no donor and the
    caller must abstain. **Never looks outside the array it is given**, and it is
    given one recording, so it cannot cross a seam.
    """
    m = np.asarray(mask, dtype=bool)
    t = int(frame)
    before = after = -1
    for k in range(t - 1, -1, -1):
        if not m[k]:
            before = k
            break
    for k in range(t + 1, m.shape[0]):
        if not m[k]:
            after = k
            break
    if before < 0:
        return after
    if after < 0:
        return before
    return before if (t - before) <= (after - t) else after


def predict(pose: npt.ArrayLike, frame: int, donor: int,
            suspect: int) -> F64 | None:
    r"""Where the rest of the animal says the suspect keypoint is.

    Fits the similarity :math:`x \mapsto sRx + c` carrying the **six non-suspect
    keypoints** of the donor frame onto their positions in this one, then applies
    it to the donor's suspect. Umeyama's closed form, with scale, because the
    apparent size of the animal changes between frames under a single overhead
    camera and a rotation-only fit would charge that to the residual.

    This is the target the pre-registration asked for, and the reason it is
    interior rather than boundary: a similarity multiplies every bone length by
    the same `s`, so a donor frame that satisfies the constraints maps to a
    position that satisfies them too, at a length strictly inside the tolerance
    wherever the donor's was. The landing is a continuous function of the donor's
    pose and of the fit, so it lies on no surface at all.

    Returns None when the fit is not determined -- fewer than three finite
    correspondences, or a degenerate configuration.
    """
    p = np.asarray(pose, dtype=np.float64)
    others = [k for k in range(p.shape[1]) if k != suspect]
    a = p[int(donor)][others]
    b = p[int(frame)][others]
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if int(ok.sum()) < 3 or not np.isfinite(p[int(donor), suspect]).all():
        return None
    a, b = a[ok], b[ok]
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ca, cb = a - mu_a, b - mu_b
    var_a = float((ca ** 2).sum())
    if var_a <= 0.0:
        return None
    u, sig, vt = np.linalg.svd(cb.T @ ca / ca.shape[0])
    d = np.eye(2)
    if np.linalg.det(u @ vt) < 0:          # keep it a rotation, not a reflection
        d[1, 1] = -1.0
    r = u @ d @ vt
    scale = float((sig * np.diag(d)).sum()) / (var_a / ca.shape[0])
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    out = r @ (p[int(donor), suspect] - mu_a) * scale + mu_b
    return np.asarray(out, dtype=np.float64) if np.isfinite(out).all() else None


def project(pose: npt.ArrayLike, frame: int, suspect: int, *,
            pairs: Sequence[tuple[int, int]], constrain: Sequence[int],
            l_hat_scaled: npt.ArrayLike, eps: float,
            target: npt.ArrayLike | None = None) -> tuple[F64, Detail]:
    r"""Move one keypoint in one frame into the feasible set. Nothing else moves.

    `constrain` holds the indices into `pairs` of the bones this corrector is
    allowed to constrain -- the **constrained group** and nothing else. Passing
    every pair incident on the suspect is what let a nose correction constrain a
    trunk bone and made the held-out gate's claim untrue; the set is now the
    caller's to state.

    `l_hat_scaled` holds the animal's reference length for every pair, in
    **scale-normalised** units -- divided by the per-frame common scale, itself
    estimated from the pairs NOT touching the suspect. The per-frame scale SD is
    0.155 log units against 0.178 for the relative geometry, so a constraint
    applied in raw pixels spends most of its tolerance on a nuisance parameter.

    The constraint is :math:`\ell_m \le \hat\ell_m (1+\varepsilon)` for every
    constrained pair touching the suspect -- an inequality, because projection can
    shorten a bone and never lengthen it. Geometrically each is a disk of radius
    :math:`\hat\ell_m(1+\varepsilon)` centred on the suspect's partner, and the
    feasible set is their intersection.

    `target` is where the suspect is believed to belong -- `predict`'s output. If
    it is feasible the suspect goes there, in the **interior**. Only when it is
    not, or when none is supplied, does the alternating projection run, and then
    the landing is on the boundary and `landed_on_boundary` says so. A boundary
    landing deposits the frame on a codimension-1 surface, which a quantizer
    downstream will find and report as a state, so its count is a result.
    """
    p = np.asarray(pose, dtype=np.float64).copy()
    lh = np.asarray(l_hat_scaled, dtype=np.float64)
    touching = [int(m) for m in constrain
                if suspect in pairs[int(m)]
                and np.isfinite(lh[int(m)]) and lh[int(m)] > 0]
    if not touching:
        return p[frame], {"moved": False, "why": "no reference for any bone",
                          "converged": True, "landed_on_boundary": False}

    def excess_of(x: F64) -> tuple[Any, float]:
        worst, most = None, 0.0
        for m in touching:
            i, j = pairs[m]
            other = p[frame, j if i == suspect else i]
            d = float(np.linalg.norm(x - other))
            limit = float(lh[m]) * (1.0 + float(eps))
            if d > limit and d - limit > most:
                worst, most = (other, d, limit), d - limit
        return worst, most

    start = p[frame, suspect].copy()
    boundary = False
    if target is not None:
        x = np.asarray(target, dtype=np.float64).copy()
    else:
        x = start.copy()
    if excess_of(x)[0] is not None:
        # Not feasible: fall back to alternating projection onto the disks. The
        # sets are convex and at most two constraints are ever active here, so
        # this converges immediately; the loop is for the rare three-bone frame.
        boundary = True
        for _ in range(32):
            worst, _e = excess_of(x)
            if worst is None:
                break
            other, d, limit = worst
            x = other + (x - other) * (limit / d)
    converged = excess_of(x)[0] is None

    moved = float(np.linalg.norm(x - start))
    p[frame, suspect] = x
    return p[frame], {"moved": moved > 0.0, "displacement_px": moved,
                      "n_constraints": len(touching),
                      "converged": bool(converged),
                      "landed_on_boundary": bool(boundary),
                      "used_target": target is not None}


def reliability(conf: npt.ArrayLike, viol: npt.ArrayLike,
                corrected: npt.ArrayLike, abstained: npt.ArrayLike) -> F64:
    r"""``(T, K)`` in [0, 1]: how much each keypoint-frame should be trusted.

    Confidence times a geometric factor. An abstained keypoint-frame goes to 0, a
    corrected one is discounted rather than trusted at face value, and everything
    else keeps its confidence.

    **Consumed at Step 4 as a covariate, never as a likelihood weight.** A
    weighted likelihood is not a code length -- a model lowers its cost by
    down-weighting whatever it predicts badly -- and charging duration
    "identically across all arms" is what makes arms comparable at all.
    """
    c = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    out = c.copy()
    v = np.asarray(viol, dtype=bool)
    corr = np.asarray(corrected, dtype=bool)
    abst = np.asarray(abstained, dtype=bool)
    if v.ndim == 1:
        v = np.broadcast_to(v[:, None], c.shape)
    if corr.ndim == 1:
        corr = np.broadcast_to(corr[:, None], c.shape)
    if abst.ndim == 1:
        abst = np.broadcast_to(abst[:, None], c.shape)
    out = np.where(corr, out * 0.5, out)
    out = np.where(abst, 0.0, out)
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)
