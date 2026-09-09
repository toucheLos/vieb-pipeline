r"""A4 for the egocentric representation: did the reduction keep the kinematics?

## The gate is the closed form, and the regressor travels beside it

The egocentric map is a bijection given the equivariance it removed, so centroid
speed and body-axis angular velocity come back from :math:`(s, \xi, \ell_a)`
**exactly**, with no fit anywhere. That is what `recover` computes and what
`a4_read` gates on.

Gating on a regressor instead has already produced one wrong-object failure on
this project: a gradient-boosted model returned 0.823 for speed and 0.392 for
angular velocity on a reduction that had lost precisely nothing. Two things
caused it, neither a fact about the representation -- a centred smoothing kernel
whose label reaches :math:`x_{t+2}`, which no causal window can contain, and a
nonlinear map the regressor simply did not learn. So the regressor arms here are
**reported and never gated**.

## What the brief's acceptance numbers actually refer to

The brief asks for R^2 >= 0.98 (speed) and >= 0.94 (turn), with a pose-only
comparison "near 0.09 / 0.07". Those are the **previous instrument's** numbers,
from ``~/reversibility-tests`` (PCA-64), not shapeflow's. shapeflow's A4 is
closed-form and exact at 1.000000, and its comparable pose-only figure is the
**single-frame** shape block at 0.175 / 0.065 -- where 0.065 lands on the old
0.067, so the like-for-like agrees.

The thresholds are kept as written and are expected to be cleared by six orders
of magnitude. A shortfall would be a bug in this module, not a property of the
corpus.

## A4 is NOT the guard against the group-logarithm error

Measured, and worth stating plainly because the opposite would be the natural
assumption. Substituting separate differencing of position and angle for the
group logarithm -- the error the brief singles out -- barely moves A4 until the
animal is turning fast:

======================  ==========
turn rate               speed R^2
======================  ==========
0.9 rad/s                 0.9973
3.0 rad/s                 0.9853
4.5 rad/s                 0.9646
12 rad/s                  0.6492
======================  ==========

against a threshold of 0.98. So a pipeline carrying the naive velocity would
**pass this gate** at every ordinary turn rate in the corpus. The guard against
that error is `tests/test_se2.py`, which compares `ego.se2_log` against
`scipy.linalg.expm` on the 3x3 matrix representation and fails the naive version
at every turn rate including zero. A4 answers a different question -- whether the
kinematics survived the reduction -- and answering it well is not the same as
catching every way the reduction could be wrong.

## Why the angular channel is exact rather than merely close

`geom.represent.raw_kinematics` measures turning as the rate of
:math:`\arg(\mathrm{nose} - \mathrm{tail\_base})`, and that angle **is**
:math:`\theta`, the heading this representation is built on. So the `omega`
channel is not an estimate of the label -- it is the label, times `fps`, before
smoothing. The one thing that has to match is the smoothing, and `recover`
reproduces `raw_kinematics`'s causal kernel exactly; if it did not, R^2 would
fall away from 1 immediately, which is the check.

## The smoothing is causal, and that is what makes the question well posed

A centred kernel makes ``label[t]`` depend on ``step[t-1..t+1]``. Backward
smoothing puts the label's whole support inside the window a causal encoder can
see. Inherited from `geom.represent` unchanged, and the reason the windowed arm
below is answerable at all.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from recur.read import Read
from recur.util import frames
from . import ego

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: The brief's acceptance. See the module docstring for what they refer to.
R2_SPEED_MIN = 0.98
R2_TURN_MIN = 0.94

#: The prior pipeline's pose-only figures, for the non-circular comparison.
PRIOR_POSE_ONLY = {"speed": 0.090, "angular": 0.067}
#: shapeflow's single-frame shape-block figures, which are the comparable ones.
SHAPEFLOW_SINGLE_FRAME = {"speed": 0.175, "angular": 0.065}

#: Inherited from `geom.represent.raw_kinematics` and must not diverge from it.
SMOOTH_S = 0.1


def _causal_smooth(values: F64, t: int, win: int) -> F64:
    """`raw_kinematics`'s backward moving average, reproduced exactly.

    Any divergence from it shows up as R^2 falling away from 1, which is the
    check -- so this is pinned by the measurement rather than by a separate
    assertion about kernels.
    """
    out = np.full(t, np.nan)
    kernel = np.ones(win) / float(win)
    pad = np.concatenate([np.full(win, np.nan), values])
    out[1:t] = np.convolve(np.nan_to_num(pad), kernel, mode="valid")[:t - 1]
    out[:win] = np.nan
    return out


def recover(x: npt.ArrayLike, ell: float, fps: float, *,
            smooth_s: float = SMOOTH_S) -> tuple[F64, F64]:
    r"""``(speed, omega)`` from the representation alone. Closed form, no fit.

    The centroid is the unweighted mean of the keypoints, so in the body frame it
    sits at :math:`\bar s_t` and its world step is

    .. math::
        R(\theta_t)^{\top}\Delta\mu_t
          = u_t + \ell_a\left(R(\Delta\theta)\bar s_{t+1} - \bar s_t\right)

    where :math:`u_t` is the group element's translation part, recovered from the
    twist by `ego.se2_exp`. A rotation preserves norm, so the speed follows
    without ever knowing :math:`\theta_t` or the arena position -- which is the
    whole claim: the equivariance was removed and the kinematics were not.
    """
    a = np.asarray(x, dtype=np.float64)
    t = a.shape[0]
    speed = np.full(t, np.nan)
    omega = np.full(t, np.nan)
    if t < 3:
        return speed, omega

    s = a[:, :ego.N_POSE].reshape(t, -1, 2)
    xi = a[:, ego.N_POSE:]
    # Undo the scaling to get the bare twist, then the group element.
    raw = np.stack([xi[:, 0] * ell / fps, xi[:, 1] * ell / fps, xi[:, 2] / fps],
                   axis=1)
    dth, u = ego.se2_exp(raw)

    bar = s.mean(axis=1)
    c, sn = np.cos(dth[:-1]), np.sin(dth[:-1])
    nxt = bar[1:]
    rot = np.stack([c * nxt[:, 0] - sn * nxt[:, 1],
                    sn * nxt[:, 0] + c * nxt[:, 1]], axis=1)
    d = u[:-1] + ell * (rot - bar[:-1])
    step = np.linalg.norm(d, axis=1) * float(fps)
    turn = xi[:-1, 2]

    win = frames(smooth_s, fps)
    return _causal_smooth(step, t, win), _causal_smooth(turn, t, win)


def r2(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Inherited from `geom.represent.r2`, restated so the units are identical."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return float("nan")
    ss_res = float(((a - b) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def exact_scores(x: npt.ArrayLike, ell: float, fps: float,
                 speed: npt.ArrayLike, omega: npt.ArrayLike, *,
                 valid: npt.ArrayLike | None = None) -> Detail:
    """The gated arm: closed-form recovery against the raw kinematic labels."""
    got_speed, got_omega = recover(x, ell, fps)
    m = np.isfinite(speed) & np.isfinite(omega) & np.isfinite(got_speed)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    return {"speed": r2(np.asarray(speed)[m], got_speed[m]),
            "angular": r2(np.asarray(omega)[m], got_omega[m]),
            "n_frames": int(m.sum())}


def _held_out_gbm(feats: F64, target: F64, groups: Sequence[str], *,
                  seed: int = 0) -> float:
    """One gradient-boosted fit, held out **by animal**.

    By animal rather than by frame: two frames 33 ms apart are not two
    observations, and a frame-level split would report how well the model
    interpolates within a session.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    g = np.asarray([str(v) for v in groups])
    uniq = np.array(sorted(set(g.tolist())))
    if uniq.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(uniq)[: max(1, uniq.size // 3)].tolist())
    te = np.array([v in held for v in g])
    if te.all() or not te.any():
        return float("nan")
    model = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
    model.fit(feats[~te], target[~te])
    return r2(target[te], model.predict(feats[te]))


def windowed_scores(x: npt.ArrayLike, speed: npt.ArrayLike,
                    omega: npt.ArrayLike, groups: Sequence[str], *,
                    k: int = 5, seed: int = 0,
                    valid: npt.ArrayLike | None = None,
                    pose_only: bool = False,
                    max_rows: int = 200_000) -> Detail:
    """A regressor on the encoder's own causal window. Reported, never gated.

    `pose_only` drops the twist, which is the evidence that the locomotor
    channels do necessary work; at ``k = 1`` it is the non-circular comparison
    against the prior pipeline's 0.090 / 0.067.
    """
    a = np.asarray(x, dtype=np.float64)
    t = a.shape[0]
    cols = slice(0, ego.N_POSE) if pose_only else slice(0, ego.N_DIMS)
    block = a[:, cols]
    rows = np.stack([block[i:t - k + 1 + i] for i in range(k)], axis=1)
    rows = rows.reshape(rows.shape[0], -1)
    lo = k - 1
    sp, om = np.asarray(speed)[lo:], np.asarray(omega)[lo:]
    g = np.asarray([str(v) for v in groups])[lo:]
    m = np.isfinite(rows).all(axis=1) & np.isfinite(sp) & np.isfinite(om)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)[lo:]
    rows, sp, om, g = rows[m], sp[m], om[m], g[m]
    if rows.shape[0] > max_rows:
        idx = np.random.default_rng(seed).permutation(rows.shape[0])[:max_rows]
        rows, sp, om, g = rows[idx], sp[idx], om[idx], g[idx]
    if rows.shape[0] < 100:
        return {"speed": float("nan"), "angular": float("nan"), "n_rows": 0}
    return {"speed": _held_out_gbm(rows, sp, g, seed=seed),
            "angular": _held_out_gbm(rows, om, g, seed=seed),
            "n_rows": int(rows.shape[0]), "k": int(k),
            "pose_only": bool(pose_only)}


def a4_read(scores: Detail, *, scored_object: Detail, n_effective: int) -> Read:
    """Did the egocentric reduction keep the kinematics it may not lose?

    Gates on `exact` only. The windowed and single-frame arms are a different
    question -- what the encoder's own input window retains -- and they travel in
    the reason rather than in the verdict.
    """
    ex = scores.get("exact", {})
    speed = float(ex.get("speed", float("nan")))
    ang = float(ex.get("angular", float("nan")))
    detail: Detail = {"scores": scores, "r2_speed_min": R2_SPEED_MIN,
                      "r2_turn_min": R2_TURN_MIN,
                      "prior_pipeline_pose_only": PRIOR_POSE_ONLY,
                      "shapeflow_single_frame": SHAPEFLOW_SINGLE_FRAME}

    tail = ""
    win = scores.get("windowed", {})
    if np.isfinite(float(win.get("speed", float("nan")))):
        tail += (f" A gradient-boosted regressor on the encoder's own {win['k']}-"
                 f"frame causal window, held out by animal, recovers "
                 f"{win['speed']:.3f} and {win['angular']:.3f}.")
    single = scores.get("pose_only_single_frame", {})
    if np.isfinite(float(single.get("speed", float("nan")))):
        tail += (f" The pose block alone at a single frame falls to "
                 f"{single['speed']:.3f} and {single['angular']:.3f}, against "
                 f"the prior pipeline's {PRIOR_POSE_ONLY['speed']:.3f} / "
                 f"{PRIOR_POSE_ONLY['angular']:.3f} and shapeflow's "
                 f"{SHAPEFLOW_SINGLE_FRAME['speed']:.3f} / "
                 f"{SHAPEFLOW_SINGLE_FRAME['angular']:.3f} -- which is the "
                 f"non-circular comparison, and is why the locomotor channels "
                 f"cannot be dropped.")

    if not (np.isfinite(speed) and np.isfinite(ang)):
        return Read("INCONCLUSIVE", scored_object,
                    "the closed-form reconstruction could not be evaluated",
                    n_effective=n_effective, detail=detail)
    if speed >= R2_SPEED_MIN and ang >= R2_TURN_MIN:
        return Read(
            "PASS", scored_object,
            f"the egocentric representation reconstructs raw centroid speed at "
            f"R^2 = {speed:.6f} (requirement >= {R2_SPEED_MIN}) and raw angular "
            f"velocity at R^2 = {ang:.6f} (requirement >= {R2_TURN_MIN}), in "
            f"closed form from (s, xi, ell_a) rather than through a regressor. "
            f"The map is a bijection given the equivariance it removed, so this "
            f"is exact and not a fit." + tail,
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        f"the representation reconstructs raw centroid speed at only "
        f"R^2 = {speed:.6f} (needs >= {R2_SPEED_MIN}) and raw angular velocity "
        f"at {ang:.6f} (needs >= {R2_TURN_MIN}). The map is a bijection, so a "
        f"shortfall is an error in this module rather than a property of the "
        f"corpus." + tail, n_effective=n_effective, detail=detail)
