r"""Stage 1. Equivariance removal: the egocentric pose and the SE(2) velocity.

.. math::
    s_t = \frac{1}{\ell_a} R(\theta_t)^{\top}(y_t - p_t), \qquad
    \xi_t = \frac{f}{\ell_a}\log\left(g_t^{-1} g_{t+1}\right)

**This is equivariance removal, not compression.** Nothing is projected away and
nothing is fitted: 17 dimensions in, 17 dimensions out, and `inverse` rebuilds
the keypoints exactly. It is the only mandatory preprocessing step, and it is
deliberately independent of the frozen Kendall gauge -- which is what makes the
A4 parity check in `parity.py` a comparison between two representations rather
than a quantity predicting itself.

## The body frame

``p_t`` is the ``center`` keypoint and ``theta_t`` is the angle of
``tail_base -> nose``, the same axis shapeflow's A4 gate uses. `heading` computes
it straight from the pose with `arctan2` rather than through
`geom.represent.reconstruct_axis_angle`, which routes the same angle through the
decomposition and the gauge. The two agree to machine precision -- that is the
decomposition identity, and `tests/test_ego.py` asserts it -- so taking the short
path costs nothing and buys independence from the gauge.

## The group logarithm, and why not separate differencing

:math:`g_t^{-1}g_{t+1}` is the motion *in the frame the animal was actually in*.
Differencing position and angle separately instead gives
:math:`(R^{\top}\Delta p, \Delta\theta)`, which is the group element's translation
part rather than the twist that generates it. The error is second order in the
step but scales as :math:`\omega\lVert v\rVert` -- **largest exactly during
turns**, which is where the behaviour is. `naive_velocity` implements the wrong
version so `tests/test_se2.py` can measure the gap rather than assert against
itself.

## Scaling: one dimensional correction to the brief

The brief writes :math:`\xi_t = (f/\ell_a)\log(g_t^{-1}g_{t+1})`, applying
:math:`f/\ell_a` to the whole twist. That is right for the two translation
components -- length over length, per second -- and **wrong for the rotation**:
an angular rate divided by a body length has units of
:math:`\mathrm{rad}\,\mathrm{s}^{-1}\mathrm{px}^{-1}`, and, worse, it makes a
large mouse's measured turning systematically smaller than a small one's. That
reintroduces exactly the between-animal size signal :math:`\ell_a` exists to
remove, straight into the channel where it is least visible.

So translation is scaled by :math:`f/\ell_a` and rotation by :math:`f` alone.
`SCALING` records the choice and `tests/test_ego.py` asserts that a uniformly
scaled animal gives an identical ``omega`` channel.

## ell_a is a per-animal constant, and that is not the same as dividing out scale

shapeflow refuses per-frame scale normalisation, correctly: for a top-down mouse
rearing appears only as apparent foreshortening, so dividing by the frame's own
size deletes rearing. :math:`\ell_a` is the **animal's** median nose-to-tail
length over all of its recordings -- one number per animal. It removes
between-animal body size and leaves every within-recording scale change intact.

It is also not per *recording*. Per-recording standardisation is what silently
disarmed a control on this project once already: it removes exactly the
between-recording differences a null has to be able to see.

## Rank: three directions are identically zero, by construction

Removing SE(2) equivariance from 14 coordinates costs three degrees of freedom
and they are visible in the output rather than projected away:

* ``s`` at ``center`` is :math:`(0,0)` -- the origin is one of the keypoints;
* the ``y`` component of ``s[nose] - s[tail_base]`` is 0 -- the axis is aligned.

So ``s`` has numerical rank 11, not 14. This is the same shape of fact as
:math:`\operatorname{Im}(c^H\Delta z)\equiv 0` in `kendall`, and it is recorded
here for the same reason: a model free to fit variance along a direction the data
never moves in earns unbounded likelihood. `basis` returns the 11-dimensional
orthonormal projection for any consumer that needs full rank. The quantizer does
not -- k-means on a dead direction is harmless, it contributes no distance -- but
the fact belongs in the open either way.

## Never across a seam

Every function takes ONE recording. :math:`\xi_t` needs frame :math:`t+1`, so the
last frame of every recording has no velocity and is marked invalid rather than
zero-filled: a zero-filled increment reads as "the animal did not move", which is
a behaviour.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: LUNA keypoint order, from `recur.qc.swap`.
LEFT_EAR, RIGHT_EAR, NOSE, CENTER, LEFT_HIP, RIGHT_HIP, TAIL_BASE = range(7)

#: Body-frame origin, and the axis that defines the heading.
ORIGIN = CENTER
AXIS = (NOSE, TAIL_BASE)

#: Dimensions. 14 egocentric coordinates + 3 SE(2) twist.
N_POSE = 14
N_TWIST = 3
N_DIMS = N_POSE + N_TWIST

#: Degrees of freedom SE(2) removal costs, hence the rank deficit in `s`.
SE2_DOF = 3
POSE_RANK = N_POSE - SE2_DOF

#: What each part of the twist is divided by. See the module docstring: applying
#: 1/ell_a to omega is dimensionally wrong AND leaks body size into the turning
#: channel, so rotation takes the frame rate alone.
SCALING = {"v_x": "fps / ell_a", "v_y": "fps / ell_a", "omega": "fps"}

#: Below this |omega| the closed-form V^-1 loses precision to cancellation in
#: 1 - cos(omega), and the series is used instead. Chosen where the two agree to
#: ~1e-14 relative, measured in tests/test_se2.py.
SMALL_OMEGA = 1e-4

#: Channel names, in column order.
CHANNELS: tuple[str, ...] = tuple(
    f"s_{k}_{c}" for k in ("left_ear", "right_ear", "nose", "center",
                           "left_hip", "right_hip", "tail_base") for c in "xy"
) + ("v_x", "v_y", "omega")


def wrap(angle: npt.ArrayLike) -> F64:
    """Wrap to (-pi, pi]. Every angular difference in this module goes here."""
    a = np.asarray(angle, dtype=np.float64)
    return np.asarray((a + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


# --------------------------------------------------------------------------
# SE(2)
# --------------------------------------------------------------------------

def se2_log(dtheta: npt.ArrayLike, u: npt.ArrayLike) -> F64:
    r"""``(T, 3)`` twist ``(v_x, v_y, omega)`` from a relative SE(2) element.

    `dtheta` is the relative rotation and `u` its ``(T, 2)`` translation part,
    already expressed in the earlier frame. The twist satisfies
    :math:`\exp(v,\omega) = (R(\omega), V(\omega)v)` with

    .. math::
        V^{-1}(\omega) = \begin{pmatrix} a & b \\ -b & a\end{pmatrix},\quad
        a = \tfrac{\omega}{2}\cot\tfrac{\omega}{2},\quad b = \tfrac{\omega}{2}

    so ``v = V^-1(omega) u``. As :math:`\omega\to 0`, :math:`a\to 1` and the map
    degenerates to the identity -- which is why differencing position and angle
    separately looks correct on straight-line motion and is wrong on turns.

    `a` is computed from its series below `SMALL_OMEGA`: the closed form divides
    by :math:`1-\cos\omega`, which loses half the mantissa to cancellation near
    zero, and near zero is where most frames sit.
    """
    w = wrap(dtheta)
    uu = np.asarray(u, dtype=np.float64)
    if uu.shape[-1] != 2:
        raise ValueError(f"u must be (T, 2), got {uu.shape}")
    half = 0.5 * w
    small = np.abs(w) < SMALL_OMEGA
    a = np.empty_like(w)
    # cot(x) = 1/x - x/3 - x^3/45 - ...  so  x*cot(x) = 1 - x^2/3 - x^4/45
    hs = half[small]
    a[small] = 1.0 - hs ** 2 / 3.0 - hs ** 4 / 45.0
    hb = half[~small]
    a[~small] = hb / np.tan(hb)
    b = half
    v_x = a * uu[:, 0] + b * uu[:, 1]
    v_y = -b * uu[:, 0] + a * uu[:, 1]
    return np.stack([v_x, v_y, w], axis=1)


def se2_exp(twist: npt.ArrayLike) -> tuple[F64, F64]:
    r"""Inverse of `se2_log`: ``(dtheta, u)`` from ``(v_x, v_y, omega)``.

    .. math:: V(\omega) = \frac{1}{\omega}
        \begin{pmatrix}\sin\omega & -(1-\cos\omega)\\ 1-\cos\omega & \sin\omega\end{pmatrix}

    Present so the round trip is testable. Nothing in the pipeline needs it.
    """
    t = np.asarray(twist, dtype=np.float64)
    v, w = t[:, :2], t[:, 2]
    small = np.abs(w) < SMALL_OMEGA
    sin_c = np.empty_like(w)     # sin(w)/w
    cos_c = np.empty_like(w)     # (1-cos w)/w
    ws = w[small]
    sin_c[small] = 1.0 - ws ** 2 / 6.0 + ws ** 4 / 120.0
    cos_c[small] = ws / 2.0 - ws ** 3 / 24.0
    wb = w[~small]
    sin_c[~small] = np.sin(wb) / wb
    cos_c[~small] = (1.0 - np.cos(wb)) / wb
    u_x = sin_c * v[:, 0] - cos_c * v[:, 1]
    u_y = cos_c * v[:, 0] + sin_c * v[:, 1]
    return w, np.stack([u_x, u_y], axis=1)


def naive_velocity(dtheta: npt.ArrayLike, u: npt.ArrayLike) -> F64:
    """The WRONG version, implemented on purpose so the gap is measurable.

    Differencing position and angle separately returns the group element's own
    translation part rather than the twist that generates it -- i.e. it takes
    ``V^-1 = I``. `tests/test_se2.py` measures where that costs, and the answer
    is: proportionally to ``omega * ||v||``, so nothing on straight-line motion
    and everything on a turn.
    """
    uu = np.asarray(u, dtype=np.float64)
    return np.stack([uu[:, 0], uu[:, 1], wrap(dtheta)], axis=1)


# --------------------------------------------------------------------------
# The body frame
# --------------------------------------------------------------------------

def heading(pose: npt.ArrayLike, *, axis: tuple[int, int] = AXIS) -> F64:
    """``(T,)`` body-axis angle, from ``tail_base -> nose``, straight from pose."""
    p = np.asarray(pose, dtype=np.float64)
    head, tail = axis
    v = p[:, head] - p[:, tail]
    return np.asarray(np.arctan2(v[:, 1], v[:, 0]), dtype=np.float64)


def body_length(pose: npt.ArrayLike, *, axis: tuple[int, int] = AXIS) -> F64:
    """``(T,)`` nose-to-tail_base distance, in pixels."""
    p = np.asarray(pose, dtype=np.float64)
    head, tail = axis
    return np.asarray(np.linalg.norm(p[:, head] - p[:, tail], axis=1),
                      dtype=np.float64)


def ell_a(pose_blocks: Sequence[npt.ArrayLike],
          usable: Sequence[npt.ArrayLike] | None = None) -> float:
    """The animal's body length: the median over ALL of its recordings.

    One number per animal. Not per frame -- that deletes rearing, which on a
    top-down camera appears only as foreshortening. Not per recording -- that
    removes the between-recording differences a null has to be able to see.
    """
    lens = []
    for i, pose in enumerate(pose_blocks):
        d = body_length(pose)
        if usable is not None:
            d = d[np.asarray(usable[i], dtype=bool)]
        lens.append(d[np.isfinite(d)])
    if not lens:
        return float("nan")
    pooled = np.concatenate(lens)
    return float(np.median(pooled)) if pooled.size else float("nan")


# --------------------------------------------------------------------------
# The transform
# --------------------------------------------------------------------------

def egocentric(pose: npt.ArrayLike, ell: float, *,
               origin: int = ORIGIN, axis: tuple[int, int] = AXIS) -> F64:
    """``(T, 14)``: every keypoint in the body frame, in body lengths."""
    p = np.asarray(pose, dtype=np.float64)
    th = heading(p, axis=axis)
    c, s = np.cos(th), np.sin(th)
    rel = p - p[:, origin][:, None, :]
    # R(theta)^T applied to each keypoint.
    x = rel[..., 0] * c[:, None] + rel[..., 1] * s[:, None]
    y = -rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None]
    return np.asarray(np.stack([x, y], axis=-1).reshape(p.shape[0], -1) / ell,
                      dtype=np.float64)


def twist(pose: npt.ArrayLike, ell: float, fps: float, *,
          origin: int = ORIGIN, axis: tuple[int, int] = AXIS,
          naive: bool = False) -> tuple[F64, BOOL]:
    r"""``(T, 3)`` scaled twist and its validity, for ONE recording.

    Row `t` is the motion from frame `t` to `t+1`, so the last row is undefined
    and `valid[-1]` is False. Never computed across a seam, because this takes
    one recording.
    """
    p = np.asarray(pose, dtype=np.float64)
    t = p.shape[0]
    out = np.zeros((t, N_TWIST), dtype=np.float64)
    valid = np.zeros(t, dtype=bool)
    if t < 2:
        return out, valid
    th = heading(p, axis=axis)
    pos = p[:, origin]
    dpos = pos[1:] - pos[:-1]
    c, s = np.cos(th[:-1]), np.sin(th[:-1])
    # The step, expressed in the frame the animal was IN when it moved.
    u = np.stack([dpos[:, 0] * c + dpos[:, 1] * s,
                  -dpos[:, 0] * s + dpos[:, 1] * c], axis=1)
    dth = wrap(th[1:] - th[:-1])
    tw = naive_velocity(dth, u) if naive else se2_log(dth, u)
    # Translation by fps/ell, rotation by fps alone -- see the module docstring.
    out[:-1, 0] = tw[:, 0] * (fps / ell)
    out[:-1, 1] = tw[:, 1] * (fps / ell)
    out[:-1, 2] = tw[:, 2] * fps
    valid[:-1] = True
    return out, valid


def transform(pose: npt.ArrayLike, ell: float, fps: float, *,
              usable: npt.ArrayLike | None = None,
              naive: bool = False) -> tuple[F64, BOOL]:
    """``(X, valid)`` for ONE recording: ``[s (14) | xi (3)]``.

    `usable` is shapeflow's own mask -- defined frames not touching a missing
    keypoint -- and is intersected in, never replaced. A frame it excludes is one
    where a keypoint was not measured, and no transform changes that.
    """
    p = np.asarray(pose, dtype=np.float64)
    s = egocentric(p, ell)
    xi, valid = twist(p, ell, fps, naive=naive)
    if usable is not None:
        u = np.asarray(usable, dtype=bool)
        valid &= u
        # xi_t reads frame t+1, so a bad frame invalidates its predecessor too.
        valid[:-1] &= u[1:]
    return np.concatenate([s, xi], axis=1), valid


def inverse(x: npt.ArrayLike, theta: npt.ArrayLike, origin_xy: npt.ArrayLike,
            ell: float) -> F64:
    """``(T, 7, 2)`` keypoints back from the representation. Exact.

    `theta` and `origin_xy` are the equivariance that was removed and are not
    part of the representation -- that is the point of removing them. They are
    supplied here so the bijection is testable, which is the only claim
    `transform` makes about information.
    """
    s = np.asarray(x, dtype=np.float64)[:, :N_POSE].reshape(-1, N_POSE // 2, 2)
    th = np.asarray(theta, dtype=np.float64)
    p0 = np.asarray(origin_xy, dtype=np.float64)
    c, si = np.cos(th), np.sin(th)
    sx, sy = s[..., 0] * ell, s[..., 1] * ell
    x_w = sx * c[:, None] - sy * si[:, None]
    y_w = sx * si[:, None] + sy * c[:, None]
    return np.asarray(np.stack([x_w, y_w], axis=-1) + p0[:, None, :],
                      dtype=np.float64)


# --------------------------------------------------------------------------
# Rank
# --------------------------------------------------------------------------

def dead_columns() -> dict[str, int]:
    """The coordinates that are identically zero after equivariance removal.

    Named rather than discovered, so a change to `ORIGIN` or `AXIS` that broke
    the identity would fail a test instead of quietly leaving live variance in a
    column something downstream assumed was dead.
    """
    return {"origin_x": 2 * ORIGIN, "origin_y": 2 * ORIGIN + 1}


def rank_read(x: npt.ArrayLike, *, tol: float = 1e-9) -> Detail:
    """Numerical rank of the pose block, and which singular values are dead.

    Expected: `POSE_RANK` = 11 of 14. Two directions go to the origin keypoint
    and one to the aligned axis, which is exactly the 3 degrees of freedom SE(2)
    removal costs.
    """
    s = np.asarray(x, dtype=np.float64)[:, :N_POSE]
    s = s[np.isfinite(s).all(axis=1)]
    if s.shape[0] < N_POSE:
        return {"rank": None, "why": "too few finite rows"}
    sv = np.linalg.svd(s - s.mean(axis=0), compute_uv=False)
    rel = sv / max(sv[0], 1e-300)
    return {"singular_values": sv.tolist(), "relative": rel.tolist(),
            "rank": int((rel > tol).sum()), "expected_rank": POSE_RANK,
            "n_dead": int((rel <= tol).sum()), "tol": tol}
