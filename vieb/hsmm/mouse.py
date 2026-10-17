"""STEP 4 C2-C3: the freeze / move / tracking_lost model for mouse keypoints.

Registered in `results/step4/REGISTRATION.md`. Everything is in body lengths and
seconds; frames appear only through `frames()` at the boundary.

* signal  -- centroid speed ``v`` and pose-change rate ``r`` over a window ``w``;
* freeze  -- the (v, r) distribution of a motionless pose plus Luna's measured
  jitter (built by the caller from `vieb.calms.degrade`), never fitted to
  behaviour;
* move    -- the residual of a two-component mixture with the freeze density
  held fixed (EM, no temporal information);
* tracking_lost -- driven by the committed tracking flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from vieb.hsmm import core
from vieb.tok import ego

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B1 = npt.NDArray[np.bool_]

STATES: tuple[str, ...] = ("freeze", "move", "tracking_lost")
FREEZE, MOVE, LOST = 0, 1, 2
W_S = 0.2
D_MAX_S = 10.0
LOG_LO, LOG_HI, N_BINS = -3.0, 1.5, 40
PSEUDO = 1e-6
P_FLAG_LOST = 0.99
EM_TOL, EM_MAX = 1e-8, 500


def half_window(fps: float) -> int:
    """Half of the registered window, in frames (at least one)."""
    return max(1, int(round(W_S * fps / 2.0)))


def d_max(fps: float) -> int:
    return int(round(D_MAX_S * fps))


def signal(pose: npt.ArrayLike, ell: float, fps: float) -> tuple[F64, F64]:
    """``(v, r)`` per frame for ONE recording, in body lengths per second.

    Displacement over ``[t - w/2, t + w/2]`` (indices clipped at the recording's
    edges, so nothing reaches across a seam). ``v``: the centroid of the seven
    keypoints. ``r``: the mean per-keypoint displacement in egocentric
    coordinates (``ego.egocentric``), i.e. posture change with heading removed.
    """
    p = np.asarray(pose, dtype=np.float64)
    T = p.shape[0]
    h = half_window(fps)
    hi = np.minimum(np.arange(T) + h, T - 1)
    lo = np.maximum(np.arange(T) - h, 0)
    dt = (hi - lo) / float(fps)
    dt = np.where(dt > 0, dt, 1.0 / float(fps))
    c = p.mean(axis=1)
    v = np.linalg.norm(c[hi] - c[lo], axis=1) / dt / float(ell)
    s = ego.egocentric(p, float(ell)).reshape(T, -1, 2)
    r = np.linalg.norm(s[hi] - s[lo], axis=2).mean(axis=1) / dt
    return np.asarray(v, dtype=np.float64), np.asarray(r, dtype=np.float64)


def bins(v: npt.ArrayLike, r: npt.ArrayLike) -> I64:
    """Flat index into the registered 40 x 40 log10 grid (clipped at its edges)."""
    edges = np.linspace(LOG_LO, LOG_HI, N_BINS + 1)

    def idx(x: npt.ArrayLike) -> I64:
        lx = np.log10(np.maximum(np.asarray(x, dtype=np.float64), 10.0 ** LOG_LO))
        return np.asarray(np.clip(np.searchsorted(edges, lx, side="right") - 1, 0, N_BINS - 1),
                          dtype=np.int64)

    return np.asarray(idx(v) * N_BINS + idx(r), dtype=np.int64)


def histogram(b: npt.ArrayLike) -> F64:
    """A normalised density over the grid, with the registered pseudocount."""
    h = np.bincount(np.asarray(b, dtype=np.int64).ravel(), minlength=N_BINS * N_BINS)
    p = h.astype(np.float64) + PSEUDO * max(float(h.sum()), 1.0)
    out = p / p.sum()
    assert abs(float(out.sum()) - 1.0) < 1e-12
    return np.asarray(out, dtype=np.float64)


def em_move(counts: npt.ArrayLike, p_freeze: npt.ArrayLike) -> tuple[F64, float, int]:
    """``(p_move, pi_freeze, iterations)``: the mixture's free component, with the
    freeze density held fixed. No temporal information is used."""
    n = np.asarray(counts, dtype=np.float64)
    pf = np.asarray(p_freeze, dtype=np.float64)
    pm = n / n.sum()
    pi = 0.5
    prev = -np.inf
    it = 0
    for it in range(1, EM_MAX + 1):
        mix = pi * pf + (1.0 - pi) * pm
        ll = float((n * np.log(mix)).sum())
        rf = pi * pf / mix
        pi = float((n * rf).sum() / n.sum())
        pm = n * (1.0 - rf)
        pm = pm + PSEUDO * max(float(pm.sum()), 1.0)
        pm = pm / pm.sum()
        if abs(ll - prev) <= EM_TOL * abs(ll):
            break
        prev = ll
    assert abs(float(pm.sum()) - 1.0) < 1e-12 and 0.0 <= pi <= 1.0
    return np.asarray(pm, dtype=np.float64), pi, it


@dataclass(frozen=True)
class Emissions:
    p_freeze: F64
    p_move: F64
    p_marginal: F64
    q_flag: float          # P(flag | freeze) = P(flag | move): the tune flag rate

    def loglik(self, b: npt.ArrayLike, flag: npt.ArrayLike) -> F64:
        """``(T, 3)`` log e_s(x_t) for freeze, move, tracking_lost."""
        bi = np.asarray(b, dtype=np.int64)
        f = np.asarray(flag, dtype=bool)
        lf = np.where(f, np.log(self.q_flag), np.log1p(-self.q_flag))
        ll_lost = np.where(f, np.log(P_FLAG_LOST), np.log1p(-P_FLAG_LOST))
        out = np.stack([np.log(self.p_freeze[bi]) + lf,
                        np.log(self.p_move[bi]) + lf,
                        np.log(self.p_marginal[bi]) + ll_lost], axis=1)
        return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------
# Parameters <-> model
# --------------------------------------------------------------------------

def n_params(family: str) -> int:
    return 3 * (1 if family == "geometric" else 2) + 3


def unpack(theta: npt.ArrayLike, family: str) -> tuple[list[tuple[float, ...]], F64]:
    """Unconstrained vector -> (per-state duration params in seconds, exit kernel)."""
    th = np.asarray(theta, dtype=np.float64)
    k = 1 if family == "geometric" else 2
    dur: list[tuple[float, ...]] = []
    for s in range(3):
        a = th[s * k:(s + 1) * k]
        if family == "geometric":
            dur.append((float(np.exp(a[0])),))
        elif family == "weibull":
            dur.append((float(np.exp(a[0])), float(np.exp(a[1]))))
        else:
            dur.append((float(a[0]), float(np.exp(a[1]))))
    q = 1.0 / (1.0 + np.exp(-th[3 * k:3 * k + 3]))
    ex = np.zeros((3, 3))
    for s in range(3):
        others = [j for j in range(3) if j != s]
        ex[s, others[0]], ex[s, others[1]] = q[s], 1.0 - q[s]
    return dur, ex


def build(theta: npt.ArrayLike, family: str, fps: float) -> core.HSMM:
    dur, ex = unpack(theta, family)
    D = d_max(fps)
    hz = np.stack([core.hazard_from_family(family, d, D, fps) for d in dur])
    return core.HSMM(hazard=hz, exit=ex, init=np.full(3, 1.0 / 3.0))


def theta0(family: str) -> F64:
    """Registered starting point: means of 3 s (freeze), 2 s (move), 1 s (lost)."""
    means = (3.0, 2.0, 1.0)
    if family == "geometric":
        d = [np.log(m) for m in means]
    elif family == "weibull":
        d = [x for m in means for x in (0.0, np.log(m))]
    else:
        d = [x for m in means for x in (np.log(m) - 0.5, np.log(1.0))]
    return np.asarray(d + [0.0, 0.0, 0.0], dtype=np.float64)


def mean_ll_per_frame(model: core.HSMM, log_e: Sequence[npt.ArrayLike]) -> float:
    ll, _f, _s = core.forward(model, log_e)
    n = sum(np.asarray(x).shape[0] for x in log_e)
    return float(ll.sum() / n)
