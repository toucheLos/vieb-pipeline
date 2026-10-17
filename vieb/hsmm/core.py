r"""A hidden semi-Markov model with explicit durations: filtering and smoothing.

Domain-agnostic: nothing here knows about mice. The interface is three pieces:

* **emissions** -- ``log e_s(x_t)``, an array ``(T, S)`` per sequence;
* **hazard** -- ``h_s(tau)``, ``(S, D)``: the probability of leaving state ``s``
  after exactly ``tau = d + 1`` frames, given that it has lasted that long;
* **exit kernel** -- ``P(j | s)``, ``(S, S)`` with a zero diagonal.

The latent state is the pair (state, elapsed frames). Elapsed time is capped at
``D`` frames. At the cap the hazard is held constant (a geometric tail), so no
probability mass is lost and the cap only coarsens the very long tail:

    stay:  alpha_t(s, d)    = e_s(x_t) alpha_{t-1}(s, d-1) (1 - h_s(d-1))
           alpha_t(s, D-1) += e_s(x_t) alpha_{t-1}(s, D-1) (1 - h_s(D-1))
    enter: alpha_t(j, 0)    = e_j(x_t) sum_{s,d} alpha_{t-1}(s, d) h_s(d) P(j | s)

Each step is normalised (the scale is the one-step predictive likelihood), and the
normalisation is asserted. **Every sequence starts from the initial distribution
at elapsed time one.** Sequences are batched for speed, padded and masked, and
never interact: a sequence's result does not depend on any other in the batch
(`tests/test_hsmm.py` checks this to the bit).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

F64 = npt.NDArray[np.float64]

TOL = 1e-9
H_MIN, H_MAX = 1e-12, 1.0 - 1e-12


@dataclass(frozen=True)
class HSMM:
    hazard: F64          # (S, D)
    exit: F64            # (S, S), zero diagonal, rows sum to 1
    init: F64            # (S,)

    def __post_init__(self) -> None:
        h, e, i = self.hazard, self.exit, self.init
        S = h.shape[0]
        assert h.ndim == 2 and e.shape == (S, S) and i.shape == (S,)
        assert np.all((h >= 0) & (h <= 1)), "hazard outside [0, 1]"
        assert np.allclose(np.diag(e), 0.0), "exit kernel must have a zero diagonal"
        if S > 1:
            assert np.allclose(e.sum(1), 1.0, atol=1e-12), "exit kernel rows must sum to 1"
        assert abs(float(i.sum()) - 1.0) < 1e-12, "initial distribution must sum to 1"

    @property
    def S(self) -> int:
        return int(self.hazard.shape[0])

    @property
    def D(self) -> int:
        return int(self.hazard.shape[1])


# --------------------------------------------------------------------------
# Duration families -> discrete hazards
# --------------------------------------------------------------------------

FAMILIES = ("geometric", "weibull", "lognormal")


def hazard_from_family(family: str, params: Sequence[float], D: int, fps: float) -> F64:
    """``(D,)`` discrete hazard for a duration law given in SECONDS.

    `geometric`: (mean_s,); `weibull`: (shape k, scale lambda_s); `lognormal`:
    (mu, sigma) of log seconds. P(dur = tau) = F(tau/fps) - F((tau-1)/fps), and
    h(tau) = P(dur = tau) / P(dur >= tau).
    """
    tau = np.arange(1, D + 1, dtype=np.float64)
    if family == "geometric":
        (mean_s,) = params
        p = 1.0 / max(float(mean_s) * fps, 1.0)
        return np.asarray(np.clip(np.full(D, p), H_MIN, H_MAX), dtype=np.float64)
    dist: Any
    if family == "weibull":
        k, lam = (float(v) for v in params)
        dist = stats.weibull_min(c=k, scale=lam)
    elif family == "lognormal":
        mu, sig = (float(v) for v in params)
        dist = stats.lognorm(s=sig, scale=float(np.exp(mu)))
    else:
        raise ValueError(family)
    s_prev = dist.sf((tau - 1.0) / fps)
    s_now = dist.sf(tau / fps)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(s_prev > 0, (s_prev - s_now) / s_prev, 1.0)
    return np.asarray(np.clip(np.nan_to_num(h, nan=1.0), H_MIN, H_MAX), dtype=np.float64)


# --------------------------------------------------------------------------
# Batched forward / backward
# --------------------------------------------------------------------------

def _pad(log_e: Sequence[npt.ArrayLike]) -> tuple[F64, npt.NDArray[np.int64]]:
    lens = np.asarray([np.asarray(x).shape[0] for x in log_e], dtype=np.int64)
    S = np.asarray(log_e[0]).shape[1]
    out = np.zeros((len(log_e), int(lens.max()), S), dtype=np.float64)
    for i, x in enumerate(log_e):
        a = np.asarray(x, dtype=np.float64)
        assert a.shape[1] == S and np.all(np.isfinite(a) | (a == -np.inf))
        out[i, :a.shape[0]] = a
    return out, lens


def _emit(le: F64) -> tuple[F64, F64]:
    """``(probabilities scaled to max 1, the log shift)`` per sequence row."""
    m = le.max(axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.exp(le - m), m[:, 0]


def _forward_batch(model: HSMM, E: F64, lens: npt.NDArray[np.int64], keep: bool
                   ) -> tuple[F64, F64, F64, F64 | None]:
    """Scaled forward over a padded batch: ``(step_ll (R,T), filt (R,T,S), last alpha,
    all alphas (R,T,S,D) if keep)``."""
    R, Tm, S = E.shape
    D = model.D
    h, ex = model.hazard, model.exit
    stay = 1.0 - h
    alpha = np.zeros((R, S, D))
    filt = np.zeros((R, Tm, S))
    step_ll = np.zeros((R, Tm))
    alphas = np.zeros((R, Tm, S, D)) if keep else None
    for t in range(Tm):
        live = t < lens
        e, shift = _emit(E[:, t, :])
        pred = np.zeros((R, S, D))
        if t == 0:
            pred[:, :, 0] = model.init[None, :]
        else:
            pred[:, :, 1:] = alpha[:, :, :-1] * stay[None, :, :-1]
            pred[:, :, D - 1] += alpha[:, :, D - 1] * stay[None, :, D - 1]
            leave = (alpha * h[None]).sum(axis=2)                # (R, S)
            # Elementwise, not `leave @ ex`: BLAS picks code paths by batch size,
            # which would make a sequence's bits depend on its batch-mates.
            pred[:, :, 0] = (leave[:, :, None] * ex[None, :, :]).sum(axis=1)
        a = pred * e[:, :, None]
        c = a.sum(axis=(1, 2))
        if np.any(c[live] <= 0):
            raise FloatingPointError(f"zero likelihood at t={t}")
        csafe = np.where(c > 0, c, 1.0)
        alpha = np.where(live[:, None, None], a / csafe[:, None, None], alpha)
        tot = alpha[live].sum(axis=(1, 2))
        assert np.all(np.abs(tot - 1.0) < TOL), "forward normalisation"
        step_ll[:, t] = np.where(live, np.log(csafe) + shift, 0.0)
        filt[:, t, :] = alpha.sum(axis=2)
        if alphas is not None:
            alphas[:, t] = alpha
    return step_ll, filt, alpha, alphas


def forward(model: HSMM, log_e: Sequence[npt.ArrayLike]) -> tuple[F64, list[F64], list[F64]]:
    """Filter every sequence (batched; sequences never interact).

    Returns ``(loglik per sequence, per-sequence (T, S) filtered state posteriors,
    per-sequence (T,) one-step predictive log-likelihoods)``.
    """
    E, lens = _pad(log_e)
    step_ll, filt, _a, _al = _forward_batch(model, E, lens, keep=False)
    R = E.shape[0]
    # Each total over its own length: a padded row's trailing zeros would change
    # the pairwise summation order, and with it the last bits.
    return (np.asarray([step_ll[i, :lens[i]].sum() for i in range(R)], dtype=np.float64),
            [filt[i, :lens[i]] for i in range(R)],
            [step_ll[i, :lens[i]] for i in range(R)])


def _smooth_one(model: HSMM, le: npt.ArrayLike) -> tuple[float, F64]:
    E, lens = _pad([le])
    step_ll, _f, _a, alphas = _forward_batch(model, E, lens, keep=True)
    assert alphas is not None
    T = int(lens[0])
    h, ex = model.hazard, model.exit
    stay = 1.0 - h
    S, D = model.S, model.D
    beta = np.ones((S, D))
    post = np.zeros((T, S))
    for t in range(T - 1, -1, -1):
        g = alphas[0, t] * beta
        z = float(g.sum())
        assert z > 0, f"zero posterior mass at t={t}"
        post[t] = (g / z).sum(axis=1)
        if t == 0:
            break
        e, _shift = _emit(E[:, t, :])
        eb = e[0][:, None] * beta                              # e_s(x_t) beta_t(s, d)
        nxt = np.empty((S, D))
        nxt[:, :-1] = stay[:, :-1] * eb[:, 1:]
        nxt[:, D - 1] = stay[:, D - 1] * eb[:, D - 1]
        enter = ex @ eb[:, 0]                                  # sum_j P(j|s) e_j beta(j, 0)
        nxt += h * enter[:, None]
        beta = nxt / max(float(nxt.max()), 1e-300)
    assert np.all(np.abs(post.sum(axis=1) - 1.0) < 1e-6), "smoothed normalisation"
    return float(step_ll[0, :T].sum()), post


def smooth(model: HSMM, log_e: Sequence[npt.ArrayLike]) -> tuple[F64, list[F64]]:
    """``(loglik per sequence, per-sequence (T, S) smoothed posteriors)``.

    One sequence at a time: the backward pass needs that sequence's full
    (T, S, D) forward array, and nothing else.
    """
    res = [_smooth_one(model, le) for le in log_e]
    return np.asarray([r[0] for r in res], dtype=np.float64), [r[1] for r in res]


# --------------------------------------------------------------------------
# Reference implementations for tests
# --------------------------------------------------------------------------

def brute_loglik(model: HSMM, log_e: npt.ArrayLike) -> float:
    """log p(x_1..T) by enumerating every state path (tiny T only)."""
    import itertools
    E = np.exp(np.asarray(log_e, dtype=np.float64))
    T, S = E.shape
    D = model.D
    total = 0.0
    for path in itertools.product(range(S), repeat=T):
        p = model.init[path[0]] * E[0, path[0]]
        d = 0
        for t in range(1, T):
            s, j = path[t - 1], path[t]
            if j == s:
                p *= 1.0 - model.hazard[s, d]
                d = min(d + 1, D - 1)
            else:
                p *= model.hazard[s, d] * model.exit[s, j]
                d = 0
            p *= E[t, j]
        total += p
    return float(np.log(total))


def as_hmm(model: HSMM) -> tuple[F64, F64]:
    """``(transition, initial)`` of the equivalent HMM when the hazard is constant."""
    h = model.hazard[:, 0]
    assert np.allclose(model.hazard, h[:, None]), "hazard is not constant"
    P = model.exit * h[:, None]
    P[np.diag_indices_from(P)] = 1.0 - h
    return np.asarray(P, dtype=np.float64), np.asarray(model.init, dtype=np.float64)


def runs(state: npt.ArrayLike) -> list[tuple[int, int, int]]:
    """``[(state, start, stop)]`` of the runs of an integer state sequence."""
    a = np.asarray(state)
    if a.size == 0:
        return []
    cut = np.flatnonzero(np.diff(a)) + 1
    starts = np.r_[0, cut]
    stops = np.r_[cut, a.size]
    return [(int(a[s]), int(s), int(e)) for s, e in zip(starts, stops)]
