"""MOSEQ §3: exact one-step predictive log-likelihood of a fitted AR-HMM, and
the single-regime AR baseline it is scored against.

Conventions are jax_moseq's (`utils/autoregression.py`): an AR(L) row regresses
x_t on ``[x_{t-L}, ..., x_{t-1}, 1]`` -- oldest lag first, bias last -- so
``Ab`` is ``(K, d, d*L + 1)`` and ``Q`` is ``(K, d, d)``.

The forward filter is scored on unbroken runs only. A run is a stretch of
frames with every keypoint observed; the first `L` frames of a run are
predictors, never predictions, and the filter restarts from the initial
distribution at every run.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

Run = tuple[int, int, int]

F64 = npt.NDArray[np.float64]

MIN_RUN = 30


def design(x: npt.ArrayLike, nlags: int) -> F64:
    """``(T - L, d*L + 1)``: the lagged predictors for x[L:]."""
    a = np.asarray(x, dtype=np.float64)
    T = a.shape[0]
    cols = [a[nlags - j:T - j] for j in range(nlags, 0, -1)]
    return np.concatenate(cols + [np.ones((T - nlags, 1))], axis=1)


def gauss_ll(x: npt.ArrayLike, Ab: npt.ArrayLike, Q: npt.ArrayLike) -> F64:
    """``(T - L, K)``: log N(x_t | Ab_k phi_t, Q_k) for every state k."""
    a = np.asarray(x, dtype=np.float64)
    A = np.asarray(Ab, dtype=np.float64)
    if A.ndim == 2:
        A, Qs = A[None], np.asarray(Q, dtype=np.float64)[None]
    else:
        Qs = np.asarray(Q, dtype=np.float64)
    d = A.shape[1]
    L = (A.shape[2] - 1) // d
    phi = design(a, L)
    tgt = a[L:]
    out = np.empty((tgt.shape[0], A.shape[0]))
    for k in range(A.shape[0]):
        r = tgt - phi @ A[k].T
        c = np.linalg.cholesky(Qs[k])
        z = np.linalg.solve(c, r.T)
        out[:, k] = (-0.5 * (z * z).sum(0) - np.log(np.diag(c)).sum()
                     - 0.5 * d * np.log(2 * np.pi))
    return out


def stationary(pi: npt.ArrayLike) -> F64:
    p = np.asarray(pi, dtype=np.float64)
    w, v = np.linalg.eig(p.T)
    s = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    return np.asarray(s / s.sum(), dtype=np.float64)


def forward(ll: npt.ArrayLike, pi: npt.ArrayLike, p0: npt.ArrayLike) -> F64:
    """``(T,)``: log p(x_t | x_{<t}) under the HMM with emission logs `ll`."""
    l = np.asarray(ll, dtype=np.float64)
    P = np.asarray(pi, dtype=np.float64)
    pred = np.asarray(p0, dtype=np.float64)
    out = np.empty(l.shape[0])
    for t in range(l.shape[0]):
        m = l[t].max()
        w = pred * np.exp(l[t] - m)
        s = w.sum()
        out[t] = m + np.log(s)
        pred = (w / s) @ P
    return out


def runs(broken: npt.ArrayLike, bounds: npt.ArrayLike,
         min_len: int = MIN_RUN) -> list[Run]:
    """``(recording index, lo, hi)`` of every unbroken run of >= `min_len` frames."""
    br = np.asarray(broken, dtype=bool)
    b = np.asarray(bounds, dtype=np.int64)
    out: list[Run] = []
    for i in range(b.size - 1):
        seg = ~br[b[i]:b[i + 1]]
        edges = np.flatnonzero(np.diff(np.r_[0, seg.astype(np.int8), 0]))
        for lo, hi in zip(edges[::2], edges[1::2]):
            if hi - lo >= min_len:
                out.append((i, int(b[i] + lo), int(b[i] + hi)))
    return out


def fit_ar(x: npt.ArrayLike, run_list: list[Run], nlags: int) -> tuple[F64, F64]:
    """One AR(`nlags`) + bias, full covariance, least squares over the runs."""
    a = np.asarray(x, dtype=np.float64)
    phis, tgts = [], []
    for _, lo, hi in run_list:
        phis.append(design(a[lo:hi], nlags))
        tgts.append(a[lo + nlags:hi])
    phi, tgt = np.concatenate(phis), np.concatenate(tgts)
    beta, *_ = np.linalg.lstsq(phi, tgt, rcond=None)
    r = tgt - phi @ beta
    return np.asarray(beta.T, np.float64), np.asarray(np.cov(r.T, bias=True), np.float64)


def score(x: npt.ArrayLike, run_list: list[Run], Ab: npt.ArrayLike,
          Q: npt.ArrayLike, pi: npt.ArrayLike, ar_Ab: npt.ArrayLike,
          ar_Q: npt.ArrayLike) -> tuple[F64, F64, npt.NDArray[np.int64]]:
    """Per scored frame: (MoSeq log-lik, AR baseline log-lik, recording index)."""
    a = np.asarray(x, dtype=np.float64)
    p0 = stationary(pi)
    ms, bs, rec = [], [], []
    for i, lo, hi in run_list:
        seg = a[lo:hi]
        ms.append(forward(gauss_ll(seg, Ab, Q), pi, p0))
        bs.append(gauss_ll(seg, ar_Ab, ar_Q)[:, 0])
        rec.append(np.full(ms[-1].size, i, dtype=np.int64))
    if not ms:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.concatenate(ms), np.concatenate(bs), np.concatenate(rec)


def log_marginal_brute(x: npt.ArrayLike, Ab: npt.ArrayLike, Q: npt.ArrayLike,
                       pi: npt.ArrayLike, p0: npt.ArrayLike) -> float:
    """Test oracle: log p(x_L:) by enumerating every state path (tiny K, T)."""
    import itertools

    ll = gauss_ll(x, Ab, Q)
    T, K = ll.shape
    P = np.asarray(pi, dtype=np.float64)
    terms = []
    for path in itertools.product(range(K), repeat=T):
        p0a = np.asarray(p0, dtype=np.float64)
        lp = np.log(p0a[path[0]]) + ll[0, path[0]]
        for t in range(1, T):
            lp += np.log(P[path[t - 1], path[t]]) + ll[t, path[t]]
        terms.append(lp)
    return float(logsumexp(terms))
