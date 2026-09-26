"""The transfer operator, its timescales, and metastable sets. STATES §3.5, §4.

Timescales come from the (non-reversible) transition matrix P(tau):
t_i = -tau / ln|lambda_i|. Metastable sets come from PCCA+ (inner-simplex
algorithm) on the eigenvectors of the REVERSIBILISED operator
P_r = (P + Pi^-1 P^T Pi) / 2, which has a real spectrum -- the construction
Costa & Stephens use for coarse-graining.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def transition_matrix(counts: npt.ArrayLike) -> tuple[F64, npt.NDArray[np.bool_]]:
    """(row-normalised P on the largest strongly connected set, which states).

    STATES Amendment 1: states outside the largest strongly connected set are
    dropped -- otherwise a closed island of microstates contributes an
    eigenvalue of exactly 1 and an infinite 'timescale' that is only a
    disconnection. Standard Markov-state-model practice.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    c = np.asarray(counts, dtype=np.float64)
    n, lab = connected_components(csr_matrix((c > 0).astype(np.int8)), directed=True,
                                  connection="strong")
    if n > 1:
        sizes = np.bincount(lab, weights=c.sum(axis=1), minlength=n)
        keep = lab == int(np.argmax(sizes))
    else:
        keep = np.ones(c.shape[0], dtype=bool)
    keep &= (c.sum(axis=1) > 0)
    c = c[np.ix_(keep, keep)]
    rs = c.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return c / rs, keep


def gap_ratios(ts: npt.ArrayLike) -> F64:
    """r_i = t_i / t_(i+1): a metastable gap is a ratio far above the ladder a
    linear process gives (about 2 for one OU coordinate). Amendment 1."""
    t = np.asarray(ts, dtype=np.float64)
    return np.asarray(t[:-1] / np.where(t[1:] > 0, t[1:], np.nan), dtype=np.float64)


def stationary(p: npt.ArrayLike) -> F64:
    w, v = np.linalg.eig(np.asarray(p).T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.abs(np.real(v[:, i]))
    return np.asarray(pi / pi.sum(), dtype=np.float64)


def timescales(p: npt.ArrayLike, tau: int, k: int = 11) -> F64:
    """t_2..t_k in frames: -tau / ln|lambda_i|, eigenvalues by modulus."""
    w = np.linalg.eigvals(np.asarray(p, dtype=np.float64))
    m = np.sort(np.abs(w))[::-1][1:k]
    m = np.clip(m, 1e-12, 1 - 1e-12)
    return np.asarray(-tau / np.log(m), dtype=np.float64)


def reversibilise(p: npt.ArrayLike) -> F64:
    pm = np.asarray(p, dtype=np.float64)
    pi = stationary(pm)
    return np.asarray(0.5 * (pm + (pm.T * pi[None, :]) / pi[:, None]),
                      dtype=np.float64)


def pcca(p_rev: npt.ArrayLike, m: int) -> F64:
    """PCCA+ memberships (``(N, m)``) by the inner-simplex algorithm.

    First m right eigenvectors of the reversibilised operator (real spectrum),
    vertices chosen greedily as the rows farthest from the span of those
    already chosen, chi = V inv(V[vertices]), clipped to [0, 1] and
    renormalised.
    """
    pr = np.asarray(p_rev, dtype=np.float64)
    w, v = np.linalg.eig(pr)
    order = np.argsort(-np.real(w))
    vv = np.real(v[:, order[:m]]).copy()
    vv[:, 0] = 1.0            # the stationary right eigenvector is constant
    x = vv.copy()
    idx = [int(np.argmax(np.linalg.norm(x[:, 1:], axis=1)))] if m > 1 else [0]
    for _ in range(1, m):
        base = x[idx]
        q, _ = np.linalg.qr(base.T)
        resid = x - (x @ q) @ q.T
        idx.append(int(np.argmax(np.linalg.norm(resid, axis=1))))
    a = np.linalg.pinv(vv[idx])
    chi = np.clip(vv @ a, 0.0, 1.0)
    chi /= np.where(chi.sum(axis=1, keepdims=True) > 0,
                    chi.sum(axis=1, keepdims=True), 1.0)
    return np.asarray(chi, dtype=np.float64)


def assign(labels: npt.ArrayLike, keep: npt.ArrayLike, chi: npt.ArrayLike,
           *, threshold: float = 0.8) -> I64:
    """Frame states: the state whose membership >= threshold, else -1."""
    lab = np.asarray(labels, dtype=np.int64)
    kp = np.asarray(keep, dtype=bool)
    ch = np.asarray(chi, dtype=np.float64)
    full = np.full(kp.size, -1, dtype=np.int64)
    top = ch.argmax(axis=1)
    conf = ch.max(axis=1)
    full[np.flatnonzero(kp)] = np.where(conf >= threshold, top, -1)
    out = np.full(lab.size, -1, dtype=np.int64)
    ok = lab >= 0
    out[ok] = full[lab[ok]]
    return out
