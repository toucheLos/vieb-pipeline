"""Microstates: PCA then k-means on delay-embedded rows. STATES §3.2-3.4.

Fitted on a uniform sample of tune rows; every recording is then assigned
chunk by chunk. Invalid frames get label -1 and never enter a transition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

N_PCS = 30
# The override exists only for the synthetic smoke test; a real run uses 1000.
N_MICRO = int(os.environ.get("VIEB_STATES_NMICRO", 1000))


@dataclass
class Micro:
    pca: Any
    km: Any
    k: int

    def assign(self, emb: npt.ArrayLike, ok: npt.ArrayLike) -> I64:
        e = np.asarray(emb, dtype=np.float32)
        v = np.asarray(ok, dtype=bool)
        lab = np.full(e.shape[0], -1, dtype=np.int64)
        if v.any():
            lab[v] = self.km.predict(self.pca.transform(e[v]))
        return lab


def fit(rows: npt.ArrayLike, k: int, *, n_micro: int = N_MICRO,
        n_pcs: int = N_PCS, seed: int = 0) -> Micro:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    r = np.asarray(rows, dtype=np.float32)
    pca = PCA(n_components=min(n_pcs, r.shape[1]), random_state=seed).fit(r)
    z = pca.transform(r)
    km = MiniBatchKMeans(n_clusters=n_micro, random_state=seed, batch_size=8192,
                         n_init=3, max_iter=200).fit(z)
    return Micro(pca=pca, km=km, k=k)


def counts(labels: npt.ArrayLike, bounds: npt.ArrayLike, tau: int,
           n: int) -> F64:
    """Transition counts at lag tau, within recordings, valid pairs only."""
    lab = np.asarray(labels, dtype=np.int64)
    b = np.asarray(bounds, dtype=np.int64)
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(b.size - 1):
        s = lab[b[i]:b[i + 1]]
        if s.size <= tau:
            continue
        a, z = s[:-tau], s[tau:]
        ok = (a >= 0) & (z >= 0)
        np.add.at(c, (a[ok], z[ok]), 1.0)
    return c


def cross_entropy(train_counts: npt.ArrayLike, test_counts: npt.ArrayLike,
                  *, alpha: float = 1.0) -> float:
    """Held-out next-microstate cross-entropy (nats/transition). STATES §3.4.

    The transition matrix from `train_counts` with add-`alpha` smoothing, so an
    unseen transition costs a finite amount; the same smoothing for every K.
    """
    tr = np.asarray(train_counts, dtype=np.float64) + alpha
    p = tr / tr.sum(axis=1, keepdims=True)
    te = np.asarray(test_counts, dtype=np.float64)
    n = te.sum()
    return float(-(te * np.log(p)).sum() / n) if n else float("nan")


def predictive_r2(lab_tr: npt.ArrayLike, fut_tr: npt.ArrayLike,
                  lab_te: npt.ArrayLike, fut_te: npt.ArrayLike, n: int) -> float:
    """Held-out share of the FUTURE features' variance explained by the state.

    STATES Amendment 1: K is chosen by how well the microstate at t predicts the
    features at t + h, beyond the end of its window. Next-microstate
    cross-entropy rewards long windows trivially, because consecutive windows
    overlap by K - 1 frames. Conditional means are fitted on `*_tr`; R^2 is
    scored on `*_te`; rows with label -1 or a non-finite future are skipped.
    """
    lt, ft = np.asarray(lab_tr), np.asarray(fut_tr, dtype=np.float64)
    le, fe = np.asarray(lab_te), np.asarray(fut_te, dtype=np.float64)
    okt = (lt >= 0) & np.isfinite(ft).all(axis=1)
    oke = (le >= 0) & np.isfinite(fe).all(axis=1)
    d = ft.shape[1]
    sums = np.zeros((n, d))
    cnt = np.zeros(n)
    np.add.at(sums, lt[okt], ft[okt])
    np.add.at(cnt, lt[okt], 1.0)
    mean_all = ft[okt].mean(axis=0)
    cond = np.where(cnt[:, None] > 0, sums / np.maximum(cnt[:, None], 1), mean_all)
    pred = cond[le[oke]]
    resid = ((fe[oke] - pred) ** 2).sum()
    total = ((fe[oke] - fe[oke].mean(axis=0)) ** 2).sum()
    return float(1.0 - resid / total) if total > 0 else float("nan")
