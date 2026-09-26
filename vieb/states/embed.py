"""Delay embedding within recordings. STATES §3.1.

x_K(t) = [x(t), x(t-1), ..., x(t-K+1)], valid only where all K frames are valid
and inside one recording. Rows are produced one recording at a time so the
corpus never has to sit in memory at 21 x 32 dimensions.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
B1 = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]


def embed_one(x: npt.ArrayLike, valid: npt.ArrayLike, k: int
              ) -> tuple[F32, B1]:
    """(``(T, K*d)`` embedding, ``(T,)`` validity) for ONE recording."""
    a = np.asarray(x, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    t, d = a.shape
    out = np.zeros((t, k * d), dtype=np.float32)
    ok = v.copy()
    for lag in range(k):
        out[lag:, lag * d:(lag + 1) * d] = a[:t - lag] if lag else a
        if lag:
            ok[:lag] = False
            ok[lag:] &= v[:t - lag]
    return out, ok


def recordings(x: npt.ArrayLike, valid: npt.ArrayLike, bounds: npt.ArrayLike,
               k: int) -> Iterator[tuple[int, F32, B1]]:
    """Yield (recording index, embedding, validity) per recording."""
    b = np.asarray(bounds, dtype=np.int64)
    a = np.asarray(x)
    v = np.asarray(valid, dtype=bool)
    for i in range(b.size - 1):
        e, ok = embed_one(a[b[i]:b[i + 1]], v[b[i]:b[i + 1]], k)
        yield i, e, ok


def sample_rows(x: npt.ArrayLike, valid: npt.ArrayLike, bounds: npt.ArrayLike,
                k: int, n: int, rng: np.random.Generator) -> F32:
    """Up to `n` valid embedded rows drawn uniformly over valid frames."""
    b = np.asarray(bounds, dtype=np.int64)
    v = np.asarray(valid, dtype=bool)
    counts = []
    for i in range(b.size - 1):
        _, ok = embed_one(np.zeros((b[i + 1] - b[i], 1), np.float32),
                          v[b[i]:b[i + 1]], k)
        counts.append(int(ok.sum()))
    total = int(sum(counts))
    if total == 0:
        return np.zeros((0, np.asarray(x).shape[1] * k), dtype=np.float32)
    frac = min(1.0, n / total)
    rows = []
    for i, e, ok in recordings(x, valid, bounds, k):
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        take = rng.random(idx.size) < frac
        rows.append(e[idx[take]])
    return np.concatenate(rows) if rows else np.zeros((0, np.asarray(x).shape[1] * k),
                                                      dtype=np.float32)
