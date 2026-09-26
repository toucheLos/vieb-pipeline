"""`vieb.states`: embedding, microstates and the operator on known systems."""
from __future__ import annotations

import numpy as np

from vieb.states import embed, micro, operator as op


def test_embed_one_stacks_lags_and_invalidates_the_start_and_gaps():
    x = np.arange(10, dtype=np.float32)[:, None]
    v = np.ones(10, dtype=bool)
    v[5] = False
    e, ok = embed.embed_one(x, v, 3)
    assert e.shape == (10, 3)
    assert list(e[4]) == [4, 3, 2]
    assert not ok[:2].any() and ok[2:5].all() and not ok[5:8].any() and ok[8:].all()


def test_counts_respect_recording_bounds_and_invalid_labels():
    lab = np.array([0, 1, 1, -1, 0, 0, 1])
    c = micro.counts(lab, np.array([0, 4, 7]), 1, 2)
    # rec 1: 0->1, 1->1 (1->-1 dropped); rec 2: 0->0, 0->1
    assert c.tolist() == [[1, 2], [0, 1]]


def _double_well(n=60000, dt=0.01, sigma=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = -1.0
    for t in range(1, n):
        f = -4 * x[t - 1] * (x[t - 1] ** 2 - 1)
        x[t] = x[t - 1] + f * dt + sigma * np.sqrt(dt) * rng.normal()
    noise = rng.normal(0, 0.3, n)
    return np.column_stack([x, noise]).astype(np.float32)


def _pipeline(x, n_micro=60, tau=20):
    v = np.ones(x.shape[0], dtype=bool)
    b = np.array([0, x.shape[0]])
    rows = embed.sample_rows(x, v, b, 1, 20000, np.random.default_rng(0))
    m = micro.fit(rows, 1, n_micro=n_micro, n_pcs=2)
    _, e, ok = next(embed.recordings(x, v, b, 1))
    lab = m.assign(e, ok)
    c = micro.counts(lab, b, tau, n_micro)
    p, keep = op.transition_matrix(c)
    return lab, p, keep


def test_double_well_has_one_slow_timescale_and_two_metastable_sets():
    x = _double_well()
    lab, p, keep = _pipeline(x)
    ts = op.timescales(p, 20, k=4)
    assert ts[0] > 5 * ts[1]                    # a clear gap after t_2
    chi = op.pcca(op.reversibilise(p), 2)
    st = op.assign(lab, keep, chi, threshold=0.8)
    well = x[:, 0] > 0
    ok = st >= 0
    agree = np.mean((st[ok] == st[ok].max()) == well[ok])
    assert max(agree, 1 - agree) > 0.95
    assert ok.mean() > 0.6


def test_white_noise_has_no_slow_timescale():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60000, 2)).astype(np.float32)
    _, p, _ = _pipeline(x)
    ts = op.timescales(p, 20, k=4)
    assert ts[0] < 20                           # decays within one lag


def test_cross_entropy_prefers_the_true_model():
    rng = np.random.default_rng(2)
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    s = [0]
    for _ in range(20000):
        s.append(int(rng.random() > p[s[-1], 0]))
    lab = np.asarray(s)
    c = micro.counts(lab, np.array([0, lab.size]), 1, 2)
    flat = np.ones((2, 2)) * 1000
    assert micro.cross_entropy(c, c) < micro.cross_entropy(flat, c)
