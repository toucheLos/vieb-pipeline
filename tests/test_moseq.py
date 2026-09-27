"""MOSEQ §3: the forward filter and the AR baseline."""
from __future__ import annotations

import numpy as np

from vieb.moseq import forward as fw


def _params(K, d, L, rng):
    Ab = rng.normal(0, 0.2, (K, d, d * L + 1))
    Q = np.stack([np.eye(d) * (0.5 + k) for k in range(K)])
    pi = rng.dirichlet(np.ones(K) * 2, size=K)
    return Ab, Q, pi


def test_forward_filter_equals_brute_force_marginal():
    rng = np.random.default_rng(0)
    Ab, Q, pi = _params(2, 2, 2, rng)
    x = rng.normal(size=(8, 2))                # 6 scored frames, 2^6 paths
    p0 = fw.stationary(pi)
    f = fw.forward(fw.gauss_ll(x, Ab, Q), pi, p0).sum()
    assert np.isclose(f, fw.log_marginal_brute(x, Ab, Q, pi, p0))


def test_design_matches_jax_moseq_lag_order():
    x = np.arange(12, dtype=float).reshape(6, 2)
    phi = fw.design(x, 3)
    # row 0 predicts x[3] from [x0, x1, x2, 1]: oldest first, bias last
    assert np.array_equal(phi[0], [0, 1, 2, 3, 4, 5, 1])


def test_one_state_hmm_is_the_ar_baseline():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(400, 3)), axis=0) * 0.1
    run_list = [(0, 0, 400)]
    A, Qa = fw.fit_ar(x, run_list, 3)
    m, b, _ = fw.score(x, run_list, A[None], Qa[None], np.ones((1, 1)), A, Qa)
    assert np.allclose(m, b)


def test_switching_data_gains_and_a_linear_process_does_not():
    """The gate's premise: a 2-regime AR gains > 0.05 nats/frame over one AR;
    a single linear AR, scored by a 2-state model of itself, gains ~0."""
    rng = np.random.default_rng(2)
    d, L, T = 10, 3, 20000
    A1 = np.zeros((d, d * L + 1)); A1[:, (L - 1) * d:L * d] = 0.9 * np.eye(d)
    A2 = np.zeros((d, d * L + 1)); A2[:, (L - 1) * d:L * d] = -0.5 * np.eye(d)
    Q = np.stack([np.eye(d) * 0.1, np.eye(d) * 1.0])
    pi = np.array([[0.98, 0.02], [0.02, 0.98]])
    z = np.zeros(T, int)
    for t in range(1, T):
        z[t] = z[t - 1] if rng.random() < 0.98 else 1 - z[t - 1]
    x = np.zeros((T, d))
    for t in range(L, T):
        A = (A1, A2)[z[t]]
        x[t] = A @ np.r_[x[t - L:t].ravel(), 1] + rng.multivariate_normal(np.zeros(d), Q[z[t]])
    tr, te = [(0, 0, T // 2)], [(0, T // 2, T)]
    ar = fw.fit_ar(x, tr, L)
    m, b, _ = fw.score(x, te, np.stack([A1, A2]), Q, pi, *ar)
    assert (m - b).mean() > 0.05
    y = np.zeros((T, d))
    for t in range(L, T):
        y[t] = 0.7 * y[t - 1] + rng.normal(size=d) * 0.5
    ar_y = fw.fit_ar(y, tr, L)
    Ab2 = np.stack([ar_y[0], ar_y[0]])
    m2, b2, _ = fw.score(y, te, Ab2, np.stack([ar_y[1], ar_y[1]]), pi, *ar_y)
    assert abs((m2 - b2).mean()) < 1e-9


def test_runs_break_at_unobserved_frames_and_drop_short_ones():
    br = np.zeros(200, bool); br[50] = True; br[90:95] = True
    r = fw.runs(br, [0, 120, 200], min_len=30)
    assert r == [(0, 0, 50), (0, 51, 90), (1, 120, 200)]
