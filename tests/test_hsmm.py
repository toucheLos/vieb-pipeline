"""STEP 4 HSMM core: brute force, HMM equivalence, normalisation, seams, planted bouts."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.hsmm import core
from vieb.moseq import forward as hmm


def model(S=2, D=4, family="weibull", params=None, seed=0, fps=30.0):
    rng = np.random.default_rng(seed)
    params = params or [(1.5, 0.08), (1.2, 0.05), (1.0, 0.06)][:S]
    hz = np.stack([core.hazard_from_family(family, p if family != "geometric" else (p[1],), D, fps)
                   for p in params])
    ex = rng.random((S, S))
    np.fill_diagonal(ex, 0)
    ex = ex / ex.sum(1, keepdims=True) if S > 1 else ex
    return core.HSMM(hazard=hz, exit=ex, init=np.full(S, 1.0 / S))


def emissions(T, S, seed=0):
    return np.log(np.random.default_rng(seed).random((T, S)) + 0.05)


@pytest.mark.parametrize("S,D,T", [(2, 3, 6), (3, 2, 5), (2, 5, 7)])
def test_forward_matches_brute_force(S, D, T):
    m = model(S=S, D=D)
    le = emissions(T, S, seed=S + D + T)
    ll, filt, _ = core.forward(m, [le])
    assert ll[0] == pytest.approx(core.brute_loglik(m, le), abs=1e-10)
    assert np.allclose(filt[0].sum(1), 1.0)


def test_smoothed_marginals_match_brute_force():
    import itertools
    S, D, T = 2, 3, 6
    m = model(S=S, D=D)
    le = emissions(T, S, seed=4)
    E = np.exp(le)
    marg = np.zeros((T, S))
    for path in itertools.product(range(S), repeat=T):
        p = m.init[path[0]] * E[0, path[0]]
        d = 0
        for t in range(1, T):
            s, j = path[t - 1], path[t]
            if j == s:
                p *= 1 - m.hazard[s, d]
                d = min(d + 1, D - 1)
            else:
                p *= m.hazard[s, d] * m.exit[s, j]
                d = 0
            p *= E[t, j]
        for t in range(T):
            marg[t, path[t]] += p
    marg /= marg.sum(1, keepdims=True)
    _ll, post = core.smooth(m, [le])
    assert np.allclose(post[0], marg, atol=1e-10)


def test_geometric_hsmm_equals_hmm():
    S, T = 3, 400
    m = model(S=S, D=50, family="geometric")
    le = emissions(T, S, seed=7)
    ll, _f, step = core.forward(m, [le])
    P, p0 = core.as_hmm(m)
    ref = hmm.forward(le, P, p0)
    assert np.allclose(step[0], ref, atol=1e-10)
    assert ll[0] == pytest.approx(ref.sum(), abs=1e-8)


def test_seam_invariance_and_batch_independence():
    m = model(S=3, D=20)
    a, b = emissions(300, 3, 1), emissions(200, 3, 2)
    ll1, f1, s1 = core.forward(m, [a, b])
    a2 = a.copy()
    a2[-50:] = a2[-50:][::-1] + 1.0
    ll2, f2, s2 = core.forward(m, [a2, b])
    assert np.array_equal(f1[1], f2[1]) and np.array_equal(s1[1], s2[1]) and ll1[1] == ll2[1]
    llb, fb, _ = core.forward(m, [b])
    assert np.array_equal(fb[0], f1[1]) and llb[0] == ll1[1]


def test_padding_does_not_leak_into_smoothing():
    m = model(S=2, D=10)
    a = emissions(120, 2, 3)
    _l, p1 = core.smooth(m, [a])
    _l, p2 = core.smooth(m, [a, emissions(400, 2, 9)])
    assert np.allclose(p1[0], p2[0], atol=1e-12)


def test_hazard_families_are_proper():
    for fam, p in (("weibull", (1.5, 2.0)), ("lognormal", (0.3, 0.8)), ("geometric", (2.0,))):
        h = core.hazard_from_family(fam, p, 3000, 30.0)
        surv = np.cumprod(np.r_[1.0, 1.0 - h[:-1]])
        pmf = surv * h
        assert np.all((h > 0) & (h < 1))
        assert pmf.sum() == pytest.approx(1.0, abs=1e-3)       # 100 s holds nearly all mass
    hw = core.hazard_from_family("weibull", (2.0, 1.0), 120, 30.0)
    assert np.all(np.diff(hw[5:60]) > 0)                     # k > 1: rising hazard


def test_planted_bouts_are_recovered():
    """Two states with separated emissions and known run structure: the
    smoothed posterior recovers the planted runs."""
    rng = np.random.default_rng(0)
    truth = []
    while len(truth) < 3000:
        s = len(truth) and 1 - truth[-1] or 0
        truth += [s] * int(rng.integers(30, 120))
    truth = np.asarray(truth[:3000])
    mu = np.where(truth == 0, -1.0, 1.0)
    x = mu + rng.normal(0, 1.2, truth.size)                  # noisy per frame
    le = np.stack([-(x + 1) ** 2 / 2.88, -(x - 1) ** 2 / 2.88], axis=1)
    m = core.HSMM(hazard=np.stack([core.hazard_from_family("lognormal", (np.log(2.5), 0.4), 300, 30.0)] * 2),
                  exit=np.array([[0.0, 1.0], [1.0, 0.0]]), init=np.array([0.5, 0.5]))
    _l, post = core.smooth(m, [le])
    acc = (post[0].argmax(1) == truth).mean()
    raw = (le.argmax(1) == truth).mean()
    assert acc > 0.95 and acc > raw + 0.1
