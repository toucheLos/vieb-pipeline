"""STABILISE 6 gate 6: the chained bracket closes around the truth."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))


def _case(q, sig_kp, sig_arm, seed=0):
    """Persistent motion -- constant velocity for 40 frames at a time, about
    2 px/frame -- because an animal's displacement grows k-fold over k frames;
    a random walk's grows like sqrt(k), no faster than the noise."""
    rng = np.random.default_rng(seed)
    n = 6000
    seg = 40
    v = rng.normal(0, 2.0, (n // seg + 1, 2))
    d = np.repeat(v, seg, axis=0)[:n]
    center = np.cumsum(d, axis=0) + 300.0
    W = np.zeros((n, 2, 3))
    W[:, 0, 0] = W[:, 1, 1] = 1.0
    W[1:, :, 2] = q * np.diff(center, axis=0) + rng.normal(0, sig_arm, (n - 1, 2))
    noisy = center + rng.normal(0, sig_kp, center.shape)
    speed = np.hypot(*np.vstack([[0, 0], np.diff(center, axis=0)]).T)
    return W, noisy, np.ones(n, dtype=bool), speed


def test_chaining_tightens_the_bracket_around_the_true_scale():
    import stabilise6 as s6
    W, c, ok, sp = _case(1.0, 2.0, 1.5)
    n1, d1, k1, _ = s6._chains(W, c, ok, sp, k=1)
    n10, d10, k10, m = s6._chains(W, c, ok, sp, k=10)
    b1, f1 = d1 / k1, n1 / d1
    b10, f10 = d10 / k10, n10 / d10
    assert b1 < b10 and f10 < f1
    assert abs(b10 - 1.0) < 0.03 and abs(f10 - 1.0) < 0.03
    assert (f10 - b10) < 0.5 * (f1 - b1)
    assert m > 50


def test_an_arm_that_under_follows_stays_below_one():
    import stabilise6 as s6
    W, c, ok, sp = _case(0.3, 2.0, 1.5)
    n, d, k, _ = s6._chains(W, c, ok, sp, k=10)
    beta, f = d / k, n / d
    # the bracket contains the true 0.3 and stays far below gate 6's band
    assert beta <= 0.3 + 0.03 and f >= 0.3 - 0.03
    assert f < 0.9
