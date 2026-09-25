"""STABILISE 5 gate 6: the keypoint-agreement statistic, on a known answer."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))


def _case(q: float, sigma: float, seed: int = 0):
    """Motion d per frame; an arm recovering q of it; keypoints with noise."""
    rng = np.random.default_rng(seed)
    n = 4000
    d = rng.normal(0, 3.0, (n, 2))
    center = np.cumsum(d, axis=0) + 300.0
    W = np.zeros((n, 2, 3))
    W[:, 0, 0] = W[:, 1, 1] = 1.0
    W[1:, :, 2] = q * (center[1:] - center[:-1])
    noisy = center + rng.normal(0, sigma, center.shape)
    speed = np.hypot(*np.vstack([[0, 0], np.diff(center, axis=0)]).T)
    return W, noisy, np.ones(n, dtype=bool), speed


def test_f_recovers_the_fraction_and_is_not_attenuated_by_keypoint_noise():
    import stabilise5 as s5
    for q in (1.0, 0.5, 0.1):
        for sigma in (0.0, 2.0):
            W, c, ok, sp = _case(q, sigma)
            num, den, k = s5._pairs(W, c, ok, sp)
            assert k > 500
            f = num / den
            # sigma = 2 px of keypoint noise on each frame is large next to the
            # ~3 px steps; an attenuated statistic would fall well below q.
            assert abs(f - q) < 0.05 * max(q, 0.2), (q, sigma, f)


def test_the_bracket_contains_the_truth_under_noise_in_both():
    import stabilise5 as s5
    rng = np.random.default_rng(1)
    W, c, ok, sp = _case(1.0, 2.0)
    W[1:, :, 2] += rng.normal(0, 1.5, (W.shape[0] - 1, 2))   # arm noise too
    num, den, k, kk = s5._pairs(W, c, ok, sp, with_kp=True)
    f, beta = num / den, den / kk
    assert beta < 1.0 < f
