"""The trend filter: its operator, its fixed point, and its knots."""
from __future__ import annotations

import numpy as np

from vieb.seg import trendfilter as tf

OBJ = {"dataset": "luna", "arm": "trendfilter", "split": "tune"}


def test_the_operator_annihilates_a_quadratic():
    """`D3` must see nothing in a piecewise-quadratic signal.

    This is the whole reason the penalty is on the THIRD difference: a
    quadratic is what the fit is allowed to be for free, so its knots are
    exactly the places the quadratic changes -- acceleration jumps.
    """
    t = np.arange(200, dtype=np.float64)
    y = np.column_stack([3.0 + 2.0 * t + 0.5 * t ** 2, -1.0 + t])
    d3, _, _ = tf.operator(200)
    assert np.abs(np.asarray(d3 @ y)).max() < 1e-6


def test_the_operator_sees_an_acceleration_jump():
    t = np.arange(200, dtype=np.float64)
    y = (0.01 * t ** 2)[:, None].copy()
    y[100:] += (0.05 * (t[100:] - t[100]) ** 2)[:, None]
    d3, _, _ = tf.operator(200)
    v = np.linalg.norm(np.asarray(d3 @ y), axis=1)
    assert int(np.argmax(v)) in (97, 98, 99, 100)


def test_the_banded_form_matches_the_sparse_product():
    d3, ab, gb = tf.operator(60)
    g = (d3.T @ d3).toarray()
    for d in range(4):
        assert np.allclose(ab[3 - d, d:], np.diag(g, d))
    h = (d3 @ d3.T).toarray()
    for d in range(4):
        assert np.allclose(gb[3 - d, d:], np.diag(h, d))


def test_a_large_lambda_leaves_no_knots():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(300, 4))
    got = tf.fit(y, np.ones(300), 1e6 * tf.scale_of(y))
    assert tf.knots(got["z"]).size == 0


def test_a_tiny_lambda_interpolates():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(300, 4))
    got = tf.fit(y, np.ones(300), 1e-9 * tf.scale_of(y))
    assert float(np.abs(got["x"] - y).max()) < 1e-2


def test_more_penalty_never_gives_more_knots():
    rng = np.random.default_rng(0)
    t = np.arange(600, dtype=np.float64)
    y = np.cumsum(rng.normal(0, 0.01, size=(600, 6)), axis=0)
    s = tf.scale_of(y)
    counts = [tf.knots(tf.fit(y, np.ones(600), a * s)["z"]).size
              for a in (0.5, 5.0, 50.0)]
    assert counts[0] >= counts[1] >= counts[2]


def test_the_weights_pull_a_confident_frame_harder():
    """The error model: a low-confidence frame moves the fit less."""
    y = np.zeros((200, 2))
    y[100] = 10.0
    w = np.ones(200)
    tight = tf.fit(y, w, 1.0)["x"][100, 0]
    w[100] = 0.01
    loose = tf.fit(y, w, 1.0)["x"][100, 0]
    assert tight > loose


def test_knots_carry_the_registered_offset():
    z = np.zeros((10, 3))
    z[4] = 1.0
    assert tf.knots(z).tolist() == [4 + tf.KNOT_OFFSET]
    assert tf.KNOT_OFFSET == 2


def test_blocked_frames_cannot_be_boundaries():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.05, size=(300, 4)), axis=0)
    blocked = np.zeros(300, bool)
    blocked[100:200] = True
    k = tf.peaks_of(y, np.ones(300), 0.5, blocked=blocked)
    assert not ((k >= 100) & (k < 200)).any()


def test_a_short_recording_returns_nothing_rather_than_raising():
    assert tf.peaks_of(np.zeros((5, 3)), np.ones(5), 1.0).size == 0


def test_the_selection_refuses_rather_than_extending_the_grid():
    rows = {a: {"corpus": 1.0, "jitter": 0.9, "jitter_hi": 0.95}
            for a in tf.ALPHAS}
    r = tf.lambda_read(rows, scored_object=OBJ, n_effective=60)
    assert r.verdict == "GRID_LIMITED" and "NOT extended" in r.reason


def test_the_selection_takes_the_smallest_passing_alpha():
    rows = {a: {"corpus": 1.0, "jitter": 0.9, "jitter_hi": 0.95}
            for a in tf.ALPHAS}
    for a in tf.ALPHAS[3:]:
        rows[a] = {"corpus": 1.0, "jitter": 0.01, "jitter_hi": 0.02}
    r = tf.lambda_read(rows, scored_object=OBJ, n_effective=60)
    assert r.verdict == "PASS"
    assert r.detail["lambda_star_alpha"] == tf.ALPHAS[3]


def test_the_share_bar_is_the_registered_one():
    assert tf.JITTER_SHARE == 0.05
