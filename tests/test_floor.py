"""The noise floor: the calibration, the injection arms, and the refusals."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.qc import bones as bn
from vieb.seg import breaks as bk, floor as fl, jitter as jit

OBJ = {"dataset": "luna", "arm": "noise_floor", "split": "tune"}


def _pose(n=8000, noise=0.01, seed=0):
    """A rigid skull translating, plus isotropic per-keypoint noise.

    8,000 frames x 3 skull bones = 24,000 residuals, which clears
    `jitter.MIN_BIN`. A smaller fixture would be refused by the calibration
    for the same reason a thin confidence bin is, and would test nothing.
    """
    rng = np.random.default_rng(seed)
    base = np.array([[-1.0, 1.0], [1.0, 1.0], [0.0, 2.0], [0.0, 0.0],
                     [-1.0, -1.0], [1.0, -1.0], [0.0, -2.0]])
    walk = np.cumsum(rng.normal(0, 0.05, size=(n, 1, 2)), axis=0)
    return base[None] + walk + rng.normal(0, noise, size=(n, 7, 2))


# --- the calibration ----------------------------------------------------

def test_the_calibration_recovers_an_injected_amplitude():
    """Var(bone length) = 2 sigma^2, so sigma = sd(r)/sqrt(2)."""
    sigma = 0.01
    p = _pose(noise=sigma)
    r, c = jit.residuals(p, np.full((p.shape[0], 7), 0.9),
                         missing=np.zeros((p.shape[0], 7), bool),
                         interpolated=np.zeros((p.shape[0], 7), bool),
                         ell=1.0)
    t = jit.calibrate(r, c)
    got = float(np.nanmax(np.asarray(t["sigma_bl"], dtype=float)))
    assert abs(got - sigma) / sigma < 0.15


def test_only_skull_bones_are_used():
    """Trunk bones flex; a length change there is posture as often as noise."""
    assert bn.SKULL == ((0, 1), (0, 2), (1, 2))
    p = _pose()
    r, _ = jit.residuals(p, np.full((p.shape[0], 7), 0.9),
                         missing=np.zeros((p.shape[0], 7), bool),
                         interpolated=np.zeros((p.shape[0], 7), bool),
                         ell=1.0)
    assert r.size == p.shape[0] * len(bn.SKULL)


def test_held_keypoint_frames_are_excluded():
    """A held keypoint has a zero residual by construction.

    Leaving them in would pull the variance toward zero exactly where
    tracking is worst -- best-behaved where it is least true.
    """
    p = _pose()
    miss = np.zeros((p.shape[0], 7), bool)
    miss[:, 0] = True                      # left_ear held throughout
    r, _ = jit.residuals(p, np.full((p.shape[0], 7), 0.9), missing=miss,
                         interpolated=np.zeros((p.shape[0], 7), bool),
                         ell=1.0)
    assert r.size == p.shape[0]            # only the (1,2) bone survives


def test_a_thin_confidence_bin_is_left_unestimated_not_guessed():
    p = _pose(n=200)
    r, c = jit.residuals(p, np.full((p.shape[0], 7), 0.9),
                         missing=np.zeros((p.shape[0], 7), bool),
                         interpolated=np.zeros((p.shape[0], 7), bool),
                         ell=1.0)
    t = jit.calibrate(r, c)
    assert not np.isfinite(np.asarray(t["sigma_bl"], dtype=float)).any()
    rd = jit.calibration_read(t, scored_object=OBJ, n_effective=2)
    assert rd.verdict == "NOT_A_RESULT"


def test_the_draw_keeps_the_measured_amplitude():
    p = _pose(noise=0.02)
    conf = np.full((p.shape[0], 7), 0.9)
    r, c = jit.residuals(p, conf, missing=np.zeros((p.shape[0], 7), bool),
                         interpolated=np.zeros((p.shape[0], 7), bool), ell=1.0)
    t = jit.calibrate(r, c)
    got = jit.draw(np.random.default_rng(0), conf, t)
    assert got.shape == (p.shape[0], 7, 2)
    assert abs(float(np.std(got)) - 0.02) / 0.02 < 0.2


def test_the_draw_scales_the_amplitude():
    p = _pose(noise=0.02)
    conf = np.full((p.shape[0], 7), 0.9)
    r, c = jit.residuals(p, conf, missing=np.zeros((p.shape[0], 7), bool),
                         interpolated=np.zeros((p.shape[0], 7), bool), ell=1.0)
    t = jit.calibrate(r, c)
    rng = np.random.default_rng(0)
    one = float(np.std(jit.draw(rng, conf, t, scale=1.0)))
    two = float(np.std(jit.draw(rng, conf, t, scale=2.0)))
    assert abs(two / one - 2.0) < 0.1


# --- the injection arms -------------------------------------------------

def test_the_static_pose_has_no_variance():
    p = _pose()
    s = fl.static_pose(p, np.ones(p.shape[0], bool))
    assert np.allclose(s.std(axis=0), 0.0)


def test_a_constant_signal_produces_no_boundaries():
    """Prediction 1, and the reason every other number is checkable."""
    x = np.ones((3000, 14))
    pk = fl.peaks_of(x, idx=range(14), fps=30.0, k_mad=3.0)
    assert pk.size == 0


def test_the_rate_never_counts_across_a_seam():
    """Two recordings: changing the second must not move the first's count."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 14))
    b = np.array([0, 2000, 4000])
    ab = np.zeros(4000, bool)
    one = fl.rate_of(x, b, ab, idx=range(14), fps=30.0, k_mad=3.0)
    y = x.copy()
    y[2000:] = rng.normal(size=(2000, 14)) * 50.0
    two = fl.rate_of(y, b, ab, idx=range(14), fps=30.0, k_mad=3.0)
    first_one = fl.peaks_of(x[:2000], idx=range(14), fps=30.0, k_mad=3.0)
    first_two = fl.peaks_of(y[:2000], idx=range(14), fps=30.0, k_mad=3.0)
    assert np.array_equal(first_one, first_two)
    assert one["n_eligible_frames"] == two["n_eligible_frames"]


def test_the_nms_ceiling_is_the_refractory_period():
    guard = bk.guard_frames(30.0, deriv_sec=bk.DERIV_SEC)
    assert fl.nms_ceiling(30.0) == 30.0 / bk.min_segment_frames(bk.DEGREE,
                                                                guard)


# --- the reads ----------------------------------------------------------

def test_a_harness_that_fires_on_a_constant_is_not_a_result():
    rd = fl.static_read({"n_peaks": 3, "eligible_seconds": 100.0},
                        scored_object=OBJ, n_effective=60)
    assert rd.verdict == "NOT_A_RESULT" and "HARNESS IS WRONG" in rd.reason


def test_a_floor_whose_interval_includes_zero_kills_the_hypothesis():
    rd = fl.floor_read({"rate_per_s": 0.0, "n_peaks": 0,
                        "ci": {"point": 0.0, "lo": 0.0, "hi": 0.01}},
                       fps=30.0, scored_object=OBJ, n_effective=60)
    assert rd.verdict == "FAIL" and "noise hypothesis is dead" in rd.reason


def test_separation_needs_non_overlapping_intervals_not_a_point_gap():
    """A point comparison crowned a cleaning arm on 0.029 px once already."""
    cell = {"refused": False, "n_frames": 50_000,
            "ci": {"point": 0.50, "lo": 0.40, "hi": 0.60}}
    floor = {"ci": {"point": 0.45, "lo": 0.35, "hi": 0.55}}
    rd = fl.separation_read(cell, floor, label="x", scored_object=OBJ,
                            n_effective=60)
    assert rd.verdict == "FAIL"          # point is higher, intervals overlap


def test_a_thin_cell_is_refused_and_not_reported_with_a_caveat():
    cell = {"refused": True, "n_frames": 900}
    rd = fl.separation_read(cell, {"ci": {}}, label="x", scored_object=OBJ,
                            n_effective=60)
    assert rd.verdict == "NOT_A_RESULT" and "Refused" in rd.reason


def test_cell_rates_refuses_below_the_registered_floor():
    cells = np.zeros(1000, dtype=np.int64)
    rows = fl.cell_rates(cells, np.zeros(1000, bool), ["a"] * 1000,
                         n_cells=1, fps=30.0)
    assert rows[0]["refused"] and fl.MIN_CELL_FRAMES == 20_000
