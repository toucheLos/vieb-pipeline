"""The order-controlled plant: its shape, its placement, and its scoring."""
from __future__ import annotations

import numpy as np

from vieb.seg import plant as pl

OBJ = {"dataset": "luna", "arm": "plant", "split": "tune"}


# --- the profile --------------------------------------------------------

def test_order_zero_jumps_in_value():
    g = pl.profile(0, 8)
    assert g[0] == 1.0


def test_order_one_is_continuous_in_value_and_jumps_in_slope():
    g = pl.profile(1, 8)
    assert g[0] == 0.0
    assert np.diff(g)[0] > 0.0 and abs(g[0]) < 1e-12


def test_order_two_is_continuous_in_value_and_slope():
    """The axiom's break: position and velocity continuous, acceleration not.

    Asserted as an ORDERING against order 1 rather than against an absolute
    tolerance. Sampled at one frame the first difference of a C1 function is
    not zero -- it picks up the curvature -- so the meaningful statement is
    that order 2 starts far flatter than order 1 does.
    """
    one, two = pl.profile(1, 16), pl.profile(2, 16)
    assert two[0] == 0.0 and one[0] == 0.0
    assert np.diff(two)[0] < 0.25 * np.diff(one)[0]
    assert np.diff(two, 2)[0] > 0.0


def test_the_taper_plants_no_break_of_its_own():
    """The reason the taper is a smootherstep and not a mirror.

    A mirrored ramp meets its own reflection with slopes +2A and -2A, planting
    an order-1 break at the midpoint SHARPER than the order-2 break under test.
    The detector would then be scored on the wrong discontinuity. Here the
    largest second difference must sit at the planted onset, not in the taper.
    """
    for order in pl.ORDERS:
        # Prepend baseline: the planted break is between the zero BEFORE the
        # instance and its first sample, which a diff inside the array alone
        # cannot see.
        g = np.concatenate([np.zeros(4), pl.profile(order, 16)])
        d2 = np.abs(np.diff(g, 2))
        assert int(np.argmax(d2)) <= 4, f"order {order} breaks inside the taper"
    # The margin, not the constant. Everything after the onset is the smooth
    # decay's own curvature, which no finite pulse avoids; what matters is
    # that the planted break stays the LARGEST second difference by a clear
    # factor, so non-maximum suppression cannot prefer anything else.
    for order in pl.ORDERS:
        g = np.concatenate([np.zeros(4), pl.profile(order, 16)])
        d2 = np.abs(np.diff(g, 2))
        assert d2[8:].max() < 0.5 * d2[:8].max(), f"order {order} margin"


def test_the_profile_returns_to_baseline():
    for order in pl.ORDERS:
        g = pl.profile(order, 8)
        assert abs(g[-1]) < 1e-12


# --- placement ----------------------------------------------------------

def test_plants_never_cross_a_recording_seam():
    """The invariant every stage in this project asserts by test."""
    rng = np.random.default_rng(0)
    bounds = np.array([0, 3000, 6000])
    ext = pl.INSTANCE_SPANS * 8
    s = pl.place(rng, bounds=bounds, abstain=np.zeros(6000, bool), w=8,
                 per_recording=20, guard=2)
    assert not any(t < 3000 < t + ext for t in s.tolist())


def test_plants_avoid_abstained_frames():
    rng = np.random.default_rng(0)
    ab = np.zeros(4000, bool)
    ab[1000:3000] = True
    s = pl.place(rng, bounds=np.array([0, 4000]), abstain=ab, w=8,
                 per_recording=30, guard=2)
    ext = pl.INSTANCE_SPANS * 8
    assert not any(ab[t:t + ext].any() for t in s.tolist())


def test_plants_do_not_overlap_each_other():
    rng = np.random.default_rng(0)
    s = np.sort(pl.place(rng, bounds=np.array([0, 4000]),
                         abstain=np.zeros(4000, bool), w=8,
                         per_recording=40, guard=2))
    assert (np.diff(s) >= pl.INSTANCE_SPANS * 8).all()


# --- the plant itself ---------------------------------------------------

def test_the_plant_moves_only_the_scored_channels():
    rng = np.random.default_rng(0)
    x = np.zeros((500, 17))
    y = pl.apply_plant(x, [100], order=2, amp=1.0, w=8, idx=range(14),
                       rng=rng)
    assert np.allclose(y[:, 14:], 0.0) and not np.allclose(y[:, :14], 0.0)


def test_the_amplitude_is_a_displacement_magnitude():
    rng = np.random.default_rng(0)
    x = np.zeros((500, 14))
    y = pl.apply_plant(x, [100], order=0, amp=3.0, w=8, idx=range(14),
                       rng=rng)
    assert abs(float(np.max(np.linalg.norm(y, axis=1))) - 3.0) < 1e-9


# --- scoring ------------------------------------------------------------

def test_recovery_needs_a_peak_inside_the_band():
    assert pl.recovered([100], [100]) == (1, 1)
    assert pl.recovered([102], [100]) == (1, 1)
    assert pl.recovered([103], [100]) == (0, 1)


def test_anywhere_separates_mislocation_from_blindness():
    """A peak in the middle of the event is a localisation failure, not an
    aperture failure, and the two need different fixes."""
    assert pl.recovered_anywhere([112], [100], extent=96) == (1, 1)
    assert pl.recovered([112], [100]) == (0, 1)
    assert pl.recovered_anywhere([300], [100], extent=96) == (0, 1)


def test_chance_is_the_rate_times_the_band():
    assert abs(pl.chance_of(100, 3000) - 100 * 5 / 3000) < 1e-12


def test_a_thin_cell_is_refused_not_caveated():
    r = pl.recovery_read([], order=2, amp=1.0, n_instances=10,
                         scored_object=OBJ, n_effective=60)
    assert r.verdict == "NOT_A_RESULT" and "Refused" in r.reason


def test_recall_at_chance_fails():
    rows = [{"animal": f"a{i}", "recall": 0.07, "chance": 0.07}
            for i in range(20)]
    r = pl.recovery_read(rows, order=2, amp=1.0, n_instances=5000,
                         scored_object=OBJ, n_effective=20)
    assert r.verdict == "FAIL" and "NOT recovered above" in r.reason


def test_ordering_reports_order_two_never_crossing():
    cells = {0: {a: 0.9 for a in pl.AMPS}, 1: {4.0: 0.6},
             2: {a: 0.05 for a in pl.AMPS}}
    r = pl.ordering_read(cells, scored_object=OBJ, n_effective=60)
    assert r.verdict == "PASS" and "NEVER RECOVERED" in r.reason


def test_the_registered_constants_are_what_the_registration_says():
    assert pl.ORDERS == (0, 1, 2)
    assert pl.AMPS == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    assert pl.TOL == 2 and pl.MIN_INSTANCES == 200
