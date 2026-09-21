"""The two controls, and the one property that lets the test fail at all."""
from __future__ import annotations

import numpy as np

from vieb.seg import controls as co

OBJ = {"dataset": "luna", "arm": "context_controls", "split": "report"}


# --- the residual -------------------------------------------------------

def test_the_residual_is_fitted_through_the_origin():
    """The whole test turns on this.

    With an intercept the residuals have mean exactly zero by construction,
    the sign-flip null could never reject, and the comparison could not have
    failed -- the failure `journeys.py` records for the raw W2 contrast and
    `METHODS_FINDINGS.md` M5 generalises.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x * 0.5 + 7.0                      # a large, real intercept
    _, r = co.residual(y, x)
    assert abs(float(np.mean(r))) > 1.0    # the intercept SURVIVES


def test_an_ordinary_least_squares_fit_would_have_zeroed_it():
    """Stated as a test so the choice cannot be undone by accident."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x * 0.5 + 7.0
    slope, icept = np.polyfit(x, y, 1)
    assert abs(float(np.mean(y - (slope * x + icept)))) < 1e-9


def test_the_residual_removes_what_the_control_explains():
    x = np.array([1.0, -2.0, 3.0, -4.0])
    beta, r = co.residual(3.0 * x, x)
    assert abs(beta - 3.0) < 1e-12 and np.allclose(r, 0.0)


def test_a_control_that_is_all_zero_leaves_the_signal_alone():
    y = np.array([1.0, 2.0, 3.0])
    beta, r = co.residual(y, np.zeros(3))
    assert beta == 0.0 and np.allclose(r, y)


# --- occupancy and the matched threshold --------------------------------

def test_occupancy_is_keyed_on_the_denominator_not_the_numerator():
    """An arm that placed nothing in a cell has occupancy zero, not a gap.

    A missing cell would drop that pair and silently change which cells the
    arms are compared on, which is the one thing `aligned_deltas` exists to
    prevent.
    """
    den = {("a", 3, "A"): 100.0, ("a", 3, "B"): 100.0}
    occ = co.occupancy({("a", 3, "A"): 5.0}, den)
    assert occ[("a", 3, "B")] == 0.0 and set(occ) == set(den)


def test_the_threshold_matches_the_target_share_exactly():
    rng = np.random.default_rng(0)
    sp = rng.random(10_000)
    thr = co.match_threshold(sp, 0.0121)
    assert abs(float(np.mean(sp < thr)) - 0.0121) < 0.002


def test_a_degenerate_target_returns_nan_rather_than_a_number():
    assert not np.isfinite(co.match_threshold([1.0, 2.0], 0.0))


# --- the vacuity guard --------------------------------------------------

def test_a_control_selecting_none_of_the_island_is_called_not_a_control():
    """M5: a control that returns nothing is usually testing the control."""
    isl = np.array([True] * 10 + [False] * 90)
    sel = np.array([False] * 10 + [True] * 10 + [False] * 80)
    r = co.overlap_read(sel, isl, arm="stillness", theta=1e-4,
                        scored_object=OBJ, n_effective=89)
    assert r.verdict == "NOT_A_RESULT"
    assert "NOT A CONTROL" in r.reason and "vacuous" in r.reason


def test_an_overlapping_control_is_reported_as_usable():
    isl = np.array([True] * 10 + [False] * 90)
    sel = np.array([True] * 8 + [False] * 2 + [True] * 10 + [False] * 80)
    r = co.overlap_read(sel, isl, arm="stillness", theta=0.05,
                        scored_object=OBJ, n_effective=89)
    assert "NOT A CONTROL" not in r.reason
    assert abs(r.detail["share_of_island_selected"] - 0.8) < 1e-12


# --- alignment ----------------------------------------------------------

def test_arms_built_on_one_denominator_are_paired_cell_for_cell():
    den = {("a", 3, "A"): 10.0, ("a", 3, "B"): 10.0,
           ("b", 4, "A"): 10.0, ("b", 4, "B"): 10.0}
    one = co.occupancy({("a", 3, "A"): 5.0}, den)
    two = co.occupancy({("b", 4, "B"): 5.0}, den)
    al = co.aligned_deltas({"one": one, "two": two}, den)
    assert al["pair_key"] == ["a|3", "b|4"] and al["n_pairs"] == 2


def test_the_read_says_freeze_scorer_when_stillness_absorbs_the_effect():
    res = {"point": -0.001, "lo": -0.004, "hi": 0.002, "n_animals": 89,
           "n_units": 439}
    r = co.controls_read(res, {"p_two_sided": 0.4}, arm="stillness",
                         beta=0.9, scored_object=OBJ, n_effective=89)
    assert r.verdict == "FAIL" and "FREEZE SCORER" in r.reason


def test_the_read_passes_when_the_residual_excludes_zero():
    res = {"point": -0.007, "lo": -0.013, "hi": -0.001, "n_animals": 89,
           "n_units": 439}
    r = co.controls_read(res, {"p_two_sided": 0.03}, arm="windows",
                         beta=0.4, scored_object=OBJ, n_effective=89)
    assert r.verdict == "PASS" and "ADDS TO WINDOWS" in r.reason
