"""The gate's arithmetic and the refusals it is obliged to make."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import recur as rc


def _ci(point, lo, hi, n=89):
    return {"point": point, "lo": lo, "hi": hi, "n_animals": n}


def _arm(per_animal):
    return {"delta_per_animal": per_animal,
            "delta": _ci(float(np.mean(list(per_animal.values()))), 0.0, 0.0)}


def test_the_k_mad_floor_is_a_registered_constant_not_a_preference():
    """Below it the detector saturates; the registration excludes it."""
    assert rc.K_MAD_FLOOR == 2.0
    assert min(rc.K_MAD_SWEEP) > rc.K_MAD_FLOOR
    assert rc.K_MAD_PRIMARY in rc.K_MAD_SWEEP


def test_the_nulls_are_the_dwell_matched_pair_and_not_the_closed_gate_s():
    assert rc.NULLS == ("microstate", "microstate0")
    assert "phase" not in rc.NULLS and "var5" not in rc.NULLS


def test_gate_fails_when_segmenting_makes_recurrence_worse():
    """The registered FAIL: the boundaries cut through behaviours."""
    seg = _arm({str(i): 0.001 for i in range(40)})
    win = _arm({str(i): 0.010 for i in range(40)})
    d = rc.excess_delta(seg, win, seed=0)
    assert d["delta"]["hi"] < 0
    rd = rc.gate_read(d, {"smallest_recovered": 0.01}, group="both",
                      scored_object={"a": 1}, n_effective=40)
    assert rd.verdict == "FAIL"
    assert "CUT THROUGH" in rd.reason
    assert "1.00%" in rd.reason          # the floor travels inside the verdict


def test_gate_passes_weakly_when_the_interval_spans_zero():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 0.002, size=60)
    seg = _arm({str(i): 0.01 + v for i, v in enumerate(vals)})
    win = _arm({str(i): 0.01 - v for i, v in enumerate(vals)})
    d = rc.excess_delta(seg, win, seed=0)
    rd = rc.gate_read(d, {"smallest_recovered": None}, group="both",
                      scored_object={"a": 1}, n_effective=60)
    assert rd.verdict == "PASS"
    assert "ADDS NOTHING" in rd.reason
    assert "did not run" in rd.reason


def test_a_gate_without_a_floor_says_so_inside_its_own_reason():
    """A FAIL that cannot say what it would have detected is not a result."""
    seg = _arm({str(i): 0.0 for i in range(30)})
    win = _arm({str(i): 0.02 for i in range(30)})
    rd = rc.gate_read(rc.excess_delta(seg, win, seed=0), {},
                      group="twist", scored_object={"a": 1}, n_effective=30)
    assert "did not run" in rd.reason


def test_excess_delta_is_paired_within_animal():
    """Unpaired would discard the pairing the design already has."""
    seg = _arm({"a": 0.05, "b": 0.01})
    win = _arm({"b": 0.02, "a": 0.03})
    d = rc.excess_delta(seg, win, seed=0)
    assert d["per_animal"] == pytest.approx({"a": 0.02, "b": -0.01})


def test_excess_delta_refuses_with_fewer_than_two_shared_animals():
    d = rc.excess_delta(_arm({"a": 0.1}), _arm({"b": 0.1}), seed=0)
    assert d["n_animals"] == 0 and "why" in d


def test_recur_read_marks_microstate_as_the_weak_arm():
    """DWELL.md's standing caveat has to travel with the number."""
    pair = {"delta": _ci(0.02, 0.01, 0.03)}
    strong = rc.recur_read(pair, null="microstate0", unit="segment",
                           scored_object={"a": 1}, n_effective=89)
    weak = rc.recur_read(pair, null="microstate", unit="segment",
                         scored_object={"a": 1}, n_effective=89)
    assert strong.verdict == weak.verdict == "PASS"
    assert "WEAKLY" in weak.reason and "WEAKLY" not in strong.reason


def test_recur_read_fails_on_an_interval_spanning_zero():
    rd = rc.recur_read({"delta": _ci(0.001, -0.002, 0.004)}, null="microstate0",
                       unit="segment", scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "FAIL" and "spans zero" in rd.reason


def test_duration_match_is_symmetric_in_ks_and_signed_in_the_quantiles():
    rng = np.random.default_rng(0)
    a = np.exp(rng.normal(0.0, 0.5, 5000))
    b = np.exp(rng.normal(0.8, 0.5, 5000))
    ab = rc.duration_match(a, b)
    ba = rc.duration_match(b, a)
    assert ab["ks"] == pytest.approx(ba["ks"], abs=1e-3)
    # The corpus arm is shorter here, and the sign says so.
    assert ab["mean_log_obs"] < ab["mean_log_null"]
    assert all(v < 0 for v in ab["quantile_diff_log"].values())
    assert ab["median_ratio"] < 1.0


def test_duration_match_refuses_rather_than_reporting_nan_quietly():
    got = rc.duration_match([1.0], [2.0, 3.0])
    assert "why" in got and not np.isfinite(got["ks"])


def test_length_match_control_detects_a_pure_length_match():
    """A rater scored +0.183 by calling the six longest clips 'same'."""
    rng = np.random.default_rng(0)
    q = np.exp(rng.normal(size=4000))
    same = q * np.exp(rng.normal(0.0, 0.01, size=4000))    # partners match
    got = rc.length_match_control(q, same, np.random.default_rng(1))
    assert got["ratio"] < 0.2
    indep = np.exp(rng.normal(size=4000))
    got2 = rc.length_match_control(q, indep, np.random.default_rng(1))
    assert got2["ratio"] == pytest.approx(1.0, abs=0.15)
