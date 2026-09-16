"""The paired context contrast, and the gate that must precede it."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import context as cx


def _cells(rows):
    """rows: (animal, day, context, clump0_frames, total_frames)."""
    lab, nf, an, dy, ct = [], [], [], [], []
    for animal, day, ctx, hit, total in rows:
        if hit:
            lab.append(0), nf.append(hit), an.append(animal)
            dy.append(day), ct.append(ctx)
        rest = total - hit
        if rest:
            lab.append(-1), nf.append(rest), an.append(animal)
            dy.append(day), ct.append(ctx)
    return (np.array(lab), np.array(nf, dtype=float), np.array(an),
            np.array(dy), np.array(ct))


def test_occupancy_is_a_rate_not_a_count():
    """Two sessions with the same clump-0 frames but different totals must
    not score the same. Counting would measure how much of each session
    survived QC rather than anything about context."""
    lab, nf, an, dy, ct = _cells([("a", 3, "A", 100, 1000),
                                  ("a", 3, "B", 100, 200)])
    occ, den = cx.cell_occupancy(lab, nf, an, dy, ct, clump=0)
    assert occ[("a", 3, "A")] == pytest.approx(0.1)
    assert occ[("a", 3, "B")] == pytest.approx(0.5)
    assert den[("a", 3, "A")] == 1000


def test_days_outside_the_registered_window_are_excluded():
    """Days 0-1 are context A only and day 2 is context C only -- that pairing
    is what CONCENTRATION.md records as completely confounded."""
    lab, nf, an, dy, ct = _cells([("a", 0, "A", 10, 100),
                                  ("a", 2, "C", 10, 100),
                                  ("a", 4, "A", 10, 100),
                                  ("a", 4, "B", 20, 100)])
    occ, _den = cx.cell_occupancy(lab, nf, an, dy, ct, clump=0)
    assert set(occ) == {("a", 4, "A"), ("a", 4, "B")}


def test_context_c_is_excluded_even_on_a_kept_day():
    lab, nf, an, dy, ct = _cells([("a", 3, "C", 10, 100)])
    occ, _den = cx.cell_occupancy(lab, nf, an, dy, ct, clump=0)
    assert occ == {}


def test_a_cell_missing_one_context_is_dropped_not_half_counted():
    lab, nf, an, dy, ct = _cells([("a", 3, "A", 10, 100),
                                  ("b", 4, "A", 10, 100),
                                  ("b", 4, "B", 30, 100)])
    occ, den = cx.cell_occupancy(lab, nf, an, dy, ct, clump=0)
    got = cx.paired_deltas(occ, den)
    assert got["n_pairs"] == 1 and got["n_dropped_incomplete"] == 1
    assert got["diff"][0] == pytest.approx(0.2)


def test_the_delta_is_signed_so_a_flip_has_something_to_flip():
    """An unsigned or unpaired quantity has nothing to flip, which is the
    failure journeys.py records for the raw W2 contrast."""
    lab, nf, an, dy, ct = _cells([("a", 3, "A", 30, 100), ("a", 3, "B", 10, 100),
                                  ("b", 3, "A", 10, 100), ("b", 3, "B", 30, 100)])
    occ, den = cx.cell_occupancy(lab, nf, an, dy, ct, clump=0)
    got = cx.paired_deltas(occ, den)
    assert sorted(got["diff"]) == pytest.approx([-0.2, 0.2])
    assert set(got["animal"]) == {"a", "b"}


def test_pair_keys_are_animal_and_day_so_one_delta_per_cell():
    lab, nf, an, dy, ct = _cells([("a", 3, "A", 10, 100), ("a", 3, "B", 20, 100),
                                  ("a", 5, "A", 10, 100), ("a", 5, "B", 40, 100)])
    got = cx.paired_deltas(*cx.cell_occupancy(lab, nf, an, dy, ct, clump=0))
    assert sorted(got["pair_key"]) == ["a|3", "a|5"]


def test_context_read_follows_the_interval_not_the_p_value():
    """LEARNING_CURVE.md fixed the precedent: a stabilisation arm at p = 0.0475
    still read FAIL because its CI spanned zero."""
    ci = {"point": 0.01, "lo": -0.002, "hi": 0.022, "n_animals": 89}
    flip = {"p_two_sided": 0.001}
    pairs = {"n_pairs": 439, "mean_A": 0.02, "mean_B": 0.03}
    rd = cx.context_read(ci, flip, pairs, clump=0, scored_object={"a": 1},
                         n_effective=89)
    assert rd.verdict == "FAIL" and "spans zero" in rd.reason


def test_context_read_passes_in_either_direction_and_names_it():
    pairs = {"n_pairs": 439, "mean_A": 0.02, "mean_B": 0.01}
    flip = {"p_two_sided": 0.005}
    up = cx.context_read({"point": 0.01, "lo": 0.004, "hi": 0.017,
                          "n_animals": 89}, flip, pairs, clump=0,
                         scored_object={"a": 1}, n_effective=89)
    down = cx.context_read({"point": -0.01, "lo": -0.017, "hi": -0.004,
                            "n_animals": 89}, flip, pairs, clump=0,
                           scored_object={"a": 1}, n_effective=89)
    assert up.verdict == down.verdict == "PASS"
    assert "context B" in up.reason and "context A" in down.reason


def test_context_read_reads_p_two_sided_and_not_the_key_that_does_not_exist():
    """`simplex.journey_read` looks for `p`, which `pair_flip_null` never
    returns, so its p is always NaN and it always takes the FAIL branch."""
    rd = cx.context_read({"point": 0.01, "lo": 0.004, "hi": 0.02,
                          "n_animals": 89},
                         {"p_two_sided": 0.0005}, {"n_pairs": 10}, clump=0,
                         scored_object={"a": 1}, n_effective=89)
    assert "0.0005" in rd.reason


def test_the_registered_constants_match_the_registration():
    assert cx.DAYS == (3, 4, 5, 6, 7)
    assert cx.CONTEXTS == ("A", "B")
    assert cx.PLAUSIBLE_EFFECT == 0.02
