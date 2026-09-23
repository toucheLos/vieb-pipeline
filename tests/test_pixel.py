"""Pixel motion, the arena floor, and the ezTrack-family freeze score.

The freeze test is the one that matters: it checks `freeze_mask` against a
**literal line-by-line transcription of ezTrack's `Measure_Freezing`**, not
against what the reimplementation was expected to do. A reimplementation tested
against its author's expectation tests the expectation.
"""
from __future__ import annotations

import numpy as np
import pytest

from vieb.pixel import freeze as fz
from vieb.pixel import motion as mo


def eztrack_measure_freezing(Motion, FreezeThresh, MinDuration):
    """Transcribed from FreezeAnalysis/FreezeAnalysis_Functions.py:345."""
    BelowThresh = (np.asarray(Motion) < FreezeThresh).astype(int)
    CumThresh = np.zeros(len(Motion))
    for x in range(1, len(Motion)):
        if BelowThresh[x] == 1:
            CumThresh[x] = CumThresh[x - 1] + BelowThresh[x]
    Freezing = (CumThresh >= MinDuration).astype(int)
    for x in range(len(Freezing) - 2, -1, -1):
        if Freezing[x] == 0 and Freezing[x + 1] > 0 \
                and Freezing[x + 1] < MinDuration:
            Freezing[x] = Freezing[x + 1] + 1
    return (Freezing > 0)


def test_freeze_mask_matches_eztrack_source_exactly():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(5, 250))
        motion = rng.random(n) * 100.0
        thresh = float(rng.random() * 100.0)
        k = int(rng.integers(1, 15))
        want = eztrack_measure_freezing(motion, thresh, k)
        got = fz.freeze_mask(motion, thresh, k)
        assert np.array_equal(want, got), (n, thresh, k)


def test_freeze_mask_keeps_the_backward_pass():
    """Dropping it truncates every bout by `min_frames - 1`, a fixed bias."""
    motion = np.array([100.0] * 3 + [0.0] * 6 + [100.0] * 3)
    got = fz.freeze_mask(motion, 50.0, 4)
    assert got[3:9].all(), got            # the ramp-up frames are recovered
    assert not got[:3].any() and not got[9:].any()


def test_freeze_mask_empty_and_never_freezing():
    assert fz.freeze_mask([], 1.0, 3).size == 0
    assert not fz.freeze_mask([9.0] * 20, 1.0, 3).any()


def test_freeze_fraction_is_a_rate_not_a_count():
    """§7: Context A sessions run ~17% longer, so only a rate may be published.

    Doubling the session doubles the freezing COUNT but must leave the fraction
    alone -- up to the single frame that ezTrack's `CumThresh[0] = 0` costs at
    the start of a series and not at an interior repeat.
    """
    m = np.array([0.0] * 10 + [100.0] * 10)
    one = fz.freeze_fraction(m, 50.0, 3)
    two = fz.freeze_fraction(np.concatenate([m, m]), 50.0, 3)
    assert abs(two - one) <= 1.0 / (2 * m.size)
    n_one = fz.freeze_mask(m, 50.0, 3).sum()
    n_two = fz.freeze_mask(np.concatenate([m, m]), 50.0, 3).sum()
    assert n_two > n_one                       # the COUNT is not a rate


def test_first_frame_can_never_freeze_as_in_eztrack():
    """Deliberate, not an off-by-one: `CumThresh[0]` is 0 in the source.

    Pinned so that a later "tidy-up" of the loop is caught rather than silently
    shifting every freeze fraction by one frame per recording.
    """
    assert not fz.freeze_mask(np.zeros(40), 1.0, 3)[0]
    assert fz.freeze_mask(np.zeros(40), 1.0, 3)[1:].all()


def test_motion_from_hist_matches_direct_thresholding():
    rng = np.random.default_rng(1)
    dif = rng.random((7, 400)) * 40.0
    hist = np.stack([np.bincount(mo.quantise(row), minlength=mo.N_BINS)
                     for row in dif])
    for cutoff in (0.0, 1.0, 7.5, 25.5, 63.0):
        want = (dif > cutoff).sum(axis=1)
        got = mo.motion_from_hist(hist, cutoff)
        # The bin holding the cutoff is dropped, so the histogram count can be
        # short by at most that bin's occupancy -- never over.
        assert (got <= want).all()
        assert (want - got <= np.array(
            [((row > cutoff) & (row < mo.bin_edge(
                int(np.floor(cutoff * mo.BIN_SCALE)) + 1))).sum()
             for row in dif])).all()


def test_quantise_clamps_into_the_overflow_bin():
    q = mo.quantise(np.array([0.0, 0.1, 63.9, 64.0, 400.0]))
    assert q[0] == 0 and q[-1] == mo.N_BINS - 1
    assert (q < mo.N_BINS).all() and (q >= 0).all()


def test_exclusion_mask_excludes_everything_when_undefined():
    """An unmaskable frame must contribute NO arena, not the animal as arena."""
    pose = np.full((7, 2), np.nan)
    pose[0] = (100.0, 100.0)
    assert mo.exclusion_mask(pose, 10.0, (48, 64)).all()


def test_exclusion_mask_is_the_box_dilated_by_the_margin():
    pose = np.array([[100.0, 100.0], [120.0, 120.0], [110.0, 110.0],
                     [105.0, 115.0], [115.0, 105.0], [108.0, 112.0],
                     [112.0, 108.0]])
    m = mo.exclusion_mask(pose, 5.0, (480, 640))
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    assert rows[0] == 95 and rows[-1] == 125
    assert cols[0] == 95 and cols[-1] == 125


def test_floor_percentile_is_never_below_the_true_percentile():
    hist = np.zeros(mo.N_BINS, dtype=np.int64)
    hist[0] = 999_000
    hist[40] = 1_000
    p = mo.floor_percentile(hist, 99.99)
    assert p >= mo.bin_edge(40)
    assert mo.cutoff_of(hist) == pytest.approx(mo.CUTOFF_MULTIPLIER * p)


def test_floor_percentile_refuses_an_empty_sample():
    assert np.isnan(mo.floor_percentile(np.zeros(mo.N_BINS, dtype=np.int64)))


def test_sweep_is_the_registered_grid_with_one_headline():
    arms = fz.sweep_arms(30.0)
    assert len(arms) == 3 * 3 * 3 + 1
    assert sum(1 for a in arms if a["headline"]) == 1
    ez = [a for a in arms if not a["derived"]]
    assert len(ez) == 1 and ez[0]["min_frames"] == 15
    assert ez[0]["freeze_thresh"] == 200.0


def test_min_duration_is_registered_in_seconds_not_frames():
    """M13: a threshold in samples is a different threshold at a different rate."""
    a30 = {a["name"]: a["min_frames"] for a in fz.sweep_arms(30.0)}
    a15 = {a["name"]: a["min_frames"] for a in fz.sweep_arms(15.0)}
    for name in a30:
        if name == "eztrack_default":
            # ezTrack's own constant IS in frames, and does not rescale. That
            # is the defect the arm exists to display.
            assert a30[name] == a15[name] == 15
        else:
            # Halving the rate must roughly halve the frame count. `frames()`
            # rounds, so 0.5 s is 15 frames at 30 fps and 8 at 15 -- exact
            # doubling is not the property, RESCALING AT ALL is.
            assert abs(a30[name] - 2 * a15[name]) <= 1, name
            assert a30[name] != a15[name], name


def test_eztrack_defaults_are_never_the_headline():
    """§9.5: they appear only as the labelled untransplanted arm."""
    arms = fz.sweep_arms(30.0)
    assert all(a["derived"] for a in arms if a["headline"])


def test_floor_read_refuses_when_nothing_is_usable():
    rd = fz.floor_read([{"usable": False, "context": "A", "mt_cutoff": 1.0}],
                       scored_object={"dataset": "luna", "arm": "t"},
                       n_effective=1)
    assert rd.verdict == "NOT_A_RESULT"


def test_floor_read_reports_the_context_gap():
    per = [{"usable": True, "context": "A", "mt_cutoff": 20.0},
           {"usable": True, "context": "A", "mt_cutoff": 22.0},
           {"usable": True, "context": "B", "mt_cutoff": 10.0},
           {"usable": True, "context": "B", "mt_cutoff": 12.0}]
    rd = fz.floor_read(per, scored_object={"dataset": "luna", "arm": "t"},
                       n_effective=4)
    assert rd.verdict == "PASS"
    assert (rd.detail or {})["context_gap"] == pytest.approx(10.0)
    assert "ezTrack's published default" in rd.reason
