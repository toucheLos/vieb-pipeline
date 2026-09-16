"""Segments as bank rows: the exclusions, the warp, and the control."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import breaks as bk, embed, planted as pl


def _rows(spans):
    return [{"start": s, "stop": e, "n_frames": e - s, "abstain_frac": 0.0,
             "fittable": True} for s, e in spans]


def test_a_segment_beside_an_abstain_block_is_marked_even_with_none_inside():
    """The population the registered exclusion exists for.

    `segment_table` cuts at every abstain run, so a segment can contain no
    abstained frame and still owe one of its edges to the dropout rather than
    to a break in acceleration.
    """
    ab = np.zeros(40, dtype=bool)
    ab[10:14] = True
    rows = _rows([(0, 10), (14, 30), (30, 40)])
    got = embed.mark_edges(rows, ab, lo=0, hi=40)
    assert got[0]["abstain_adjacent"] is True      # stop touches the run
    assert got[1]["abstain_adjacent"] is True      # start follows the run
    assert got[2]["abstain_adjacent"] is False
    assert [bool(x) for x in embed.selectable(got)] == [False, False, True]


def test_selectable_needs_all_three_conditions_not_two():
    ab = np.zeros(40, dtype=bool)
    rows = embed.mark_edges(_rows([(0, 20), (20, 40)]), ab, lo=0, hi=40)
    rows[0]["fittable"] = False
    rows[1]["abstain_frac"] = 0.02
    assert not embed.selectable(rows).any()


def test_recording_seams_are_marked_but_not_excluded():
    """A recording edge is a real edge, not a tracking failure."""
    ab = np.zeros(30, dtype=bool)
    got = embed.mark_edges(_rows([(0, 15), (15, 30)]), ab, lo=0, hi=30)
    assert got[0]["seam_adjacent"] and got[1]["seam_adjacent"]
    assert embed.selectable(got).all()


def test_stack_gives_the_rectangular_shape_bank_build_requires():
    """The whole reason segments could not previously enter the search."""
    x = np.cumsum(np.random.default_rng(0).normal(size=(200, 5)), axis=0)
    rows = _rows([(0, 7), (7, 90), (90, 200)])
    xs, start = embed.stack_segments(x, rows, cols=(0, 1, 2))
    assert xs.shape == (3 * embed.SEG_GRID, 3)
    assert start.tolist() == [0, embed.SEG_GRID, 2 * embed.SEG_GRID]
    # Rectangular gather, the thing `iter_chunks` does:
    gathered = xs[start[:, None] + np.arange(embed.SEG_GRID)[None, :]]
    assert gathered.shape == (3, embed.SEG_GRID, 3)


def test_the_warp_preserves_endpoints_and_is_length_invariant():
    """A line resampled from any length is the same line."""
    long = np.linspace(0, 1, 137)[:, None] * np.array([[1.0, -2.0]])
    short = np.linspace(0, 1, 9)[:, None] * np.array([[1.0, -2.0]])
    a = embed.resample_segment(long, 0, 137)
    b = embed.resample_segment(short, 0, 9)
    assert np.allclose(a, b, atol=1e-9)
    assert np.allclose(a[0], long[0]) and np.allclose(a[-1], long[-1])


def test_matched_windows_match_the_length_distribution_and_avoid_abstain():
    rng = np.random.default_rng(0)
    ab = np.zeros(6000, dtype=bool)
    ab[2000:2400] = True
    bounds = np.array([0, 3000, 6000], dtype=np.int64)
    durs = np.array([10, 40, 90, 200], dtype=np.int64)
    win = embed.matched_windows(rng, bounds=bounds, abstain=ab,
                                durations=durs, n_target=400)
    assert len(win) > 300
    assert set(w["n_frames"] for w in win) <= set(durs.tolist())
    for w in win:
        assert not ab[w["start"]:w["stop"]].any()


def test_matched_windows_never_straddle_a_recording_seam():
    """The invariant every stage in this project asserts by test."""
    rng = np.random.default_rng(1)
    bounds = np.array([0, 500, 1000], dtype=np.int64)
    win = embed.matched_windows(rng, bounds=bounds,
                                abstain=np.zeros(1000, dtype=bool),
                                durations=np.array([120], dtype=np.int64),
                                n_target=300)
    assert win
    for w in win:
        assert not (w["start"] < 500 < w["stop"])


def test_open_end_distance_is_per_frame_and_not_a_length_readout():
    a = np.ones((10, 3))
    b = np.ones((400, 3))
    assert embed.open_end_distance(a, b) == pytest.approx(0.0, abs=1e-12)
    c = np.zeros((50, 3))
    # Same per-frame gap, very different lengths: same distance.
    assert embed.open_end_distance(a, c) == pytest.approx(
        embed.open_end_distance(np.ones((300, 3)), c), abs=1e-12)


def test_planting_realised_occupancy_tracks_the_request_at_the_low_end():
    """Probabilistic rounding, not `round()`.

    With integer rounding 0.25% goes to zero instances and 0.5% and 1.0% both
    go to one, so two cells meant to differ by a factor of two report the same
    realised occupancy. A dose-response whose dose does not move is not one.
    """
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(60000, 4)), axis=0)
    bounds = np.array([0, 20000, 40000, 60000], dtype=np.int64)
    tpl = pl.ego_template(x, 60, rng, n_avg=20)
    got = {}
    for occ in (0.0025, 0.005, 0.01):
        _y, mask, meta = pl.ego_plant(x, tpl, np.random.default_rng(7),
                                      bounds=bounds, occupancy=occ)
        got[occ] = meta["realized_occupancy"]
        assert meta["realized_occupancy"] == pytest.approx(mask.mean())
    assert got[0.0025] < got[0.005] < got[0.01]
    assert got[0.01] == pytest.approx(0.01, rel=0.4)


def test_planting_never_crosses_a_recording_seam():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=(9000, 3)), axis=0)
    bounds = np.array([0, 3000, 6000, 9000], dtype=np.int64)
    tpl = pl.ego_template(x, 50, rng, n_avg=20)
    _y, mask, _m = pl.ego_plant(x, tpl, rng, bounds=bounds, occupancy=0.1)
    run = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(run == 1), np.flatnonzero(run == -1)):
        for edge in (3000, 6000):
            assert not (s < edge < e)


def test_planting_avoids_blocked_frames():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=(8000, 3)), axis=0)
    blocked = np.zeros(8000, dtype=bool)
    blocked[1000:5000] = True
    tpl = pl.ego_template(x, 40, rng, n_avg=20, valid=~blocked)
    _y, mask, _m = pl.ego_plant(x, tpl, rng,
                                bounds=np.array([0, 8000], dtype=np.int64),
                                occupancy=0.2, blocked=blocked)
    assert not (mask & blocked).any()


def test_template_refuses_when_too_few_clean_windows_exist():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=(300, 3)), axis=0)
    bad = np.ones(300, dtype=bool)
    bad[:80] = False
    with pytest.raises(ValueError, match="stereotype"):
        pl.ego_template(x, 50, rng, n_avg=40, valid=~bad)


def test_planted_segments_are_stereotyped_and_not_identical():
    """An exact repeat is recoverable by any method and overstates the floor."""
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(size=(20000, 3)), axis=0)
    tpl = pl.ego_template(x, 60, rng, n_avg=20)
    y, mask, _m = pl.ego_plant(x, tpl, rng,
                               bounds=np.array([0, 20000], dtype=np.int64),
                               occupancy=0.2, jitter=0.05)
    run = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(run == 1)
    assert starts.size >= 2
    a, b = y[starts[0]:starts[0] + 60], y[starts[1]:starts[1] + 60]
    assert not np.allclose(a, b)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.8


def test_the_channel_groups_partition_the_ego_dimensions():
    g = bk.CHANNEL_GROUPS
    assert set(g["shape"]) | set(g["twist"]) == set(g["both"])
    assert not set(g["shape"]) & set(g["twist"])
