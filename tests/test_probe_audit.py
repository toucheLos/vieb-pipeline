"""The probe autopsy's arithmetic, and the anchor that makes it refusable."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import probe_audit as pa

OBJ = {"dataset": "luna", "arm": "probe_audit", "group": "shape"}


def test_percentile_of_is_uniform_on_its_own_distribution():
    """Scoring every frame against its own distribution averages to 0.5.

    This is the identity the anchor relies on. If it ever stops holding, the
    whole stage is unreadable and `anchor_read` is what says so.
    """
    d = np.random.default_rng(0).normal(size=5000)
    p = pa.percentile_of(d, np.arange(d.size))
    assert abs(float(p.mean()) - 0.5) < 0.01
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_percentile_of_averages_ties():
    """A constant stretch must not hand every frame in it a rank of zero."""
    d = np.array([0.0, 0.0, 0.0, 0.0])
    p = pa.percentile_of(d, np.arange(4))
    assert np.allclose(p, 0.5)


def test_percentile_of_ranks_a_spike_high():
    d = np.concatenate([np.zeros(99), [10.0]])
    assert pa.percentile_of(d, [99])[0] > 0.99


def test_edge_frames_are_slice_local_and_cover_both_ends():
    """Indices come back zero-based within the slice, never corpus-global.

    Indexing a per-recording `D` with a corpus-global frame is the seam bug this
    repo asserts against everywhere else.
    """
    mask = np.zeros(100, dtype=bool)
    mask[60:70] = True                      # one run, inside recording [50,100)
    got = pa.edge_frames(mask, lo=50, hi=100, tol=2)
    assert got.min() >= 0 and got.max() < 50
    # start at slice-local 10, stop at slice-local 20
    assert set(range(8, 13)) <= set(got.tolist())
    assert set(range(18, 23)) <= set(got.tolist())


def test_edge_frames_clips_at_the_slice_boundary():
    mask = np.zeros(20, dtype=bool)
    mask[0:3] = True
    got = pa.edge_frames(mask, lo=0, hi=20, tol=5)
    assert got.min() == 0


def test_edge_frames_empty_when_nothing_planted():
    assert pa.edge_frames(np.zeros(50, dtype=bool), lo=0, hi=50, tol=2).size == 0


def test_populations_random_is_count_matched():
    d = np.random.default_rng(1).normal(size=1000)
    rng = np.random.default_rng(2)
    got = pa.populations(d, np.arange(40), np.arange(10), rng)
    assert got["n_planted"] == 40 and got["n_fired"] == 10
    assert abs(got["pct_random"] - 0.5) < 0.2


def test_populations_reports_nan_rather_than_zero_when_empty():
    """A recording that contributes nothing is dropped, not counted as zero."""
    d = np.random.default_rng(3).normal(size=100)
    got = pa.populations(d, np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                         np.random.default_rng(4))
    assert np.isnan(got["pct_planted"]) and np.isnan(got["pct_fired"])


def _rows(pct_planted, pct_random=0.5, pct_fired=0.9, n=12):
    return [{"animal": f"a{i}", "pct_planted": pct_planted,
             "pct_random": pct_random, "pct_fired": pct_fired}
            for i in range(n)]


def test_anchor_refuses_when_the_random_draw_is_not_at_half():
    r = pa.anchor_read(_rows(0.9, pct_random=0.72), scored_object=OBJ,
                       n_effective=12)
    assert r.verdict == "NOT_A_RESULT"
    assert "ANCHOR MOVED" in r.reason


def test_anchor_passes_at_half():
    assert pa.anchor_read(_rows(0.9), scored_object=OBJ,
                          n_effective=12).verdict == "PASS"


def test_separation_withdraws_step_b_when_the_plant_has_no_edge():
    """The registered first row: interval includes zero."""
    rng = np.random.default_rng(5)
    rows = [{"animal": f"a{i}", "pct_planted": 0.5 + rng.normal(0, 0.01),
             "pct_random": 0.5, "pct_fired": 0.9} for i in range(12)]
    r = pa.separation_read(rows, scored_object=OBJ, n_effective=12)
    assert r.verdict == "NOT_A_RESULT"
    assert "NO EDGE TO FIND" in r.reason


def test_separation_is_inconclusive_on_a_weak_edge():
    """Above zero, but below the lower bound of the firing interval."""
    r = pa.separation_read(_rows(0.60, pct_fired=0.95), scored_object=OBJ,
                           n_effective=12)
    assert r.verdict == "INCONCLUSIVE"
    assert "WEAK edge" in r.reason


def test_separation_lets_step_b_stand_when_the_plant_is_findable():
    r = pa.separation_read(_rows(0.95, pct_fired=0.95), scored_object=OBJ,
                           n_effective=12)
    assert r.verdict == "PASS"
    assert "Step B stands" in r.reason


def test_frame_level_interval_is_reported_but_named_unquotable():
    r = pa.separation_read(_rows(0.9), scored_object=OBJ, n_effective=12)
    assert "frame_level_interval_do_not_quote" in r.detail


def test_every_read_carries_n_effective():
    """Required by the convention; a Read without it is not a claim."""
    for r in (pa.anchor_read(_rows(0.9), scored_object=OBJ, n_effective=12),
              pa.separation_read(_rows(0.9), scored_object=OBJ,
                                 n_effective=12)):
        assert r.n_effective == 12
        assert r.scored_object and r.reason


def test_n_effective_is_refused_when_absent():
    with pytest.raises((TypeError, ValueError)):
        pa.anchor_read(_rows(0.9), scored_object=OBJ, n_effective=None)
