"""Two distances, and the property that separates them."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import dist as ds


def _ramp(n, c=3, slope=1.0):
    return (np.arange(n, dtype=np.float64)[:, None] * slope
            * np.ones((1, c)))


def test_open_end_is_blind_to_duration_and_union_end_is_not():
    """The single property the two metrics exist to differ on.

    A short segment and a long one that agree on their shared prefix are
    identical under `open` and far apart under `union`. If this ever stopped
    holding, Step A would be running one metric twice.
    """
    short = _ramp(16)
    long = _ramp(400)
    assert ds.open_end(short, long) == pytest.approx(0.0, abs=1e-12)
    assert ds.union_end(short, long) > 1.0


def test_both_metrics_are_zero_on_identical_segments():
    x = _ramp(50)
    assert ds.open_end(x, x) == pytest.approx(0.0, abs=1e-12)
    assert ds.union_end(x, x) == pytest.approx(0.0, abs=1e-12)


def test_both_metrics_are_symmetric():
    a, b = _ramp(20), _ramp(37, slope=0.4)
    assert ds.open_end(a, b) == pytest.approx(ds.open_end(b, a))
    assert ds.union_end(a, b) == pytest.approx(ds.union_end(b, a))


def test_per_frame_normalisation_keeps_them_from_being_length_readouts():
    """A constant offset over any length must score the same.

    Without dividing by the extent, a longer comparison would score worse for
    being longer, and the metric would rank on duration rather than on shape.
    """
    off = np.ones((1, 3))
    a10, b10 = np.zeros((10, 3)), np.zeros((10, 3)) + off
    a99, b99 = np.zeros((99, 3)), np.zeros((99, 3)) + off
    assert ds.open_end(a10, b10) == pytest.approx(ds.open_end(a99, b99))
    assert ds.union_end(a10, b10) == pytest.approx(ds.union_end(a99, b99))


def test_union_holds_the_last_frame_rather_than_padding_with_zero():
    """Zero is "no change" in this space, not "no data".

    A short constant segment continued against a long identical constant one
    must be zero distance. Zero-padding would instead charge the whole tail.
    """
    short = np.full((8, 3), 2.0)
    long = np.full((200, 3), 2.0)
    assert ds.union_end(short, long) == pytest.approx(0.0, abs=1e-12)


def test_metric_fn_refuses_an_unknown_name():
    with pytest.raises(ValueError, match="unknown metric"):
        ds.metric_fn("dtw")


def test_rerank_finds_the_true_nearest_among_candidates():
    blocks = [_ramp(20), _ramp(20, slope=1.001), _ramp(20, slope=9.0)]
    i, d = ds.rerank(blocks, 0, [1, 2], metric="open")
    assert i == 1 and d < 0.1


def test_rerank_never_returns_the_query_or_a_forbidden_candidate():
    """The forbidden set is the query's own animal: this is a CROSS-animal
    statistic, and a segment's own animal supplies near-copies of itself."""
    blocks = [_ramp(20), _ramp(20, slope=1.0), _ramp(20, slope=5.0)]
    i, _d = ds.rerank(blocks, 0, [0, 1, 2], metric="open",
                      forbid=np.array([False, True, False]))
    assert i == 2


def test_rerank_refuses_when_every_candidate_is_masked():
    blocks = [_ramp(12), _ramp(12)]
    i, d = ds.rerank(blocks, 0, [1], metric="union",
                     forbid=np.array([True]))
    assert i == -1 and not np.isfinite(d)


def test_pairwise_excludes_the_query_s_own_animal():
    blocks = [_ramp(10), _ramp(10, slope=1.0), _ramp(10, slope=7.0)]
    animal = np.array(["a", "a", "b"])
    idx, _d = ds.pairwise(blocks, [0], animal, metric="open")
    assert idx[0] == 2          # 1 is the same animal, so it cannot win


def test_recovery_read_passes_when_the_index_finds_the_exact_answer():
    n = 200
    ex = np.arange(n)
    d = np.full(n, 2.0)
    rd = ds.recovery_read(ex, ex, d, d, metric="open",
                          scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "PASS" and "100.0%" in rd.reason


def test_recovery_read_fails_when_the_index_loses_the_answer():
    """A retrieval that misses the true neighbour shifts every distance
    quantile the statistic reads, so it is a FAIL and not a footnote."""
    n = 200
    ex = np.arange(n)
    ap = np.where(np.arange(n) % 2 == 0, ex, -1 + 0 * ex)
    rd = ds.recovery_read(ap, ex, np.full(n, 3.0), np.full(n, 2.0),
                          metric="open", scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "FAIL" and "losing the answer" in rd.reason


def test_recovery_read_reports_inflation_beside_agreement():
    """They fail differently: missing the true neighbour but landing on one
    equally close has not damaged the statistic."""
    n = 300
    ex = np.arange(n)
    ap = ex.copy()
    ap[:30] = -1
    rd = ds.recovery_read(ap, ex, np.full(n, 2.02), np.full(n, 2.0),
                          metric="union", scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "PASS"
    assert rd.detail["median_distance_ratio"] == pytest.approx(1.01, abs=1e-6)


def test_recovery_read_refuses_on_too_few_queries():
    rd = ds.recovery_read([1, 2], [1, 2], [1.0, 1.0], [1.0, 1.0],
                          metric="open", scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "INCONCLUSIVE"


def test_the_registered_constants_are_what_the_registration_says():
    assert ds.PREFIX_L == 16 and ds.CANDIDATES == 64
    assert ds.METRICS == ("open", "union")
