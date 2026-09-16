"""Step 3: clumps or a continuum, and the refusals in between."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import vocab as vb


def _knn(points, k=vb.KNN):
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx, np.take_along_axis(d, idx, axis=1)


def test_components_finds_two_planted_groups_without_being_told_two():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, size=(60, 3))
    b = rng.normal(8, 0.05, size=(60, 3)) + 8
    idx, dist = _knn(np.vstack([a, b]))
    lab, s = vb.components(idx, dist, theta=1.0, min_size=20)
    assert s["n_clumps"] == 2
    assert set(np.unique(lab).tolist()) == {0, 1}
    assert s["unassigned_fraction"] == 0.0


def test_components_leaves_a_continuum_unassigned_or_in_one_blob():
    """A Gaussian cloud must not come back as a vocabulary.

    Below every pairwise distance it is all unassigned; above the bulk of them
    it is ONE blob. Neither is a vocabulary, and both must be reachable without
    the labelling crashing.
    """
    rng = np.random.default_rng(1)
    idx, dist = _knn(rng.normal(size=(300, 6)))
    lab, s = vb.components(idx, dist, theta=0.05, min_size=20)
    assert s["n_clumps"] == 0 and s["unassigned_fraction"] == 1.0
    assert (lab == -1).all()
    lab2, s2 = vb.components(idx, dist, theta=5.0, min_size=20)
    assert s2["n_clumps"] == 1 and s2["largest_clump_fraction"] > 0.9


def test_clumps_are_numbered_largest_first():
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.normal(0, 0.05, size=(100, 2)),
                     rng.normal(20, 0.05, size=(30, 2))])
    idx, dist = _knn(pts)
    lab, s = vb.components(idx, dist, theta=1.0, min_size=20)
    assert s["n_clumps"] == 2
    assert (lab == 0).sum() > (lab == 1).sum()


def test_a_component_below_the_minimum_is_unassigned_not_a_clump():
    rng = np.random.default_rng(3)
    pts = np.vstack([rng.normal(0, 0.05, size=(40, 2)),
                     rng.normal(30, 0.05, size=(5, 2))])
    idx, dist = _knn(pts)
    lab, s = vb.components(idx, dist, theta=1.0, min_size=20)
    assert s["n_clumps"] == 1
    assert s["unassigned_fraction"] == pytest.approx(5 / 45)


def test_bimodality_separates_two_peaks_from_a_heavy_tail():
    """BC alone cannot -- it rises on both -- so the ingredients are reported.

    The denominator takes EXCESS kurtosis. With the raw fourth moment the
    statistic inverts: a clean two-peak mixture scored 0.237 and an exponential
    0.427, which is backwards.
    """
    rng = np.random.default_rng(4)
    two = np.concatenate([rng.normal(0, 1, 5000), rng.normal(8, 1, 5000)])
    tail = rng.standard_exponential(10000)
    b, t = vb.bimodality(two), vb.bimodality(tail)
    assert b["bimodality_coefficient"] > 5 / 9
    assert b["excess_kurtosis"] < 0            # flat-topped: two peaks
    assert t["excess_kurtosis"] > 3.0          # heavy-tailed, not two peaks
    assert b["bimodality_coefficient"] > t["bimodality_coefficient"]


def test_bimodality_refuses_on_degenerate_input():
    assert "why" in vb.bimodality([1.0, 1.0])
    assert "why" in vb.bimodality(np.ones(100))


def test_bic_gain_prefers_two_components_only_when_there_are_two():
    rng = np.random.default_rng(5)
    two = np.concatenate([rng.normal(0, 1, 4000), rng.normal(7, 1, 4000)])
    one = rng.normal(0, 1, 8000)
    assert vb.gmm1d_bic_gain(two, seed=0)["bic_gain"] > 0
    assert vb.gmm1d_bic_gain(one, seed=0)["bic_gain"] < 0


def test_participation_reports_concentration_beside_the_count():
    """Many animals with one supplying most members is a different object."""
    lab = np.array([0] * 10 + [1] * 10)
    an = ["a"] * 8 + ["b", "c"] + ["d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    rows = vb.participation(lab, an, n_animals_total=20)
    assert rows[0]["n_animals"] == 3 and rows[0]["top_animal_share"] == 0.8
    assert rows[1]["n_animals"] == 10 and rows[1]["top_animal_share"] == 0.1


def test_participation_read_fails_when_no_clump_is_widely_shared():
    rows = [{"clump": i, "size": 50, "n_animals": 9, "animal_fraction": 0.1,
             "top_animal_share": 0.5} for i in range(6)]
    rd = vb.participation_read(rows, n_animals_total=89, scored_object={"a": 1},
                               n_effective=89)
    assert rd.verdict == "FAIL" and "artifacts" in rd.reason


def test_participation_read_refuses_with_no_clumps_at_all():
    rd = vb.participation_read([], n_animals_total=89, scored_object={"a": 1},
                               n_effective=89)
    assert rd.verdict == "FAIL" and "nothing" in rd.reason


def test_clump_read_calls_a_continuum_when_the_nulls_match_the_corpus():
    obs = {"n_clumps": 12, "unassigned_fraction": 0.4, "bic_gain": 50.0}
    nulls = {"microstate": {"n_clumps": 14, "unassigned_fraction": 0.42,
                            "bic_gain": 60.0}}
    rd = vb.clump_read(obs, nulls, scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "FAIL" and "CONTINUUM" in rd.reason.upper()


def test_clump_read_needs_the_corpus_to_beat_every_null_on_both_statistics():
    obs = {"n_clumps": 40, "unassigned_fraction": 0.3, "bic_gain": 900.0}
    one = {"microstate": {"n_clumps": 5, "unassigned_fraction": 0.9,
                          "bic_gain": 10.0}}
    assert vb.clump_read(obs, one, scored_object={"a": 1},
                         n_effective=89).verdict == "PASS"
    # A null that matches on ONE statistic is enough to withhold the PASS.
    two = {**one, "microstate0": {"n_clumps": 60, "unassigned_fraction": 0.3,
                                  "bic_gain": 20.0}}
    assert vb.clump_read(obs, two, scored_object={"a": 1},
                         n_effective=89).verdict == "FAIL"


def test_clump_read_refuses_without_a_null():
    rd = vb.clump_read({"n_clumps": 9, "unassigned_fraction": 0.1,
                        "bic_gain": 5.0}, {},
                       scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "INCONCLUSIVE"


def test_coverage_read_refuses_a_vocabulary_that_covers_almost_nothing():
    rd = vb.coverage_read({"unassigned_fraction": 0.97, "min_size": 20},
                          scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "NOT_A_RESULT"
    rd2 = vb.coverage_read({"unassigned_fraction": 0.35, "min_size": 20},
                           scored_object={"a": 1}, n_effective=89)
    assert rd2.verdict == "PASS" and "65.0%" in rd2.reason
