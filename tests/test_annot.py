"""The boundary matcher, the ceiling, and the two refusals that protect them."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import annot as an

OBJ = {"dataset": "luna", "arm": "annot", "split": "tune"}


# --- the matcher --------------------------------------------------------

def test_match_is_one_to_one():
    """Three hypotheses around one reference earn ONE true positive.

    This is the bug `scripts/breaks.py:165 _agreement` has and this must not:
    counting unmatched hits inflates recall exactly where a detector is
    noisiest.
    """
    assert len(an.match([100], [99, 100, 101], tol=5)) == 1


def test_match_prefers_the_nearer_candidate():
    """Nearest-first, so a far pairing cannot steal a boundary from a near one."""
    got = dict(an.match([100, 108], [107, 101], tol=5))
    a = np.sort([100, 108])
    b = np.sort([107, 101])
    assert b[got[0]] == 101 and b[got[1]] == 107
    assert a[0] == 100 and a[1] == 108


def test_match_respects_the_tolerance():
    assert an.match([100], [106], tol=5) == []
    assert len(an.match([100], [105], tol=5)) == 1


def test_match_is_symmetric_in_count():
    a, b = [10, 40, 70], [11, 39, 200]
    assert len(an.match(a, b, 3)) == len(an.match(b, a, 3))


def test_match_is_order_independent():
    assert len(an.match([70, 10, 40], [39, 200, 11], 3)) == \
           len(an.match([10, 40, 70], [11, 39, 200], 3))


def test_match_empty_sets():
    assert an.match([], [1, 2], 5) == []
    assert an.match([1, 2], [], 5) == []


# --- precision / recall / F1 -------------------------------------------

def test_prf_counts_extra_hypotheses_as_false_positives():
    got = an.prf([100], [99, 100, 101], tol=5)
    assert got["tp"] == 1.0
    assert got["recall"] == 1.0
    assert got["precision"] == pytest.approx(1 / 3)


def test_prf_f1_is_symmetric():
    a, b = [10, 40, 70, 90], [11, 39, 200]
    assert an.prf(a, b, 3)["f1"] == pytest.approx(an.prf(b, a, 3)["f1"])


def test_two_empty_sets_agree_completely():
    """Both raters saying "nothing changed" is agreement, not a zero.

    Scoring it zero would punish the exact observation the continuum question
    turns on.
    """
    assert an.prf([], [], tol=5)["f1"] == 1.0


def test_one_empty_set_is_total_disagreement():
    assert an.prf([], [5], tol=5)["f1"] == 0.0
    assert an.prf([5], [], tol=5)["f1"] == 0.0


def test_perfect_agreement_is_one():
    assert an.prf([10, 50, 90], [10, 50, 90], tol=2)["f1"] == 1.0


# --- the chance level ---------------------------------------------------

def test_chance_f1_rises_with_tolerance():
    """The reason a chance level is computed at all: at +/-10 it is not small."""
    rng = np.random.default_rng(0)
    lo = an.chance_f1(8, 8, 300, 2, rng, n_restarts=60)
    hi = an.chance_f1(8, 8, 300, 10, rng, n_restarts=60)
    assert hi > lo


def test_chance_f1_of_two_empty_sets_is_one():
    assert an.chance_f1(0, 0, 300, 5, np.random.default_rng(0)) == 1.0


def test_chance_f1_is_far_below_one_for_sparse_marks():
    rng = np.random.default_rng(1)
    assert an.chance_f1(3, 3, 3000, 2, rng, n_restarts=60) < 0.1


# --- the pairing rows ---------------------------------------------------

CLIPS = {"c0": {"animal": "a1", "n_frames": 300, "fps": 30.0},
         "c1": {"animal": "a2", "n_frames": 300, "fps": 30.0}}


def test_pair_rows_only_compares_clips_both_raters_rated():
    """A clip one rater never opened is missing data, not a disagreement."""
    marks = {"r1": {"c0": [10, 100], "c1": [50]}, "r2": {"c0": [11, 99]}}
    rows = an.pair_rows(marks, CLIPS)
    assert {r["clip"] for r in rows} == {"c0"}


def test_pair_rows_emits_every_tolerance():
    marks = {"r1": {"c0": [10]}, "r2": {"c0": [11]}}
    rows = an.pair_rows(marks, CLIPS)
    assert sorted(r["tol"] for r in rows) == sorted(an.TOLERANCES)


def test_pair_rows_covers_every_rater_pair():
    marks = {"r1": {"c0": [10]}, "r2": {"c0": [11]}, "r3": {"c0": [12]}}
    rows = an.pair_rows(marks, CLIPS, tolerances=(5,))
    assert {(r["rater_a"], r["rater_b"]) for r in rows} == \
           {("r1", "r2"), ("r1", "r3"), ("r2", "r3")}


# --- the ceiling --------------------------------------------------------

def _rows(f1, chance=0.05, n=12, tol=5):
    return [{"rater_a": "r1", "rater_b": "r2", "clip": f"c{i}",
             "animal": f"a{i}", "tol": tol, "n_frames": 300,
             "chance_f1": chance, "f1": f1, "precision": f1, "recall": f1,
             "tp": 1.0, "n_ref": 1.0, "n_hyp": 1.0} for i in range(n)]


def test_ceiling_refuses_a_single_rater():
    """One rater is no ceiling, and a detector scored against it scores nothing."""
    rows = [dict(r, rater_b="r1") for r in _rows(0.8)]
    r = an.ceiling_read(rows, tol=5, scored_object=OBJ, n_effective=1)
    assert r.verdict == "NOT_A_RESULT"
    assert "NO CEILING" in r.reason


def test_ceiling_fails_when_agreement_reaches_only_chance():
    r = an.ceiling_read(_rows(0.06, chance=0.30), tol=5, scored_object=OBJ,
                        n_effective=12)
    assert r.verdict == "FAIL"
    assert "DO NOT AGREE" in r.reason


def test_ceiling_passes_and_says_it_is_not_one():
    r = an.ceiling_read(_rows(0.72), tol=5, scored_object=OBJ, n_effective=12)
    assert r.verdict == "PASS"
    assert "NOT 1.0" in r.reason


def test_ceiling_selects_its_own_tolerance():
    rows = _rows(0.9, tol=2) + _rows(0.2, chance=0.4, tol=10)
    assert an.ceiling_read(rows, tol=2, scored_object=OBJ,
                           n_effective=12).verdict == "PASS"
    assert an.ceiling_read(rows, tol=10, scored_object=OBJ,
                           n_effective=12).verdict == "FAIL"


def test_coverage_is_descriptive_and_claims_no_verdict():
    """No threshold for 'near-zero' was registered, so none is invented."""
    marks = {"r1": {"c0": [10, 100], "c1": []}}
    r = an.coverage_read(marks, CLIPS, scored_object=OBJ, n_effective=2)
    assert r.verdict == "NOT_A_RESULT"
    assert r.detail["n_empty"] == 1
    assert "after the fact" in r.reason


def test_every_read_carries_n_effective_and_a_reason():
    for r in (an.ceiling_read(_rows(0.7), tol=5, scored_object=OBJ,
                              n_effective=12),
              an.coverage_read({"r1": {"c0": [1]}}, CLIPS, scored_object=OBJ,
                               n_effective=1)):
        assert r.n_effective is not None and r.scored_object and r.reason


def test_tolerances_are_the_registered_three():
    assert an.TOLERANCES == (2, 5, 10)


# --- the decomposition, and the offset distribution ---------------------

def test_decompose_partitions_every_row_exactly_once():
    marks = {"r1": {"c0": [10, 100], "c1": []},
             "r2": {"c0": [11], "c1": []}}
    rows = an.pair_rows(marks, CLIPS)
    got = an.decompose(rows, tol=5)
    assert (got["n_both_empty"] + got["n_one_empty"]
            + got["n_both_marked"]) == got["n_rows"]


def test_a_both_empty_clip_scores_one_on_the_observed_AND_on_chance():
    """The reason the decomposition exists.

    Two raters who both say "nothing changed" agree, so `prf` returns 1.0
    deliberately -- and `chance_f1` returns 1.0 too. Such a clip therefore
    contributes the maximum to BOTH sides of the comparison and cannot
    separate them. A headline F1 quoted without saying how many of these it
    contains overstates agreement.
    """
    marks = {"r1": {"c0": []}, "r2": {"c0": []}}
    row = an.pair_rows(marks, CLIPS)[0]
    assert row["f1"] == 1.0 and row["chance_f1"] == 1.0


def test_decompose_read_claims_no_verdict_and_says_so():
    marks = {"r1": {"c0": [10, 100], "c1": []},
             "r2": {"c0": [11], "c1": []}}
    rows = an.pair_rows(marks, CLIPS)
    r = an.decompose_read(rows, tol=5, scored_object=OBJ, n_effective=2)
    assert r.verdict == "NOT_A_RESULT"
    assert "does not restate or replace" in r.reason


def test_offsets_are_signed_with_a_later_mark_positive():
    marks = {"aa": {"c0": [110]}, "bb": {"c0": [100]}}
    got = an.offsets(marks, CLIPS)
    assert [r["offset"] for r in got] == [10]


def test_offsets_are_undefined_where_one_rater_marked_nothing():
    """There is no nearest mark to measure against, so no row is emitted."""
    marks = {"aa": {"c0": [110], "c1": [5]}, "bb": {"c0": [100], "c1": []}}
    assert {r["clip"] for r in an.offsets(marks, CLIPS)} == {"c0"}


def test_offset_read_refuses_rather_than_dividing_by_nothing():
    r = an.offset_read([], scored_object=OBJ, n_effective=1)
    assert r.verdict == "NOT_A_RESULT" and r.detail["n_marks"] == 0


def test_source_read_splits_on_how_the_mark_was_placed():
    marks = {"aa": {"c0": [110], "c1": [90]}, "bb": {"c0": [100], "c1": [100]}}
    src = {"aa": {"c0": "rvfc", "c1": "currentTime"}}
    rows = an.offsets(marks, CLIPS, sources=src)
    r = an.source_read(rows, scored_object=OBJ, n_effective=2)
    assert r.verdict == "NOT_A_RESULT"
    assert r.detail["rvfc"]["median_offset_frames"] == 10.0
    assert r.detail["currentTime"]["median_offset_frames"] == -10.0


def test_the_descriptive_reads_never_return_a_verdict():
    """None of them may pass or fail. The registered ceiling is the verdict."""
    marks = {"aa": {"c0": [110], "c1": []}, "bb": {"c0": [100], "c1": []}}
    rows = an.pair_rows(marks, CLIPS)
    offs = an.offsets(marks, CLIPS, sources={"aa": {"c0": "rvfc"}})
    for r in (an.decompose_read(rows, tol=5, scored_object=OBJ,
                                n_effective=2),
              an.offset_read(offs, scored_object=OBJ, n_effective=2),
              an.source_read(offs, scored_object=OBJ, n_effective=2)):
        assert r.verdict == "NOT_A_RESULT"
        assert r.n_effective is not None and r.scored_object and r.reason
