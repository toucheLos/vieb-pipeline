"""Runs, not frames -- and the two ways a run stream gets silently corrupted.

A run that spans a recording seam and a self-repeat manufactured by dropping an
abstain gap are both invisible downstream: they produce a plausible transition
table that contains a transition nobody observed. Both are asserted here.
"""
import numpy as np
import pytest

from vieb.tok import rle
from vieb.tok.rle import ABSTAIN


class TestEncode:

    def test_a_run_never_spans_a_seam(self):
        """BLOCKING. The same symbol either side of a boundary is two runs."""
        y = np.zeros(10, dtype=np.int32)
        r = rle.encode(y, [0, 5, 10])
        assert r["code"].tolist() == [0, 0]
        assert r["duration"].tolist() == [5, 5]
        assert r["recording"].tolist() == [0, 1]

    def test_durations_sum_to_the_frame_count(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, size=500)
        r = rle.encode(y, [0, 200, 500])
        assert int(r["duration"].sum()) == 500

    def test_an_empty_recording_contributes_nothing(self):
        r = rle.encode(np.arange(4) // 2, [0, 0, 4])
        assert r["recording"].tolist() == [1, 1]


class TestSelfTransitions:

    def test_encoding_leaves_none(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 4, size=2000)
        assert rle.self_transitions(rle.encode(y, [0, 700, 2000])) == 0

    def test_a_manufactured_repeat_is_caught(self):
        """BLOCKING. This is what splicing out an abstain run produces."""
        runs = {"code": np.array([3, 3, 1]), "duration": np.array([4, 5, 2]),
                "recording": np.array([0, 0, 0])}
        assert rle.self_transitions(runs) == 1

    def test_the_same_code_across_a_seam_is_not_one(self):
        runs = {"code": np.array([3, 3]), "duration": np.array([4, 5]),
                "recording": np.array([0, 1])}
        assert rle.self_transitions(runs) == 0


class TestSequencesSplitOnAbstain:

    def test_x_abstain_x_does_not_become_xx(self):
        """BLOCKING. Measured at 32.3% of adjacent pairs when it went wrong."""
        y = np.array([5] * 4 + [ABSTAIN] * 3 + [5] * 4)
        seqs = rle.sequences(rle.encode(y, [0, 11]), ["a"])
        assert [s.tolist() for s in seqs["a"]] == [[5], [5]]

    def test_dropping_the_gap_instead_would_have_made_one(self):
        """The counterfactual, so the test above is known to be load-bearing."""
        y = np.array([5] * 4 + [ABSTAIN] * 3 + [5] * 4)
        runs = rle.encode(y, [0, 11])
        kept = runs["code"][runs["code"] != ABSTAIN]
        assert kept.tolist() == [5, 5]          # what the wrong handling yields

    def test_a_sequence_never_spans_a_recording(self):
        y = np.array([1, 2] * 4)
        seqs = rle.sequences(rle.encode(y, [0, 4, 8]), ["a", "a"])
        assert len(seqs["a"]) == 2

    def test_gap_none_concatenates_and_is_only_a_control(self):
        y = np.array([5] * 4 + [ABSTAIN] * 3 + [5] * 4)
        seqs = rle.sequences(rle.encode(y, [0, 11]), ["a"], gap=None)
        assert [s.tolist() for s in seqs["a"]] == [[5, 5]]


class TestDistribution:

    def _runs(self):
        y = np.concatenate([np.full(30, 1), np.full(9, 2),
                            np.full(60, ABSTAIN), np.full(21, 3)])
        return rle.encode(y, [0, 120])

    def test_abstain_is_out_of_the_behavioural_distribution(self):
        d = rle.distribution(self._runs(), 30.0)
        assert d["n_runs"] == 3 and d["n_abstain_runs"] == 1
        assert d["frames"]["median"] == pytest.approx(21.0)

    def test_the_abstain_frame_share_is_reported_separately(self):
        d = rle.distribution(self._runs(), 30.0)
        assert d["abstain_frame_share"] == pytest.approx(60 / 120)

    def test_including_abstain_changes_the_median(self):
        """Folding it in would let a tracking dropout read as a behaviour."""
        r = self._runs()
        a = rle.distribution(r, 30.0)["median_frames"]
        b = rle.distribution(r, 30.0, exclude_abstain=False)["median_frames"]
        assert a != b

    def test_seconds_are_frames_over_fps(self):
        d = rle.distribution(self._runs(), 30.0)
        assert d["median_seconds"] == pytest.approx(d["median_frames"] / 30.0)

    def test_fps_is_carried_so_a_reader_can_check_it(self):
        assert rle.distribution(self._runs(), 250.0)["fps"] == 250.0


class TestFrameVersusRunMass:

    def test_a_long_dwell_holds_more_frames_than_runs(self):
        """The measured bias, in miniature: one symbol dwells, one flickers."""
        y = np.concatenate([np.full(100, 0), np.tile([1, 0], 10)])
        m = rle.frame_versus_run_mass(rle.encode(y, [0, len(y)]), 2)
        assert m["frame_share"][0] > m["run_share"][0]
        assert m["frame_share"][1] < m["run_share"][1]

    def test_shares_sum_to_one(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 6, size=3000)
        m = rle.frame_versus_run_mass(rle.encode(y, [0, 3000]), 6)
        assert sum(m["frame_share"]) == pytest.approx(1.0)
        assert sum(m["run_share"]) == pytest.approx(1.0)

    def test_abstain_is_excluded(self):
        y = np.array([0] * 5 + [ABSTAIN] * 50 + [1] * 5)
        m = rle.frame_versus_run_mass(rle.encode(y, [0, 60]), 2)
        assert sum(m["frame_share"]) == pytest.approx(1.0)


class TestRunlenRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 256}

    def _read(self, y, fps=30.0):
        d = rle.distribution(rle.encode(y, [0, len(y)]), fps)
        return rle.runlen_read(d, scored_object=self.OBJ, n_effective=60)

    def test_a_flickering_alphabet_fails(self):
        r = self._read(np.arange(600) % 7)
        assert r.verdict == "FAIL"
        assert "retires THIS alphabet" in r.reason

    def test_the_failure_says_it_retires_the_alphabet_not_the_sweep(self):
        r = self._read(np.arange(600) % 7)
        assert "not the sweep" in r.reason and "coarser N" in r.reason

    def test_a_usable_alphabet_passes(self):
        y = np.repeat(np.arange(60) % 5, 12)
        assert self._read(y).verdict == "PASS"

    def test_the_boundary_is_at_or_below(self):
        """Median exactly 3 fails: the floor is inclusive, as the docstring says."""
        y = np.repeat(np.arange(200) % 6, 3)
        d = rle.distribution(rle.encode(y, [0, len(y)]), 30.0)
        assert d["median_frames"] == 3.0
        assert rle.runlen_read(d, scored_object=self.OBJ,
                               n_effective=60).verdict == "FAIL"

    def test_a_self_transition_is_degenerate_not_merely_failing(self):
        runs = {"code": np.array([3, 3]), "duration": np.array([40, 50]),
                "recording": np.array([0, 0])}
        r = rle.runlen_read(rle.distribution(runs, 30.0),
                            scored_object=self.OBJ, n_effective=60)
        assert r.verdict == "FAIL" and r.degenerate

    def test_all_abstain_is_not_a_result(self):
        r = self._read(np.full(100, ABSTAIN))
        assert r.verdict == "NOT_A_RESULT" and r.degenerate

    def test_every_read_carries_its_numbers(self):
        r = self._read(np.repeat(np.arange(60) % 5, 12))
        for k in ("median_frames", "median_seconds", "n_runs",
                  "abstain_frame_share", "self_transitions"):
            assert k in r.detail


class TestBuckets:
    """Share of runs and share of frames are different numbers, and conflating
    them is how a stream that is half one-frame noise reads as survivable."""

    def test_runs_and_frames_disagree(self):
        y = np.concatenate([np.arange(100) % 2, np.full(400, 7)])
        d = rle.distribution(rle.encode(y, [0, len(y)]), 30.0)
        b1 = [b for b in d["buckets"] if b["at_most"] == 1][0]
        assert b1["share_runs"] > 0.9
        assert b1["share_frames"] < 0.25

    def test_the_widest_bucket_reaches_everything(self):
        y = np.repeat(np.arange(50) % 5, 2)
        d = rle.distribution(rle.encode(y, [0, len(y)]), 30.0)
        assert d["buckets"][-1]["share_runs"] == pytest.approx(1.0)
        assert d["buckets"][-1]["share_frames"] == pytest.approx(1.0)

    def test_thresholds_are_inherited_not_chosen_here(self):
        assert [b["at_most"] for b in
                rle.distribution(rle.encode(np.arange(20) % 3, [0, 20]),
                                 30.0)["buckets"]] == list(rle.BUCKETS)
