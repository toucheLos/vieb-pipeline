"""The dispersion ratio must mean what the write-up says it means.

Its whole value is having an exact null: one is what independent frames give.
If the statistic is wrong the 350x is a number about nothing.
"""
import numpy as np
import pytest

from vieb.qc import concentration as cc


class TestDispersion:

    def test_independent_frames_give_about_one(self):
        """BLOCKING. The null. Simulate the model the ratio is measured against
        and check the statistic recovers it."""
        rng = np.random.default_rng(0)
        n = np.full(400, 6000)
        k = rng.binomial(n, 0.0216)
        d = cc.dispersion(k, n)
        assert 0.6 < d["dispersion"] < 1.6, d["dispersion"]

    def test_structure_reads_far_above_one(self):
        """Half the recordings clean, half at ten times the rate."""
        rng = np.random.default_rng(0)
        n = np.full(400, 6000)
        p = np.where(np.arange(400) < 200, 0.002, 0.04)
        d = cc.dispersion(rng.binomial(n, p), n)
        assert d["dispersion"] > 50, d["dispersion"]

    def test_the_pooled_rate_is_frame_weighted(self):
        d = cc.dispersion([10, 0], [1000, 9000])
        assert d["pooled_rate"] == pytest.approx(0.001)

    def test_one_recording_is_refused(self):
        assert not np.isfinite(cc.dispersion([5], [100])["dispersion"])


class TestLorenz:

    def test_a_uniform_process_matches_its_baseline(self):
        k = np.full(100, 50)
        out = cc.lorenz(k)
        for cell in out["shares"].values():
            assert cell["share_of_violations"] == pytest.approx(
                cell["uniform_baseline"], abs=0.02)
        assert out["gini"] == pytest.approx(0.0, abs=0.02)

    def test_total_concentration_reads_high(self):
        k = np.zeros(100); k[0] = 1000
        out = cc.lorenz(k)
        assert out["shares"]["worst_1pct"]["share_of_violations"] == 1.0
        assert out["gini"] > 0.95

    def test_every_share_carries_its_baseline(self):
        """The number must never be readable without what uniform would give."""
        out = cc.lorenz(np.arange(1, 101))
        for cell in out["shares"].values():
            assert "uniform_baseline" in cell


class TestEdgeness:

    def test_the_middle_is_zero_and_the_edge_is_large(self):
        rng = np.random.default_rng(0)
        p = rng.normal(0, 10, size=(5000, 2))
        e = cc.edgeness(p)
        assert np.nanmin(e) < 0.1
        assert np.nanmax(e) > 2.0

    def test_it_is_invariant_to_translation_and_scale(self):
        """Different boxes and camera mounts must be comparable."""
        rng = np.random.default_rng(1)
        p = rng.normal(0, 10, size=(4000, 2))
        a = cc.edgeness(p)
        b = cc.edgeness(p * 7.0 + np.array([500.0, -300.0]))
        np.testing.assert_allclose(a, b, rtol=1e-9)

    def test_too_few_frames_gives_nan_rather_than_a_number(self):
        assert np.isnan(cc.edgeness(np.zeros((2, 2)))).all()


class TestTheRead:

    OBJ = {"dataset": "luna", "arm": "concentration", "split": "report"}

    @staticmethod
    def _lor():
        return cc.lorenz(np.arange(1, 201))

    def test_a_high_ratio_passes_and_carries_the_caveat(self):
        rd = cc.concentration_read({"dispersion": 350.0}, self._lor(),
                                   scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "PASS"
        assert "Only across-network variance decides it" in rd.reason
        assert "UPPER bound" in rd.reason

    def test_a_ratio_near_one_fails_and_says_the_ensemble_has_more_to_work_with(self):
        rd = cc.concentration_read({"dispersion": 1.2}, self._lor(),
                                   scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "FAIL"
        assert "more to average away" in rd.reason
        assert "Only across-network variance decides it" in rd.reason

    def test_every_verdict_carries_the_both_hypotheses_caveat(self):
        """It must be in the reason string, not left to the surrounding prose."""
        for d in (350.0, 1.2):
            rd = cc.concentration_read({"dispersion": d}, self._lor(),
                                       scored_object=self.OBJ, n_effective=89)
            assert cc.BOTH_HYPOTHESES in rd.reason
            assert rd.detail["caveat"] == cc.BOTH_HYPOTHESES

    def test_a_missing_ratio_is_refused(self):
        rd = cc.concentration_read({"dispersion": float("nan"), "why": "x"},
                                   self._lor(), scored_object=self.OBJ,
                                   n_effective=89)
        assert rd.verdict == "NOT_A_RESULT" and rd.degenerate
