"""The surrogates must be structureless, smooth, and shaped like the corpus.

A null that differs from the data in two ways at once tests neither. The
egocentric representation has three rank-deficient directions by construction,
so a surrogate carrying variance in one of them would be trivially separable
from the corpus and would hand the quantizer structure the corpus does not have.
"""
import numpy as np
import pytest

from vieb.tok import ego, surrogate as sr


def _signal(n=1200, c=ego.N_DIMS, seed=0):
    """A smooth, correlated signal with two identically-zero columns and one
    exact linear dependence -- the shape the ego array actually has."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    base = np.stack([np.sin(2 * np.pi * t / (1.5 + 0.3 * i))
                     + 0.1 * rng.normal(size=n) for i in range(c)], axis=1)
    base[:, ego.dead_columns()["origin_x"]] = 0.0
    base[:, ego.dead_columns()["origin_y"]] = 0.0
    base[:, 5] = base[:, 1] - base[:, 3]          # an exact linear dependence
    return base


class TestPhasePreservesWhatItMustPreserve:

    def test_the_power_spectrum_of_every_channel_is_unchanged(self):
        """The whole reason `phase` is the primary arm: same smoothness, same
        autocorrelation, hence the same run-length structure under quantizing."""
        x = _signal()
        out = sr.generate(x, [0, x.shape[0]], "phase",
                          np.random.default_rng(0))["x"]
        a = np.abs(np.fft.rfft(x, axis=0))
        b = np.abs(np.fft.rfft(np.asarray(out, np.float64), axis=0))
        assert np.allclose(a, b, atol=1e-6)

    def test_it_is_not_the_same_signal(self):
        x = _signal()
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                     np.random.default_rng(0))["x"], np.float64)
        assert not np.allclose(out, x, atol=1e-3)

    def test_an_exact_linear_dependence_between_columns_survives(self):
        """BLOCKING. Shared phase multiplies every column by the same factor, so
        the SE(2) rank deficit is preserved for free. A surrogate that broke it
        would carry variance in a direction the corpus never moves in."""
        x = _signal()
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                     np.random.default_rng(0))["x"], np.float64)
        assert np.allclose(out[:, 5], out[:, 1] - out[:, 3], atol=1e-4)

    def test_the_mean_of_each_channel_is_kept(self):
        x = _signal() + 3.0
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                     np.random.default_rng(0))["x"], np.float64)
        assert np.allclose(out.mean(0), x.mean(0), atol=1e-3)


class TestDeadColumns:

    @pytest.mark.parametrize("kind", sr.KINDS)
    def test_a_zero_column_stays_zero(self, kind):
        """BLOCKING for both arms. A VAR's residual noise is full-rank in the
        subspace it was fitted on, so the dead columns are copied, not
        simulated -- and this is what checks that they were."""
        x = _signal()
        out = sr.generate(x, [0, x.shape[0]], kind, np.random.default_rng(0))
        got = np.asarray(out["x"], np.float64)
        for idx in ego.dead_columns().values():
            assert np.allclose(got[:, idx], 0.0), (kind, idx)
        assert out["n_dead_columns"] == 2

    def test_live_columns_finds_the_dead_ones(self):
        x = _signal()
        live = sr.live_columns(x)
        assert not live[ego.dead_columns()["origin_x"]]
        assert not live[ego.dead_columns()["origin_y"]]
        assert int(live.sum()) == ego.N_DIMS - 2


class TestSeams:

    @pytest.mark.parametrize("kind", sr.KINDS)
    def test_a_surrogate_never_crosses_a_recording_boundary(self, kind):
        """BLOCKING. A surrogate built across a seam would splice two recordings
        into one spectrum, then be compared against data where crossing a seam
        is forbidden everywhere else in this repo."""
        x = _signal(n=800)
        bounds = [0, 400, 800]
        a = sr.generate(x, bounds, kind, np.random.default_rng(0))["x"]
        y = x.copy()
        y[400:] = _signal(n=400, seed=7)          # change ONLY the second half
        b = sr.generate(y, bounds, kind, np.random.default_rng(0))["x"]
        assert np.allclose(np.asarray(a)[:400], np.asarray(b)[:400], atol=1e-6)

    def test_a_recording_too_short_to_randomise_is_copied(self):
        x = _signal(n=6)
        out = sr.generate(x, [0, 3, 6], "phase", np.random.default_rng(0))["x"]
        assert np.allclose(np.asarray(out, np.float64), x)

    def test_every_recording_is_covered(self):
        x = _signal(n=900)
        out = sr.generate(x, [0, 300, 600, 900], "phase",
                          np.random.default_rng(0))
        assert out["n_recordings"] == 3
        assert np.asarray(out["x"]).shape == x.shape


class TestVar:

    def test_it_produces_a_different_signal_of_the_same_shape(self):
        x = _signal()
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "var5",
                                     np.random.default_rng(0))["x"], np.float64)
        assert out.shape == x.shape
        assert not np.allclose(out, x, atol=1e-3)

    def test_a_degenerate_fit_falls_back_and_is_counted(self):
        """Recorded per recording rather than hidden: a fallback means that
        recording's null is a phase surrogate, not a VAR."""
        x = _signal(n=40)
        out = sr.generate(x, [0, 40], "var5", np.random.default_rng(0))
        assert out["n_fallback"] == 1

    def test_a_healthy_fit_does_not_fall_back(self):
        out = sr.generate(_signal(n=2000), [0, 2000], "var5",
                          np.random.default_rng(0))
        assert out["n_fallback"] == 0


class TestGenerate:

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            sr.generate(_signal(), [0, 1200], "white",
                        np.random.default_rng(0))

    @pytest.mark.parametrize("kind", sr.KINDS)
    def test_the_output_is_float32_like_the_shards(self, kind):
        out = sr.generate(_signal(), [0, 1200], kind, np.random.default_rng(0))
        assert np.asarray(out["x"]).dtype == np.float32


class TestFalsifierRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 256}

    def _read(self, lo, hi, point, corpus=11.3, surr=4.0):
        return sr.falsifier_read(
            {"animal_interval": {"lo": lo, "hi": hi, "point": point},
             "frame_interval": {"lo": lo, "hi": hi}},
            {"mean_delta": corpus}, {"mean_delta": surr},
            kind="phase", scored_object=self.OBJ, n_effective=89)

    def test_a_gap_excluding_zero_passes(self):
        r = self._read(3.0, 9.0, 6.0)
        assert r.verdict == "PASS" and "not explained by quantizing" in r.reason

    def test_the_pass_refuses_the_overclaim(self):
        """Two linear-Gaussian surrogates are two alternatives, not all."""
        r = self._read(3.0, 9.0, 6.0)
        assert "does NOT license" in r.reason
        assert "two alternatives, not all of them" in r.reason

    def test_a_gap_containing_zero_fails_and_closes_task_3(self):
        r = self._read(-2.0, 4.0, 1.0)
        assert r.verdict == "FAIL"
        assert "coarse sweep does not run" in r.reason

    def test_a_surrogate_that_beats_the_corpus_withdraws_the_ladder(self):
        r = self._read(-9.0, -3.0, -6.0, corpus=4.0, surr=11.3)
        assert r.verdict == "FAIL" and "withdrawn" in r.reason

    def test_a_non_finite_gap_is_not_a_result(self):
        r = self._read(np.nan, np.nan, np.nan)
        assert r.verdict == "NOT_A_RESULT" and r.degenerate

    def test_the_frame_interval_is_carried_but_never_gates(self):
        r = self._read(3.0, 9.0, 6.0)
        assert "frame_interval" in r.detail
        assert "does not read it" in r.detail["frame_interval_note"]
