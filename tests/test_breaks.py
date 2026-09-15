"""The boundary detector, and the invariants that make it testable.

Four of these are lifted from `~/exbias/tests/test_exbias.py`, because ExBias
records that its *calibration* tests — not its unit tests — are what caught the
defect behind its zero-state run. The one that matters most is the last:
adjusted R² must have the same null expectation at every length, or the
smoothness check becomes a length filter in disguise.
"""
import numpy as np
import pytest

from vieb.seg import breaks as bk
from vieb.tok import ego


def _smooth(n=600, c=ego.N_DIMS, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6, n)
    return np.stack([np.sin(2 * np.pi * t / (2.0 + 0.2 * i)) for i in range(c)],
                    axis=1)


def _kinked(n=600, at=300, c=ego.N_DIMS):
    """Constant acceleration, reversed at `at`. A C2 discontinuity and nothing
    else: position and velocity are continuous through it."""
    t = np.arange(n, dtype=np.float64)
    a = np.where(t < at, 1.0, -1.0)
    vel = np.concatenate([[0.0], np.cumsum(a)[:-1]])
    pos = np.concatenate([[0.0], np.cumsum(vel)[:-1]])
    return np.tile(pos[:, None], (1, c)) / 1000.0


class TestSecondDerivWeights:

    @pytest.mark.parametrize("side", ("left", "right"))
    def test_they_recover_a_known_quadratic_exactly(self, side):
        """BLOCKING. If the weights are wrong the whole detector is."""
        h = 7
        w = bk.second_deriv_weights(h, side)
        s = np.arange(h, dtype=float)
        if side == "left":
            s = -s[::-1]
        for c2 in (0.0, 1.0, -2.5, 7.25):
            y = 3.0 - 0.5 * s + c2 * s ** 2
            assert float(w @ y) == pytest.approx(2.0 * c2, abs=1e-9)

    def test_a_bad_side_is_refused(self):
        with pytest.raises(ValueError):
            bk.second_deriv_weights(5, "centred")


class TestDiscontinuity:

    def test_a_kink_is_localized_to_within_one_frame(self):
        """BLOCKING, lifted from ExBias."""
        x = _kinked(at=300)
        d = bk.discontinuity(x, 4)
        assert abs(int(np.argmax(d)) - 300) <= 1, int(np.argmax(d))

    def test_the_kink_stands_far_above_the_bulk(self):
        x = _kinked(at=300)
        d = bk.discontinuity(x, 4)
        peak = float(d.max())
        bulk = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
        assert peak > 50 * max(bulk, 1e-12), (peak, bulk)

    def test_a_smooth_signal_has_a_tiny_score(self):
        d = bk.discontinuity(_smooth(), 4)
        assert float(d.max()) < 0.05, float(d.max())

    def test_constant_velocity_produces_no_score(self):
        """BLOCKING, and the speed-invariance check. A fast straight line is not
        a boundary; a detector that fires on speed is measuring locomotion."""
        n = 400
        x = np.tile(np.arange(n, dtype=np.float64)[:, None], (1, ego.N_DIMS))
        d = bk.discontinuity(x * 37.0, 4)
        assert float(np.abs(d).max()) < 1e-9

    def test_the_windows_are_strictly_one_sided(self):
        """BLOCKING. A centred convolution lets both windows straddle t, so they
        share the frames whose disagreement is being measured and D collapses.
        That is a recorded past bug, so the counterfactual is built here: slide
        the two windows toward each other so they overlap, and the kink's
        contrast against the bulk must fall away.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        def score(x, h, shift):
            """`shift = 0` is the real detector; `shift > 0` overlaps the
            windows by `2*shift` frames."""
            wl = bk.second_deriv_weights(h, "left")
            wr = bk.second_deriv_weights(h, "right")
            pad = h + shift
            xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
            win = sliding_window_view(xp, h, axis=0)          # starts at -pad
            # left window ENDS at t + shift; right window STARTS at t - shift.
            left = win[pad - h + 1 + shift:pad - h + 1 + shift + x.shape[0]] @ wl
            right = win[pad - shift:pad - shift + x.shape[0]] @ wr
            d = np.sqrt(((right - left) ** 2).sum(axis=1))
            d[:pad] = 0.0
            d[-pad:] = 0.0
            return d

        # Compared on the PEAK mismatch, not on peak-over-bulk: `_kinked` is
        # exactly piecewise-quadratic, so away from the kink both versions sit
        # at float noise and a contrast ratio would be dividing one
        # near-zero by another.
        x, h = _kinked(at=300), 6
        sharp = float(score(x, h, 0).max())
        blunt = float(score(x, h, h // 2).max())
        assert sharp > 2 * blunt, (sharp, blunt)

    def test_the_real_detector_matches_the_zero_shift_reference(self):
        """The helper above must be the same computation, or the counterfactual
        is comparing against something other than the detector."""
        from numpy.lib.stride_tricks import sliding_window_view
        x, h = _kinked(at=300), 6
        wl = bk.second_deriv_weights(h, "left")
        wr = bk.second_deriv_weights(h, "right")
        left = sliding_window_view(
            np.pad(x, ((h - 1, 0), (0, 0)), mode="edge"), h, axis=0) @ wl
        right = sliding_window_view(
            np.pad(x, ((0, h - 1), (0, 0)), mode="edge"), h, axis=0) @ wr
        ref = np.sqrt(((right - left) ** 2).sum(axis=1))
        ref[:h] = 0.0
        ref[-h:] = 0.0
        assert np.allclose(bk.discontinuity(x, h), ref)

    def test_a_short_recording_returns_zeros_rather_than_padding_artefacts(self):
        assert not bk.discontinuity(_smooth(n=5), 4).any()

    def test_it_refuses_a_non_matrix(self):
        with pytest.raises(ValueError):
            bk.discontinuity(np.zeros(50), 4)


class TestBoundaries:

    def test_nms_keeps_the_strongest_peak_not_the_earliest(self):
        """BLOCKING, lifted from ExBias. A left-to-right rule lets an earlier
        weaker peak suppress a later stronger one, so the boundary lands on
        noise instead of on the kink."""
        d = np.zeros(100)
        d[40] = 5.0
        d[44] = 9.0
        got = bk.boundaries(d, 1.0, min_gap=10)
        assert got.tolist() == [44]

    def test_nothing_above_threshold_gives_nothing(self):
        d = np.zeros(100)
        d[50] = 1.0
        assert bk.boundaries(d, 5.0, min_gap=4).size == 0

    def test_peaks_respect_the_refractory_gap(self):
        d = np.zeros(200)
        d[[30, 33, 120]] = [9.0, 8.0, 7.0]
        got = bk.boundaries(d, 1.0, min_gap=10)
        assert got.tolist() == [30, 120]

    def test_a_blocked_frame_cannot_be_a_boundary(self):
        """An abstained frame's pose is partly an interpolant, so its
        acceleration is partly the gap policy's."""
        d = np.zeros(100)
        d[50] = 9.0
        blocked = np.zeros(100, bool)
        blocked[50] = True
        assert bk.boundaries(d, 1.0, min_gap=4, blocked=blocked).size == 0

    def test_a_smooth_signal_yields_no_boundaries_at_the_reference_threshold(self):
        x = _smooth()
        d = bk.discontinuity(x, 4)
        got = bk.boundaries(d, bk.mad_threshold(d, 8.0), min_gap=12)
        assert got.size == 0, got


class TestFloorsAndGuard:

    def test_the_floor_is_samples_per_parameter(self):
        assert bk.identifiability_floor(3, 3) == 12
        assert bk.identifiability_floor(1, 3) == 6

    def test_the_guard_defaults_to_half_the_derivative_window(self):
        assert bk.guard_frames(30.0, deriv_sec=0.133) == 2
        assert bk.guard_frames(250.0, deriv_sec=0.133) == 17

    def test_the_guard_is_never_zero(self):
        assert bk.guard_frames(30.0, guard_sec=0.0) == 1

    def test_the_minimum_segment_includes_both_guards(self):
        assert bk.min_segment_frames(3, 2) == 16


class TestFitQuality:

    def test_a_polynomial_is_fitted_exactly(self):
        s = np.linspace(-1, 1, 40)
        y = np.stack([1 + 2 * s - 3 * s ** 2 + 0.5 * s ** 3] * 4, axis=1)
        r2, adj = bk.fit_quality(y, 3)
        assert r2 == pytest.approx(1.0, abs=1e-9)
        assert adj == pytest.approx(1.0, abs=1e-9)

    def test_fitting_fewer_samples_than_parameters_raises(self):
        """BLOCKING. Silently clamping is how a 3-frame floor -- a degree-3 fit
        to 3 samples -- got into ExBias's pipeline."""
        with pytest.raises(ValueError, match="interpolation"):
            bk.fit_quality(np.zeros((4, 3)), 3)

    def test_adjusted_r2_has_the_same_null_at_every_length(self):
        """BLOCKING, and the calibration test that matters most. Raw R^2 rises
        with shortness under a white-noise null, so thresholding it makes the
        smoothness check a length filter that keeps under-determined short
        segments and discards the long ones."""
        rng = np.random.default_rng(0)
        raw, adj = {}, {}
        for n in (16, 40, 200):
            r = [bk.fit_quality(rng.normal(size=(n, 6)), 3) for _ in range(200)]
            raw[n] = float(np.mean([a for a, _ in r]))
            adj[n] = float(np.mean([b for _, b in r]))
        assert raw[16] > raw[200] + 0.1, raw
        assert max(abs(v) for v in adj.values()) < 0.05, adj


class TestSegmentTable:

    def _rows(self, n=200, peaks=(80,), abstain=None, guard=2):
        x = _smooth(n=n)
        ab = np.zeros(n, bool) if abstain is None else abstain
        return bk.segment_table(x, np.asarray(peaks), lo=0, hi=n, abstain=ab,
                                guard=guard, fps=30.0)

    def test_a_peak_splits_the_stream(self):
        rows = self._rows(peaks=(80,))
        assert [(r["start"], r["stop"]) for r in rows] == [(0, 80), (80, 200)]

    def test_abstain_splits_and_is_never_fused_away(self):
        """BLOCKING. Dropping an abstained stretch would fuse the segments
        either side into one and manufacture a duration the animal never held."""
        ab = np.zeros(200, bool)
        ab[90:120] = True
        rows = self._rows(peaks=(), abstain=ab)
        spans = [(r["start"], r["stop"]) for r in rows]
        assert spans == [(0, 90), (90, 120), (120, 200)]
        assert [r["abstain_frac"] for r in rows][1] == 1.0

    def test_an_all_abstain_segment_is_not_fittable(self):
        ab = np.zeros(200, bool)
        ab[90:120] = True
        rows = self._rows(peaks=(), abstain=ab)
        assert not rows[1]["fittable"]
        assert not np.isfinite(rows[1]["fit_r2_adj"])

    def test_a_segment_shorter_than_the_floor_is_kept_and_flagged(self):
        """How much of the stream is unfittable is part of the result, not
        something to discard quietly."""
        rows = self._rows(peaks=(4,))
        assert rows[0]["n_frames"] == 4 and not rows[0]["fittable"]
        assert rows[1]["fittable"]

    def test_the_fit_interval_is_inset_by_the_guard_band(self):
        """The guard is what makes segmentation testable: without it every
        segment is a smooth piece with a kink glued to its front."""
        rows = self._rows(peaks=(80,), guard=3)
        assert rows[0]["fit_start"] == 3 and rows[0]["fit_stop"] == 77

    def test_durations_are_in_seconds(self):
        rows = self._rows(peaks=(60,))
        assert rows[0]["duration_s"] == pytest.approx(2.0)

    def test_nothing_spans_a_recording_boundary(self):
        x = _smooth(n=200)
        rows = bk.segment_table(x, np.asarray([]), lo=50, hi=150,
                                abstain=np.zeros(200, bool), guard=2, fps=30.0)
        assert all(50 <= r["start"] and r["stop"] <= 150 for r in rows)


class TestBreaksRead:

    OBJ = {"dataset": "luna", "arm": "raw"}

    def test_it_refuses_to_call_a_rate_a_finding(self):
        r = bk.breaks_read({"boundary_rate_per_s": 1.4, "n_segments": 4000,
                            "median_duration_s": 0.7, "frac_fittable": 0.8,
                            "mean_fit_r2_adj": 0.6},
                           scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "GRID_LIMITED"
        assert "NOT a finding" in r.reason and "fires on noise" in r.reason

    def test_a_degenerate_detector_is_not_a_result(self):
        r = bk.breaks_read({"boundary_rate_per_s": float("nan"),
                            "n_segments": 0},
                           scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "NOT_A_RESULT" and r.degenerate
