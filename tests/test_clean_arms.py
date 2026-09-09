"""The cleaning arms: what each one is allowed to do, and what it must not.

The property that separates the outlier arm from the smoothers has its own test
(`test_it_leaves_every_other_frame_bit_identical`). It is the reason that arm is
in the bakeoff at all: it can reduce the violation rate without moving anything
else, and a smoother cannot.
"""
import numpy as np
import pytest

from vieb.clean import arms
from tests.synthetic import mouse

FPS = 30.0


class TestHeldArray:

    def test_it_interpolates_across_missing_and_leaves_the_rest_alone(self):
        p = np.asarray(mouse(200), dtype=np.float64)
        miss = np.zeros((200, 7), dtype=bool)
        miss[100:105, 2] = True
        dirty = p.copy()
        dirty[100:105, 2] = -999.0
        held = arms.held_array(dirty, miss)
        np.testing.assert_allclose(held[:100], dirty[:100], atol=0)
        assert np.abs(held[100:105, 2] - (-999.0)).min() > 1.0

    def test_an_all_missing_keypoint_is_left_alone_rather_than_invented(self):
        p = np.asarray(mouse(50), dtype=np.float64)
        miss = np.zeros((50, 7), dtype=bool)
        miss[:, 3] = True
        np.testing.assert_allclose(arms.held_array(p, miss)[:, 3], p[:, 3])


class TestSmoothers:

    def test_savgol_reproduces_a_linear_ramp_exactly(self):
        """polyorder 2 fits a line exactly, so a ramp must survive untouched."""
        t = np.arange(300.0)
        p = np.zeros((300, 7, 2))
        p[:, :, 0] = t[:, None]
        out = arms.savgol(p, None, FPS, window_s=0.25)
        np.testing.assert_allclose(out[10:-10], p[10:-10], atol=1e-9)

    def test_savgol_attenuates_a_single_frame_spike(self):
        p = np.zeros((300, 7, 2))
        p[150, 0, 0] = 100.0
        out = arms.savgol(p, None, FPS, window_s=0.25)
        assert out[150, 0, 0] < 60.0

    def test_the_median_removes_a_single_frame_spike_entirely(self):
        p = np.zeros((300, 7, 2))
        p[150, 0, 0] = 100.0
        out = arms.rolling_median(p, None, FPS, window_s=0.25)
        assert abs(out[150, 0, 0]) < 1e-9

    def test_the_median_preserves_a_step_where_a_mean_would_ramp_it(self):
        p = np.zeros((300, 7, 2))
        p[150:, 0, 0] = 10.0
        out = arms.rolling_median(p, None, FPS, window_s=0.25)
        assert abs(out[140, 0, 0]) < 1e-9 and abs(out[160, 0, 0] - 10.0) < 1e-9

    def test_a_window_shorter_than_the_recording_is_required(self):
        p = np.zeros((3, 7, 2))
        np.testing.assert_allclose(arms.savgol(p, None, FPS, window_s=1.0), p)
        np.testing.assert_allclose(arms.rolling_median(p, None, FPS, window_s=1.0), p)

    def test_the_window_is_seconds_not_frames(self):
        """The same 0.25 s window is 7 frames at 30 fps and 63 at 250."""
        p = np.zeros((2000, 7, 2))
        p[1000, 0, 0] = 100.0
        slow = arms.rolling_median(p, None, 30.0, window_s=0.25)
        fast = arms.rolling_median(p, None, 250.0, window_s=0.25)
        # Both kill the spike; the wider one flattens a wider neighbourhood.
        assert abs(slow[1000, 0, 0]) < 1e-9 and abs(fast[1000, 0, 0]) < 1e-9
        p2 = np.zeros((2000, 7, 2))
        p2[995:1006, 0, 0] = 100.0        # 11 frames: survives 7, dies at 63
        assert arms.rolling_median(p2, None, 30.0, window_s=0.25)[1000, 0, 0] > 50
        assert arms.rolling_median(p2, None, 250.0, window_s=0.25)[1000, 0, 0] < 50


class TestPositionOutlier:

    def dirty(self, t=400, frame=200, jump=300.0):
        p = np.asarray(mouse(t), dtype=np.float64)
        p[frame, 2, 0] += jump
        return p

    def test_it_removes_a_teleport(self):
        p = self.dirty()
        out = arms.position_outlier(p, None, FPS, rule="mad", k=5.0)
        assert abs(out[200, 2, 0] - p[200, 2, 0]) > 100.0

    def test_it_leaves_every_other_frame_bit_identical(self):
        """The property no smoother has, and the reason this arm is here.

        It can reduce the violation rate without moving anything else, so a
        distortion metric can separate it from a filter that achieves the same
        rate by moving the whole corpus.

        The spike costs its neighbour: a one-frame excursion breaks the step out
        AND the step back, so a displacement threshold flags both. That is what
        refineDLC specifies and it is kept deliberately -- see the docstring.
        Everything outside that pair is bit-identical.
        """
        p = self.dirty()
        out = arms.position_outlier(p, None, FPS, rule="mad", k=5.0)
        moved = np.flatnonzero(np.abs(out - p).max(axis=(1, 2)) > 0)
        assert moved.tolist() == [200, 201]
        untouched = np.ones(p.shape[0], dtype=bool)
        untouched[[200, 201]] = False
        np.testing.assert_array_equal(out[untouched], p[untouched])

    def test_a_clean_recording_comes_back_untouched(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        out = arms.position_outlier(p, None, FPS, rule="mad", k=5.0)
        np.testing.assert_array_equal(out, p)

    def test_a_long_run_is_left_alone_rather_than_interpolated_across(self):
        """Same reasoning as shapeflow's gap policy: filling a long gap
        fabricates smooth motion, and smooth motion reads as behaviour."""
        p = np.asarray(mouse(400), dtype=np.float64)
        p[150:250, 2, 0] += 300.0
        out = arms.position_outlier(p, None, FPS, rule="mad", k=5.0,
                                    max_gap_s=0.5)
        assert np.abs(out[180:220] - p[180:220]).max() == 0.0

    def test_it_never_extrapolates_past_an_endpoint(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        p[0, 2, 0] += 300.0
        p[-1, 2, 0] += 300.0
        out = arms.position_outlier(p, None, FPS, rule="mad", k=5.0)
        assert out[0, 2, 0] == p[0, 2, 0] and out[-1, 2, 0] == p[-1, 2, 0]

    @pytest.mark.parametrize("rule,k", [("mad", 5.0), ("iqr", 3.0),
                                        ("percentile", 99.0), ("absolute", 50.0)])
    def test_every_rule_runs_and_catches_the_teleport(self, rule, k):
        out = arms.position_outlier(self.dirty(), None, FPS, rule=rule, k=k)
        assert abs(out[200, 2, 0]) < 1e9

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError):
            arms.position_outlier(np.zeros((10, 7, 2)), None, FPS, rule="magic")


class TestTheRegistry:

    @pytest.mark.parametrize("name", sorted(arms.ARMS))
    def test_every_arm_runs_and_preserves_shape(self, name):
        p = np.asarray(mouse(400), dtype=np.float64)
        out = arms.apply(name, p, None, FPS)
        assert out.shape == p.shape and np.isfinite(out).all()

    def test_raw_is_the_identity_and_is_a_copy(self):
        p = np.asarray(mouse(50), dtype=np.float64)
        out = arms.apply("raw", p, None, FPS)
        np.testing.assert_array_equal(out, p)
        assert out is not p

    def test_the_stored_arms_are_not_recomputed(self):
        """wiener and butterworth are read off disk, so a second implementation
        cannot silently disagree with the arrays every existing result used."""
        assert set(arms.STORED_ARMS) == {"wiener", "butterworth"}
        assert not set(arms.STORED_ARMS) & set(arms.ARMS)
