"""Anipose's Viterbi filter: that it loads, what it fixes, and what it leaves.

This file also pins the correction. An earlier version of this repo recorded the
filter as unavailable because the corpus carries one detection per bodypart-frame
-- a conclusion drawn from the paper's wording rather than the code.
`test_it_runs_on_single_detection_input` is that claim's replacement, and it
fails if anyone reinstates the old one.
"""
import numpy as np
import pytest

from vieb.clean import arms, viterbi as V
from tests.synthetic import mouse

FPS = 30.0


def straight(t=600, kp=7):
    """Constant-velocity motion, so any change the filter makes is its own."""
    p = np.zeros((t, kp, 2))
    p[:, :, 0] = (np.arange(t) * 0.5)[:, None]
    p[:, :, 1] = 100.0
    p[:, 0, 1] += 20.0
    p[:, 1, 1] -= 20.0
    return p


class TestItLoads:

    def test_the_two_functions_come_from_anipose_not_from_here(self):
        assert V.VERSION != "unknown"
        assert V.SOURCE.endswith("anipose/filter_pose.py")
        assert set(V.NEEDED) == {"remove_dups", "viterbi_path"}

    def test_the_import_chain_is_bypassed_deliberately(self):
        """`.common` -> aniposelib -> numba is calibration code the filter never
        touches, and importing it would drag opencv-contrib into a venv shared
        with recur."""
        with pytest.raises(ModuleNotFoundError):
            import anipose.filter_pose            # noqa: F401


class TestWhatItDoes:

    def test_it_runs_on_single_detection_input(self):
        """The correction, as an assertion.

        One detection per bodypart-frame is what this corpus has, and the filter
        works on it: the candidate set at frame i is the previous n_back frames'
        detections as well as frame i's.
        """
        p = straight()
        p[300, 2, 0] += 400.0
        out = V.viterbi(p, None, FPS)
        assert abs(out[300, 2, 0] - 150.0) < 5.0

    def test_it_declines_the_jump_rather_than_averaging_it_away(self):
        """A smoother would leave a fraction of the excursion behind. Path
        selection replaces it with a plausible position outright."""
        p = straight()
        p[300, 2, 0] += 400.0
        out = V.viterbi(p, None, FPS)
        assert abs(out[300, 2, 0] - p[300, 2, 0]) > 380.0

    def test_it_leaves_a_clean_recording_almost_entirely_alone(self):
        p = straight()
        r = V.reassignment(p, V.viterbi(p, None, FPS))
        assert r["reassigned"] < 0.01

    def test_a_short_recording_is_returned_unchanged(self):
        p = straight(t=4)
        np.testing.assert_array_equal(V.viterbi(p, None, FPS), p)

    def test_no_missing_sentinel_survives_into_the_output(self):
        """A chosen missing particle comes back as (-1, -1); leaving one in a
        pose array reads downstream as the animal teleporting to the origin."""
        p = straight()
        p[200:210, 3, :] += 900.0
        out = V.viterbi(p, None, FPS)
        assert not ((out[:, :, 0] == -1) & (out[:, :, 1] == -1)).any()

    def test_it_preserves_shape_and_finiteness_on_a_real_shaped_fixture(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        out = V.viterbi(p, None, FPS)
        assert out.shape == p.shape and np.isfinite(out).all()


class TestReassignment:

    def test_it_bounds_how_much_the_filter_changed(self):
        """The number that says whether this is de-glitching or re-trajectorying."""
        p = straight()
        p[300, 2, 0] += 400.0
        r = V.reassignment(p, V.viterbi(p, None, FPS))
        assert 0.0 < r["reassigned"] < 0.01
        assert r["median_move_px"] > 100.0
        assert len(r["reassigned_per_keypoint"]) == 7

    def test_an_untouched_recording_reassigns_nothing(self):
        p = straight()
        assert V.reassignment(p, p)["reassigned"] == 0.0


class TestTheArm:

    def test_it_is_registered_and_dispatches(self):
        assert "viterbi" in arms.ARMS
        p = np.asarray(mouse(300), dtype=np.float64)
        out = arms.apply("viterbi", p, None, FPS)
        assert out.shape == p.shape

    def test_it_is_not_recorded_as_unavailable_anywhere(self):
        """The retired claim must not come back through a docstring."""
        src = open("vieb/clean/arms.py").read()
        assert "Viterbi filter is unavailable" not in src
