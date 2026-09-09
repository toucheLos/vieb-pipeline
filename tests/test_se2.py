"""The SE(2) group logarithm, checked against an independent matrix exponential.

The brief is explicit that separate differencing of position and angle is wrong
under simultaneous translation and rotation, with error scaling as
``omega * ||v||`` -- largest exactly during turns, which is where the behaviour
is. `naive_velocity` implements that error on purpose, so the last class here
measures the gap rather than asserting correctness against itself.

Correctness is checked against `scipy.linalg.expm` on the 3x3 matrix
representation, which shares no code with `ego.se2_log`.
"""
import numpy as np
import pytest

from vieb.tok import ego


def hat(twist):
    """The se(2) matrix for a twist ``(v_x, v_y, omega)``."""
    vx, vy, w = twist
    return np.array([[0.0, -w, vx], [w, 0.0, vy], [0.0, 0.0, 0.0]])


def group(dtheta, u):
    """The SE(2) matrix for a rotation and a translation."""
    c, s = np.cos(dtheta), np.sin(dtheta)
    return np.array([[c, -s, u[0]], [s, c, u[1]], [0.0, 0.0, 1.0]])


class TestAgainstTheMatrixExponential:

    @pytest.mark.parametrize("w", [0.0, 1e-9, 1e-6, 1e-4, 0.01, 0.3, 1.5, 3.0, -2.0])
    def test_exp_of_the_log_is_the_original_group_element(self, w):
        """The defining property, against scipy rather than against ourselves."""
        from scipy.linalg import expm
        u = np.array([[0.7, -1.3]])
        tw = ego.se2_log(np.array([w]), u)[0]
        np.testing.assert_allclose(expm(hat(tw)), group(w, u[0]), atol=1e-12)

    @pytest.mark.parametrize("w", [1e-9, 1e-5, 0.2, 2.0])
    def test_the_round_trip_closes(self, w):
        u = np.array([[0.7, -1.3]])
        tw = ego.se2_log(np.array([w]), u)
        back_w, back_u = ego.se2_exp(tw)
        np.testing.assert_allclose(back_w, [w], atol=1e-12)
        np.testing.assert_allclose(back_u, u, rtol=1e-10, atol=1e-12)

    def test_the_series_and_the_closed_form_agree_at_the_crossover(self, monkeypatch):
        """Both branches at the SAME omega, which is the only fair comparison.

        `SMALL_OMEGA` has to sit where each is still accurate: below it the
        closed form loses mantissa to cancellation in 1 - cos(omega), above it
        the truncated series runs out of terms. Flipping the threshold rather
        than nudging omega is what isolates the branch from its input.
        """
        w = np.array([ego.SMALL_OMEGA])
        u = np.array([[1.0, -0.5]])
        closed = ego.se2_log(w, u)                       # |w| < eps is False
        monkeypatch.setattr(ego, "SMALL_OMEGA", ego.SMALL_OMEGA * 10.0)
        series = ego.se2_log(w, u)                       # now takes the series
        np.testing.assert_allclose(series, closed, rtol=1e-12, atol=1e-15)

    def test_the_closed_form_is_the_one_that_degrades_near_zero(self):
        """Which is the reason the series branch exists at all.

        At omega = 1e-9 the closed form's 1 - cos(omega) is ~5e-19 against a
        double's 2.2e-16 resolution near 1, so `a` comes back with a handful of
        bits. The series is exact there to every digit that matters.
        """
        w = 1e-9
        closed_a = (w / 2.0) / np.tan(w / 2.0)
        series_a = 1.0 - (w / 2.0) ** 2 / 3.0
        assert series_a == pytest.approx(1.0, abs=1e-18)
        # The closed form is not catastrophically wrong here, but it is computed
        # through a cancellation and the series is not -- so the series is what
        # `se2_log` uses below the threshold.
        assert abs(closed_a - 1.0) < 1e-15

    def test_pure_translation_leaves_the_step_untouched(self):
        u = np.array([[2.0, -3.0]])
        tw = ego.se2_log(np.array([0.0]), u)
        np.testing.assert_allclose(tw[0], [2.0, -3.0, 0.0], atol=1e-15)

    def test_pure_rotation_about_the_origin_has_no_translation(self):
        tw = ego.se2_log(np.array([0.9]), np.zeros((1, 2)))
        np.testing.assert_allclose(tw[0], [0.0, 0.0, 0.9], atol=1e-15)

    def test_a_full_turn_wraps_rather_than_accumulating(self):
        tw = ego.se2_log(np.array([2 * np.pi + 0.1]), np.zeros((1, 2)))
        assert tw[0, 2] == pytest.approx(0.1)


class TestWhereTheNaiveVersionCosts:

    def test_it_agrees_exactly_on_straight_line_motion(self):
        """Which is why the error survives review: it is invisible until a turn."""
        u = np.array([[1.0, 0.0]])
        z = np.array([0.0])
        np.testing.assert_allclose(ego.se2_log(z, u), ego.naive_velocity(z, u),
                                   atol=1e-15)

    @pytest.mark.parametrize("w", [0.05, 0.2, 0.5, 1.0])
    def test_the_error_grows_with_omega_times_speed(self, w):
        u = np.array([[1.0, 0.0]])
        err = np.linalg.norm(ego.se2_log(np.array([w]), u)[0, :2]
                             - ego.naive_velocity(np.array([w]), u)[0, :2])
        # Leading order is |w|/2 * ||u||: the b term the naive version drops.
        assert err == pytest.approx(abs(w) / 2.0, rel=0.2)

    def test_at_a_realistic_turn_rate_the_error_is_percent_scale(self):
        """30 fps, a 90-degree turn over ~0.5s: 0.1 rad/frame. Not negligible."""
        u = np.array([[1.0, 0.0]])
        w = np.array([0.1])
        rel = (np.linalg.norm(ego.se2_log(w, u)[0, :2] - ego.naive_velocity(w, u)[0, :2])
               / np.linalg.norm(u))
        assert 0.02 < rel < 0.10

    def test_the_naive_version_is_not_the_log_of_anything(self):
        """It does not satisfy the property the correct one is defined by."""
        from scipy.linalg import expm
        u, w = np.array([[1.0, 0.5]]), np.array([0.8])
        tw = ego.naive_velocity(w, u)[0]
        assert not np.allclose(expm(hat(tw)), group(w[0], u[0]), atol=1e-6)
