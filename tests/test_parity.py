"""A4 for the egocentric representation.

The gate is a closed form, so the interesting assertions are that it really is
closed-form -- exact, and computed without ever seeing the heading or the arena
position it discarded -- and that the read gates on that rather than on the
regressor arms travelling beside it.
"""
import numpy as np
import pytest

from recur.geom import represent as rep
from vieb.tok import ego, parity
from tests.test_ego import moving, rot

FPS = 30.0
OBJ = {"dataset": "luna", "arm": "ego", "split": "report"}


def wiggly(t=900, seed=0):
    """A moving mouse with real posture variation, so R^2 has a denominator."""
    p = np.asarray(moving(t, turn=0.03, speed=2.5), dtype=np.float64)
    return p + np.random.default_rng(seed).normal(0, 0.5, p.shape)


def scored(p):
    ell = ego.ell_a([p])
    x, valid = ego.transform(p, ell, FPS)
    speed, omega = rep.raw_kinematics(p, FPS, axis=ego.AXIS, causal=True)
    return parity.exact_scores(x, ell, FPS, speed, omega, valid=valid), x, ell


class TestTheClosedForm:

    def test_speed_and_turn_come_back_exactly(self):
        sc, _, _ = scored(wiggly())
        assert sc["speed"] == pytest.approx(1.0, abs=1e-9)
        assert sc["angular"] == pytest.approx(1.0, abs=1e-9)
        assert sc["n_frames"] > 800

    def test_it_clears_the_brief_thresholds_by_orders_of_magnitude(self):
        sc, _, _ = scored(wiggly())
        assert sc["speed"] > parity.R2_SPEED_MIN
        assert sc["angular"] > parity.R2_TURN_MIN

    def test_recovery_never_sees_the_heading_or_the_arena_position(self):
        """`recover` takes the representation and ell_a. Nothing else.

        A rigidly moved copy of the recording has a different heading and a
        different position at every frame, and must give the identical answer --
        which it cannot do if either had leaked into the reconstruction.
        """
        p = wiggly(400)
        q = (p @ rot(1.1).T) + np.array([4321.0, -876.0])
        ell = ego.ell_a([p])
        a = parity.recover(ego.transform(p, ell, FPS)[0], ell, FPS)
        b = parity.recover(ego.transform(q, ell, FPS)[0], ell, FPS)
        np.testing.assert_allclose(a[0][3:-1], b[0][3:-1], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(a[1][3:-1], b[1][3:-1], rtol=1e-7, atol=1e-9)

    def test_the_gate_can_fail_at_all(self):
        """A gate that cannot fail is not a gate. Break the twist hard enough."""
        p = wiggly()
        ell = ego.ell_a([p])
        x, valid = ego.transform(p, ell, FPS)
        x[:, ego.N_POSE:] *= 0.5
        speed, omega = rep.raw_kinematics(p, FPS, axis=ego.AXIS, causal=True)
        sc = parity.exact_scores(x, ell, FPS, speed, omega, valid=valid)
        assert sc["speed"] < parity.R2_SPEED_MIN

    @pytest.mark.parametrize("turn,caught", [(0.03, False), (0.10, False),
                                             (0.25, True), (0.40, True)])
    def test_a4_does_NOT_catch_the_naive_twist_at_ordinary_turn_rates(
            self, turn, caught):
        """Measured, and the reason `tests/test_se2.py` exists.

        Substituting separate differencing for the group logarithm -- the error
        the brief singles out -- costs almost nothing on A4 until the animal is
        turning fast. R^2 on speed runs 0.9973 at 0.9 rad/s, 0.9853 at 3 rad/s,
        and only breaches the 0.98 threshold above roughly 4.5 rad/s.

        So **A4 is not the guard against this error**, and it must not be relied
        on as one. The guard is the direct comparison of `se2_log` against
        `scipy.linalg.expm` in `tests/test_se2.py`, which fails on the naive
        version at every turn rate including zero.
        """
        p = np.asarray(moving(900, turn=turn, speed=2.5), dtype=np.float64)
        p = p + np.random.default_rng(0).normal(0, 0.5, p.shape)
        ell = ego.ell_a([p])
        bad, valid = ego.transform(p, ell, FPS, naive=True)
        speed, omega = rep.raw_kinematics(p, FPS, axis=ego.AXIS, causal=True)
        sc = parity.exact_scores(bad, ell, FPS, speed, omega, valid=valid)
        assert (sc["speed"] < parity.R2_SPEED_MIN) is caught
        # The correct twist is exact at every one of these turn rates.
        good, gv = ego.transform(p, ell, FPS)
        assert parity.exact_scores(good, ell, FPS, speed, omega,
                                   valid=gv)["speed"] == pytest.approx(1.0, abs=1e-9)

    def test_a_short_recording_returns_nan_rather_than_a_number(self):
        s, o = parity.recover(np.zeros((2, ego.N_DIMS)), 40.0, FPS)
        assert not np.isfinite(s).any() and not np.isfinite(o).any()


class TestTheRead:

    def test_it_passes_on_the_closed_form_alone(self):
        rd = parity.a4_read({"exact": {"speed": 1.0, "angular": 1.0}},
                            scored_object=OBJ, n_effective=89)
        assert rd.verdict == "PASS" and "bijection" in rd.reason

    def test_a_failing_regressor_arm_does_not_fail_the_gate(self):
        """The wrong-object failure this project already paid for once."""
        rd = parity.a4_read(
            {"exact": {"speed": 1.0, "angular": 1.0},
             "windowed": {"speed": 0.823, "angular": 0.392, "k": 5}},
            scored_object=OBJ, n_effective=89)
        assert rd.verdict == "PASS"
        assert "0.823" in rd.reason and "0.392" in rd.reason

    def test_a_failing_closed_form_fails_and_blames_this_module(self):
        rd = parity.a4_read({"exact": {"speed": 0.4, "angular": 0.2}},
                            scored_object=OBJ, n_effective=89)
        assert rd.verdict == "FAIL"
        assert "error in this module rather than a property of the corpus" in rd.reason

    def test_it_refuses_when_the_closed_form_was_not_evaluated(self):
        rd = parity.a4_read({"exact": {}}, scored_object=OBJ, n_effective=89)
        assert rd.verdict == "INCONCLUSIVE"

    def test_the_single_frame_arm_carries_both_comparison_numbers(self):
        rd = parity.a4_read(
            {"exact": {"speed": 1.0, "angular": 1.0},
             "pose_only_single_frame": {"speed": 0.17, "angular": 0.06}},
            scored_object=OBJ, n_effective=89)
        assert "0.090" in rd.reason and "0.175" in rd.reason


class TestTheRegressorArms:

    def test_the_pose_block_alone_loses_the_locomotor_channels(self):
        """Reported, never gated -- and the evidence they cannot be dropped."""
        p = wiggly(1200)
        ell = ego.ell_a([p])
        x, valid = ego.transform(p, ell, FPS)
        speed, omega = rep.raw_kinematics(p, FPS, axis=ego.AXIS, causal=True)
        groups = ["a"] * 600 + ["b"] * 600
        full = parity.windowed_scores(x, speed, omega, groups, k=3,
                                      valid=valid, max_rows=5000)
        pose = parity.windowed_scores(x, speed, omega, groups, k=1,
                                      valid=valid, pose_only=True,
                                      max_rows=5000)
        assert full["speed"] > pose["speed"]

    def test_it_returns_nan_rather_than_fitting_one_animal(self):
        p = wiggly(300)
        ell = ego.ell_a([p])
        x, valid = ego.transform(p, ell, FPS)
        speed, omega = rep.raw_kinematics(p, FPS, axis=ego.AXIS, causal=True)
        out = parity.windowed_scores(x, speed, omega, ["only"] * 300, k=2,
                                     valid=valid)
        assert not np.isfinite(out["speed"])
