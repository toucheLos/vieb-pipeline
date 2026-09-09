"""The reversal audit, extended to the egocentric channels.

recur's seven checks are asserted to still run and still pass through the
composition, and the
twist's parity is verified **from an actually reversed trajectory** rather than
from the algebra that motivated it. That distinction is the point of the whole
guard: the module's own docstring records that a sign error here is invisible
downstream, while a structural error -- pairing a quantity with the wrong frames
-- is not.
"""
import numpy as np
import pytest

from recur.geom import reversal as base
from vieb.tok import ego
from vieb.tok import reversal as rv
from tests.test_ego import moving

FPS = 30.0


class TestTheInheritedAuditStillHolds:

    def test_it_runs_at_least_seven_checks(self):
        assert rv.audit()["n_checks"] >= 7

    def test_recurs_own_checks_are_carried_through_unchanged(self):
        """Composed, not edited: a regression in recur shows up here."""
        a = rv.audit()
        assert a["n_checks_inherited"] == base.audit()["n_checks"] == 7
        assert a["inherited_verdict"] == "PASS"
        assert a["n_checks"] == a["n_checks_inherited"] + 3

    def test_every_check_passes(self):
        a = rv.audit()
        assert a["verdict"] == "PASS" and a["n_failed"] == 0

    def test_the_original_seven_are_all_still_there(self):
        """Ported verbatim means the old checks survive, not that they are
        replaced by new ones that happen to number the same."""
        names = {c["check"] for c in rv.audit()["checks"]}
        for want in ("odd coordinate reverses sign",
                     "derivative of an odd coordinate does NOT reverse sign",
                     "centroid_speed is EVEN under reversal",
                     "angular_velocity is ODD under reversal",
                     "reversing a delay stack flips the lag order",
                     "NOT flipping the lag order is detectable",
                     "reversal is per recording"):
            assert want in names


class TestTheTwistUnderReversal:

    def make(self, t=300):
        p = np.asarray(moving(t, turn=0.04, speed=2.0), dtype=np.float64)
        ell = ego.ell_a([p])
        xi, _ = ego.twist(p, ell, FPS)
        return p, ell, xi

    def test_reversing_the_trajectory_gives_the_reversed_twist(self):
        """Measured, not asserted: reverse the POSE, recompute, compare.

        If `reverse_twist` were wrong in sign or in offset this would fail,
        because the right-hand side never touches `reverse_twist` at all.
        """
        p, ell, xi = self.make()
        t = p.shape[0]
        xi_of_reversed, _ = ego.twist(p[::-1], ell, FPS)
        np.testing.assert_allclose(rv.reverse_twist(xi, np.array([0, t])),
                                   xi_of_reversed, atol=1e-9)

    def test_every_component_flips_sign_including_v_x(self):
        """`v_x` is signed, so it is ODD -- unlike `centroid_speed`, which is a
        magnitude and therefore EVEN. Confusing the two is the trap."""
        p, ell, xi = self.make()
        t = p.shape[0]
        back = rv.reverse_twist(xi, np.array([0, t]))
        np.testing.assert_allclose(back[:t - 1], -xi[:t - 1][::-1], atol=1e-12)
        assert np.abs(xi[:t - 1, 0]).max() > 0.1        # v_x is not trivially 0

    def test_dropping_the_one_frame_shift_is_detectable(self):
        p, _, xi = self.make()
        t = p.shape[0]
        good = rv.reverse_twist(xi, np.array([0, t]))
        bad = rv.naive_reverse_twist(xi, np.array([0, t]))
        assert not np.allclose(good, bad, atol=1e-6)
        # And the signature of the error: the undefined row lands at the front.
        assert np.allclose(bad[0], 0.0) and not np.allclose(good[0], 0.0)

    def test_the_egocentric_pose_block_is_even(self):
        p = np.asarray(moving(200, turn=0.04), dtype=np.float64)
        s = ego.egocentric(p, ego.ell_a([p]))
        s_of_reversed = ego.egocentric(p[::-1], ego.ell_a([p]))
        np.testing.assert_allclose(s[::-1], s_of_reversed, atol=1e-9)

    def test_reversal_of_a_twist_never_crosses_a_seam(self):
        a = np.asarray(moving(120, turn=0.04), dtype=np.float64)
        b = np.asarray(moving(90, turn=-0.02, speed=5.0), dtype=np.float64)
        ell = ego.ell_a([a, b])
        xa, _ = ego.twist(a, ell, FPS)
        xb, _ = ego.twist(b, ell, FPS)
        both = np.concatenate([xa, xb])
        bounds = np.array([0, 120, 210])
        got = rv.reverse_twist(both, bounds)
        np.testing.assert_allclose(got[:120], rv.reverse_twist(xa, np.array([0, 120])), atol=1e-12)
        np.testing.assert_allclose(got[120:], rv.reverse_twist(xb, np.array([0, 90])), atol=1e-12)


class TestTheParityTableMatchesTheChannels:

    def test_every_ego_channel_has_a_declared_parity(self):
        for name in ego.CHANNELS:
            key = "s" if name.startswith("s_") else name
            assert key in rv.EGO_PARITY, name

    def test_the_twist_channels_are_all_odd(self):
        assert [rv.EGO_PARITY[c] for c in ("v_x", "v_y", "omega")] == [base.ODD] * 3

    def test_the_pose_block_is_even_and_the_table_says_so(self):
        assert rv.EGO_PARITY["s"] == base.EVEN
