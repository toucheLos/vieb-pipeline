"""The egocentric transform: what it removes, what it keeps, and what it costs.

Three claims carry the stage and each has a test here that could fail:

* **it removes SE(2) equivariance** -- a rigidly translated and rotated copy of a
  recording gives an identical representation;
* **it removes between-animal body size and nothing else** -- a uniformly scaled
  animal gives identical `s` AND an identical `omega`, the second being the
  dimensional correction to the brief;
* **it loses nothing** -- `inverse` rebuilds the keypoints to float64.

Plus the rank fact, which is not a defect but must not be a surprise.
"""
import numpy as np
import pytest

from recur import kendall as kd
from recur.geom import represent as rep
from vieb.tok import ego
from tests.synthetic import mouse

FPS = 30.0


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def moving(t=600, *, turn=0.0, speed=3.0, body_scale=1.0):
    """A rigid mouse driven along a path, so the twist has a known answer.

    `body_scale` resizes the ANIMAL without touching the path, and `speed` moves
    the path without touching the animal. Keeping them separate is the whole
    point: scaling the returned array multiplies both at once, which is what
    made the first version of the body-lengths-per-second test assert something
    it was not testing.
    """
    body = np.asarray(mouse(1))[0]
    body = (body - body[ego.ORIGIN]) * body_scale
    out = np.empty((t, 7, 2))
    th, pos = 0.0, np.zeros(2)
    for f in range(t):
        out[f] = pos + body @ rot(th).T
        pos = pos + rot(th) @ np.array([speed, 0.0])
        th += turn
    return out


class TestEquivarianceRemoval:

    def test_a_translated_and_rotated_copy_is_identical(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        q = (p @ rot(0.77).T) + np.array([1234.0, -567.0])
        a, va = ego.transform(p, 40.0, FPS)
        b, vb = ego.transform(q, 40.0, FPS)
        np.testing.assert_allclose(a[va], b[vb], atol=1e-9)

    def test_it_is_not_invariant_to_something_it_should_see(self):
        """A control on the control: a real posture change must move it."""
        p = np.asarray(mouse(400), dtype=np.float64)
        q = p.copy()
        q[:, ego.NOSE, 1] += 4.0
        a, _ = ego.transform(p, 40.0, FPS)
        b, _ = ego.transform(q, 40.0, FPS)
        assert np.abs(a - b).max() > 1e-3

    def test_heading_agrees_with_the_kendall_reconstruction(self):
        """The short path and the gauge path are the same angle, exactly.

        `heading` reads the axis straight off the pose; `reconstruct_axis_angle`
        routes it through the decomposition and the frozen gauge. They must agree
        -- that is the decomposition identity -- and agreeing is what lets this
        module stay independent of the gauge without inventing a new heading.
        """
        p = np.asarray(mouse(400), dtype=np.float64)
        c = kd.fit_gauge(p)
        _, _, theta, z = kd.decompose(p, c)
        via_gauge = rep.reconstruct_axis_angle(theta, z, axis=ego.AXIS)
        direct = ego.heading(p)
        np.testing.assert_allclose(ego.wrap(via_gauge - direct), 0.0, atol=1e-9)


class TestBodyLengthNormalisation:

    def test_a_uniformly_scaled_animal_gives_identical_shape(self):
        p = np.asarray(moving(200), dtype=np.float64)
        big = p * 2.5
        a, _ = ego.transform(p, ego.ell_a([p]), FPS)
        b, _ = ego.transform(big, ego.ell_a([big]), FPS)
        np.testing.assert_allclose(a[:-1, :ego.N_POSE], b[:-1, :ego.N_POSE],
                                   atol=1e-9)

    def test_a_scaled_animal_gives_an_identical_omega(self):
        """The dimensional correction to the brief, as an assertion.

        Dividing the whole twist by `ell_a` -- as written -- would make this
        animal's turning rate 2.5x smaller than its twin's purely because it is
        bigger, which is body size leaking into the channel where it is least
        visible. Rotation takes the frame rate alone.
        """
        p = np.asarray(moving(200, turn=0.05), dtype=np.float64)
        big = np.asarray(moving(200, turn=0.05, body_scale=2.5), dtype=np.float64)
        a, _ = ego.transform(p, ego.ell_a([p]), FPS)
        b, _ = ego.transform(big, ego.ell_a([big]), FPS)
        np.testing.assert_allclose(a[:-1, -1], b[:-1, -1], atol=1e-9)
        assert np.abs(a[:-1, -1]).max() > 1.0          # omega is not trivially 0

    def test_the_same_pixel_speed_on_a_bigger_animal_reads_slower(self):
        """Body lengths per second, so a bigger animal covering the same pixels
        is moving less far in its own units."""
        p = np.asarray(moving(200, speed=3.0), dtype=np.float64)
        big = np.asarray(moving(200, speed=3.0, body_scale=2.5), dtype=np.float64)
        a, _ = ego.transform(p, ego.ell_a([p]), FPS)
        b, _ = ego.transform(big, ego.ell_a([big]), FPS)
        np.testing.assert_allclose(a[:-1, -3] / b[:-1, -3], 2.5, rtol=1e-6)

    def test_a_bigger_animal_moving_proportionally_faster_reads_identical(self):
        """Same behaviour at a different body size is the same representation."""
        p = np.asarray(moving(200, speed=3.0), dtype=np.float64)
        big = np.asarray(moving(200, speed=7.5, body_scale=2.5), dtype=np.float64)
        a, _ = ego.transform(p, ego.ell_a([p]), FPS)
        b, _ = ego.transform(big, ego.ell_a([big]), FPS)
        np.testing.assert_allclose(a[:-1, -3:-1], b[:-1, -3:-1], rtol=1e-6,
                                   atol=1e-9)

    def test_ell_a_pools_every_recording_of_the_animal(self):
        """One number per animal, not per recording."""
        small = np.asarray(moving(100))
        big = np.asarray(moving(100, body_scale=3.0))
        pooled = ego.ell_a([small, big])
        assert (ego.ell_a([small]) < pooled < ego.ell_a([big]))


class TestItLosesNothing:

    def test_the_transform_is_a_bijection_to_float64(self):
        p = np.asarray(mouse(300), dtype=np.float64)
        ell = ego.ell_a([p])
        x, _ = ego.transform(p, ell, FPS)
        back = ego.inverse(x, ego.heading(p), p[:, ego.ORIGIN], ell)
        np.testing.assert_allclose(back, p, atol=1e-9)

    def test_the_twist_reproduces_the_path_it_came_from(self):
        """Integrating the twist must return the trajectory, or it is not one."""
        p = np.asarray(moving(200, turn=0.03), dtype=np.float64)
        ell = ego.ell_a([p])
        xi, valid = ego.twist(p, ell, FPS)
        raw = np.stack([xi[:, 0] * ell / FPS, xi[:, 1] * ell / FPS,
                        xi[:, 2] / FPS], axis=1)
        dth, u = ego.se2_exp(raw)
        th, pos = ego.heading(p), p[:, ego.ORIGIN]
        for t in np.flatnonzero(valid):
            c, s = np.cos(th[t]), np.sin(th[t])
            step = np.array([c * u[t, 0] - s * u[t, 1],
                             s * u[t, 0] + c * u[t, 1]])
            np.testing.assert_allclose(pos[t] + step, pos[t + 1], atol=1e-8)
            assert ego.wrap(th[t] + dth[t] - th[t + 1]) == pytest.approx(0, abs=1e-9)


class TestRank:

    def test_the_named_dead_columns_are_exactly_zero(self):
        x, _ = ego.transform(np.asarray(mouse(300)), 40.0, FPS)
        for name, col in ego.dead_columns().items():
            assert np.abs(x[:, col]).max() == 0.0, name

    def test_the_axis_alignment_kills_a_third_direction(self):
        x, _ = ego.transform(np.asarray(mouse(300)), 40.0, FPS)
        head, tail = ego.AXIS
        assert np.abs(x[:, 2 * head + 1] - x[:, 2 * tail + 1]).max() < 1e-9

    def test_the_pose_block_has_rank_eleven_not_fourteen(self):
        """Three lost directions is exactly SE(2)'s three degrees of freedom."""
        p = np.asarray(mouse(600), dtype=np.float64)
        # A rigid fixture has rank 0; give the body real posture variation.
        rng = np.random.default_rng(0)
        p = p + rng.normal(0, 0.8, p.shape)
        x, _ = ego.transform(p, 40.0, FPS)
        r = ego.rank_read(x)
        assert r["rank"] == ego.POSE_RANK == 11
        assert r["n_dead"] == ego.SE2_DOF == 3


class TestSeams:

    def test_the_last_frame_of_a_recording_has_no_velocity(self):
        x, valid = ego.transform(np.asarray(mouse(300)), 40.0, FPS)
        assert not valid[-1] and valid[:-1].all()

    def test_an_unusable_frame_invalidates_its_predecessor_too(self):
        """xi_t reads frame t+1, so a bad t+1 makes t undefined as well."""
        usable = np.ones(300, dtype=bool)
        usable[100] = False
        _, valid = ego.transform(np.asarray(mouse(300)), 40.0, FPS, usable=usable)
        assert not valid[100] and not valid[99] and valid[98] and valid[101]

    def test_a_two_frame_recording_yields_one_velocity(self):
        _, valid = ego.transform(np.asarray(mouse(2)), 40.0, FPS)
        assert valid.tolist() == [True, False]

    def test_a_one_frame_recording_yields_none(self):
        _, valid = ego.transform(np.asarray(mouse(1)), 40.0, FPS)
        assert not valid.any()
