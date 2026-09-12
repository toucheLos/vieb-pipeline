"""The continuity residual, and the four things it must not get wrong.

Three of these are blocking, in the sense that a failure means the number is
measuring something other than what it is reported as measuring:

* `test_it_agrees_with_the_committed_predictor` -- the closed form and Phase D's
  SVD fit are the same fit. If they drift, the two phases stop being comparable.
* `TestSpikeVersusStep` -- a one-frame excursion must read as SPIKE and a
  persistent shift as STEP. Summing them would erase the distinction the whole
  measure exists to make.
* `test_a_rigid_motion_of_the_whole_animal_reads_zero` -- if the animal moving
  registers, the instrument is measuring locomotion and not tracking error.
"""
import numpy as np
import pytest

from vieb.qc import continuity as ct, disposition as dp
from tests.synthetic import mouse

NOSE = 2


def ell_of(pose):
    """A body length for the fixture, so residuals are in the reported units."""
    return float(np.median(np.linalg.norm(pose[:, NOSE] - pose[:, 3], axis=-1)))


class TestTheFit:

    def test_it_agrees_with_the_committed_predictor(self):
        """BLOCKING. The closed form replaces an SVD for speed, not for answers.

        `disposition.predict` ran the committed Phase D corpus pass. This module
        must not quietly be doing a different fit.
        """
        p = np.asarray(mouse(120), dtype=np.float64)
        others = [j for j in range(7) if j != NOSE]
        for t in range(1, 40):
            a = p[t - 1][None, others, :]
            b = p[t][None, others, :]
            s, th, off = ct.similarity(a, b)
            got = ct._apply(s, th, off, p[t - 1][None, NOSE, :])[0]
            want = dp.predict(p, t, t - 1, NOSE)
            assert want is not None
            np.testing.assert_allclose(got, want, atol=1e-9)

    def test_it_recovers_a_similarity_it_is_shown(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 10, size=(5, 6, 2))
        th = 0.7
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        b = 2.5 * a @ rot.T + np.array([13.0, -4.0])
        s, theta, off = ct.similarity(a, b)
        np.testing.assert_allclose(s, 2.5, atol=1e-9)
        np.testing.assert_allclose(theta, th, atol=1e-9)
        np.testing.assert_allclose(ct._apply(s, theta, off, a[:, 0, :]),
                                   b[:, 0, :], atol=1e-8)

    def test_it_never_returns_a_reflection(self):
        """The SVD version needs a determinant guard; this one cannot express one."""
        rng = np.random.default_rng(1)
        a = rng.normal(0, 10, size=(20, 6, 2))
        b = a.copy()
        b[..., 1] *= -1.0                      # a reflection, not a rotation
        s, theta, off = ct.similarity(a, b)
        got = ct._apply(s, theta, off, a[:, 0, :])
        assert not np.allclose(got, b[:, 0, :], atol=1e-6)


class TestTheResiduals:

    def test_a_clean_recording_is_nearly_continuous(self):
        p = np.asarray(mouse(300), dtype=np.float64)
        rm, rp = ct.residuals(p, ell_of(p))
        spike, _ = ct.decompose(rm, rp)
        assert np.nanmedian(spike) < 0.05

    def test_the_first_and_last_frames_have_no_neighbour(self):
        """Seam invariance. Reaching past either end would cross a recording."""
        p = np.asarray(mouse(50), dtype=np.float64)
        rm, rp = ct.residuals(p, ell_of(p))
        assert np.isnan(rm[0]).all()
        assert np.isnan(rp[-1]).all()
        assert np.isfinite(rm[1:]).any() and np.isfinite(rp[:-1]).any()

    def test_a_one_frame_recording_yields_nothing_rather_than_a_number(self):
        p = np.asarray(mouse(1), dtype=np.float64)
        rm, rp = ct.residuals(p, ell_of(np.asarray(mouse(50), dtype=np.float64)))
        assert np.isnan(rm).all() and np.isnan(rp).all()

    def test_a_rigid_motion_of_the_whole_animal_reads_zero(self):
        """BLOCKING. Translation, rotation and scale are the animal moving and
        the camera's foreshortening -- not tracking error. If they register, this
        measures locomotion."""
        p = np.asarray(mouse(200), dtype=np.float64)
        base = p[100]
        moved = np.zeros((5, 7, 2))
        for i in range(5):
            th = 0.3 * i
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            moved[i] = (1.0 + 0.2 * i) * base @ rot.T + np.array([40.0 * i, -9.0 * i])
        rm, rp = ct.residuals(moved, ell_of(p))
        assert np.nanmax(rm) < 1e-9
        assert np.nanmax(rp) < 1e-9

    def test_it_refuses_a_non_positive_body_length(self):
        p = np.asarray(mouse(50), dtype=np.float64)
        rm, rp = ct.residuals(p, 0.0)
        assert np.isnan(rm).all() and np.isnan(rp).all()


class TestSpikeVersusStep:
    """BLOCKING. These are different errors and the whole measure is the split."""

    @pytest.mark.parametrize("vec", [(45.0, 0.0), (0.0, -45.0), (30.0, 30.0)])
    def test_a_one_frame_teleport_reads_as_spike(self, vec):
        p = np.asarray(mouse(200), dtype=np.float64)
        dirty = p.copy()
        dirty[100, NOSE] += np.asarray(vec)
        spike, step = ct.decompose(*ct.residuals(dirty, ell_of(p)))
        # It leaves and comes back, so BOTH neighbours disagree.
        assert spike[100, NOSE] > 0.10
        assert spike[100, NOSE] > 4 * np.nanmedian(spike[:, NOSE])
        assert step[100, NOSE] < spike[100, NOSE]

    def test_a_persistent_shift_reads_as_step_not_spike(self):
        p = np.asarray(mouse(200), dtype=np.float64)
        dirty = p.copy()
        dirty[100:, NOSE] += np.array([45.0, 0.0])     # never comes back
        spike, step = ct.decompose(*ct.residuals(dirty, ell_of(p)))
        assert step[100, NOSE] > 0.10
        assert spike[100, NOSE] < step[100, NOSE]

    def test_a_one_sided_frame_is_neither(self):
        rm = np.array([[np.nan], [0.3], [0.4]])
        rp = np.array([[0.2], [0.5], [np.nan]])
        spike, step = ct.decompose(rm, rp)
        assert np.isnan(spike[0, 0]) and np.isnan(step[0, 0])
        assert np.isnan(spike[2, 0]) and np.isnan(step[2, 0])
        assert spike[1, 0] == pytest.approx(0.3)
        assert step[1, 0] == pytest.approx(0.2)

    def test_the_displaced_keypoint_is_the_largest_by_a_clear_margin(self):
        """The hypothesis this instrument was built to test: one keypoint moves,
        the other six say so.

        Not by a clean zero, though. The fit for each keypoint uses the other
        six, so a badly wrong keypoint contaminates its neighbours' predictions
        too. Attribution by argmax is sound; the innocent keypoints are not
        silent, and the module docstring records the measured margin.
        """
        p = np.asarray(mouse(200), dtype=np.float64)
        dirty = p.copy()
        dirty[100, NOSE] += np.array([45.0, 0.0])
        spike, _ = ct.decompose(*ct.residuals(dirty, ell_of(p)))
        others = [k for k in range(7) if k != NOSE]
        assert int(np.nanargmax(spike[100])) == NOSE
        assert spike[100, NOSE] > 2 * np.nanmax(spike[100, others])


class TestTheHeldOutSide:

    def test_it_scores_the_neighbour_the_corrector_did_not_use(self):
        t_n = 10
        rm = np.full((t_n, 1), 0.01)
        rp = np.full((t_n, 1), 0.90)
        donor = np.full(t_n, -1, dtype=np.int64)
        donor[4] = 3                     # corrector used t-1, so score t+1
        out = ct.held_out_side(donor, rm, rp)
        assert out[4, 0] == pytest.approx(0.90)

    def test_the_other_direction_too(self):
        t_n = 10
        rm = np.full((t_n, 1), 0.90)
        rp = np.full((t_n, 1), 0.01)
        donor = np.full(t_n, -1, dtype=np.int64)
        donor[4] = 5                     # corrector used t+1, so score t-1
        out = ct.held_out_side(donor, rm, rp)
        assert out[4, 0] == pytest.approx(0.90)

    def test_untouched_frames_keep_the_two_sided_spike(self):
        t_n = 10
        rm = np.full((t_n, 1), 0.20)
        rp = np.full((t_n, 1), 0.80)
        out = ct.held_out_side(np.full(t_n, -1, dtype=np.int64), rm, rp)
        assert np.allclose(out, 0.20)    # min(0.20, 0.80)


class TestTheOverlap:

    def test_two_disjoint_nets_do_not_read_as_agreement(self):
        """The reason this is a 2x2. Plain accuracy would read 0.96 here."""
        a = np.zeros(1000, dtype=bool); a[:20] = True
        b = np.zeros(1000, dtype=bool); b[20:40] = True
        o = ct.overlap_2x2(a, b)
        assert o["spike_and_bone"] == 0
        assert o["jaccard"] == 0.0
        assert o["spike_only"] == 20 and o["bone_only"] == 20

    def test_identical_masks_read_as_total_agreement(self):
        a = np.zeros(500, dtype=bool); a[10:30] = True
        o = ct.overlap_2x2(a, a)
        assert o["jaccard"] == pytest.approx(1.0)
        assert o["spike_only"] == 0 and o["bone_only"] == 0

    def test_the_cells_account_for_every_frame(self):
        rng = np.random.default_rng(0)
        a = rng.random(2000) < 0.03
        b = rng.random(2000) < 0.05
        o = ct.overlap_2x2(a, b)
        assert (o["spike_and_bone"] + o["spike_only"]
                + o["bone_only"] + o["neither"]) == 2000


class TestTheRead:

    OBJ = {"dataset": "luna", "arm": "continuity", "split": "report"}

    @staticmethod
    def iv(spike, hf):
        return {"spike": {"lo": spike[0], "hi": spike[1]},
                "hf_retained": {"lo": hf[0], "hi": hf[1]}}

    def test_overlapping_intervals_do_not_crown_an_arm(self):
        rows = [{"arm": "wiener"}, {"arm": "viterbi"}]
        iv = {"wiener": self.iv((0.010, 0.014), (0.20, 0.24)),
              "viterbi": self.iv((0.009, 0.013), (0.55, 0.60))}
        rd = ct.continuity_read(rows, iv, scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "INCONCLUSIVE"
        assert "0.029 px" in rd.reason

    def test_an_arm_that_bought_continuity_by_flattening_is_named_as_such(self):
        rows = [{"arm": "wiener"}, {"arm": "median_0.50"}]
        iv = {"wiener": self.iv((0.010, 0.014), (0.20, 0.24)),
              "median_0.50": self.iv((0.004, 0.006), (0.10, 0.14))}
        rd = ct.continuity_read(rows, iv, scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "INCONCLUSIVE"
        assert "deleting the movement" in rd.reason
        assert rd.detail["separates_by_flattening"] == ["median_0.50"]

    def test_a_genuine_improvement_passes_and_says_it_is_not_a_verdict(self):
        rows = [{"arm": "wiener"}, {"arm": "viterbi"}]
        iv = {"wiener": self.iv((0.010, 0.014), (0.20, 0.24)),
              "viterbi": self.iv((0.004, 0.006), (0.55, 0.60))}
        rd = ct.continuity_read(rows, iv, scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "PASS"
        assert "not restated here" in rd.reason

    def test_a_missing_incumbent_is_refused(self):
        rows = [{"arm": "viterbi"}]
        iv = {"viterbi": self.iv((0.004, 0.006), (0.55, 0.60))}
        rd = ct.continuity_read(rows, iv, scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "NOT_A_RESULT" and rd.degenerate
