"""The disposition, and the nulls that must pass before it touches data.

Two of these are the pre-registered blocking checks:

* `test_it_changes_nothing_on_clean_frames` -- if it fires where there is no
  violation, suspect identification is keying on geometry rather than on error.
* `TestInjectionRecovery` -- recovers the injected DIRECTION, not just its
  magnitude. A corrector that merely lands on the constraint surface recovers
  magnitude and would pass a distance test while being wrong about where the
  keypoint belongs.
"""
import numpy as np
import pytest

from recur.qc.swap import runs_of
from vieb.qc import bones, disposition as dp, lobo
from tests.synthetic import mouse

EPS = 0.10
LEFT_EAR, RIGHT_EAR, NOSE = 0, 1, 2
PAIRS = bones.pair_indices(7)
ALL = list(range(len(PAIRS)))
SKULL_IDX = [PAIRS.index(b) for b in bones.SKULL]


def refs(pose, group, eps=EPS, keep=None):
    L = bones.metric_lengths(pose, group, "raw", pairs=PAIRS, keep=keep)
    return np.array([bones.reference_length(L[:, m], eps)["l_hat"]
                     for m in range(len(group))])


def scaled_refs(pose, eps=EPS):
    """Reference length for every one of the 21 pairs, in raw px."""
    L = bones.bone_lengths(pose, PAIRS)
    return np.array([bones.reference_length(L[:, m], eps)["l_hat"]
                     for m in range(len(PAIRS))])


def displace(pose, frame, keypoint, vector):
    out = np.array(pose, dtype=np.float64, copy=True)
    out[frame, keypoint] = out[frame, keypoint] + np.asarray(vector, float)
    return out


class TestTheEnvelope:

    def test_short_runs_are_correctable_and_long_ones_are_not(self):
        m = np.zeros(100, dtype=bool)
        m[10:12] = True          # 2 frames -> correct
        m[40:50] = True          # 10 frames -> abstain
        corr, abst = dp.envelope(m)
        assert corr[10] and corr[11] and not abst[10]
        assert abst[45] and not corr[45]

    def test_the_boundary_is_inclusive_at_three_frames(self):
        m = np.zeros(60, dtype=bool)
        m[10:13] = True          # exactly 3
        m[30:34] = True          # 4
        corr, abst = dp.envelope(m)
        assert corr[10:13].all() and not abst[10:13].any()
        assert abst[30:34].all() and not corr[30:34].any()

    def test_no_frame_is_both(self):
        rng = np.random.default_rng(0)
        m = rng.random(2000) < 0.05
        corr, abst = dp.envelope(m)
        assert not (corr & abst).any()

    def test_nearby_abstain_runs_are_merged(self):
        """Abstention costs are per-RUN: each splits the symbol sequence and
        destroys ~(n-1) n-gram positions, so few long blocks beat many short."""
        m = np.zeros(200, dtype=bool)
        m[20:30] = True
        m[32:42] = True          # 2-frame gap
        _, abst = dp.envelope(m)
        assert runs_of(abst).shape[0] == 1

    def test_a_clean_recording_yields_neither(self):
        corr, abst = dp.envelope(np.zeros(500, dtype=bool))
        assert not corr.any() and not abst.any()


class TestSuspectIdentification:

    def test_the_keypoint_common_to_every_violating_bone_is_named(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        p = displace(p, 200, LEFT_EAR, [60.0, 0.0])
        L = bones.metric_lengths(p, bones.SKULL, "raw", pairs=PAIRS)
        v = bones.violations(L, refs(p, bones.SKULL), EPS)
        assert dp.suspects(v, bones.SKULL)[200] == LEFT_EAR

    def test_confidence_breaks_the_tie_when_one_bone_is_violated(self):
        """A single violated bone leaves both endpoints eligible; the lower
        confidence is what orders them, and calibration is irrelevant to that."""
        v = np.zeros((3, 3), dtype=bool)
        v[1, bones.SKULL.index((LEFT_EAR, NOSE))] = True
        conf = np.ones((3, 7))
        conf[1, NOSE] = 0.1
        assert dp.suspects(v, bones.SKULL, conf)[1] == NOSE
        conf[1, NOSE], conf[1, LEFT_EAR] = 1.0, 0.1
        assert dp.suspects(v, bones.SKULL, conf)[1] == LEFT_EAR

    def test_frames_with_no_violation_have_no_suspect(self):
        v = np.zeros((10, 3), dtype=bool)
        assert (dp.suspects(v, bones.SKULL) == -1).all()

    def test_disjoint_violating_bones_name_nobody(self):
        """Two bones sharing no keypoint cannot be one bad landmark, so the
        frame is left to the abstain path rather than guessed at."""
        v = np.zeros((2, len(bones.TRUNK)), dtype=bool)
        v[0, bones.TRUNK.index((3, 4))] = True
        v[0, bones.TRUNK.index((5, 6))] = True
        assert dp.suspects(v, bones.TRUNK)[0] == -1


class TestTheIdentityNull:

    def test_it_changes_nothing_on_clean_frames(self):
        """BLOCKING, pre-registered. Bit-identical, not approximately."""
        p = np.asarray(mouse(400), dtype=np.float64)
        L = bones.metric_lengths(p, bones.SKULL, "raw", pairs=PAIRS)
        v = bones.violations(L, refs(p, bones.SKULL), EPS)
        assert not v.any(), "fixture must be clean for this test to mean anything"
        out = p.copy()
        for t in range(p.shape[0]):
            s = dp.suspects(v, bones.SKULL)[t]
            if s >= 0:
                out[t], _ = dp.project(out, t, int(s), pairs=PAIRS,
                                       constrain=ALL,
                                       l_hat_scaled=scaled_refs(p), eps=EPS)
        np.testing.assert_array_equal(out, p)


class TestInjectionRecovery:
    """BLOCKING, pre-registered.

    A corrector that lands anywhere on the constraint surface recovers the
    magnitude of the error and would pass a distance test. Only the direction
    says it moved the keypoint back towards where it belongs.

    These exercise the whole path the corpus run uses -- donor, predict, project
    -- because that is what has to recover the direction. The projection alone is
    the fallback, and it is tested as the fallback below.
    """

    @staticmethod
    def correct(dirty, t, keypoint, lh):
        """The whole path: only frame `t` violates, so the donor is `t-1`."""
        viol = np.zeros(dirty.shape[0], dtype=bool)
        viol[t] = True
        tgt = dp.predict(dirty, t, dp.donor_frame(viol, t), keypoint)
        return dp.project(dirty, t, keypoint, pairs=PAIRS, constrain=SKULL_IDX,
                          l_hat_scaled=lh, eps=EPS, target=tgt)

    @pytest.mark.parametrize("vec", [(70.0, 0.0), (0.0, 70.0),
                                     (50.0, 50.0), (-60.0, 20.0)])
    def test_it_recovers_the_direction_not_just_the_magnitude(self, vec):
        p = np.asarray(mouse(400), dtype=np.float64)
        truth = p[200, NOSE].copy()
        dirty = displace(p, 200, NOSE, vec)
        fixed, info = self.correct(dirty, 200, NOSE, scaled_refs(p))
        assert info["moved"]
        applied = fixed[NOSE] - dirty[200, NOSE]
        wanted = truth - dirty[200, NOSE]
        cos = float(applied @ wanted /
                    (np.linalg.norm(applied) * np.linalg.norm(wanted)))
        assert cos > 0.8, f"direction cosine {cos:.3f}"

    def test_it_moves_the_suspect_towards_the_truth(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        truth = p[200, NOSE].copy()
        dirty = displace(p, 200, NOSE, (70.0, 0.0))
        fixed, _ = self.correct(dirty, 200, NOSE, scaled_refs(p))
        assert (np.linalg.norm(fixed[NOSE] - truth)
                < np.linalg.norm(dirty[200, NOSE] - truth))

    def test_only_the_suspect_moves(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        dirty = displace(p, 200, NOSE, (70.0, 0.0))
        fixed, _ = self.correct(dirty, 200, NOSE, scaled_refs(p))
        other = [k for k in range(7) if k != NOSE]
        np.testing.assert_array_equal(fixed[other], dirty[200, other])

    def test_a_short_bone_is_never_pushed_out(self):
        """The constraint is an inequality: projection can shorten a bone in the
        image plane and never lengthen it, so a foreshortened frame is left."""
        p = np.asarray(mouse(400), dtype=np.float64)
        short = displace(p, 200, NOSE, (-20.0, 0.0))   # nose pulled IN
        fixed, info = dp.project(short, 200, NOSE, pairs=PAIRS, constrain=ALL,
                                 l_hat_scaled=scaled_refs(p), eps=EPS)
        assert not info["moved"]
        np.testing.assert_array_equal(fixed[NOSE], short[200, NOSE])


class TestTheConstraintSet:

    def test_it_constrains_only_what_it_is_given(self):
        """The first corpus run passed every pair incident on the suspect, so a
        nose correction constrained trunk bone (2, 3) directly and the held-out
        gate's claim was untrue. Amendment 1. The set is the caller's to state."""
        p = np.asarray(mouse(400), dtype=np.float64)
        dirty = displace(p, 200, NOSE, (70.0, 0.0))
        lh = scaled_refs(p)
        _, info = dp.project(dirty, 200, NOSE, pairs=PAIRS,
                             constrain=SKULL_IDX, l_hat_scaled=lh, eps=EPS)
        # NOSE touches two SKULL bones, (0,2) and (1,2). Nothing else.
        assert info["n_constraints"] == 2

    def test_no_trunk_bone_is_reachable_from_the_skull_set(self):
        touched = {PAIRS[m] for m in SKULL_IDX}
        assert not (touched & set(bones.TRUNK))


class TestThePredictor:

    def test_the_donor_is_the_nearest_clean_frame(self):
        m = np.zeros(100, dtype=bool)
        m[50:52] = True
        assert dp.donor_frame(m, 50) == 49
        assert dp.donor_frame(m, 51) == 52

    def test_a_run_against_a_recording_boundary_has_no_donor(self):
        m = np.ones(10, dtype=bool)
        assert dp.donor_frame(m, 0) == -1
        assert dp.donor_frame(m, 9) == -1

    def test_it_never_looks_outside_the_recording_it_is_given(self):
        """Seam invariance: it is handed one recording and indexes only that."""
        m = np.zeros(20, dtype=bool)
        m[0] = True
        d = dp.donor_frame(m, 0)
        assert 0 <= d < 20

    def test_it_recovers_an_undisplaced_keypoint_almost_exactly(self):
        """On a clean pair of frames the similarity fit should land on the
        observed position -- if it does not, the predictor is not predicting."""
        p = np.asarray(mouse(400), dtype=np.float64)
        got = dp.predict(p, 200, 199, NOSE)
        assert got is not None
        assert np.linalg.norm(got - p[200, NOSE]) < 0.15 * bones.bone_lengths(
            p[200:201], PAIRS).mean()

    def test_it_refuses_a_degenerate_fit(self):
        p = np.zeros((3, 7, 2), dtype=np.float64)
        assert dp.predict(p, 1, 0, NOSE) is None


class TestTheAtom:

    def test_a_feasible_target_lands_in_the_interior(self):
        """The whole point of Amendment 1. The first run put 100% of corrections
        exactly on a codimension-1 surface, which a quantizer finds downstream
        and reports as a state."""
        p = np.asarray(mouse(400), dtype=np.float64)
        dirty = displace(p, 200, NOSE, (70.0, 0.0))
        tgt = dp.predict(dirty, 200, 199, NOSE)
        _, info = dp.project(dirty, 200, NOSE, pairs=PAIRS, constrain=SKULL_IDX,
                             l_hat_scaled=scaled_refs(p), eps=EPS, target=tgt)
        assert info["moved"] and not info["landed_on_boundary"]
        assert info["converged"] and info["used_target"]

    def test_without_a_target_it_lands_on_the_surface_and_says_so(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        dirty = displace(p, 200, NOSE, (70.0, 0.0))
        lh = scaled_refs(p)
        fixed, info = dp.project(dirty, 200, NOSE, pairs=PAIRS,
                                 constrain=SKULL_IDX, l_hat_scaled=lh, eps=EPS)
        assert info["moved"] and info["landed_on_boundary"]
        ratios = [np.linalg.norm(fixed[NOSE] - fixed[i if j == NOSE else j])
                  / (lh[m] * (1 + EPS))
                  for m in SKULL_IDX for i, j in [PAIRS[m]] if NOSE in (i, j)]
        assert any(abs(r - 1.0) < 1e-9 for r in ratios)

    def test_convergence_is_reported_and_not_assumed(self):
        p = np.asarray(mouse(400), dtype=np.float64)
        dirty = displace(p, 200, NOSE, (70.0, 0.0))
        _, info = dp.project(dirty, 200, NOSE, pairs=PAIRS, constrain=SKULL_IDX,
                             l_hat_scaled=scaled_refs(p), eps=EPS)
        assert info["converged"] is True


class TestReliability:

    def test_abstained_keypoint_frames_go_to_zero(self):
        conf = np.full((10, 7), 0.9)
        abst = np.zeros(10, dtype=bool); abst[3] = True
        r = dp.reliability(conf, np.zeros(10, bool), np.zeros(10, bool), abst)
        assert (r[3] == 0.0).all() and (r[0] == 0.9).all()

    def test_corrected_frames_are_discounted_not_trusted(self):
        conf = np.full((10, 7), 0.8)
        corr = np.zeros(10, dtype=bool); corr[5] = True
        r = dp.reliability(conf, np.zeros(10, bool), corr, np.zeros(10, bool))
        assert 0.0 < r[5].max() < 0.8

    def test_it_stays_in_the_unit_interval(self):
        rng = np.random.default_rng(0)
        r = dp.reliability(rng.random((100, 7)) * 2 - 0.5,
                           np.zeros(100, bool), rng.random(100) < 0.1,
                           rng.random(100) < 0.1)
        assert r.min() >= 0.0 and r.max() <= 1.0


class TestTheGate:

    OBJ = {"dataset": "luna", "arm": "disposition", "split": "report"}

    def test_a_fall_on_the_held_out_group_passes(self):
        rd = lobo.lobo_read(0.030, 0.028, constrained="skull",
                            evaluated="trunk", scored_object=self.OBJ,
                            n_effective=89)
        assert rd.verdict == "PASS" and "never saw" in rd.reason

    def test_no_fall_fails_and_says_why_the_other_axes_cannot_see_it(self):
        rd = lobo.lobo_read(0.030, 0.030, constrained="skull",
                            evaluated="trunk", scored_object=self.OBJ,
                            n_effective=89)
        assert rd.verdict == "FAIL"
        assert "Distortion and retention cannot show this" in rd.reason

    def test_a_rise_fails(self):
        rd = lobo.lobo_read(0.030, 0.034, constrained="skull",
                            evaluated="trunk", scored_object=self.OBJ,
                            n_effective=89)
        assert rd.verdict == "FAIL"

    def test_nothing_to_evaluate_is_inconclusive(self):
        rd = lobo.lobo_read(0.0, 0.0, constrained="skull", evaluated="trunk",
                            scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "INCONCLUSIVE"
