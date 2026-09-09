"""The bone-length diagnostic: what it must catch, and what it must not claim.

The first test in this file is the one the brief names as mandatory. Without it
a units or indexing bug returns zero violations over 22 million frames and reads
as clean data -- the quietest possible failure, and the only one that would let
every number downstream be computed on a corpus nobody checked.

The second load-bearing test is `test_a_uniform_rescale_fires_raw_and_is_invisible
_to_scalefree`. It is the reason there are two metrics at all: it plants the
exact confound the raw sweep cannot distinguish from tracking error, and asserts
that the scale-free arm can.
"""
import numpy as np
import pytest

from vieb.qc import bones
from tests.synthetic import mouse

FPS = 30.0
LEFT_EAR, RIGHT_EAR, NOSE = 0, 1, 2
EAR_NOSE = bones.SKULL.index((LEFT_EAR, NOSE))


def rigid(t=900):
    """A rigid synthetic mouse: every bone length is constant by construction."""
    return mouse(t)


def refs(lengths, eps, group=None):
    """One reference per bone, which is the only shape `violations` accepts."""
    group = bones.SKULL if group is None else group
    return [bones.reference_length(lengths[:, m], eps)["l_hat"]
            for m in range(len(group))]


def teleport(pose, frame, keypoint, along, factor):
    """Displace one keypoint at one frame so a named bone grows by `factor`."""
    out = np.array(pose, dtype=np.float64, copy=True)
    anchor = out[frame, along]
    out[frame, keypoint] = anchor + (out[frame, keypoint] - anchor) * factor
    return out


# --------------------------------------------------------------------------
# The mandatory injection test
# --------------------------------------------------------------------------

class TestTheInjectedTeleport:

    @pytest.mark.parametrize("delta", [0.12, 0.35, 0.75])
    def test_a_synthetic_teleport_is_flagged_at_the_expected_eps(self, delta):
        """Flagged at every eps below the planted excess, and at none above it.

        Both halves matter. Firing below is the detection; NOT firing above is
        what says the threshold is the thing doing the work rather than a bug
        that flags everything.
        """
        pose = rigid()
        f = 400
        dirty = teleport(pose, f, LEFT_EAR, NOSE, 1.0 + delta)
        lengths = bones.metric_lengths(dirty, bones.SKULL, "raw")

        for eps in bones.EPS:
            ref = refs(lengths, eps)
            viol = bones.violations(lengths, ref, eps)
            assert bool(viol[f, EAR_NOSE]) is (eps < delta), (
                f"eps={eps}, planted delta={delta}: bone ratio is "
                f"{lengths[f, EAR_NOSE] / ref[EAR_NOSE]:.4f}")

    def test_the_clean_frames_around_it_are_untouched(self):
        pose = rigid()
        dirty = teleport(pose, 400, LEFT_EAR, NOSE, 1.12)
        lengths = bones.metric_lengths(dirty, bones.SKULL, "raw")
        mask = bones.frame_mask(
            bones.violations(lengths, refs(lengths, 0.02), 0.02))
        assert mask.sum() == 1 and mask[400]

    def test_a_clean_rigid_mouse_violates_nothing_at_any_eps(self):
        lengths = bones.metric_lengths(rigid(), bones.SKULL, "raw")
        for eps in bones.EPS:
            assert not bones.violations(lengths, refs(lengths, eps), eps).any()

    def test_attribution_names_the_keypoint_that_moved(self):
        pose = rigid()
        dirty = teleport(pose, 400, LEFT_EAR, NOSE, 1.5)
        lengths = bones.metric_lengths(dirty, bones.SKULL, "raw")
        att = bones.attribution(
            bones.violations(lengths, refs(lengths, 0.10), 0.10), bones.SKULL, 7)
        assert int(np.argmax(att["share"])) == LEFT_EAR


# --------------------------------------------------------------------------
# The confound the two metrics exist to separate
# --------------------------------------------------------------------------

class TestPerspectiveVersusTracking:

    def test_a_uniform_rescale_fires_raw_and_is_invisible_to_scalefree(self):
        """Rearing multiplies every distance by a common factor.

        This is the whole argument for the second metric. The raw sweep cannot
        tell this frame from a mistracked one; the scale-free sweep must, because
        a common factor is a common offset in log space and the per-frame median
        removes it exactly.
        """
        pose = np.array(rigid(), dtype=np.float64)
        f, factor = 400, 1.30
        centre = pose[f].mean(axis=0)
        pose[f] = centre + (pose[f] - centre) * factor

        raw = bones.metric_lengths(pose, bones.SKULL, "raw")
        assert bones.frame_mask(bones.violations(raw, refs(raw, 0.10), 0.10))[f]

        sf = bones.metric_lengths(pose, bones.SKULL, "scalefree")
        assert not bones.frame_mask(bones.violations(sf, refs(sf, 0.10), 0.10)).any()

    def test_scalefree_still_catches_a_single_displaced_keypoint(self):
        """The scale-free arm must not be blind to real error.

        A common-scale subtraction that also removed one-keypoint failures would
        be useless, so this is the other half of the previous test.
        """
        dirty = teleport(rigid(), 400, LEFT_EAR, NOSE, 1.30)
        sf = bones.metric_lengths(dirty, bones.SKULL, "scalefree")
        assert bones.frame_mask(bones.violations(sf, refs(sf, 0.10), 0.10))[400]


# --------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------

class TestReferenceLength:

    def test_the_iteration_is_monotone_non_increasing(self):
        rng = np.random.default_rng(0)
        v = np.concatenate([rng.normal(50.0, 1.0, 5000),
                            rng.normal(200.0, 5.0, 200)])
        ref = bones.reference_length(v, 0.05)
        trace = ref["trace"]
        assert len(trace) == bones.REF_ITERS + 1
        assert all(trace[i + 1] <= trace[i] + 1e-9 for i in range(len(trace) - 1))

    def test_it_converges_on_a_clean_population_and_says_so(self):
        v = np.full(5000, 42.0)
        ref = bones.reference_length(v, 0.05)
        assert ref["converged"] and ref["l_hat"] == pytest.approx(42.0)

    def test_one_mistracked_frame_cannot_define_the_anatomy(self):
        v = np.full(5000, 42.0)
        v[17] = 4200.0
        assert bones.reference_length(v, 0.05)["l_hat"] == pytest.approx(42.0)

    def test_it_refuses_rather_than_guessing_on_too_little_data(self):
        ref = bones.reference_length(np.array([1.0, 2.0]), 0.05)
        assert not np.isfinite(ref["l_hat"]) and "why" in ref


class TestViolationSemantics:

    def test_an_unmeasurable_length_is_not_a_violation(self):
        """Missingness must not enter a statistic about geometry."""
        lengths = np.full((10, 3), 10.0)
        lengths[4, 1] = np.nan
        assert not bones.violations(lengths, [10.0] * 3, 0.02)[4, 1]

    def test_a_scalar_reference_across_several_bones_is_refused(self):
        """Bones have different true lengths; one reference for all of them
        flags the longer ones on every frame and says nothing."""
        with pytest.raises(ValueError, match="own true length"):
            bones.violations(np.full((10, 3), 10.0), 10.0, 0.02)

    def test_one_bone_is_enough(self):
        """Deliberately not shapeflow's three-pairs rule -- different question."""
        viol = np.zeros((10, 3), dtype=bool)
        viol[3, 1] = True
        assert bones.frame_mask(viol)[3]


# --------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------

class TestShuffledCeiling:

    def test_shuffling_landmarks_makes_the_violation_common(self):
        pose = rigid()
        lengths = bones.metric_lengths(pose, bones.SKULL, "raw")
        ref = refs(lengths, 0.10)
        ceiling = bones.shuffled_ceiling(pose, bones.SKULL, ref, 0.10,
                                         np.random.default_rng(0))
        observed = float(bones.frame_mask(
            bones.violations(lengths, ref, 0.10)).mean())
        assert observed == 0.0 and ceiling > 0.5

    def test_it_never_crosses_a_recording_seam(self):
        """Two recordings far apart in the arena, shuffled together, inflate it.

        The API takes ONE recording for this reason. The test plants the seam it
        would have to cross and asserts the concatenated rate is the higher one,
        so a caller that pools recordings is measuring the arena rather than the
        animal.
        """
        a = np.array(rigid(300), dtype=np.float64)
        b = np.array(rigid(300), dtype=np.float64) + 5000.0
        ref = refs(bones.metric_lengths(a, bones.SKULL, "raw"), 0.10)

        per_recording = np.mean([
            bones.shuffled_ceiling(r, bones.SKULL, ref, 0.10,
                                   np.random.default_rng(0))
            for r in (a, b)])
        pooled = bones.shuffled_ceiling(np.concatenate([a, b]), bones.SKULL,
                                        ref, 0.10, np.random.default_rng(0))
        assert pooled > per_recording


# --------------------------------------------------------------------------
# The second opinion
# --------------------------------------------------------------------------

class TestOverlap:

    def test_the_cells_partition_the_frames_and_match_the_marginals(self):
        rng = np.random.default_rng(0)
        a = rng.random(1000) < 0.05
        b = rng.random(1000) < 0.10
        o = bones.overlap(a, b)
        assert o["both"] + o["new_only"] + o["other_only"] + o["neither"] == 1000
        assert o["both"] + o["new_only"] == int(a.sum())
        assert o["both"] + o["other_only"] == int(b.sum())

    def test_disjoint_masks_have_zero_jaccard_and_full_only_cells(self):
        a = np.array([True, True, False, False])
        b = np.array([False, False, True, True])
        o = bones.overlap(a, b)
        assert o["jaccard"] == 0.0 and o["new_only"] == 2 and o["other_only"] == 2

    def test_it_refuses_masks_of_different_length(self):
        with pytest.raises(ValueError):
            bones.overlap(np.zeros(10, bool), np.zeros(11, bool))


# --------------------------------------------------------------------------
# The join
# --------------------------------------------------------------------------

class TestTheR2Join:

    def test_segment_rates_read_exbias_bounds_in_recording_frame_index(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True
        rates = bones.segment_rates(mask, np.array([[0, 10], [10, 20], [20, 100]]))
        assert rates == pytest.approx([0.0, 1.0, 0.0])

    def test_out_of_range_bounds_are_clipped_not_wrapped(self):
        mask = np.zeros(50, dtype=bool)
        rates = bones.segment_rates(mask, np.array([[40, 200]]))
        assert np.isfinite(rates[0])

    def test_deciles_refuse_rather_than_bin_too_few_segments(self):
        out = bones.r2_deciles(np.zeros(20), np.linspace(0, 1, 20))
        assert out["bins"] == [] and "why" in out

    def test_deciles_recover_a_planted_monotone_relationship(self):
        rng = np.random.default_rng(0)
        r2 = rng.random(4000)
        rate = np.clip(0.5 - 0.5 * r2 + rng.normal(0, 0.02, 4000), 0, 1)
        out = bones.r2_deciles(rate, r2)
        means = [b["mean_violation_rate"] for b in out["bins"]]
        assert len(means) == 10 and means[0] > means[-1]

    def test_deciles_carry_duration_and_flag_a_non_monotone_rate(self):
        """The two statistics can disagree, and the table has to show it.

        Planted: violations concentrated in the middle deciles, which is what a
        duration confound looks like. `rate_is_monotone` must say False so a
        reader does not read the coefficient as a slope.
        """
        rng = np.random.default_rng(0)
        r2 = np.linspace(0.0, 1.0, 4000)
        rate = 0.5 - 2.0 * (r2 - 0.5) ** 2 + rng.normal(0, 0.01, 4000)
        dur = np.full(4000, 0.7)
        out = bones.r2_deciles(rate, r2, durations=dur)
        assert out["rate_is_monotone"] is False
        assert all(b["mean_duration_s"] == pytest.approx(0.7) for b in out["bins"])

    def test_deciles_without_durations_report_nan_not_a_warning(self):
        out = bones.r2_deciles(np.zeros(4000), np.linspace(0, 1, 4000))
        assert all(not np.isfinite(b["mean_duration_s"]) for b in out["bins"])

    def test_a_correlation_that_is_entirely_the_control_partials_to_zero(self):
        """Both variables driven by duration, and nothing else between them.

        This is the shape the corpus actually has: ExBias R^2 falls with segment
        length and the violation rate's denominator IS segment length, so the
        marginal rho is large while the controlled one is not.
        """
        rng = np.random.default_rng(0)
        dur = rng.uniform(0.3, 4.0, 6000)
        rate = 1.0 / dur + rng.normal(0, 0.01, 6000)
        r2 = 1.0 - 0.2 * dur + rng.normal(0, 0.01, 6000)
        animals = [f"a{i % 10}" for i in range(6000)]

        # Both fall with duration, so they rise together: the marginal rho is
        # strongly POSITIVE and describes a relationship neither variable has
        # with the other.
        marginal, _ = bones.per_animal_spearman(rate, r2, animals)
        partial, kept = bones.per_animal_partial_spearman(rate, r2, dur, animals)
        assert float(np.mean(marginal)) > 0.8
        assert abs(float(np.mean(partial))) < 0.25 and len(kept) == 10

    def test_a_real_association_survives_the_control(self):
        rng = np.random.default_rng(1)
        dur = rng.uniform(0.3, 4.0, 6000)
        rate = 0.5 * rng.random(6000)
        r2 = 1.0 - 0.15 * dur - 0.8 * rate + rng.normal(0, 0.01, 6000)
        animals = [f"a{i % 10}" for i in range(6000)]
        partial, _ = bones.per_animal_partial_spearman(rate, r2, dur, animals)
        assert float(np.mean(partial)) < -0.7

    def test_the_join_read_prefers_the_controlled_rho_and_says_so(self):
        rd = bones.join_read(
            {"point": -0.55, "lo": -0.60, "hi": -0.50}, {},
            scored_object=OBJ, n_effective=89,
            partial_ci={"point": -0.02, "lo": -0.05, "hi": 0.01})
        assert rd.verdict == "FAIL"
        assert "duration held fixed" in rd.reason
        assert rd.detail["controlled_for"] == "segment duration"

    def test_spearman_is_per_animal_and_survives_simpsons_paradox(self):
        """A pooled correlation can be strong where every within-animal one is nil.

        Planted here on purpose: each animal sits at its own (rate, R^2) offset
        along a descending line, and within each animal the two are independent.
        Pooling reports a large negative rho for a relationship no animal has.
        """
        from scipy import stats
        rng = np.random.default_rng(0)
        rate, r2, animals = [], [], []
        for k in range(10):
            rate.append(rng.normal(0.05 * k, 0.002, 200))
            r2.append(rng.normal(0.9 - 0.05 * k, 0.002, 200))
            animals += [f"a{k}"] * 200
        rate, r2 = np.concatenate(rate), np.concatenate(r2)

        pooled = float(stats.spearmanr(rate, r2).statistic)
        rhos, kept = bones.per_animal_spearman(rate, r2, animals)
        assert pooled < -0.9
        assert len(kept) == 10 and abs(float(np.mean(rhos))) < 0.2


# --------------------------------------------------------------------------
# The reads
# --------------------------------------------------------------------------

OBJ = {"dataset": "luna", "arm": "bones", "split": "report"}


def rows_at(rate):
    return [{"eps": e, "rate": rate, "n_recordings_above_1pct": 0,
             "shuffled_ceiling": 0.9} for e in bones.EPS]


class TestReads:

    def test_ceiling_read_refuses_when_shuffling_does_not_raise_the_rate(self):
        rd = bones.ceiling_read(0.20, 0.10, scored_object=OBJ, n_effective=89)
        assert rd.verdict == "NOT_A_RESULT" and rd.degenerate

    def test_ceiling_read_passes_with_a_margin(self):
        assert bones.ceiling_read(0.01, 0.90, scored_object=OBJ,
                                  n_effective=89).verdict == "PASS"

    @pytest.mark.parametrize("rate,verdict", [
        (0.005, "PASS"), (0.08, "GRID_LIMITED"), (0.40, "FAIL")])
    def test_the_skull_branch(self, rate, verdict):
        rd = bones.curve_read(rows_at(rate), group="skull", metric="raw",
                              scored_object=OBJ, n_effective=89)
        assert rd.verdict == verdict

    def test_trunk_is_never_the_branch(self):
        rd = bones.curve_read(rows_at(0.005), group="trunk", metric="raw",
                              scored_object=OBJ, n_effective=89)
        assert rd.verdict == "NOT_A_RESULT"
        assert "posture" in rd.reason

    def test_a_failed_r2_join_overrides_a_low_rate(self):
        """Uncorrelated with R^2 is blocking however small the rate is."""
        join = bones.join_read({"point": 0.01, "lo": -0.2, "hi": 0.2}, {},
                               scored_object=OBJ, n_effective=89)
        assert join.verdict == "FAIL"
        rd = bones.curve_read(rows_at(0.005), group="skull", metric="raw",
                              scored_object=OBJ, n_effective=89, join=join)
        assert rd.verdict == "FAIL" and "re-tracking" in rd.reason

    def test_a_small_but_real_rho_is_bounded_rather_than_claimed(self):
        rd = bones.join_read({"point": 0.04, "lo": 0.01, "hi": 0.07}, {},
                             scored_object=OBJ, n_effective=89)
        assert rd.verdict == "GRID_LIMITED" and rd.saturation

    def test_a_standout_drop_is_an_elbow(self):
        sharp = [{"eps": e, "rate": r} for e, r in
                 zip(bones.EPS, [0.40, 0.38, 0.02, 0.019, 0.018])]
        rd = bones.curve_read(sharp, group="skull", metric="raw",
                              scored_object=OBJ, n_effective=89)
        assert rd.detail["curve_shape"] == "elbow"
        assert rd.detail["elbow_eps"] == 0.05
        assert "ELBOW" in rd.reason

    def test_monotonically_increasing_drops_are_a_smooth_decay(self):
        """The shape this corpus actually has, and the bug that misread it.

        Relative drops of 34%, 37%, 44%, 54% are one population thinning out.
        The first version of the rule fired on "largest drop >= 50%" alone and
        wrote "two populations are separable and this is where to cut" into the
        generated document, which was false.
        """
        smooth = [{"eps": e, "rate": r} for e, r in
                  zip(bones.EPS, [0.0513, 0.0341, 0.0213, 0.0120, 0.0055])]
        rd = bones.curve_read(smooth, group="skull", metric="raw",
                              scored_object=OBJ, n_effective=89)
        assert rd.detail["curve_shape"] == "smooth decay"
        assert rd.detail["drops_are_monotone"] is True
        assert rd.detail["elbow_eps"] is None
        assert "no threshold on this grid is principled" in rd.reason

    def test_a_large_but_unremarkable_drop_is_not_an_elbow(self):
        flat = [{"eps": e, "rate": r} for e, r in
                zip(bones.EPS, [1.0, 0.45, 0.20, 0.09, 0.04])]
        rd = bones.curve_read(flat, group="skull", metric="raw",
                              scored_object=OBJ, n_effective=89)
        assert rd.detail["curve_shape"] == "smooth decay"
