"""What the cleaning changes: the statistic, and the baseline it is measured from.

The load-bearing test here is `test_measuring_the_gap_policy_against_its_own
_output_returns_zero`. It plants the exact mistake the module exists to avoid --
using `pose_unfiltered` as the baseline when `pose_unfiltered` IS the gap
policy's output -- and shows it returns the one answer that cannot be right.
"""
import numpy as np
import pytest

from vieb.qc import effect
from tests.synthetic import mouse

FPS = 30.0
OBJ = {"dataset": "luna", "arm": "effect", "split": "all"}


def gapped(pose, bad_frames, keypoint=2):
    """A crude gap policy: linearly interpolate one keypoint over `bad_frames`."""
    out = np.array(pose, dtype=np.float64, copy=True)
    a, b = bad_frames.start, bad_frames.stop
    for c in (0, 1):
        out[a:b, keypoint, c] = np.linspace(out[a - 1, keypoint, c],
                                            out[b, keypoint, c], b - a + 2)[1:-1]
    return out


class TestTheBaseline:

    def test_measuring_the_gap_policy_against_its_own_output_returns_zero(self):
        """The trap. `pose_unfiltered` is the OUTPUT, not the input.

        Against the raw array the gap policy has a visible effect; against its
        own output it has exactly none, and zero is the one answer that cannot
        be right for a layer that demonstrably moved something.
        """
        raw = np.asarray(mouse(300), dtype=np.float64)
        filled = gapped(raw, slice(100, 103))

        right = effect.displacement(raw, filled)
        wrong = effect.displacement(filled, filled)

        assert right.max() > 0.0
        assert wrong.max() == 0.0

    def test_the_effect_is_confined_to_the_frames_that_were_filled(self):
        raw = np.asarray(mouse(300), dtype=np.float64)
        d = effect.displacement(raw, gapped(raw, slice(100, 103)))
        moved = np.flatnonzero(d.max(axis=1) > 0)
        assert moved.tolist() == [100, 101, 102]


class TestDisplacement:

    def test_a_known_shift_measures_its_own_length(self):
        p = np.asarray(mouse(50), dtype=np.float64)
        q = p + np.array([3.0, 4.0])
        assert effect.displacement(p, q) == pytest.approx(5.0)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError):
            effect.displacement(np.zeros((10, 7, 2)), np.zeros((9, 7, 2)))


class TestSummarize:

    def test_the_threshold_curve_is_monotone(self):
        rng = np.random.default_rng(0)
        out = effect.summarize(np.abs(rng.normal(0, 2, (500, 7))))
        fr = [out["frac_moved_over"][str(t)] for t in effect.THRESHOLDS_PX]
        assert all(fr[i] >= fr[i + 1] for i in range(len(fr) - 1))

    def test_body_lengths_divide_by_the_animal(self):
        d = np.full((100, 7), 11.0)
        out = effect.summarize(d, ell=110.0)
        assert out["body_lengths"]["median"] == pytest.approx(0.1)

    def test_without_ell_there_is_no_body_length_row(self):
        assert "body_lengths" not in effect.summarize(np.ones((10, 7)))

    def test_a_mask_restricts_the_distribution(self):
        d = np.zeros((100, 7))
        d[:10] = 5.0
        m = np.zeros(100, dtype=bool)
        m[:10] = True
        assert effect.summarize(d, mask=m)["px"]["median"] == pytest.approx(5.0)


class TestConcentration:

    def base(self, inside, outside, n=1000, frac=0.1):
        d = np.full((n, 7), float(outside))
        m = np.zeros(n, dtype=bool)
        m[: int(n * frac)] = True
        d[m] = inside
        return effect.by_flag(d, {"f": m})["f"]

    def test_it_uses_the_median_when_the_median_is_informative(self):
        c = self.base(4.0, 1.0)
        assert c["concentration_on"] == "median"
        assert c["concentration"] == pytest.approx(4.0)

    def test_it_falls_back_to_the_mean_when_both_medians_are_zero(self):
        """A layer touching 1% of frames has a zero median on both sides, and
        0/0 is the wrong statistic rather than an absence of concentration."""
        d = np.zeros((1000, 7))
        m = np.zeros(1000, dtype=bool)
        m[:10] = True
        # Fewer than half the flagged frames move, so the median is zero on
        # BOTH sides -- which is the situation the fallback exists for.
        d[:3] = 3.0
        c = effect.by_flag(d, {"f": m})["f"]
        assert c["median_inside"] == 0.0 and c["median_outside"] == 0.0
        assert c["concentration_on"] == "mean"

    def test_touching_only_flagged_frames_is_infinite_not_missing(self):
        d = np.zeros((1000, 7))
        m = np.zeros(1000, dtype=bool)
        m[:10] = True
        d[m] = 2.0
        c = effect.by_flag(d, {"f": m})["f"]
        assert np.isinf(c["concentration"])
        rd = effect.concentration_read("gap_policy", c, scored_object=OBJ,
                                       n_effective=1)
        assert rd.verdict == "PASS" and "only the frames it marked" in rd.reason

    def test_a_transform_indifferent_to_the_flag_fails(self):
        c = self.base(1.0, 1.0)
        rd = effect.concentration_read("wiener", c, scored_object=OBJ,
                                       n_effective=1)
        assert rd.verdict == "FAIL"
        assert "'cleaning' overstates what it does" in rd.reason

    def test_a_targeted_transform_passes(self):
        rd = effect.concentration_read("gap", self.base(10.0, 1.0),
                                       scored_object=OBJ, n_effective=1)
        assert rd.verdict == "PASS"

    def test_a_per_frame_flag_is_broadcast_over_keypoints(self):
        d = np.ones((100, 7))
        out = effect.by_flag(d, {"f": np.ones(100, dtype=bool)})["f"]
        assert out["n_inside"] == 700 and out["n_outside"] == 0


class TestKinematicShift:

    def test_a_uniform_slowdown_shows_as_a_flat_ratio(self):
        rng = np.random.default_rng(0)
        sp = np.abs(rng.normal(5, 2, 5000))
        out = effect.kinematic_shift(sp, sp * 0.6, sp, sp * 0.6)
        assert all(r == pytest.approx(0.6, rel=1e-6) for r in out["speed"]["ratio"])

    def test_deleting_only_the_tail_shows_only_in_the_tail(self):
        rng = np.random.default_rng(0)
        sp = np.abs(rng.normal(5, 2, 20000))
        clipped = np.minimum(sp, np.quantile(sp, 0.95))
        out = effect.kinematic_shift(sp, clipped, sp, clipped)
        r = out["speed"]["ratio"]
        assert r[0] == pytest.approx(1.0, abs=1e-9)   # median untouched
        assert r[3] < 0.85                            # p99.9 cut

    def test_it_refuses_rather_than_summarising_a_handful_of_frames(self):
        out = effect.kinematic_shift(np.ones(5), np.ones(5), np.ones(5), np.ones(5))
        assert "why" in out["speed"]


class TestByKeypoint:

    def test_every_landmark_gets_a_row_in_order(self):
        names = ["a", "b", "c"]
        rows = effect.by_keypoint(np.ones((50, 3)), names)
        assert [r["keypoint"] for r in rows] == names
