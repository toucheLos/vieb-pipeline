"""The injection benchmark, and the checks that say the harness works.

Two of these are blocking in the strict sense that if they fail, nothing else in
`results/injection.json` means anything:

* `test_raw_scores_exactly_zero_damage` -- the baseline moves nothing, so its
  damage is identically zero. A non-zero value means the harness is measuring
  itself.
* `test_an_oracle_recovers_everything` -- if handing the harness the true pose
  does not score ~0 repair, the benchmark cannot detect a repair at all and a
  PASS from it would be meaningless.
"""
import numpy as np
import pytest

from vieb.qc import bones, inject, recover, truth
from tests.synthetic import mouse

PAIRS = bones.pair_indices(7)
FPS = 30.0
CENTER = 3


def ell_of(pose):
    return float(np.median(np.linalg.norm(pose[:, 2] - pose[:, CENTER], axis=-1)))


def clean_fixture(t=600):
    """A synthetic mouse and a pool covering all of it."""
    p = np.asarray(mouse(t), dtype=np.float64)
    segs = np.array([[0, t]], dtype=np.int64)
    return p, segs, ell_of(p)


class TestThePool:

    def test_a_clean_recording_is_almost_all_pool(self):
        p = np.asarray(mouse(600), dtype=np.float64)
        conf = np.full((600, 7), 0.9)
        miss = np.zeros((600, 7), dtype=bool)
        mask, why = truth.clean_mask(p, conf, miss, ell_of(p), pairs=PAIRS)
        assert why["frac_bone_ok"] == 1.0
        assert mask.mean() > 0.9

    def test_low_confidence_frames_are_excluded(self):
        p = np.asarray(mouse(300), dtype=np.float64)
        conf = np.full((300, 7), 0.9)
        conf[100:120, 2] = 0.1
        miss = np.zeros((300, 7), dtype=bool)
        mask, _ = truth.clean_mask(p, conf, miss, ell_of(p), pairs=PAIRS)
        assert not mask[100:120].any()

    def test_a_missing_keypoint_excludes_the_frame(self):
        p = np.asarray(mouse(300), dtype=np.float64)
        conf = np.full((300, 7), 0.9)
        miss = np.zeros((300, 7), dtype=bool)
        miss[50:60, 4] = True
        mask, _ = truth.clean_mask(p, conf, miss, ell_of(p), pairs=PAIRS)
        assert not mask[50:60].any()

    def test_a_discontinuity_excludes_its_frames(self):
        p = np.asarray(mouse(300), dtype=np.float64)
        p[150, 2] += np.array([60.0, 0.0])
        conf = np.full((300, 7), 0.9)
        miss = np.zeros((300, 7), dtype=bool)
        mask, _ = truth.clean_mask(p, conf, miss, ell_of(p), pairs=PAIRS)
        assert not mask[150]

    def test_short_segments_are_dropped(self):
        m = np.zeros(200, dtype=bool)
        m[10:20] = True             # 10 frames, too short
        m[100:180] = True           # 80 frames
        segs = truth.segments(m, min_frames=30)
        assert segs.shape == (1, 2)
        assert tuple(segs[0]) == (100, 180)

    def test_speed_bins_are_quantiles_not_equal_width(self):
        v = np.concatenate([np.linspace(0.0, 0.1, 90), np.linspace(5.0, 100.0, 10)])
        edges = truth.speed_bins(v, n_bins=5)
        assert edges.size == 6
        # Quantile edges: four of the five bins fall inside the dense low range,
        # where equal-width bins would put 90 of 100 segments in one bin.
        assert truth.assign_bin(0.0, edges) == 0
        assert truth.assign_bin(100.0, edges) == 4
        assert edges[-2] < 100.0

    def test_a_degenerate_speed_distribution_yields_fewer_strata(self):
        """Tied quantile edges would file the slowest segments as the fastest."""
        v = np.concatenate([np.full(90, 0.01), np.full(10, 100.0)])
        edges = truth.speed_bins(v, n_bins=5)
        assert edges.size == 2, "five tied edges collapse to two"
        assert truth.assign_bin(0.01, edges) == 0
        assert truth.assign_bin(100.0, edges) == 0


class TestTheCorruptions:

    def test_it_does_not_modify_its_input(self):
        p, segs, ell = clean_fixture()
        before = p.copy()
        inject.corrupt(p, segs, np.random.default_rng(0), ell=ell)
        np.testing.assert_array_equal(p, before)

    def test_every_event_is_inside_a_segment(self):
        p = np.asarray(mouse(600), dtype=np.float64)
        segs = np.array([[100, 300], [400, 550]], dtype=np.int64)
        _, d = inject.corrupt(p, segs, np.random.default_rng(0), ell=ell_of(p))
        for ev in d["events"]:
            assert any(int(a) <= ev["a"] and ev["b"] <= int(b) for a, b in segs), ev

    def test_corruptions_never_overlap(self):
        """Two on one keypoint-frame would make 'the truth' ambiguous."""
        p, segs, ell = clean_fixture(1200)
        _, d = inject.corrupt(p, segs, np.random.default_rng(1), ell=ell)
        spans = sorted((e["a"], e["b"]) for e in d["events"])
        assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))

    def test_the_realised_rate_tracks_the_requested_one(self):
        p, segs, ell = clean_fixture(6000)
        _, d = inject.corrupt(p, segs, np.random.default_rng(2), ell=ell,
                              rates={"teleport": 0.02, "park": 0.0,
                                     "swap": 0.0, "dropout": 0.0})
        # Teleports are 1-3 frames on one keypoint of seven, so the
        # keypoint-frame rate is the event rate times about 2/7.
        assert d["n_events"]["teleport"] > 50
        # The rate is a share of KEYPOINT-frames and must be hit as one.
        assert 0.85 < d["realised_rate"]["teleport"] / 0.02 < 1.15

    def test_a_park_holds_one_position_and_does_not_track(self):
        """This is what makes a park unreachable by a temporal filter."""
        p, segs, ell = clean_fixture()
        out, d = inject.corrupt(p, segs, np.random.default_rng(3), ell=ell,
                                rates={"teleport": 0, "park": 0.02,
                                       "swap": 0, "dropout": 0})
        parks = [e for e in d["events"] if e["kind"] == "park"]
        assert parks
        ev = parks[0]
        k = ev["keypoints"][0]
        held = out[ev["a"]:ev["b"], k]
        assert np.allclose(held, held[0]), "a park must not move"

    def test_a_teleport_tracks_the_animal_at_a_constant_offset(self):
        p, segs, ell = clean_fixture()
        out, d = inject.corrupt(p, segs, np.random.default_rng(4), ell=ell,
                                rates={"teleport": 0.02, "park": 0,
                                       "swap": 0, "dropout": 0})
        ev = [e for e in d["events"] if e["kind"] == "teleport"][0]
        k = ev["keypoints"][0]
        off = out[ev["a"]:ev["b"], k] - p[ev["a"]:ev["b"], k]
        assert np.allclose(off, off[0])

    def test_a_dropout_is_nan_and_not_a_sentinel(self):
        """A (0, 0) sentinel reads downstream as a teleport to the origin."""
        p, segs, ell = clean_fixture()
        out, d = inject.corrupt(p, segs, np.random.default_rng(5), ell=ell,
                                rates={"teleport": 0, "park": 0,
                                       "swap": 0, "dropout": 0.05})
        ev = [e for e in d["events"] if e["kind"] == "dropout"][0]
        assert np.isnan(out[ev["a"]:ev["b"], ev["keypoints"][0]]).all()

    def test_a_swap_exchanges_a_bilateral_pair(self):
        p, segs, ell = clean_fixture()
        out, d = inject.corrupt(p, segs, np.random.default_rng(6), ell=ell,
                                rates={"teleport": 0, "park": 0,
                                       "swap": 0.05, "dropout": 0})
        ev = [e for e in d["events"] if e["kind"] == "swap"][0]
        i, j = ev["keypoints"]
        assert (i, j) in inject.SWAP_PAIRS
        np.testing.assert_allclose(out[ev["a"]:ev["b"], i], p[ev["a"]:ev["b"], j])

    def test_the_mask_marks_exactly_the_corrupted_keypoint_frames(self):
        p, segs, ell = clean_fixture()
        out, d = inject.corrupt(p, segs, np.random.default_rng(7), ell=ell)
        changed = ~np.isclose(out, p, equal_nan=True).all(axis=-1)
        # Everything marked changed must be masked. The converse can fail
        # legitimately: a zero-length displacement draw changes nothing.
        assert not (changed & ~d["mask"]).any()

    def test_the_displacement_sampler_matches_its_measured_median(self):
        v = inject.sample_displacement(np.random.default_rng(0), 40000)
        assert 20.0 < float(np.median(v)) < 25.0        # measured 22.15 px
        assert 80.0 < float(np.percentile(v, 90)) < 105.0   # measured 92.23
        assert v.min() > 0.0

    def test_the_keypoint_sampler_prefers_the_landmarks_that_really_fail(self):
        k = inject.sample_keypoint(np.random.default_rng(0), 40000)
        share = np.bincount(k, minlength=7) / k.size
        assert share[2] > share[3] * 5, "nose should beat centre by ~24x"
        assert int(np.argmax(share)) == 2

    def test_no_segment_means_no_corruption_rather_than_a_crash(self):
        p = np.asarray(mouse(100), dtype=np.float64)
        out, d = inject.corrupt(p, np.zeros((0, 2), dtype=np.int64),
                                np.random.default_rng(0), ell=ell_of(p))
        np.testing.assert_array_equal(out, p)
        assert d["events"] == []


class TestTheScoring:

    @staticmethod
    def setup(seed=0, t=600):
        p, segs, ell = clean_fixture(t)
        out, d = inject.corrupt(p, segs, np.random.default_rng(seed), ell=ell)
        pool = np.zeros(t, dtype=bool)
        for a, b in segs:
            pool[a:b] = True
        return p, out, d, pool, ell

    def test_raw_scores_exactly_zero_damage(self):
        """BLOCKING. The baseline moves nothing, so it breaks nothing."""
        p, out, d, pool, ell = self.setup()
        s = recover.score(p, out, d["mask"], pool, ell)
        assert s["damage"] == 0.0
        assert s["repair"] > 0.0

    def test_raw_repair_is_exactly_the_injected_magnitude(self):
        """BLOCKING. If the arm does nothing, the error IS the corruption."""
        p, out, d, pool, ell = self.setup()
        s = recover.score(p, out, d["mask"], pool, ell)
        err = recover.errors(p, out, ell)
        hit = d["mask"] & pool[:, None] & np.isfinite(err)
        assert s["repair"] == pytest.approx(float(err[hit].mean()))

    def test_an_oracle_recovers_everything(self):
        """BLOCKING. If handing it the truth does not score ~0, it cannot see a
        repair at all and a PASS would mean nothing."""
        p, out, d, pool, ell = self.setup()
        s = recover.score(p, p, d["mask"], pool, ell)
        assert s["repair"] == pytest.approx(0.0, abs=1e-12)
        assert s["damage"] == pytest.approx(0.0, abs=1e-12)

    def test_a_constant_offset_arm_is_all_damage(self):
        """The failure mode no other axis in this repo can charge for."""
        p, out, d, pool, ell = self.setup()
        shifted = out + np.array([2.0, 0.0])
        s = recover.score(p, shifted, d["mask"], pool, ell)
        assert s["damage"] > 0.0
        assert s["damage"] == pytest.approx(2.0 / ell, rel=1e-6)

    def test_nothing_outside_the_pool_is_scored(self):
        p, out, d, _pool, ell = self.setup()
        pool = np.zeros(p.shape[0], dtype=bool)
        pool[:100] = True
        s = recover.score(p, out, d["mask"], pool, ell)
        # Dropouts are NaN and land in neither mean; they are counted, not lost.
        assert (s["n_corrupted"] + s["n_clean"]
                + s["n_unscoreable"]) == 100 * 7

    def test_by_kind_reports_the_fraction_of_error_removed(self):
        p, out, d, pool, ell = self.setup()
        k = recover.by_kind(p, out, d["events"], ell)
        for kind, cell in k.items():
            if kind == "dropout":
                continue
            # raw removes nothing, so error == injected and the fraction is 0.
            assert cell["fraction_of_error_removed"] == pytest.approx(0.0, abs=1e-9)


class TestTheRead:

    OBJ = {"dataset": "luna", "arm": "injection", "split": "report"}

    @staticmethod
    def rows(raw_damage=0.0):
        return [{"arm": "raw", "damage": raw_damage, "repair": 0.3},
                {"arm": "viterbi", "damage": 0.002, "repair": 0.18},
                {"arm": "median_0.50", "damage": 0.04, "repair": 0.12}]

    @staticmethod
    def iv(viterbi, median):
        return {"viterbi": {"net": {"lo": viterbi[0], "hi": viterbi[1]}},
                "median_0.50": {"net": {"lo": median[0], "hi": median[1]}}}

    def test_a_non_zero_baseline_damage_voids_the_table(self):
        rd = recover.recovery_read(self.rows(raw_damage=1e-6),
                                   self.iv((-0.5, -0.2), (0.1, 0.4)),
                                   {}, scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "NOT_A_RESULT" and rd.degenerate
        assert "measuring itself" in rd.reason

    def test_an_arm_that_helps_in_every_stratum_passes(self):
        rd = recover.recovery_read(
            self.rows(), self.iv((-0.5, -0.2), (0.1, 0.4)),
            {"viterbi": [-0.4, -0.5, -0.3, -0.6, -0.2],
             "median_0.50": [0.2, 0.3, 0.1, 0.4, 0.5]},
            scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "PASS"
        assert "viterbi" in rd.detail["helped"]
        assert "median_0.50" in rd.detail["harmed"]

    def test_an_arm_that_helps_only_slow_frames_is_refused(self):
        """The pool is biased towards still behaviour, so this must not average."""
        rd = recover.recovery_read(
            self.rows(), self.iv((-0.5, -0.2), (-0.3, -0.1)),
            {"viterbi": [-0.4, -0.5, -0.3, -0.6, -0.2],
             "median_0.50": [-0.9, -0.6, -0.1, 0.3, 0.8]},
            scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "GRID_LIMITED"
        assert rd.detail["mixed_by_speed"] == ["median_0.50"]
        assert "report a failure as a success" in rd.reason

    def test_an_interval_touching_zero_does_not_count_as_helping(self):
        rd = recover.recovery_read(
            self.rows(), self.iv((-0.5, 0.0), (0.1, 0.4)),
            {"viterbi": [-0.4] * 5}, scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "FAIL"
        assert set(rd.detail["harmed"]) == {"viterbi", "median_0.50"}

    def test_a_missing_baseline_is_refused(self):
        rd = recover.recovery_read(
            [{"arm": "viterbi", "damage": 0.002, "repair": 0.18}],
            self.iv((-0.5, -0.2), (0.1, 0.4)), {},
            scored_object=self.OBJ, n_effective=89)
        assert rd.verdict == "NOT_A_RESULT" and rd.degenerate
