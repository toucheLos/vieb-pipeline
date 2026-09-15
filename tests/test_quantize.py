"""The alphabet must be a tokenization of the representation, not of the clock.

Three things here can go wrong silently and all three have precedents on this
project: a bout crossing a recording seam, abstain being dropped instead of
carried, and the two arms being standardised differently so the MDL comparison
between them measures the standardisation.
"""
import numpy as np
import pytest

from recur.label.nonparametric import ABSTAIN
from vieb.tok import quantize as q


def _frames(n, rng, *, scale=1.0):
    """n frames of 17 channels: 14 pose, then v_x, v_y, omega.

    Speed is deliberately AUTOCORRELATED. `represent.speed_terciles` requires a
    bout to last 0.25 s, and iid speed flips tercile almost every frame, so a
    white-noise fixture has no bouts at all and `corpus_sd` correctly returns
    nothing. Real speed is smooth; a fixture that is not would be testing the
    absence of bouts rather than the SD.
    """
    x = rng.normal(0, 0.2, size=(n, 17))
    t = np.arange(n) / 30.0
    slow = np.sin(2 * np.pi * t / 6.0) + 0.3 * np.sin(2 * np.pi * t / 1.7)
    x[:, 14] = scale * (1.5 + slow) + 0.05 * rng.normal(size=n)
    x[:, 15] = scale * 0.2 * np.cos(2 * np.pi * t / 5.0)
    return x


class TestSpeed:

    def test_it_is_the_translation_magnitude(self):
        x = np.zeros((3, 17))
        x[:, 14] = [3.0, 0.0, -3.0]
        x[:, 15] = [4.0, 0.0, 4.0]
        assert q.speed(x) == pytest.approx([5.0, 0.0, 5.0])

    def test_omega_is_not_in_it(self):
        """BLOCKING. A fast pivot must not land in the fast-translation stratum:
        rotation carries a different unit and is a different behaviour."""
        x = np.zeros((2, 17))
        x[:, 16] = [0.0, 50.0]
        assert q.speed(x) == pytest.approx([0.0, 0.0])


class TestBoutsAndSeams:

    def test_no_bout_crosses_a_seam(self):
        """BLOCKING. The seam rule, asserted rather than assumed."""
        rng = np.random.default_rng(0)
        sp = np.abs(rng.normal(1.0, 0.5, size=600))
        bounds = np.array([0, 200, 450, 600])
        for a, b in q.bouts(sp, 30.0, bounds):
            r = np.searchsorted(bounds, a, side="right") - 1
            assert b <= bounds[r + 1], (a, b, bounds)

    def test_a_bout_list_is_the_concatenation_of_its_recordings(self):
        """Cutting the concatenated array would make the terciles a property of
        whichever recordings happened to be adjacent."""
        rng = np.random.default_rng(1)
        one = np.abs(rng.normal(1.0, 0.4, size=300))
        two = np.abs(rng.normal(4.0, 0.4, size=300))
        joint = q.bouts(np.concatenate([one, two]), 30.0, [0, 300, 600])
        alone = (q.bouts(one, 30.0, [0, 300])
                 + [(a + 300, b + 300) for a, b in q.bouts(two, 30.0, [0, 300])])
        assert joint == alone

    def test_the_sd_is_finite_and_per_channel(self):
        rng = np.random.default_rng(2)
        x = _frames(600, rng)
        sd, n = q.corpus_sd(x, q.speed(x), 30.0, [0, 300, 600])
        assert sd.shape == (17,)
        assert np.isfinite(sd).all() and (sd > 0).all()
        assert n > 0

    def test_no_bouts_returns_zeros_rather_than_none(self):
        x = _frames(5, np.random.default_rng(3))
        sd, n = q.corpus_sd(x, np.full(5, np.nan), 30.0, [0, 5])
        assert n == 0 and sd.shape == (17,) and not sd.any()


class TestStrata:

    def test_edges_are_interior_quantiles(self):
        sp = np.arange(1000, dtype=float)
        e = q.strata_edges(sp, np.ones(1000, bool))
        assert e.shape == (q.N_STRATA - 1,)
        assert (np.diff(e) > 0).all()

    def test_strata_are_balanced_on_the_frames_they_were_cut_on(self):
        sp = np.arange(1000, dtype=float)
        e = q.strata_edges(sp, np.ones(1000, bool))
        counts = np.bincount(q.stratum_of(sp, e), minlength=q.N_STRATA)
        assert counts.min() >= 190 and counts.max() <= 210, counts

    def test_edges_come_from_the_mask_only(self):
        """A stratum boundary is a hyperparameter, so it is cut on tune."""
        sp = np.concatenate([np.arange(100.0), np.full(100, 1e6)])
        mask = np.concatenate([np.ones(100, bool), np.zeros(100, bool)])
        assert q.strata_edges(sp, mask).max() < 100.0

    def test_non_finite_speed_gets_no_stratum(self):
        e = np.array([1.0, 2.0, 3.0, 4.0])
        assert q.stratum_of(np.array([np.nan, 2.5]), e).tolist() == [-1, 2]

    def test_no_finite_speed_is_refused(self):
        with pytest.raises(ValueError):
            q.strata_edges(np.full(10, np.nan), np.ones(10, bool))


class TestAllocate:

    @pytest.mark.parametrize("n", q.STATE_GRID)
    def test_it_sums_to_n_exactly(self, n):
        """The MDL codebook term charges |codebook|; an arm quietly carrying
        2045 symbols where the other carries 2048 is charged differently."""
        assert sum(q.allocate(n)) == n

    def test_the_remainder_goes_to_the_low_strata(self):
        assert q.allocate(12, 5) == [3, 3, 2, 2, 2]

    def test_it_refuses_fewer_states_than_strata(self):
        with pytest.raises(ValueError):
            q.allocate(3, 5)


class TestFit:

    def _setup(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        x = _frames(n, rng, scale=3.0)
        sp = q.speed(x)
        sd, _ = q.corpus_sd(x, sp, 30.0, [0, n // 2, n])
        return x, sp, np.where(sd > 0, sd, 1.0), np.arange(17), np.ones(n, bool)

    def test_plain_gives_exactly_n_centroids(self):
        x, sp, sd, cols, m = self._setup()
        mod = q.fit(x, sd, cols, m, arm="plain", n_states=16)
        assert mod["centroids"].shape == (16, 17)

    def test_speed_gives_exactly_n_centroids_across_strata(self):
        x, sp, sd, cols, m = self._setup()
        mod = q.fit(x, sd, cols, m, arm="speed", n_states=20, sp=sp)
        assert mod["centroids"].shape == (20, 17)
        assert mod["offsets"].tolist() == [0, 4, 8, 12, 16, 20]

    def test_both_arms_carry_the_identical_sd(self):
        """BLOCKING. Two arms standardised differently are two representations,
        and the MDL comparison between them would measure the wrong difference."""
        x, sp, sd, cols, m = self._setup()
        a = q.fit(x, sd, cols, m, arm="plain", n_states=16)
        b = q.fit(x, sd, cols, m, arm="speed", n_states=20, sp=sp)
        assert np.array_equal(a["sd"], b["sd"])

    def test_the_speed_arm_needs_a_speed_vector(self):
        x, sp, sd, cols, m = self._setup()
        with pytest.raises(ValueError):
            q.fit(x, sd, cols, m, arm="speed", n_states=20)

    def test_an_unknown_arm_is_refused(self):
        x, sp, sd, cols, m = self._setup()
        with pytest.raises(ValueError):
            q.fit(x, sd, cols, m, arm="pca", n_states=16)

    def test_a_stratum_too_small_for_its_budget_is_refused(self):
        x, sp, sd, cols, m = self._setup(n=100)
        with pytest.raises(ValueError):
            q.fit(x, sd, cols, m, arm="speed", n_states=2000, sp=sp)

    def test_the_partition_is_fitted_on_the_mask_only(self):
        x, sp, sd, cols, m = self._setup()
        half = m.copy()
        half[len(m) // 2:] = False
        mod = q.fit(x, sd, cols, half, arm="plain", n_states=16)
        assert mod["n_fit_frames"] == int(half.sum())

    def test_it_records_an_input_hash(self):
        x, sp, sd, cols, m = self._setup()
        a = q.fit(x, sd, cols, m, arm="plain", n_states=16, seed=0)
        b = q.fit(x, sd, cols, m, arm="plain", n_states=16, seed=1)
        assert a["input_hash"] and a["input_hash"] != b["input_hash"]


class TestAssign:

    def _model(self, arm, n_states, seed=0):
        rng = np.random.default_rng(seed)
        x = _frames(4000, rng, scale=3.0)
        sp = q.speed(x)
        sd, _ = q.corpus_sd(x, sp, 30.0, [0, 2000, 4000])
        sd = np.where(sd > 0, sd, 1.0)
        m = np.ones(4000, bool)
        return x, sp, q.fit(x, sd, np.arange(17), m, arm=arm,
                            n_states=n_states, sp=sp)

    def test_output_is_frame_aligned(self):
        x, sp, mod = self._model("plain", 16)
        assert q.assign(x, mod).shape == (x.shape[0],)

    def test_abstain_is_applied_without_dropping_rows(self):
        """BLOCKING. Dropping abstained rows would splice two distant moments
        into one run, which is exactly what RLE must not see."""
        x, sp, mod = self._model("plain", 16)
        ab = np.zeros(x.shape[0], bool)
        ab[100:200] = True
        y = q.assign(x, mod, abstain=ab)
        assert y.shape == (x.shape[0],)
        assert (y[100:200] == ABSTAIN).all()
        assert (y[:100] != ABSTAIN).all()

    def test_every_speed_symbol_stays_inside_its_stratum(self):
        """BLOCKING. The whole point of the arm: a fast frame must get a symbol
        from the fast budget, or the stratification bought nothing."""
        x, sp, mod = self._model("speed", 20)
        y = q.assign(x, mod, sp=sp)
        strat = q.stratum_of(sp, mod["edges"])
        off = mod["offsets"]
        ok = y != ABSTAIN
        assert (y[ok] >= off[strat[ok]]).all()
        assert (y[ok] < off[strat[ok] + 1]).all()

    def test_the_speed_arm_needs_a_speed_vector(self):
        x, sp, mod = self._model("speed", 20)
        with pytest.raises(ValueError):
            q.assign(x, mod)

    def test_assignment_is_deterministic(self):
        x, sp, mod = self._model("plain", 16)
        assert np.array_equal(q.assign(x, mod), q.assign(x, mod))


class TestOccupancy:

    def test_abstain_is_out_of_the_denominator(self):
        y = np.array([0, 0, 1, ABSTAIN, ABSTAIN])
        occ = q.occupancy(y, 2)
        assert occ["n_labelled"] == 3
        assert occ["top_share"] == pytest.approx(2 / 3)
        assert occ["abstain_frac"] == pytest.approx(0.4)

    def test_dead_symbols_are_counted(self):
        occ = q.occupancy(np.zeros(10, int), 10)
        assert occ["dead_frac"] == pytest.approx(0.9)

    def test_a_uniform_alphabet_has_gini_near_zero(self):
        occ = q.occupancy(np.arange(1000) % 50, 50)
        assert abs(occ["gini"]) < 0.02

    def test_all_abstain_gives_no_occupancy(self):
        occ = q.occupancy(np.full(5, ABSTAIN), 4)
        assert occ["n_labelled"] == 0 and not np.isfinite(occ["top_share"])


class TestAlphabetRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 16}

    def _read(self, occ):
        return q.alphabet_read(occ, scored_object=self.OBJ, n_effective=60)

    def test_a_dominant_symbol_fails(self):
        y = np.concatenate([np.zeros(60, int), np.arange(40) % 15 + 1])
        assert self._read(q.occupancy(y, 16)).verdict == "FAIL"

    def test_a_mostly_dead_alphabet_is_grid_limited(self):
        y = np.arange(100) % 3
        assert self._read(q.occupancy(y, 100)).verdict == "GRID_LIMITED"

    def test_a_healthy_alphabet_passes(self):
        y = np.arange(1600) % 16
        assert self._read(q.occupancy(y, 16)).verdict == "PASS"

    def test_all_abstain_is_not_a_result(self):
        r = self._read(q.occupancy(np.full(20, ABSTAIN), 16))
        assert r.verdict == "NOT_A_RESULT" and r.degenerate

    def test_every_read_carries_the_numbers_it_claims(self):
        r = self._read(q.occupancy(np.arange(1600) % 16, 16))
        for k in ("top_share", "dead_frac", "abstain_frac", "gini"):
            assert k in r.detail
        assert r.n_effective == 60
