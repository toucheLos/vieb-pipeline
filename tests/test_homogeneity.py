"""Does a symbol share a future -- and is the statistic comparable across N?

Plug-in total variation is biased upward by roughly sqrt(K/n). At N = 256 with
~1,800 runs per quartile over 255 successors that is about 0.37; at N = 8 it is
about 0.01. Reported bare it would make every fine alphabet look heterogeneous
and every coarse one look coherent, which is the exact opposite of the truth and
would have driven Task 3's floor the wrong way.
"""
import numpy as np
import pytest

from vieb.tok import homogeneity as hm


def _symbol(rng, n=1600, k=255, shift=0.0, n_animals=8):
    """One symbol's runs. `shift` moves the FAR quartile's next-symbol
    distribution away from the near one; 0 means the null is true."""
    animal = np.repeat(np.arange(n_animals), n // n_animals)
    n = animal.shape[0]
    dist = rng.random(n)
    quart = hm.quartiles_within(dist, animal)
    far = quart == 3
    nxt = rng.integers(0, k, size=n)
    if shift > 0:
        bias = rng.random(n) < shift
        nxt = np.where(far & bias, rng.integers(0, max(2, k // 20), size=n), nxt)
    return {"code": np.zeros(n, dtype=np.int64), "next_code": nxt,
            "censored": np.zeros(n, dtype=bool),
            "duration_bin": rng.integers(0, 12, size=n),
            "quartile": quart, "duration": rng.integers(1, 30, size=n),
            "animal": np.array([f"a{i}" for i in animal])}


def _table(d, *, k=255, n_perm=200, min_runs=100, n_fix=80):
    return hm.symbol_table(d["code"], d["next_code"], d["censored"],
                           d["duration_bin"], d["quartile"], d["duration"],
                           n_states=k, n_dur_bins=12, n_perm=n_perm,
                           n_fix=n_fix, min_runs=min_runs, seed=0)


class TestTotalVariation:

    def test_identical_distributions_are_zero(self):
        p = np.array([0.2, 0.3, 0.5])
        assert hm.total_variation(p, p) == 0.0

    def test_disjoint_support_is_one(self):
        assert hm.total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_it_is_symmetric(self):
        a, b = [0.1, 0.9], [0.7, 0.3]
        assert hm.total_variation(a, b) == pytest.approx(hm.total_variation(b, a))


class TestQuartilesWithin:

    def test_quartiles_are_cut_inside_each_group(self):
        """BLOCKING. An animal whose frames sit far from every centroid would
        otherwise fill the far quartile of every symbol, and the test would be
        measuring that animal against the others."""
        v = np.concatenate([np.arange(8.0), 100 + np.arange(8.0)])
        g = np.concatenate([np.zeros(8, int), np.ones(8, int)])
        q = hm.quartiles_within(v, g)
        assert sorted(q[:8]) == [0, 0, 1, 1, 2, 2, 3, 3]
        assert sorted(q[8:]) == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_a_group_of_three_gets_no_quartile(self):
        q = hm.quartiles_within(np.arange(3.0), np.zeros(3, int))
        assert (q == -1).all()

    def test_non_finite_values_are_excluded(self):
        v = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        q = hm.quartiles_within(v, np.zeros(9, int))
        assert q[2] == -1 and (q[[0, 1, 3, 4, 5, 6, 7, 8]] >= 0).all()


class TestTheBiasAndItsCancellation:
    """The reason the effect is an excess and never a raw TV."""

    def test_raw_tv_is_large_under_a_true_null_at_large_K(self):
        """BLOCKING. Two halves of ONE distribution, and the plug-in TV is 0.3+
        purely from finite samples over a wide support."""
        rng = np.random.default_rng(0)
        row = _table(_symbol(rng, n=1600, k=255), k=255)[0]
        assert row["tv_next"] > 0.25, row["tv_next"]

    def test_the_excess_is_about_zero_under_a_true_null(self):
        rng = np.random.default_rng(0)
        row = _table(_symbol(rng, n=1600, k=255), k=255)[0]
        assert abs(row["excess_next"]) < 0.05, row

    def test_the_raw_bias_depends_on_K_but_the_excess_does_not(self):
        """BLOCKING, and the whole design rests on it. Same true null, two
        support sizes: the raw TV differs by an order of magnitude, the excess
        does not differ at all."""
        rng = np.random.default_rng(1)
        wide = _table(_symbol(rng, n=1600, k=255), k=255)[0]
        narrow = _table(_symbol(rng, n=1600, k=7), k=7)[0]
        assert wide["tv_next"] > 4 * narrow["tv_next"], (wide["tv_next"],
                                                         narrow["tv_next"])
        assert abs(wide["excess_next"]) < 0.05
        assert abs(narrow["excess_next"]) < 0.05

    def test_a_real_difference_shows_as_a_positive_excess(self):
        rng = np.random.default_rng(2)
        row = _table(_symbol(rng, n=1600, k=255, shift=0.5), k=255)[0]
        assert row["excess_next"] > 0.05, row
        assert row["p_next"] <= 0.01

    def test_the_null_is_true_means_p_is_not_small(self):
        rng = np.random.default_rng(3)
        row = _table(_symbol(rng, n=1600, k=255), k=255)[0]
        assert row["p_next"] > 0.01, row


class TestSymbolTable:

    def test_a_symbol_below_the_minimum_is_skipped(self):
        rng = np.random.default_rng(0)
        d = _symbol(rng, n=400, k=255)
        assert _table(d, k=255, min_runs=10_000) == []

    def test_only_the_extreme_quartiles_are_compared(self):
        rng = np.random.default_rng(0)
        d = _symbol(rng, n=1600, k=255)
        row = _table(d, k=255)[0]
        assert row["n_runs"] == int(((d["quartile"] == 0)
                                     | (d["quartile"] == 3)).sum())
        assert row["n_near"] + row["n_far"] == row["n_runs"]

    def test_duration_is_compared_as_well_as_next_symbol(self):
        rng = np.random.default_rng(0)
        row = _table(_symbol(rng, n=1600, k=255), k=255)[0]
        for key in ("tv_duration", "excess_duration", "p_duration"):
            assert key in row and np.isfinite(row[key])

    def test_censored_runs_are_out_of_the_next_symbol_comparison(self):
        """A censored run has no next symbol; counting one would invent it."""
        rng = np.random.default_rng(0)
        d = _symbol(rng, n=1600, k=255)
        d["censored"][:] = True
        row = _table(d, k=255)[0]
        assert not np.isfinite(row["tv_next"])
        assert np.isfinite(row["tv_duration"])


class TestAggregateAndRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 255}

    def _rows(self, excess, frames, p=0.5):
        return [{"symbol": i, "frames": f, "n_runs": 500,
                 "excess_next": e, "tv_next": 0.3 + e, "null_next": 0.3,
                 "p_next": p}
                for i, (e, f) in enumerate(zip(excess, frames))]

    def test_the_aggregate_is_frame_mass_weighted(self):
        a = hm.aggregate(self._rows([0.0, 0.4], [9000, 1000]))
        assert a["excess"] == pytest.approx(0.04)
        assert a["excess_unweighted"] == pytest.approx(0.2)

    def test_the_fdr_fraction_is_carried_and_labelled_descriptive(self):
        a = hm.aggregate(self._rows([0.1] * 4, [100] * 4, p=0.0001))
        assert a["fdr_failing_fraction"] == pytest.approx(1.0)
        assert "gates nothing" in a["fdr_note"]

    def test_an_interval_excluding_zero_reads_under_resolved(self):
        a = hm.aggregate(self._rows([0.1, 0.1], [100, 100]))
        r = hm.homogeneity_read(a, {"lo": 0.05, "hi": 0.15, "n_animals": 89},
                                scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "PASS" and "UNDER-RESOLVED" in r.reason

    def test_an_interval_containing_zero_reads_as_sharing_a_future(self):
        a = hm.aggregate(self._rows([0.0, 0.0], [100, 100]))
        r = hm.homogeneity_read(a, {"lo": -0.02, "hi": 0.02, "n_animals": 89},
                                scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "PASS" and "SHARE A FUTURE" in r.reason
        assert "as many as share a future" in r.reason

    def test_the_verdict_never_reads_the_fdr_fraction(self):
        """Every symbol significant, but the effect interval contains zero."""
        a = hm.aggregate(self._rows([0.0] * 4, [100] * 4, p=0.0001))
        r = hm.homogeneity_read(a, {"lo": -0.01, "hi": 0.01, "n_animals": 89},
                                scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "PASS" and "SHARE A FUTURE" in r.reason
        assert "not for the verdict" in r.reason

    def test_a_negative_excess_is_inconclusive_not_a_coherence_result(self):
        a = hm.aggregate(self._rows([-0.1, -0.1], [100, 100]))
        r = hm.homogeneity_read(a, {"lo": -0.15, "hi": -0.05, "n_animals": 89},
                                scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "INCONCLUSIVE" and "mis-specified" in r.reason

    def test_nothing_testable_is_not_a_result(self):
        r = hm.homogeneity_read(hm.aggregate([]),
                                {"lo": float("nan"), "hi": float("nan"),
                                 "n_animals": 0},
                                scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "NOT_A_RESULT" and r.degenerate


class TestJackknife:

    def _setup(self, shift=0.6):
        rng = np.random.default_rng(0)
        d = _symbol(rng, n=1600, k=63, shift=shift, n_animals=12)
        rows = _table(d, k=63, n_perm=200)
        agg = hm.aggregate(rows)
        ci = hm.jackknife(rows, d["code"], d["next_code"], d["censored"],
                          d["quartile"], d["duration"], d["animal"],
                          n_states=63, point=float(agg["excess"]), n_fix=80,
                          seed=0)
        return agg, ci

    def test_it_returns_an_animal_level_interval(self):
        _agg, ci = self._setup()
        assert ci["n_animals"] == 12 and ci["n_jack"] > 0
        assert ci["se"] > 0

    def test_the_interval_brackets_the_full_data_estimate(self):
        """BLOCKING. Four resampling schemes failed exactly this, and an
        interval that excludes its own estimate is not an interval."""
        _agg, ci = self._setup()
        assert ci["lo"] < ci["point"] < ci["hi"], ci

    def test_it_brackets_under_the_null_too(self):
        _agg, ci = self._setup(shift=0.0)
        assert ci["lo"] < ci["point"] < ci["hi"], ci

    def test_no_animal_is_duplicated_so_the_mean_tracks_the_full_data(self):
        _agg, ci = self._setup()
        assert abs(ci["jackknife_minus_full_data"]) < 0.03, ci

    def test_the_method_string_records_what_it_does(self):
        _agg, ci = self._setup()
        assert "leave-one-animal-out" in ci["method"]
        assert "no clustering artifact" in ci["method"]

    def test_too_few_animals_is_refused(self):
        rng = np.random.default_rng(0)
        d = _symbol(rng, n=400, k=63, n_animals=2)
        ci = hm.jackknife([], d["code"], d["next_code"], d["censored"],
                          d["quartile"], d["duration"], d["animal"],
                          n_states=63, point=0.1, n_fix=80)
        assert not np.isfinite(ci["point"])
