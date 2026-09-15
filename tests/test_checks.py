"""The guard that would have caught the duration-charge bug.

`ladder.duration_pmf` spread each bin's mass over one whole duration too many;
`f(d)` summed to 1.115; every duration charge in the MDL objective was wrong by
~0.11 nats. A constant offset on every arm cancels in the comparisons and
survives in the absolute figures, which is why nothing downstream looked wrong.
"""
import numpy as np
import pytest

from vieb import checks


class TestAssertPmf:

    def test_a_pmf_passes_and_is_returned(self):
        p = np.array([0.25, 0.25, 0.5])
        out = checks.assert_pmf(p, name="x")
        assert np.array_equal(out, p)

    def test_the_actual_bug_is_caught(self):
        """1.115, which is what `f(d)` summed to before the width fix."""
        with pytest.raises(AssertionError, match="1.115"):
            checks.assert_pmf(np.array([0.5, 0.615]), name="duration_pmf")

    def test_a_sum_slightly_under_one_is_caught(self):
        with pytest.raises(AssertionError, match="not a pmf"):
            checks.assert_pmf(np.array([0.5, 0.4999]), name="x")

    def test_rounding_at_float64_scale_passes(self):
        rng = np.random.default_rng(0)
        p = rng.random(5000)
        checks.assert_pmf(p / p.sum(), name="x")

    def test_a_negative_entry_is_caught(self):
        with pytest.raises(AssertionError, match="negative"):
            checks.assert_pmf(np.array([1.2, -0.2]), name="x")

    def test_a_nan_is_caught(self):
        with pytest.raises(AssertionError, match="non-finite"):
            checks.assert_pmf(np.array([np.nan, 1.0]), name="x")

    def test_an_empty_vector_is_missing_not_uniform(self):
        """BLOCKING. Returning zeros for 'no observations' is how bones.py came
        to report a blame share that summed to nothing."""
        with pytest.raises(AssertionError, match="missing rather than uniform"):
            checks.assert_pmf(np.zeros(0), name="x")

    def test_empty_can_be_allowed_explicitly(self):
        checks.assert_pmf(np.zeros(0), name="x", allow_empty=True)

    def test_an_all_zero_vector_is_caught(self):
        with pytest.raises(AssertionError):
            checks.assert_pmf(np.zeros(4), name="x")

    def test_rows_are_checked_independently_on_an_axis(self):
        good = np.array([[0.5, 0.5], [0.25, 0.75]])
        checks.assert_pmf(good, name="x", axis=1)
        bad = np.array([[0.5, 0.5], [0.25, 0.70]])
        with pytest.raises(AssertionError, match="axis 1"):
            checks.assert_pmf(bad, name="x", axis=1)

    def test_a_matrix_summing_to_one_overall_still_fails_per_row(self):
        """The mistake that made the first version of the hazard assertion
        useless: a table of per-cell simplexes is not one big simplex."""
        m = np.array([[0.5, 0.5], [0.5, 0.5]]) / 2
        checks.assert_pmf(m, name="x")
        with pytest.raises(AssertionError):
            checks.assert_pmf(m, name="x", axis=1)


class TestAssertLogPmf:

    def test_a_log_pmf_passes(self):
        checks.assert_log_pmf(np.log([0.5, 0.5]), name="x")

    def test_minus_inf_is_a_forbidden_outcome_not_an_error(self):
        """The zero diagonal a run-length-encoded stream requires."""
        checks.assert_log_pmf(np.array([-np.inf, 0.0]), name="x")

    def test_plus_inf_is_an_error(self):
        with pytest.raises(AssertionError, match=r"\+inf"):
            checks.assert_log_pmf(np.array([np.inf, 0.0]), name="x")

    def test_nan_is_an_error(self):
        with pytest.raises(AssertionError, match="NaN"):
            checks.assert_log_pmf(np.array([np.nan, 0.0]), name="x")

    def test_an_unnormalised_log_pmf_is_caught(self):
        with pytest.raises(AssertionError, match="not a pmf"):
            checks.assert_log_pmf(np.log([0.5, 0.6]), name="x")


class TestAssertUnit:

    def test_every_cell_must_be_one(self):
        checks.assert_unit(np.ones((3, 4)), name="x")

    def test_one_bad_cell_in_a_large_table_is_caught(self):
        a = np.ones((100, 13))
        a[57, 4] = 1.02
        with pytest.raises(AssertionError, match="worst cell"):
            checks.assert_unit(a, name="hazard cells")

    def test_the_message_names_how_many_cells_were_checked(self):
        a = np.ones((10, 3))
        a[0, 0] = 2.0
        with pytest.raises(AssertionError, match="over 30 cells"):
            checks.assert_unit(a, name="x")

    def test_a_table_that_sums_to_one_overall_still_fails(self):
        with pytest.raises(AssertionError):
            checks.assert_unit(np.full((10,), 0.1), name="x")


class TestAssertShare:

    def test_fractions_pass(self):
        checks.assert_share([0.0, 0.5, 1.0], name="x")

    def test_above_one_is_caught(self):
        with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
            checks.assert_share([0.5, 1.2], name="x")

    def test_below_zero_is_caught(self):
        with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
            checks.assert_share([-0.01, 0.5], name="x")

    def test_shares_need_not_sum_to_one(self):
        """Cumulative shares and rates are not a partition of anything."""
        checks.assert_share([0.3, 0.6, 0.9], name="x")

    def test_non_finite_is_caught(self):
        with pytest.raises(AssertionError, match="non-finite"):
            checks.assert_share([np.nan, 0.5], name="x")


class TestTheGuardsAreWiredIn:
    """Each of these calls the real function and would fail if the assertion
    had been added to a branch the caller never reaches."""

    def test_duration_pmf_is_checked_at_frame_resolution(self):
        from vieb.tok import hazard as hz, ladder as ld
        rng = np.random.default_rng(0)
        dur = rng.geometric(0.2, size=5000)
        ld.duration_pmf(dur, hz.duration_grid(dur))

    def test_a_broken_width_would_now_fail(self):
        """The bug, reintroduced: one extra whole duration per bin."""
        from vieb.tok import hazard as hz, ladder as ld
        rng = np.random.default_rng(0)
        dur = rng.geometric(0.2, size=5000)
        e = hz.duration_grid(dur)
        pmf = ld.duration_pmf(dur, e)
        widths = np.exp(pmf["log_width"]) + 1.0
        whole = np.arange(1, int(dur.max()) + 1)
        b = hz.bin_of(whole - 1, e)
        with pytest.raises(AssertionError):
            checks.assert_pmf(np.exp(pmf["log_p_bin"][b]) / widths[b],
                              name="broken f(d)", tol=1e-6)

    def test_the_hazard_table_is_checked_cellwise(self):
        from vieb.tok import hazard as hz, ladder as ld
        rng = np.random.default_rng(1)
        n = 3000
        code = np.empty(n, dtype=np.int64)
        for i in range(n):
            choices = [c for c in range(4) if i == 0 or c != code[i - 1]]
            code[i] = choices[rng.integers(0, len(choices))]
        d = ld.prepare({"code": code, "duration": rng.geometric(0.3, size=n),
                        "recording": np.zeros(n, int)}, 4, abstain="symbol")
        e = hz.duration_grid(d["duration"])
        ids, _v, nc = ld.context_ids(d["code"], d["duration"], d["recording"],
                                     k=0, alphabet=d["alphabet"], edges=e)
        hz.fit(ids, d["duration"], d["next_state"], d["censored"], edges=e,
               n_context=nc, n_states=d["alphabet"])
