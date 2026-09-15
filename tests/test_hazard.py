"""The hazard model, the censoring, and the history that must not span a seam.

Censoring is the failure that matters most here: scoring a run that was cut off
by the end of a recording as though the animal had chosen to stop biases every
duration estimate downward, and nothing downstream looks wrong when it happens.
"""
import numpy as np
import pytest

from vieb.tok import hazard as hz, ladder as ld
from vieb.tok.rle import ABSTAIN


def _runs(code, dur, rec):
    return {"code": np.asarray(code), "duration": np.asarray(dur),
            "recording": np.asarray(rec)}


class TestDurationGrid:

    def test_it_is_log_spaced_and_monotone(self):
        e = hz.duration_grid(np.arange(1, 5000))
        assert (np.diff(e) > 0).all()
        assert e[0] == 0 and e[-1] == np.iinfo(np.int64).max

    def test_the_tail_quantile_keeps_one_long_run_from_owning_the_grid(self):
        """A single 40-second run must not push eleven bins below one frame."""
        d = np.concatenate([np.ones(10000, int), [1200]])
        e = hz.duration_grid(d)
        assert e[-2] < 100

    def test_no_durations_is_refused(self):
        with pytest.raises(ValueError):
            hz.duration_grid(np.zeros(0))

    def test_bin_of_is_half_open(self):
        e = np.array([0, 2, 5, np.iinfo(np.int64).max])
        assert hz.bin_of([0, 1, 2, 4, 5, 999], e).tolist() == [0, 0, 1, 1, 2, 2]


class TestCensoring:

    def _fit(self, code, dur, rec, abstain="conditioned", n_states=3):
        d = ld.prepare(_runs(code, dur, rec), n_states, abstain=abstain)
        e = hz.duration_grid(d["duration"])
        ids, vocab, n = ld.context_ids(d["code"], d["duration"], d["recording"],
                                       k=0, alphabet=d["alphabet"], edges=e)
        return d, e, hz.fit(ids, d["duration"], d["next_state"], d["censored"],
                            edges=e, n_context=n, n_states=d["alphabet"])

    def test_the_last_run_of_every_recording_is_censored(self):
        """BLOCKING. Two recordings, so the fixture has two terminal runs."""
        d, _e, _m = self._fit([0, 1, 0, 1], [5, 5, 5, 5], [0, 0, 1, 1])
        assert d["censored"].tolist() == [False, True, False, True]

    def test_a_censored_run_contributes_no_exit(self):
        _d, _e, m = self._fit([0, 1], [5, 5], [0, 0])
        assert m["n_censored"] == 1
        assert float(m["exits_per_cell"].sum()) == 1.0

    def test_all_censored_means_no_exits_at_all(self):
        _d, _e, m = self._fit([0, 1], [5, 5], [0, 1])
        assert float(m["exits_per_cell"].sum()) == 0.0
        assert m["exit_keys"].size == 0

    def test_a_run_ended_by_abstain_is_censored_when_conditioned_away(self):
        """BLOCKING, and the other half of the pair below."""
        d, _e, _m = self._fit([0, ABSTAIN, 1], [5, 5, 5], [0, 0, 0])
        assert d["n_runs"] == 2 and d["n_dropped"] == 1
        assert d["censored"].tolist() == [True, True]

    def test_the_same_run_is_a_genuine_exit_when_abstain_is_a_symbol(self):
        d, _e, _m = self._fit([0, ABSTAIN, 1], [5, 5, 5], [0, 0, 0],
                              abstain="symbol")
        assert d["n_runs"] == 3 and d["n_dropped"] == 0
        assert d["censored"].tolist() == [False, False, True]
        assert d["next_state"][0] == 3           # abstain became symbol n_states

    def test_an_unknown_abstain_mode_is_refused(self):
        with pytest.raises(ValueError):
            ld.prepare(_runs([0], [1], [0]), 3, abstain="drop")


class TestTheTableIsAProbability:

    def _model(self, seed=0, n=20000, p=0.2, n_states=4):
        rng = np.random.default_rng(seed)
        dur = rng.geometric(p, size=n)
        code = rng.integers(0, n_states, size=n)
        rec = np.zeros(n, dtype=np.int64)
        d = ld.prepare(_runs(code, dur, rec), n_states, abstain="symbol")
        e = hz.duration_grid(d["duration"])
        ids, _v, nc = ld.context_ids(d["code"], d["duration"], d["recording"],
                                     k=0, alphabet=d["alphabet"], edges=e)
        return d, e, ids, hz.fit(ids, d["duration"], d["next_state"],
                                 d["censored"], edges=e, n_context=nc,
                                 n_states=d["alphabet"])

    def test_stay_plus_exits_sum_to_one_in_every_observed_cell(self):
        _d, _e, _i, m = self._model()
        stay = np.exp(m["log_stay"])
        exits = (m["exits_per_cell"] + m["alpha"] * 0) / np.exp(m["log_denom"])
        live = m["at_risk"] > 0
        total = stay + exits + (m["alpha"] * (m["n_states"] - 1)
                                / np.exp(m["log_denom"]))
        assert np.allclose(total[live], 1.0, atol=1e-9)

    def test_a_constant_hazard_is_recovered(self):
        """Geometric dwell has a flat hazard; the fitted profile must be flat."""
        _d, _e, _i, m = self._model(p=0.2)
        haz = m["exits_per_cell"] / np.maximum(m["at_risk"], 1e-9)
        live = m["at_risk"] > 500
        assert np.allclose(haz[live], 0.2, atol=0.05), haz[live]

    def test_the_likelihood_matches_a_brute_force_expansion(self):
        """BLOCKING. The bin-overlap accumulation is an optimisation, and this
        is the thing it is optimising -- one row per elapsed frame."""
        d, e, ids, m = self._model(n=3000)
        fast = hz.log_likelihood(m, ids, d["duration"], d["next_state"],
                                 d["censored"])
        slow = np.zeros_like(fast)
        for i in range(d["duration"].shape[0]):
            c, dd = int(ids[i]), int(d["duration"][i])
            for tau in range(dd - 1):
                slow[i] += m["log_stay"][c, int(hz.bin_of([tau], e)[0])]
            if not d["censored"][i]:
                b = int(hz.bin_of([dd - 1], e)[0])
                slow[i] += hz._log_exit(m, np.array([c]), np.array([b]),
                                        np.array([int(d["next_state"][i])]))[0]
        assert np.allclose(fast, slow, atol=1e-9)


class TestHistoryAndSeams:

    def test_k0_context_is_just_the_state(self):
        ids, vocab, n = ld.context_ids([0, 1, 2, 1], [3, 3, 3, 3], [0, 0, 0, 0],
                                       k=0, alphabet=3,
                                       edges=np.array([0, 2, 9223372036854775807]))
        assert ids.tolist() == [0, 1, 2, 1]

    def test_a_context_never_spans_a_recording_seam(self):
        """BLOCKING. The first run of a recording has no predecessor, and must
        not borrow the last run of the previous one.

        Compared on the RAW key rather than the dense id: two separately built
        vocabularies number their contexts independently, so equal ids would be
        a coincidence and unequal ids would prove nothing.
        """
        e = np.array([0, 2, 9223372036854775807])
        nb, cov = int(e.shape[0] - 1), ld.covariate_block(2)

        def key(code, dur):
            c = np.asarray(code, dtype=np.int64)
            d = np.asarray(dur, dtype=np.int64)
            return ld._raw_keys(c, hz.bin_of(np.maximum(d - 1, 0), e),
                                np.array([0, 1]), 1, 3, nb, cov)

        a, b = key([0, 1], [3, 3]), key([2, 1], [9, 3])
        assert a[1] == b[1], "the first run of recording 1 borrowed a predecessor"

    def test_the_predecessor_is_used_when_there_is_one(self):
        """The counterfactual, so the seam test above is known to be testing
        something that would otherwise differ."""
        e = np.array([0, 2, 9223372036854775807])
        nb, cov = int(e.shape[0] - 1), ld.covariate_block(2)

        def key(code):
            c = np.asarray(code, dtype=np.int64)
            d = np.array([3, 3], dtype=np.int64)
            return ld._raw_keys(c, hz.bin_of(d - 1, e), np.array([0, 0]), 1, 3,
                                nb, cov)

        assert key([0, 1])[1] != key([2, 1])[1]

    def test_an_unseen_context_falls_into_one_reserved_bucket(self):
        e = np.array([0, 2, 9223372036854775807])
        _i, vocab, n = ld.context_ids([0, 1], [3, 3], [0, 0], k=1, alphabet=3,
                                      edges=e)
        ids, _v, _n = ld.context_ids([2, 2], [3, 3], [0, 0], k=1, alphabet=3,
                                     edges=e, vocab=vocab)
        assert (ids == vocab.shape[0]).all()
        assert n == vocab.shape[0] + 1

    def test_deeper_history_makes_more_contexts(self):
        rng = np.random.default_rng(0)
        code = rng.integers(0, 4, 5000)
        dur = rng.integers(1, 9, 5000)
        rec = np.zeros(5000, dtype=np.int64)
        e = hz.duration_grid(dur)
        sizes = [ld.context_ids(code, dur, rec, k=k, alphabet=4, edges=e)[2]
                 for k in ld.K_GRID]
        assert sizes[0] < sizes[1] < sizes[2]


class TestCovariateSlot:

    def test_it_is_zero_width_by_default(self):
        """BLOCKING, per the pre-registration: the empty case is asserted so
        that adding the slot later is not a change to every model's shape."""
        b = ld.covariate_block(17)
        assert b.shape == (17, 0)

    def test_a_populated_block_is_accepted_and_changes_the_context(self):
        e = np.array([0, 2, 9223372036854775807])
        cov = np.array([[0], [1]], dtype=np.float64)
        plain, _v, _n = ld.context_ids([1, 1], [3, 3], [0, 0], k=0, alphabet=3,
                                       edges=e)
        with_cov, _v, _n = ld.context_ids([1, 1], [3, 3], [0, 0], k=0,
                                          alphabet=3, edges=e, covariates=cov)
        assert plain[0] == plain[1]
        assert with_cov[0] != with_cov[1]

    def test_a_wrong_length_block_is_refused(self):
        with pytest.raises(ValueError):
            ld.covariate_block(5, np.zeros((3, 2)))


class TestShapeByState:

    def test_it_reports_a_hazard_per_bin_for_the_busiest_states(self):
        rng = np.random.default_rng(1)
        n = 20000
        dur = rng.geometric(0.25, size=n)
        code = rng.integers(0, 3, size=n)
        d = ld.prepare(_runs(code, dur, np.zeros(n, int)), 3, abstain="symbol")
        e = hz.duration_grid(d["duration"])
        ids, _v, nc = ld.context_ids(d["code"], d["duration"], d["recording"],
                                     k=0, alphabet=d["alphabet"], edges=e)
        m = hz.fit(ids, d["duration"], d["next_state"], d["censored"], edges=e,
                   n_context=nc, n_states=d["alphabet"])
        rows = hz.shape_by_state(m, np.arange(nc), top=3)
        assert len(rows) == 3
        assert len(rows[0]["hazard_by_bin"]) == m["n_bins"]
        seen = [v for v in rows[0]["hazard_by_bin"] if v is not None]
        assert seen and all(0.0 <= v <= 1.0 for v in seen)
