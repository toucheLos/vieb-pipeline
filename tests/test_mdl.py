"""Description length, and the one thing that makes it a fair comparison.

Duration must be charged. Without it a coarse vocabulary wins by hiding
information in dwell time -- a two-symbol alphabet gets an excellent NLL per
transition and has described nothing. Every rung here charges both, and the
discretization is fixed across arms so that two rungs with different duration
models are still comparable.
"""
import numpy as np
import pytest

from vieb.tok import hazard as hz, ladder as ld, mdl


def _data(n=4000, n_states=5, seed=0, n_rec=8):
    """A run stream that obeys the run-length invariant.

    Adjacent runs within a recording must differ, because that is what RLE
    guarantees and what the zero diagonal assumes. A fixture drawing codes
    independently would put self-repeats in the stream and test the model
    against data it can never see.
    """
    rng = np.random.default_rng(seed)
    rec = np.sort(rng.integers(0, n_rec, size=n))
    code = np.empty(n, dtype=np.int64)
    for i in range(n):
        choices = [c for c in range(n_states)
                   if i == 0 or rec[i] != rec[i - 1] or c != code[i - 1]]
        code[i] = choices[rng.integers(0, len(choices))]
    dur = rng.geometric(0.25, size=n)
    return ld.prepare({"code": code, "duration": dur, "recording": rec},
                      n_states, abstain="symbol")


class TestDurationIsCharged:

    def test_every_rung_charges_duration(self):
        """BLOCKING. A rung that charged only the symbol would win by hiding
        information in dwell time."""
        d = _data()
        e = hz.duration_grid(d["duration"])
        for rung in (0, 1):
            m = ld.fit_rung(rung, d, edges=e)
            long = dict(d, duration=d["duration"] * 3)
            assert (ld.score_rung(m, long, edges=e).sum()
                    != ld.score_rung(m, d, edges=e).sum())

    def test_a_censored_run_pays_no_symbol_cost(self):
        d = _data()
        e = hz.duration_grid(d["duration"])
        m = ld.fit_rung(0, d, edges=e)
        nats = ld.score_rung(m, d, edges=e)
        only_dur = ld._duration_nats(m["pmf"], d["duration"])
        assert np.allclose(nats[d["censored"]], only_dur[d["censored"]])
        assert (nats[~d["censored"]] > only_dur[~d["censored"]]).all()

    def test_the_discretization_is_one_frame_and_identical_across_rungs(self):
        """Registered: `-log f(d) + log delta` at one-frame resolution,
        identically in every arm. Held as a pmf over frames, the two terms
        collapse to `-log p_bin + log width`, so this IS that charge."""
        d = _data()
        e = hz.duration_grid(d["duration"])
        a = ld.fit_rung(0, d, edges=e)["pmf"]
        b = ld.fit_rung(1, d, edges=e)["pmf"]
        assert np.allclose(a["log_p_bin"], b["log_p_bin"])
        assert np.allclose(a["log_width"], b["log_width"])
        assert np.array_equal(a["edges"], b["edges"])

    def test_the_duration_pmf_normalises_over_frames(self):
        """`f(d) = p_bin / width` must be a pmf over whole frames, or the
        duration charge is not a code length.

        Summed over 1..d_max, because that is the range the model allocates mass
        to: the last bin is open at infinity and coding is truncated at the
        longest duration seen.
        """
        rng = np.random.default_rng(0)
        dur = rng.geometric(0.2, size=20000)
        e = hz.duration_grid(dur)
        pmf = ld.duration_pmf(dur, e)
        d = np.arange(1, int(dur.max()) + 1)
        b = hz.bin_of(d - 1, e)
        total = float(np.exp(pmf["log_p_bin"][b] - pmf["log_width"][b]).sum())
        assert total == pytest.approx(1.0, abs=1e-9), total

    def test_rungs_zero_and_one_share_a_duration_model_by_design(self):
        """So that rung1 - rung0 isolates the transition table."""
        d = _data()
        e = hz.duration_grid(d["duration"])
        a = ld.score_rung(ld.fit_rung(0, d, edges=e), d, edges=e)
        b = ld.score_rung(ld.fit_rung(1, d, edges=e), d, edges=e)
        cen = d["censored"]
        assert np.allclose(a[cen], b[cen])


class TestZeroDiagonal:

    def test_rung_one_cannot_emit_a_self_repeat(self):
        """RLE makes AA impossible; a model able to emit it inflates every real
        2-gram."""
        d = _data()
        m = ld.fit_rung(1, d, edges=hz.duration_grid(d["duration"]))
        assert np.isneginf(np.diag(m["log_p"])).all()

    def test_each_row_is_a_simplex_over_the_other_states(self):
        d = _data()
        m = ld.fit_rung(1, d, edges=hz.duration_grid(d["duration"]))
        p = np.exp(m["log_p"])
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)


class TestCodeLength:

    def test_the_terms_are_reported_apart(self):
        """A rung can lose entirely on the codebook while describing the data
        better, and a total that hides which term dominates cannot be argued
        with."""
        c = mdl.code_length(np.full(100, 2.0), np.full(100, 30), 30.0,
                            n_params=10, n_runs_fit=1000)
        assert c["data_nats"] == pytest.approx(200.0)
        assert c["codebook_nats"] == pytest.approx(10 * mdl.bic_c(1000))
        assert c["total_nats"] == pytest.approx(c["data_nats"] + c["codebook_nats"])

    def test_nats_per_second_uses_scored_time(self):
        c = mdl.code_length(np.full(10, 1.0), np.full(10, 30), 30.0,
                            n_params=0, n_runs_fit=100)
        assert c["t_seconds"] == pytest.approx(10.0)
        assert c["nats_per_second"] == pytest.approx(1.0)

    def test_the_parameter_charge_is_half_log_n(self):
        assert mdl.bic_c(1000) == pytest.approx(0.5 * np.log(1000))


class TestPerAnimal:

    def _pa(self, vals, animals, n_params=4, n_runs_fit=100):
        return mdl.per_animal(vals, np.full(len(vals), 30), animals, 30.0,
                              n_params=n_params, n_runs_fit=n_runs_fit)

    def test_the_animal_mean_reproduces_the_pooled_figure(self):
        """BLOCKING. The codebook is shared out by time so the offset is
        identical for every animal; charging each the whole book would weight it
        by 1/T_a and punish short animals for a cost they did not incur."""
        vals = np.array([1.0, 3.0, 2.0, 4.0])
        pa = self._pa(vals, ["a", "a", "b", "b"])
        pooled = mdl.code_length(vals, np.full(4, 30), 30.0, n_params=4,
                                 n_runs_fit=100)
        assert float(np.mean(pa["nats_per_second"])) == pytest.approx(
            pooled["nats_per_second"])

    def test_every_animal_carries_the_same_codebook_offset(self):
        a = self._pa(np.ones(4), ["a", "a", "b", "b"], n_params=0)
        b = self._pa(np.ones(4), ["a", "a", "b", "b"], n_params=40)
        d = b["nats_per_second"] - a["nats_per_second"]
        assert np.allclose(d, d[0])

    def test_it_reports_one_row_per_animal(self):
        pa = self._pa(np.ones(6), list("aabbcc"))
        assert pa["n_animals"] == 3 and len(pa["animals"]) == 3


class TestImprovementAndRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 256}

    def _pa(self, vals):
        return mdl.per_animal(vals, np.full(len(vals), 30),
                              [f"a{i}" for i in range(len(vals))], 30.0,
                              n_params=0, n_runs_fit=100)

    def test_misaligned_animals_are_refused_not_realigned(self):
        a = mdl.per_animal([1.0], [30], ["x"], 30.0, n_params=0, n_runs_fit=10)
        b = mdl.per_animal([1.0], [30], ["y"], 30.0, n_params=0, n_runs_fit=10)
        with pytest.raises(ValueError):
            mdl.improvement(a, b)

    def test_a_clear_improvement_passes(self):
        lo, hi = self._pa(np.full(40, 10.0)), self._pa(np.full(40, 8.0))
        r = mdl.ladder_read(mdl.improvement(lo, hi), name="rung 2 over rung 1",
                            scored_object=self.OBJ)
        assert r.verdict == "PASS" and r.n_effective == 40

    def test_a_worse_rung_fails_and_says_which_way(self):
        lo, hi = self._pa(np.full(40, 8.0)), self._pa(np.full(40, 10.0))
        r = mdl.ladder_read(mdl.improvement(lo, hi), name="rung 2 over rung 1",
                            scored_object=self.OBJ)
        assert r.verdict == "FAIL" and "WORSE" in r.reason

    def test_an_interval_containing_zero_fails(self):
        rng = np.random.default_rng(0)
        lo = self._pa(rng.normal(10, 3, 40))
        hi = self._pa(rng.normal(10, 3, 40))
        r = mdl.ladder_read(mdl.improvement(lo, hi), name="rung 2 over rung 1",
                            scored_object=self.OBJ)
        assert r.verdict == "FAIL" and "contains zero" in r.reason

    def test_the_frame_interval_is_carried_but_never_gates(self):
        """It is computed on purpose so the document can show how much narrower
        the wrong method looks."""
        lo, hi = self._pa(np.full(40, 10.0)), self._pa(np.full(40, 9.0))
        r = mdl.ladder_read(mdl.improvement(lo, hi), name="x",
                            scored_object=self.OBJ)
        assert "frame_interval" in r.detail and "animal_interval" in r.detail
        assert "licenses nothing" in r.detail["frame_interval_note"]

    def test_too_few_animals_is_not_a_result(self):
        lo, hi = self._pa(np.array([1.0])), self._pa(np.array([2.0]))
        r = mdl.ladder_read(mdl.improvement(lo, hi), name="x",
                            scored_object=self.OBJ)
        assert r.verdict == "NOT_A_RESULT" and r.degenerate


class TestRareTransitions:

    def test_common_edges_are_chosen_on_fit_not_on_the_scored_data(self):
        counts = np.zeros((3, 3))
        counts[0, 1] = 100.0
        nats = np.array([1.0, 9.0, 9.0])
        code = np.array([0, 2, 2])
        nxt = np.array([1, 0, 0])
        cen = np.zeros(3, dtype=bool)
        out = mdl.rare_transition_recall(nats, code, nxt, cen, counts, common=1)
        assert out["n_common"] == 1 and out["n_rare"] == 2
        assert out["common_nats_per_run"] == pytest.approx(1.0)
        assert out["rare_nats_per_run"] == pytest.approx(9.0)

    def test_all_censored_gives_no_split(self):
        out = mdl.rare_transition_recall(np.ones(2), np.zeros(2), np.zeros(2),
                                         np.ones(2, bool), np.zeros((2, 2)))
        assert out["n_common"] == 0 and out["n_rare"] == 0


class TestSelfRepeatsAreRefused:
    """An inf code length is worse than an error: it propagates silently."""

    def test_a_scored_self_repeat_raises_with_the_diagnosis(self):
        d = _data(n=400)
        e = hz.duration_grid(d["duration"])
        m = ld.fit_rung(1, d, edges=e)
        broken = dict(d, next_state=d["code"].copy(),
                      censored=np.zeros_like(d["censored"]))
        with pytest.raises(ValueError, match="self-repeats"):
            ld.score_rung(m, broken, edges=e)

    def test_a_clean_stream_scores_finite(self):
        d = _data(n=400)
        e = hz.duration_grid(d["duration"])
        for rung in (0, 1, 2):
            nats = ld.score_rung(ld.fit_rung(rung, d, edges=e), d, edges=e)
            assert np.isfinite(nats).all(), rung
