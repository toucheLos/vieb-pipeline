"""The surrogates must be structureless, smooth, and shaped like the corpus.

A null that differs from the data in two ways at once tests neither. The
egocentric representation has three rank-deficient directions by construction,
so a surrogate carrying variance in one of them would be trivially separable
from the corpus and would hand the quantizer structure the corpus does not have.
"""
import numpy as np
import pytest

from vieb.tok import ego, surrogate as sr


def _signal(n=1200, c=ego.N_DIMS, seed=0):
    """A smooth, correlated signal with two identically-zero columns and one
    exact linear dependence -- the shape the ego array actually has."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    base = np.stack([np.sin(2 * np.pi * t / (1.5 + 0.3 * i))
                     + 0.1 * rng.normal(size=n) for i in range(c)], axis=1)
    base[:, ego.dead_columns()["origin_x"]] = 0.0
    base[:, ego.dead_columns()["origin_y"]] = 0.0
    base[:, 5] = base[:, 1] - base[:, 3]          # an exact linear dependence
    return base


class TestPhasePreservesWhatItMustPreserve:

    def test_the_power_spectrum_of_every_channel_is_unchanged(self):
        """The whole reason `phase` is the primary arm: same smoothness, same
        autocorrelation, hence the same run-length structure under quantizing."""
        x = _signal()
        out = sr.generate(x, [0, x.shape[0]], "phase",
                          np.random.default_rng(0))["x"]
        a = np.abs(np.fft.rfft(x, axis=0))
        b = np.abs(np.fft.rfft(np.asarray(out, np.float64), axis=0))
        assert np.allclose(a, b, atol=1e-6)

    def test_it_is_not_the_same_signal(self):
        x = _signal()
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                     np.random.default_rng(0))["x"], np.float64)
        assert not np.allclose(out, x, atol=1e-3)

    def test_an_exact_linear_dependence_between_columns_survives(self):
        """BLOCKING. Shared phase multiplies every column by the same factor, so
        the SE(2) rank deficit is preserved for free. A surrogate that broke it
        would carry variance in a direction the corpus never moves in."""
        x = _signal()
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                     np.random.default_rng(0))["x"], np.float64)
        assert np.allclose(out[:, 5], out[:, 1] - out[:, 3], atol=1e-4)

    def test_the_mean_of_each_channel_is_kept(self):
        x = _signal() + 3.0
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                     np.random.default_rng(0))["x"], np.float64)
        assert np.allclose(out.mean(0), x.mean(0), atol=1e-3)


class TestDeadColumns:

    @pytest.mark.parametrize("kind", ("phase", "var5"))
    def test_a_zero_column_stays_zero(self, kind):
        """BLOCKING for both arms. A VAR's residual noise is full-rank in the
        subspace it was fitted on, so the dead columns are copied, not
        simulated -- and this is what checks that they were."""
        x = _signal()
        out = sr.generate(x, [0, x.shape[0]], kind, np.random.default_rng(0))
        got = np.asarray(out["x"], np.float64)
        for idx in ego.dead_columns().values():
            assert np.allclose(got[:, idx], 0.0), (kind, idx)
        assert out["n_dead_columns"] == 2

    def test_live_columns_finds_the_dead_ones(self):
        x = _signal()
        live = sr.live_columns(x)
        assert not live[ego.dead_columns()["origin_x"]]
        assert not live[ego.dead_columns()["origin_y"]]
        assert int(live.sum()) == ego.N_DIMS - 2


class TestSeams:

    @pytest.mark.parametrize("kind", ("phase", "var5"))
    def test_a_surrogate_never_crosses_a_recording_boundary(self, kind):
        """BLOCKING. A surrogate built across a seam would splice two recordings
        into one spectrum, then be compared against data where crossing a seam
        is forbidden everywhere else in this repo."""
        x = _signal(n=800)
        bounds = [0, 400, 800]
        a = sr.generate(x, bounds, kind, np.random.default_rng(0))["x"]
        y = x.copy()
        y[400:] = _signal(n=400, seed=7)          # change ONLY the second half
        b = sr.generate(y, bounds, kind, np.random.default_rng(0))["x"]
        assert np.allclose(np.asarray(a)[:400], np.asarray(b)[:400], atol=1e-6)

    def test_a_recording_too_short_to_randomise_is_copied(self):
        x = _signal(n=6)
        out = sr.generate(x, [0, 3, 6], "phase", np.random.default_rng(0))["x"]
        assert np.allclose(np.asarray(out, np.float64), x)

    def test_every_recording_is_covered(self):
        x = _signal(n=900)
        out = sr.generate(x, [0, 300, 600, 900], "phase",
                          np.random.default_rng(0))
        assert out["n_recordings"] == 3
        assert np.asarray(out["x"]).shape == x.shape


class TestVar:

    def test_it_produces_a_different_signal_of_the_same_shape(self):
        x = _signal()
        out = np.asarray(sr.generate(x, [0, x.shape[0]], "var5",
                                     np.random.default_rng(0))["x"], np.float64)
        assert out.shape == x.shape
        assert not np.allclose(out, x, atol=1e-3)

    def test_a_degenerate_fit_falls_back_and_is_counted(self):
        """Recorded per recording rather than hidden: a fallback means that
        recording's null is a phase surrogate, not a VAR."""
        x = _signal(n=40)
        out = sr.generate(x, [0, 40], "var5", np.random.default_rng(0))
        assert out["n_fallback"] == 1

    def test_a_healthy_fit_does_not_fall_back(self):
        out = sr.generate(_signal(n=2000), [0, 2000], "var5",
                          np.random.default_rng(0))
        assert out["n_fallback"] == 0


class TestGenerate:

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            sr.generate(_signal(), [0, 1200], "brownian",
                        np.random.default_rng(0))

    @pytest.mark.parametrize("kind", ("phase", "var5"))
    def test_the_output_is_float32_like_the_shards(self, kind):
        out = sr.generate(_signal(), [0, 1200], kind, np.random.default_rng(0))
        assert np.asarray(out["x"]).dtype == np.float32


class TestFalsifierRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 256}

    def _read(self, lo, hi, point, corpus=11.3, surr=4.0):
        return sr.falsifier_read(
            {"animal_interval": {"lo": lo, "hi": hi, "point": point},
             "frame_interval": {"lo": lo, "hi": hi}},
            {"mean_delta": corpus}, {"mean_delta": surr},
            kind="phase", scored_object=self.OBJ, n_effective=89)

    def test_a_gap_excluding_zero_passes(self):
        r = self._read(3.0, 9.0, 6.0)
        assert r.verdict == "PASS" and "not explained by quantizing" in r.reason
        assert "power spectrum" in r.reason

    def test_the_pass_refuses_the_overclaim(self):
        """Two linear-Gaussian surrogates are two alternatives, not all."""
        r = self._read(3.0, 9.0, 6.0)
        assert "does NOT license" in r.reason
        assert "one alternative is not" in r.reason

    def test_a_gap_containing_zero_fails_and_closes_task_3(self):
        r = self._read(-2.0, 4.0, 1.0)
        assert r.verdict == "FAIL"
        assert "coarse sweep does not run" in r.reason

    def test_a_surrogate_that_beats_the_corpus_withdraws_the_ladder(self):
        r = self._read(-9.0, -3.0, -6.0, corpus=4.0, surr=11.3)
        assert r.verdict == "FAIL" and "withdrawn" in r.reason

    def test_a_non_finite_gap_is_not_a_result(self):
        r = self._read(np.nan, np.nan, np.nan)
        assert r.verdict == "NOT_A_RESULT" and r.degenerate

    def test_the_frame_interval_is_carried_but_never_gates(self):
        r = self._read(3.0, 9.0, 6.0)
        assert "frame_interval" in r.detail
        assert "does not read it" in r.detail["frame_interval_note"]


def _fit(x, n_states=12, seed=0):
    """A small microstate partition and chain over one block."""
    from recur.null import microstate as ms
    sd = np.where(x.std(0) > 0, x.std(0), 1.0)
    cols = np.arange(x.shape[1], dtype=np.int64)
    part = ms.fit_partition(x, sd, cols, np.ones(x.shape[0], bool),
                            n_states=n_states, seed=seed)
    st = np.asarray(ms.assign_states(x, part["centroids"], sd, cols), np.int64)
    v = np.asarray(ms.visits(st), np.int64)
    chain = ms.pooled_chain([v[:, 0]], n_states, order=1)
    return {"centroids": part["centroids"], "sd": sd, "cols": cols,
            "chain": chain}, st


class TestSimulateMarginal:

    def test_it_never_repeats_when_it_can_avoid_it(self):
        """BLOCKING, and registered in DWELL_PREREGISTRATION.md SS3. RLE
        guarantees AA cannot occur; an unconditioned marginal draw would hand
        the surrogate a transition the corpus cannot contain."""
        rng = np.random.default_rng(0)
        pool = rng.integers(0, 9, size=4000)
        seq, n_rep = sr.simulate_marginal(pool, 3000, rng)
        assert n_rep == 0
        assert not (seq[1:] == seq[:-1]).any()

    def test_a_single_state_cannot_avoid_repeating_and_says_so(self):
        """A recording that visits one microstate is a fact about the
        recording, not a failure to be hidden."""
        rng = np.random.default_rng(0)
        seq, n_rep = sr.simulate_marginal(np.zeros(50, int), 40, rng)
        assert n_rep == 39 and (seq == 0).all()

    def test_it_follows_the_marginal(self):
        rng = np.random.default_rng(1)
        pool = np.concatenate([np.zeros(900, int), np.ones(100, int),
                               np.full(1000, 2)])
        seq, _ = sr.simulate_marginal(pool, 20000, rng)
        share = np.bincount(seq, minlength=3) / seq.size
        assert share[1] < share[0] and share[1] < share[2]


class TestEmitVisits:

    def test_every_emitted_frame_is_a_real_observed_frame(self):
        """The manifold is preserved by construction -- that is the whole
        reason this null exists rather than a generative one."""
        rng = np.random.default_rng(0)
        x = _signal(n=1500)
        _f, st = _fit(x)
        seq, _ = sr.simulate_marginal(np.asarray(
            __import__("recur.null.microstate", fromlist=["x"]).visits(st))[:, 0],
            600, rng)
        out, diag = sr.emit_visits(x, st, seq, rng)
        rows = {tuple(np.round(r, 9)) for r in x}
        assert all(tuple(np.round(r, 9)) in rows for r in out)
        assert not diag["degenerate"]

    def test_the_dwell_distribution_survives(self):
        """BLOCKING. The null's defining property: re-assigning the emitted
        signal to the same partition must give back the source's visit-length
        distribution.

        Gated on the **mean**, because the mean is what run rate is, and run
        rate is the quantity the falsifier's phase and VAR arms got wrong by a
        factor of 2.1. The visit-length distribution is heavy-tailed -- a third
        of visits are one frame while the mean is eleven -- so the median of a
        few hundred draws is noisy in a way the mean is not, and gating on it
        would be gating on sampling noise.
        """
        from recur.null import microstate as ms
        rng = np.random.default_rng(0)
        x = _signal(n=12000)
        f, st = _fit(x)
        v = np.asarray(ms.visits(st), np.int64)
        seq, _ = sr.simulate_marginal(v[:, 0], v.shape[0], rng)
        out, _d = sr.emit_visits(x, st, seq, rng)
        st2 = np.asarray(ms.assign_states(out, f["centroids"], f["sd"],
                                          f["cols"]), np.int64)
        v2 = np.asarray(ms.visits(st2), np.int64)
        src = float((v[:, 2] - v[:, 1]).mean())
        got = float((v2[:, 2] - v2[:, 1]).mean())
        assert abs(got - src) / src < 0.25, (src, got)

    def test_it_is_far_better_dwell_matched_than_phase(self):
        """The whole point of this arm. Phase randomisation gave 2.1x the run
        rate; stitching real visits must be very much closer than that."""
        from recur.null import microstate as ms
        rng = np.random.default_rng(0)
        x = _signal(n=12000)
        f, st = _fit(x)
        v = np.asarray(ms.visits(st), np.int64)
        seq, _ = sr.simulate_marginal(v[:, 0], v.shape[0], rng)
        out, _d = sr.emit_visits(x, st, seq, rng)
        st2 = np.asarray(ms.assign_states(out, f["centroids"], f["sd"],
                                          f["cols"]), np.int64)
        ratio = (np.asarray(ms.visits(st2)).shape[0]) / v.shape[0]
        assert 0.8 < ratio < 1.25, ratio

        ph = np.asarray(sr.generate(x, [0, x.shape[0]], "phase",
                                    np.random.default_rng(0))["x"], np.float64)
        st_p = np.asarray(ms.assign_states(ph, f["centroids"], f["sd"],
                                           f["cols"]), np.int64)
        phase_ratio = np.asarray(ms.visits(st_p)).shape[0] / v.shape[0]
        assert abs(ratio - 1.0) < abs(phase_ratio - 1.0), (ratio, phase_ratio)

    def test_a_recording_with_too_few_visits_is_copied_and_flagged(self):
        rng = np.random.default_rng(0)
        x = _signal(n=20)
        out, diag = sr.emit_visits(x, np.zeros(20, np.int64),
                                   np.zeros(4, np.int64), rng)
        assert diag["degenerate"] and np.allclose(out, x)

    def test_it_fills_exactly_the_input_length(self):
        rng = np.random.default_rng(0)
        x = _signal(n=1200)
        _f, st = _fit(x)
        out, _d = sr.emit_visits(x, st, np.arange(12, dtype=np.int64), rng)
        assert out.shape == x.shape


class TestTheNewKinds:

    @pytest.mark.parametrize("kind", ("white", "ou"))
    def test_they_produce_a_different_signal_of_the_same_shape(self, kind):
        x = _signal()
        out = sr.generate(x, [0, x.shape[0]], kind, np.random.default_rng(0),
                          fps=30.0)
        got = np.asarray(out["x"], np.float64)
        assert got.shape == x.shape and not np.allclose(got, x, atol=1e-3)

    def test_white_has_no_temporal_structure(self):
        x = _signal(n=4000)
        out = np.asarray(sr.generate(x, [0, 4000], "white",
                                     np.random.default_rng(0))["x"], np.float64)
        lag1 = np.corrcoef(out[1:, 14], out[:-1, 14])[0, 1]
        assert abs(lag1) < 0.1, lag1

    def test_ou_is_smooth_where_white_is_not(self):
        x = _signal(n=4000)
        w = np.asarray(sr.generate(x, [0, 4000], "white",
                                   np.random.default_rng(0))["x"], np.float64)
        o = np.asarray(sr.generate(x, [0, 4000], "ou", np.random.default_rng(0),
                                   fps=30.0)["x"], np.float64)
        assert (np.corrcoef(o[1:, 0], o[:-1, 0])[0, 1]
                > np.corrcoef(w[1:, 0], w[:-1, 0])[0, 1] + 0.5)

    @pytest.mark.parametrize("kind", sr.VISIT_KINDS)
    def test_a_visit_kind_without_a_fit_is_refused(self, kind):
        with pytest.raises(ValueError, match="needs a fitted partition"):
            sr.generate(_signal(), [0, 1200], kind, np.random.default_rng(0))

    @pytest.mark.parametrize("kind", sr.VISIT_KINDS)
    def test_visit_kinds_emit_only_real_frames(self, kind):
        x = _signal(n=1500)
        f, _st = _fit(x)
        out = sr.generate(x, [0, 1500], kind, np.random.default_rng(0), fit=f)
        got = np.asarray(out["x"], np.float64)
        # The reference is cast to float32 first: the shards are float32 and an
        # exact tuple match against float64 input fails on rounding alone.
        rows = {tuple(np.round(r, 5))
                for r in np.asarray(x, np.float32).astype(np.float64)}
        assert all(tuple(np.round(r, 5)) in rows for r in got)

    @pytest.mark.parametrize("kind", ("white", "ou") + sr.VISIT_KINDS)
    def test_dead_columns_stay_dead(self, kind):
        x = _signal()
        f = _fit(x)[0] if kind in sr.VISIT_KINDS else None
        out = sr.generate(x, [0, x.shape[0]], kind, np.random.default_rng(0),
                          fps=30.0, fit=f)
        got = np.asarray(out["x"], np.float64)
        for idx in ego.dead_columns().values():
            assert np.allclose(got[:, idx], 0.0), (kind, idx)

    @pytest.mark.parametrize("kind", ("white", "ou") + sr.VISIT_KINDS)
    def test_no_surrogate_crosses_a_seam(self, kind):
        x = _signal(n=1600)
        f = _fit(x)[0] if kind in sr.VISIT_KINDS else None
        a = sr.generate(x, [0, 800, 1600], kind, np.random.default_rng(0),
                        fps=30.0, fit=f)["x"]
        y = x.copy()
        y[800:] = _signal(n=800, seed=11)
        b = sr.generate(y, [0, 800, 1600], kind, np.random.default_rng(0),
                        fps=30.0, fit=f)["x"]
        assert np.allclose(np.asarray(a)[:800], np.asarray(b)[:800], atol=1e-6)

    def test_microstate0_reports_its_residual_repeats(self):
        x = _signal(n=1500)
        f, _st = _fit(x)
        out = sr.generate(x, [0, 1500], "microstate0",
                          np.random.default_rng(0), fit=f)
        assert "n_residual_repeats" in out and "median_visit_frames" in out


class TestTheVerdictNamesTheRightNull:
    """The first version said "second-order-matched" for every arm, which is
    false of the visit-stitching pair -- they match dwell, not the spectrum."""

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 256}

    def _read(self, kind):
        return sr.falsifier_read(
            {"animal_interval": {"lo": 0.9, "hi": 1.7, "point": 1.3},
             "frame_interval": {}},
            {"mean_delta": 11.3}, {"mean_delta": 10.0},
            kind=kind, scored_object=self.OBJ, n_effective=89)

    @pytest.mark.parametrize("kind", sr.KINDS)
    def test_every_arm_has_a_description(self, kind):
        assert kind in sr.WHAT_IS_MATCHED
        assert sr.WHAT_IS_MATCHED[kind] in self._read(kind).reason

    def test_the_dwell_arms_do_not_claim_a_spectrum_match(self):
        for kind in ("microstate", "microstate0"):
            assert "spectrum" not in self._read(kind).reason

    def test_the_circular_arm_says_so_in_its_own_verdict(self):
        r = self._read("microstate")
        assert "READ THIS ONE WEAKLY" in r.reason and r.detail["circular"]

    def test_the_clean_arm_is_not_flagged_circular(self):
        r = self._read("microstate0")
        assert "READ THIS ONE WEAKLY" not in r.reason
        assert not r.detail["circular"]
