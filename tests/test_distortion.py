"""Distortion, and the omitted term that would have made it wrong everywhere.

`microstate.assign_states` computes `cn - 2*(F @ C.T)` and drops `||f||^2`
because it only takes an argmin. That is correct for its purpose and fatal for
this one: the dropped term varies across frames, so a distortion built on it is
wrong by a different amount for every frame and looks perfectly plausible.
"""
import numpy as np
import pytest
from recur.null import microstate

from vieb.tok import distortion as dz, ego


def _fixture(n=400, n_states=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.0, size=(n, ego.N_DIMS))
    sd = np.full(ego.N_DIMS, 2.0)
    cols = np.arange(ego.N_DIMS, dtype=np.int64)
    cent = rng.normal(0, 0.5, size=(n_states, ego.N_DIMS)).astype(np.float32)
    labels = microstate.assign_states(x, cent, sd, cols)
    return x, labels, cent, sd, cols, n_states


class TestTheOmittedTerm:

    def test_assign_states_surrogate_is_not_a_squared_distance(self):
        """BLOCKING. The premise of this whole module: if the surrogate WERE a
        distance, reusing it would be free and this test would fail."""
        x, labels, cent, sd, cols, _n = _fixture()
        f = microstate.standardise(x, sd, cols).astype(np.float64)
        c = np.asarray(cent, dtype=np.float64)
        surrogate = (c ** 2).sum(1)[None, :] - 2.0 * (f @ c.T)
        true = ((f[:, None, :] - c[None, :, :]) ** 2).sum(2)
        assert not np.allclose(surrogate, true)
        assert (surrogate < 0).any(), "the surrogate is negative about half the time"

    def test_the_two_differ_by_exactly_the_dropped_term(self):
        x, labels, cent, sd, cols, _n = _fixture()
        f = microstate.standardise(x, sd, cols).astype(np.float64)
        c = np.asarray(cent, dtype=np.float64)
        surrogate = (c ** 2).sum(1)[None, :] - 2.0 * (f @ c.T)
        true = ((f[:, None, :] - c[None, :, :]) ** 2).sum(2)
        assert np.allclose(true - surrogate, (f ** 2).sum(1)[:, None])

    def test_the_argmin_is_unchanged_which_is_why_labels_are_fine(self):
        x, labels, cent, sd, cols, _n = _fixture()
        f = microstate.standardise(x, sd, cols).astype(np.float64)
        c = np.asarray(cent, dtype=np.float64)
        true = ((f[:, None, :] - c[None, :, :]) ** 2).sum(2)
        assert np.array_equal(np.argmin(true, axis=1), labels)


class TestResiduals:

    def test_it_measures_against_the_assigned_centroid(self):
        x, labels, cent, sd, cols, _n = _fixture()
        r = dz.residuals(x, labels, cent, sd, cols)
        f = microstate.standardise(x, sd, cols).astype(np.float64)
        assert np.allclose(r, f - np.asarray(cent, np.float64)[labels])

    def test_frame_error_matches_brute_force(self):
        x, labels, cent, sd, cols, _n = _fixture()
        got = dz.frame_sq_error(x, labels, cent, sd, cols)
        f = microstate.standardise(x, sd, cols).astype(np.float64)
        want = ((f - np.asarray(cent, np.float64)[labels]) ** 2).sum(1)
        assert np.allclose(got, want)

    def test_perfect_centroids_give_zero(self):
        cent = np.zeros((1, ego.N_DIMS), dtype=np.float32)
        x = np.zeros((10, ego.N_DIMS))
        sd, cols = np.ones(ego.N_DIMS), np.arange(ego.N_DIMS)
        assert dz.frame_sq_error(x, np.zeros(10, int), cent, sd, cols).sum() == 0.0


class TestAccumulate:

    def test_chunking_does_not_change_the_answer(self):
        x, labels, cent, sd, cols, n = _fixture(n=1000)
        a = dz.accumulate(x, labels, cent, sd, cols, n_states=n, chunk=1000)
        b = dz.accumulate(x, labels, cent, sd, cols, n_states=n, chunk=37)
        assert np.allclose(a["sse_channel"], b["sse_channel"])
        assert np.allclose(a["sse_symbol"], b["sse_symbol"])
        assert a["n_frames"] == b["n_frames"]

    def test_merge_is_additive(self):
        x, labels, cent, sd, cols, n = _fixture(n=600)
        whole = dz.accumulate(x, labels, cent, sd, cols, n_states=n)
        halves = [dz.accumulate(x[:300], labels[:300], cent, sd, cols, n_states=n),
                  dz.accumulate(x[300:], labels[300:], cent, sd, cols, n_states=n)]
        m = dz.merge(halves)
        assert np.allclose(whole["sse_channel"], m["sse_channel"])
        assert whole["n_frames"] == m["n_frames"]

    def test_keep_excludes_frames_entirely(self):
        """An abstained frame has a symbol by convention only, and its residual
        is not a quantization error."""
        x, labels, cent, sd, cols, n = _fixture(n=200)
        keep = np.zeros(200, bool)
        keep[:50] = True
        a = dz.accumulate(x, labels, cent, sd, cols, n_states=n, keep=keep)
        b = dz.accumulate(x[:50], labels[:50], cent, sd, cols, n_states=n)
        assert a["n_frames"] == 50
        assert np.allclose(a["sse_channel"], b["sse_channel"])

    def test_merge_refuses_nothing(self):
        with pytest.raises(ValueError):
            dz.merge([])


class TestSummarise:

    def _s(self, **kw):
        x, labels, cent, sd, cols, n = _fixture(n=2000, **kw)
        acc = dz.accumulate(x, labels, cent, sd, cols, n_states=n)
        return dz.summarise(acc, sd), sd

    def test_the_channel_groups_partition_the_total(self):
        s, _sd = self._s()
        parts = sum(g["d"] for g in s["by_group"].values())
        assert parts == pytest.approx(s["d_total"])

    def test_the_groups_cover_every_channel_exactly_once(self):
        seen = [i for idx in dz.CHANNEL_GROUPS.values() for i in idx]
        assert sorted(seen) == list(range(ego.N_DIMS))

    def test_native_units_are_standardised_times_sd(self):
        s, sd = self._s()
        for i, row in enumerate(s["per_channel"]):
            assert row["rms_native"] == pytest.approx(
                np.sqrt(row["mse_standardised"]) * sd[i])

    def test_the_worst_decile_share_is_a_fraction(self):
        s, _sd = self._s()
        assert 0.0 <= s["per_symbol"]["worst_decile_share_of_error"] <= 1.0

    def test_no_frames_is_reported_not_divided_by(self):
        acc = {"sse_channel": np.zeros(17), "sse_symbol": np.zeros(4),
               "n_symbol": np.zeros(4, int), "n_frames": 0, "n_states": 4}
        assert dz.summarise(acc, np.ones(17))["n_frames"] == 0


class TestVerifyAssignment:

    def _model(self, cent, sd, cols, arm="plain", offsets=(0,), edges=()):
        return {"centroids": cent, "sd": sd, "cols": cols,
                "arm": np.array(arm), "offsets": np.asarray(offsets, np.int64),
                "edges": np.asarray(edges, np.float64)}

    def test_stored_labels_agree_with_the_true_argmin(self):
        x, labels, cent, sd, cols, _n = _fixture(n=800)
        rng = np.random.default_rng(0)
        out = dz.verify_assignment(x, labels, self._model(cent, sd, cols),
                                   rng=rng)
        assert out["frac_agree"] == 1.0

    def test_a_wrong_basis_is_caught(self):
        """BLOCKING. If the labels were produced under a different sd than the
        one handed here, every residual is measured against the wrong centroid
        and nothing else would notice."""
        x, labels, cent, sd, cols, _n = _fixture(n=800)
        rng = np.random.default_rng(0)
        wrong = sd * np.linspace(0.3, 3.0, sd.shape[0])
        out = dz.verify_assignment(x, labels, self._model(cent, wrong, cols),
                                   rng=rng)
        assert out["frac_agree"] < 0.9

    def test_a_stratified_arm_is_checked_against_ITS_rule(self):
        """BLOCKING, and the bug this replaced. A speed-arm frame is assigned
        the nearest centroid WITHIN its quintile block; a global argmin scores
        a correct stratified arm at ~30% and looks like corruption."""
        from vieb.tok import quantize as qz
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1.0, size=(2000, ego.N_DIMS))
        x[:, 14] = rng.normal(0, 3.0, size=2000)
        sd = np.full(ego.N_DIMS, 1.0)
        cols = np.arange(ego.N_DIMS, dtype=np.int64)
        sp = qz.speed(x)
        model = qz.fit(x, sd, cols, np.ones(2000, bool), arm="speed",
                       n_states=20, sp=sp)
        labels = qz.assign(x, model, sp=sp)
        out = dz.verify_assignment(x, labels, model, rng=rng, sp=sp)
        assert out["frac_agree"] == 1.0 and out["arm"] == "speed"

    def test_the_stratified_arm_needs_a_speed_vector(self):
        x, labels, cent, sd, cols, _n = _fixture(n=100)
        m = self._model(cent, sd, cols, arm="speed", offsets=(0, 6),
                        edges=(0.0,))
        with pytest.raises(ValueError):
            dz.verify_assignment(x, labels, m, rng=np.random.default_rng(0))


class TestDistortionRead:

    OBJ = {"dataset": "luna", "arm": "plain", "n_states": 6}

    def _read(self, frac):
        x, labels, cent, sd, cols, n = _fixture(n=1000)
        acc = dz.accumulate(x, labels, cent, sd, cols, n_states=n)
        s = dz.summarise(acc, sd)
        return dz.distortion_read(s, {"frac_agree": frac, "n_checked": 1000,
                                      "arm": "plain"},
                                  scored_object=self.OBJ, n_effective=89)

    def test_a_disagreeing_assignment_fails_degenerately(self):
        r = self._read(0.83)
        assert r.verdict == "FAIL" and r.degenerate

    def test_agreement_passes_and_the_reason_refuses_to_grade_distortion(self):
        r = self._read(1.0)
        assert r.verdict == "PASS"
        assert "coordinate and not a verdict" in r.reason

    def test_the_read_carries_both_channel_groups(self):
        r = self._read(1.0)
        assert set(r.detail["by_group"]) == {"shape", "twist"}
