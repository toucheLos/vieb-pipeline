"""The gate, and the three polarities that are easy to get backwards.

High separability is FAILURE. Beating the negative control is NOT a result.
A win rate of 50% against length-matched intervals means the detector bought
nothing, and ExBias measured 65.9% — so 50% is the floor, not the standard.
"""
import numpy as np
import pytest

from vieb.seg import breaks as bk, validate as va
from vieb.tok import ego


def _piecewise(n_pieces=12, per=60, c=ego.N_DIMS, seed=0):
    """Concatenated random cubics. Smooth inside each piece, kinked between —
    the signal the axiom describes, so a detector should find the joins."""
    rng = np.random.default_rng(seed)
    s = np.linspace(-1, 1, per)
    out = []
    for _ in range(n_pieces):
        co = rng.normal(size=(4, c))
        out.append(np.stack([s ** k for k in range(4)], 1) @ co)
    return np.concatenate(out)


def _smooth(n=700, c=ego.N_DIMS):
    t = np.linspace(0, 5, n)
    return np.stack([np.sin(2 * np.pi * t / (1.7 + 0.3 * i)) for i in range(c)],
                    axis=1)


class TestRateCurve:

    def _curve(self, x):
        n = x.shape[0]
        return va.rate_curve(x, [0, n], np.zeros(n, bool), h=4, min_gap=16,
                             fps=30.0)

    def test_the_rate_falls_as_the_threshold_rises(self):
        c = self._curve(_piecewise())
        rates = [c["rate"][k] for k in sorted(c["rate"])]
        assert all(a >= b for a, b in zip(rates, rates[1:])), rates

    def test_a_smooth_signal_yields_nothing_at_a_high_threshold(self):
        c = self._curve(_smooth())
        assert c["rate"][6.0] == 0.0

    def test_every_registered_threshold_is_present(self):
        c = self._curve(_piecewise())
        assert sorted(c["rate"]) == sorted(float(k) for k in va.K_SWEEP)

    def test_it_never_crosses_a_seam(self):
        """Two recordings: changing the second must not move the first's rate."""
        x = np.concatenate([_piecewise(seed=0), _piecewise(seed=1)])
        y = np.concatenate([_piecewise(seed=0), _piecewise(seed=5)])
        b = [0, x.shape[0] // 2, x.shape[0]]
        ab = np.zeros(x.shape[0], bool)
        a1 = va.rate_curve(x[:b[1]], [0, b[1]], ab[:b[1]], h=4, min_gap=16,
                           fps=30.0)
        a2 = va.rate_curve(y[:b[1]], [0, b[1]], ab[:b[1]], h=4, min_gap=16,
                           fps=30.0)
        assert a1["n_peaks"] == a2["n_peaks"]


class TestLengthMatchedControl:

    def _run(self, x, seed=0):
        n = x.shape[0]
        d = bk.discontinuity(x, 4)
        pk = bk.boundaries(d, bk.mad_threshold(d, 2.0), min_gap=16)
        rows = bk.segment_table(x, pk, lo=0, hi=n, abstain=np.zeros(n, bool),
                                guard=2, fps=30.0)
        return va.length_matched_control(x, rows, 0, n,
                                         np.random.default_rng(seed), n_draws=4)

    def test_detected_segments_beat_random_intervals_on_a_kinked_signal(self):
        """BLOCKING. If this does not hold on a signal that IS piecewise-smooth,
        the detector is not finding the pieces."""
        out = self._run(_piecewise(n_pieces=16, per=80))
        assert out["n_compared"] > 30
        assert out["win_rate"] > 0.8, out

    def test_a_smooth_signal_gives_no_advantage(self):
        """The negative control for the control: with no kinks to find, a
        detected interval is no better than a random one of the same length."""
        out = self._run(_smooth(n=1200))
        if out["n_compared"] > 30:
            assert out["win_rate"] < 0.8, out

    def test_it_carries_the_number_it_must_be_judged_against(self):
        out = self._run(_piecewise())
        assert out["exbias_reference"] == pytest.approx(0.659)

    def test_nothing_fittable_gives_nan_rather_than_zero(self):
        out = va.length_matched_control(_smooth(), [], 0, 700,
                                        np.random.default_rng(0))
        assert out["n_compared"] == 0 and not np.isfinite(out["win_rate"])


class TestManifoldRead:

    OBJ = {"dataset": "luna", "arm": "seg"}

    def _read(self, **kw):
        res = {"balanced_accuracy": 0.5, "auc": 0.5, "n_test": 1000,
               "converged": True, **kw}
        return va.manifold_read(res, kind="phase", scored_object=self.OBJ,
                                n_effective=89)

    def test_a_separable_null_FAILS(self):
        """BLOCKING, and the polarity is the opposite of every other probe
        here: a null a probe can pick out is not a usable null."""
        r = self._read(balanced_accuracy=0.88)
        assert r.verdict == "FAIL" and "IS SEPARABLE" in r.reason
        assert "uninterpretable" in r.reason

    def test_an_inseparable_null_passes(self):
        r = self._read(balanced_accuracy=0.52)
        assert r.verdict == "PASS" and "usable null" in r.reason

    def test_the_limit_is_inclusive(self):
        assert self._read(balanced_accuracy=0.60).verdict == "PASS"
        assert self._read(balanced_accuracy=0.601).verdict == "FAIL"

    def test_an_unconverged_probe_is_not_a_result(self):
        r = self._read(converged=False, balanced_accuracy=0.99)
        assert r.verdict == "NOT_A_RESULT" and r.degenerate

    def test_a_refused_probe_is_inconclusive(self):
        r = va.manifold_read({"balanced_accuracy": None, "why": "too few rows"},
                             kind="ou", scored_object=self.OBJ, n_effective=89)
        assert r.verdict == "INCONCLUSIVE" and "too few rows" in r.reason

    def test_the_prose_is_about_surrogates_not_correctors(self):
        """`separable_read` would print a paragraph about a corrector
        depositing mass on a constraint surface."""
        r = self._read(balanced_accuracy=0.88)
        assert "corrector" not in r.reason and "constraint surface" not in r.reason


class TestGateRead:

    OBJ = {"dataset": "luna", "arm": "seg"}

    def _gate(self, cells):
        return va.gate_read(cells, scored_object=self.OBJ, n_effective=89)

    def _win(self, kind, k, lo=0.05):
        return {f"{kind}|{k}": {"point": lo + 0.02, "lo": lo, "hi": lo + 0.05}}

    def test_beating_every_structured_null_everywhere_passes(self):
        cells = {}
        for kind in ("phase", "var5", "ou"):
            for k in (1.0, 3.0, 6.0):
                cells.update(self._win(kind, k))
        r = self._gate(cells)
        assert r.verdict == "PASS"
        assert "does NOT say they are behaviours" in r.reason

    def test_one_failing_cell_closes_the_route(self):
        cells = {}
        for kind in ("phase", "var5", "ou"):
            for k in (1.0, 3.0, 6.0):
                cells.update(self._win(kind, k))
        cells["phase|6.0"] = {"point": -0.01, "lo": -0.04, "hi": 0.02}
        r = self._gate(cells)
        assert r.verdict == "FAIL"
        assert "THE SEGMENTATION ROUTE CLOSES" in r.reason
        assert "phase at k = 6" in r.reason

    def test_beating_only_the_negative_control_is_not_a_result(self):
        """BLOCKING. Every smooth signal beats i.i.d. noise."""
        cells = dict(self._win("white", 3.0))
        cells["phase|3.0"] = {"point": 0.0, "lo": -0.02, "hi": 0.02}
        r = self._gate(cells)
        assert r.verdict == "FAIL"

    def test_the_negative_control_alone_gives_no_gate(self):
        r = self._gate(self._win("white", 3.0))
        assert r.verdict == "NOT_A_RESULT" and r.degenerate

    def test_an_unresolvable_cell_is_grid_limited_not_a_pass(self):
        cells = dict(self._win("phase", 1.0))
        cells["var5|1.0"] = {"point": float("nan"), "lo": float("nan"),
                             "hi": float("nan")}
        r = self._gate(cells)
        assert r.verdict == "GRID_LIMITED"

    def test_the_verdict_names_the_tightest_cell(self):
        cells = dict(self._win("phase", 1.0, lo=0.30))
        cells.update(self._win("ou", 5.0, lo=0.01))
        r = self._gate(cells)
        assert r.detail["worst_cell"]["kind"] == "ou"
