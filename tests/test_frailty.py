"""Can the instrument tell frailty from duration dependence?

Two controls, and they are the whole point. A falling pooled hazard is produced
BOTH by a process whose members become harder to leave over time AND by a
mixture of short- and long-lived members each of whose hazard is flat. If this
module cannot separate those on simulated data where the answer is known, it
cannot be trusted on the corpus -- which is exactly how the dwell result came to
be retracted.
"""
import numpy as np
import pytest

from vieb.tok import frailty as fr, hazard as hz


def _geometric_mixture(rng, n_animals=30, per_animal=8000, p_fast=0.5,
                       p_slow=0.012):
    """Two flat-hazard populations mixed. Pooled hazard MUST fall; within-group
    hazard is exactly constant, because geometric dwell is memoryless."""
    dur, grp, ani = [], [], []
    for a in range(n_animals):
        g = rng.integers(0, 2, size=per_animal)
        d = np.where(g == 0, rng.geometric(p_fast, size=per_animal),
                     rng.geometric(p_slow, size=per_animal))
        dur.append(d)
        grp.append(g)
        ani += [f"a{a}"] * per_animal
    return (np.concatenate(dur), np.concatenate(grp).astype(np.int64),
            np.asarray(ani))


def _heavy_tail(rng, n_animals=30, per_animal=8000):
    """One population with a genuinely falling hazard, and a slice label that
    carries no information about duration."""
    dur, grp, ani = [], [], []
    for a in range(n_animals):
        d = np.ceil(rng.pareto(1.1, size=per_animal) * 4).astype(np.int64) + 1
        dur.append(np.clip(d, 1, 4000))
        grp.append(rng.integers(0, 2, size=per_animal))
        ani += [f"a{a}"] * per_animal
    return (np.concatenate(dur), np.concatenate(grp).astype(np.int64),
            np.asarray(ani))


class TestQuintiles:

    def test_edges_come_from_the_mask_only(self):
        s = np.concatenate([np.arange(100.0), np.full(100, 1e6)])
        mask = np.concatenate([np.ones(100, bool), np.zeros(100, bool)])
        assert fr.quintile_edges(s, mask).max() < 100.0

    def test_there_are_n_minus_one_edges(self):
        s = np.arange(1000.0)
        assert fr.quintile_edges(s, np.ones(1000, bool)).shape == (4,)

    def test_non_finite_speed_gets_no_quintile(self):
        e = np.array([1.0, 2.0, 3.0, 4.0])
        assert fr.quintile_of(np.array([np.nan, 2.5]), e).tolist() == [-1, 2]

    def test_no_finite_speed_is_refused(self):
        with pytest.raises(ValueError):
            fr.quintile_edges(np.full(10, np.nan), np.ones(10, bool))


class TestCurve:

    def test_a_geometric_dwell_gives_a_flat_hazard(self):
        """The null shape. If this is not flat the accounting is wrong."""
        rng = np.random.default_rng(0)
        d = rng.geometric(0.2, size=200_000)
        c = fr.curve(d, np.zeros(d.shape[0], bool), hz.duration_grid(d))
        h = c["hazard"][0][np.isfinite(c["hazard"][0])]
        assert h.size >= 5
        assert np.allclose(h, 0.2, atol=0.03), h

    def test_a_thin_cell_is_nan_not_a_hazard(self):
        d = np.array([1, 1, 900])
        c = fr.curve(d, np.zeros(3, bool), hz.duration_grid(np.arange(1, 400)))
        assert not np.isfinite(c["hazard"][0]).all()

    def test_runs_with_no_slice_are_dropped_and_counted(self):
        d = np.full(100, 5)
        sid = np.where(np.arange(100) < 40, -1, 0)
        c = fr.curve(d, np.zeros(100, bool), hz.duration_grid(d),
                     slice_id=sid, n_slices=1)
        assert c["n_runs"] == 60 and c["n_dropped"] == 40


class TestFallRatio:

    def test_it_uses_the_first_and_last_REPORTABLE_bins(self):
        h = np.array([np.nan, 0.5, 0.25, np.nan])
        f = fr.fall_ratio(h)
        assert f["first_bin"] == 1 and f["last_bin"] == 2
        assert f["fall"] == pytest.approx(2.0)

    def test_fewer_than_two_live_bins_is_nan(self):
        assert not np.isfinite(fr.fall_ratio(np.array([np.nan, 0.5]))["fall"])

    def test_a_flat_curve_falls_by_one(self):
        assert fr.fall_ratio(np.full(6, 0.3))["fall"] == pytest.approx(1.0)
        assert fr.fall_ratio(np.full(6, 0.3))["log_fall"] == pytest.approx(0.0)


class TestTheTwoControls:
    """The instrument must return opposite verdicts on these."""

    OBJ = {"dataset": "sim", "arm": "control"}

    def _run(self, dur, grp, ani):
        cen = np.zeros(dur.shape[0], dtype=bool)
        edges = hz.duration_grid(dur)
        pooled = fr.fall_ratio(fr.curve(dur, cen, edges)["hazard"][0])
        c = fr.curve(dur, cen, edges, slice_id=grp, n_slices=2)
        rows = [fr.fall_ratio(c["hazard"][s]) for s in range(2)]
        w = np.array([c["exits"][s].sum() for s in range(2)])
        lf = np.array([r["log_fall"] for r in rows])
        ok = np.isfinite(lf)
        within = {"fall": float(np.exp(np.average(lf[ok], weights=w[ok])))}
        pa = fr.retained_by_animal(dur, cen, ani, grp, edges, n_slices=2)
        return pooled, within, pa

    def test_a_geometric_mixture_reads_as_frailty(self):
        """BLOCKING. Each group is memoryless by construction, so every bit of
        the pooled fall is mixing."""
        rng = np.random.default_rng(0)
        pooled, within, pa = self._run(*_geometric_mixture(rng))
        assert pooled["fall"] > 3.0, pooled
        r = fr.frailty_read(pa, pooled, within, slice_name="true group",
                            scored_object=self.OBJ, n_effective=30, n_boot=400)
        assert r.verdict == "PASS" and "FRAILTY" in r.reason
        assert float(np.nanmean(pa["retained"])) < fr.RETAINED_FLOOR

    def test_the_within_group_hazard_of_the_mixture_is_flat(self):
        """The mechanism, shown directly rather than inferred from the verdict."""
        rng = np.random.default_rng(0)
        dur, grp, _a = _geometric_mixture(rng)
        c = fr.curve(dur, np.zeros(dur.shape[0], bool), hz.duration_grid(dur),
                     slice_id=grp, n_slices=2)
        for s in (0, 1):
            f = fr.fall_ratio(c["hazard"][s])
            assert abs(f["log_fall"]) < 0.5, (s, f)

    def test_a_genuine_falling_hazard_survives_an_uninformative_slice(self):
        """BLOCKING, and the other direction. One population, slice label is
        noise, so conditioning must remove none of the fall."""
        rng = np.random.default_rng(1)
        pooled, within, pa = self._run(*_heavy_tail(rng))
        assert pooled["fall"] > 3.0, pooled
        r = fr.frailty_read(pa, pooled, within, slice_name="noise",
                            scored_object=self.OBJ, n_effective=30, n_boot=400)
        assert r.verdict == "PASS" and "SURVIVES" in r.reason
        assert float(np.nanmean(pa["retained"])) > fr.RETAINED_FLOOR

    def test_the_surviving_verdict_still_refuses_per_bout_memory(self):
        rng = np.random.default_rng(1)
        pooled, within, pa = self._run(*_heavy_tail(rng))
        r = fr.frailty_read(pa, pooled, within, slice_name="noise",
                            scored_object=self.OBJ, n_effective=30, n_boot=400)
        assert "unobserved mixture would look the same" in r.reason


class TestFrailtyRead:

    OBJ = {"dataset": "sim", "arm": "control"}

    def _pa(self, retained):
        r = np.asarray(retained, dtype=np.float64)
        return {"animals": [f"a{i}" for i in range(r.shape[0])],
                "retained": r, "log_fall_pooled": np.full(r.shape[0], 1.0),
                "log_fall_within": r, "n_animals": int(r.shape[0]),
                "n_usable": int(np.isfinite(r).sum()), "n_refused": 0}

    def test_a_straddling_interval_is_inconclusive(self):
        rng = np.random.default_rng(0)
        r = fr.frailty_read(self._pa(rng.normal(0.5, 0.3, 60)),
                            {"fall": 10.0}, {"fall": 3.0}, slice_name="x",
                            scored_object=self.OBJ, n_effective=60, n_boot=400)
        assert r.verdict == "INCONCLUSIVE"

    def test_too_few_usable_animals_is_grid_limited(self):
        r = fr.frailty_read(self._pa(np.array([np.nan, np.nan, 0.4])),
                            {"fall": 10.0}, {"fall": 3.0}, slice_name="thin",
                            scored_object=self.OBJ, n_effective=3)
        assert r.verdict == "GRID_LIMITED" and r.degenerate
        assert "not a flat hazard" in r.reason

    def test_the_frame_interval_is_carried_but_never_gates(self):
        r = fr.frailty_read(self._pa(np.full(40, 0.1)), {"fall": 10.0},
                            {"fall": 1.1}, slice_name="x",
                            scored_object=self.OBJ, n_effective=40, n_boot=400)
        assert "frame_interval" in r.detail
        assert "does not read it" in r.detail["frame_interval_note"]

    def test_every_read_carries_the_realised_counts(self):
        r = fr.frailty_read(self._pa(np.full(40, 0.1)), {"fall": 10.0},
                            {"fall": 1.1}, slice_name="x",
                            scored_object=self.OBJ, n_effective=40, n_boot=400)
        for k in ("n_animals_usable", "n_animals_refused",
                  "min_exits_per_cell", "retained_floor"):
            assert k in r.detail


class TestTheCommonSpan:
    """The confound that reversed the first reading of the real data.

    A fast quintile's runs never reach the late elapsed bins, so its curve is
    reportable only over a short prefix. Comparing its fall to the pooled fall
    then compares a 6-bin fall against a 13-bin fall, and the slice looks
    flatter because its range is shorter rather than because its hazard is.
    """

    def test_a_shorter_range_mechanically_gives_a_smaller_fall(self):
        """The mechanism, shown on a curve that falls identically throughout."""
        h = np.array([0.5 / (2 ** i) for i in range(13)])
        assert fr.fall_ratio(h)["fall"] == pytest.approx(2 ** 12)
        assert fr.fall_ratio(h, span=(0, 5))["fall"] == pytest.approx(2 ** 5)

    def test_common_span_is_the_overlap(self):
        a = np.array([0.5, 0.4, 0.3, np.nan, np.nan])
        b = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        assert fr.common_span(a, b) == (0, 2)

    def test_no_overlap_is_refused(self):
        a = np.array([0.5, np.nan, np.nan])
        b = np.array([np.nan, np.nan, 0.5])
        assert fr.common_span(a, b) == (-1, -1)

    def test_an_identical_curve_on_a_short_range_retains_everything(self):
        """BLOCKING, and the regression test for the bug. A slice whose hazard
        equals the pool's over its own range must read as retaining ALL of the
        fall, however short that range is. The unrestricted comparison scored
        this at 1/6 of the fall and called it a collapse."""
        h = np.array([0.5 / (2 ** i) for i in range(13)])
        short = h.copy()
        short[6:] = np.nan
        span = fr.common_span(short, h)
        assert span == (0, 5)
        f_s = fr.fall_ratio(short, span=span)
        f_p = fr.fall_ratio(h, span=span)
        assert f_s["log_fall"] / f_p["log_fall"] == pytest.approx(1.0)
        unrestricted = (fr.fall_ratio(short)["log_fall"]
                        / fr.fall_ratio(h)["log_fall"])
        assert unrestricted == pytest.approx(5 / 12, abs=0.01)

    def test_the_read_carries_the_span_it_used(self):
        rng = np.random.default_rng(0)
        dur, grp, ani = _geometric_mixture(rng, n_animals=20, per_animal=6000)
        cen = np.zeros(dur.shape[0], bool)
        edges = hz.duration_grid(dur)
        pa = fr.retained_by_animal(dur, cen, ani, grp, edges, n_slices=2)
        assert "mean_span_bins" in pa
        r = fr.frailty_read(pa, {"fall": 40.0}, {"fall": 1.0},
                            slice_name="true group",
                            scored_object={"dataset": "sim"}, n_effective=20,
                            n_boot=300)
        assert r.detail["mean_span_bins"] is not None
        assert "reportable in BOTH" in r.detail["span_note"]
