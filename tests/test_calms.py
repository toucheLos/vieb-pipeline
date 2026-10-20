"""STEP 1: operators, seam invariance, the label-import guard, scoring on planted toys."""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pytest

from vieb.calms import degrade as dg
from vieb.calms import io as cio
from vieb.calms import score as sc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rigid_pose(T=6000, seed=0):
    """A walking rigid mouse in Luna order, on Luna's pixel scale: skull bones
    have constant length, so the skull estimator's residual is pure noise."""
    rng = np.random.default_rng(seed)
    # Luna's skull geometry: ear-ear and ear-nose ~0.27-0.30 BL. A shorter skull
    # makes the bone length nonlinear in the noise and the estimator under-reads.
    body = np.array([[-15, 15], [-15, -15], [10, 0], [-30, 0], [-60, 12], [-60, -12],
                     [-100, 0]], dtype=np.float64)
    body *= cio.LUNA_ELL_PX / np.linalg.norm(body[2] - body[6])
    th = np.cumsum(rng.normal(0, 0.02, T))
    pos = np.cumsum(rng.normal(0, 1.0, (T, 2)), axis=0)
    c, s = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)     # (T, 2, 2)
    return np.einsum("tij,kj->tki", R, body) + pos[:, None, :]


# --------------------------------------------------------------------------
# Jitter
# --------------------------------------------------------------------------

def test_profile_skull_rms_is_the_luna_level():
    p = dg.luna_profile()
    skull = [dg.LUNA_ORDER.index(k) for k in dg.SKULL_NAMES]
    assert np.sqrt(np.mean(p[skull] ** 2)) == pytest.approx(dg.sigma_bar_luna(), rel=1e-12)
    assert dg.sigma_bar_luna() == pytest.approx(0.0618, abs=5e-5)


def test_added_noise_is_quadrature_and_unreachable_below_native():
    prof = np.ones(7)
    assert dg.added_bl(1.0, 0.6, prof) == pytest.approx(np.full(7, 0.8))
    assert np.all(dg.added_bl(0.5, 0.6, prof) == 0)
    assert not dg.reachable(0.5, 0.6) and dg.reachable(0.0, 0.6) and dg.reachable(1.0, 0.6)


LUNA_SKULL = (0.27, 0.30, 0.275)


def test_calibration_pose_has_the_requested_skull():
    from vieb.qc import bones as bn
    p = dg.calibration_pose(LUNA_SKULL, n_frames=50)
    got = np.median(bn.bone_lengths(p, bn.SKULL), 0) / cio.LUNA_ELL_PX
    assert got == pytest.approx(LUNA_SKULL, abs=1e-9)


def test_uncalibrated_operator_under_reads_and_the_gain_fixes_it():
    """Amendment 1: the estimator reads the bare operator ~9% low; with g it
    reads Luna's level, on a fresh seed it was not calibrated on."""
    s1, prof, sb = dg.s1(), dg.luna_profile(), dg.sigma_bar_luna()
    pose = dg.calibration_pose(LUNA_SKULL, n_frames=100_000, seed=5)
    bare = dg.estimator_reading(pose, 1.0, prof, s1, 1.0, seed=7)
    assert bare < 0.95 * sb
    g = dg.gain(LUNA_SKULL, prof, s1, sb)
    assert 1.03 < g < 1.2
    assert dg.estimator_reading(pose, 1.0, prof, s1, g, seed=7) == pytest.approx(sb, rel=0.02)


def test_jitter_keeps_luna_colour():
    s1 = dg.s1()
    x = dg.jitter_noise(np.random.default_rng(1), 200_000, np.full(7, 0.05), s1)
    acf1 = np.mean([np.corrcoef(x[:-1, k, 0], x[1:, k, 0])[0, 1] for k in range(7)])
    assert acf1 > 0.4, "the jitter came out white; Luna's lag-1 is +0.58"
    assert np.std(x) == pytest.approx(0.05 * cio.LUNA_ELL_PX, rel=0.03)


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------

def donors(seed=0):
    rng = np.random.default_rng(seed)
    T = 5000
    miss = np.zeros((2 * T, 7), dtype=bool)
    for _ in range(200):
        a, k = int(rng.integers(0, 2 * T - 20)), int(rng.integers(7))
        miss[a:a + int(rng.integers(1, 20)), k] = True
    ab = np.zeros(2 * T, dtype=bool)
    ab[100:130] = True
    return {"missing": miss, "abstain": ab, "bounds": np.array([0, T, 2 * T])}


def test_thin_keeps_whole_runs():
    d = donors()
    rng = np.random.default_rng(0)
    assert (dg.thin(d["missing"], 1.0, rng) == d["missing"]).all()
    assert not dg.thin(d["missing"], 0.0, rng).any()
    half = dg.thin(d["missing"], 0.5, rng)
    kept = {r for k in range(7) for r in dg.runs(half[:, k])}
    orig = {r for k in range(7) for r in dg.runs(d["missing"][:, k])}
    assert kept <= orig                     # no partial runs, no new runs


def test_dropout_rate_scales_with_level():
    d = donors()
    rates = []
    for m in (0.5, 1.0, 2.0):
        mi, _ab = dg.dropout(50_000, m, d, np.random.default_rng(1))
        rates.append(mi.mean())
    assert rates[0] < rates[1] < rates[2]


def test_level_zero_has_no_dropout_or_noise():
    pose = rigid_pose(500)
    out, mi, ab = dg.degraded(pose, 0.0, r=0.3, profile=dg.luna_profile(), s1_px=1.0,
                              donors=donors(), rng=np.random.default_rng(0))
    assert (out == pose).all() and not mi.any() and not ab.any()


def test_seam_invariance():
    """Perturb sequence 1's tail: sequence 2's degraded draw is unchanged to the bit."""
    s1 = dg.s1()
    prof = dg.luna_profile()
    a, b = rigid_pose(900, 1), rigid_pose(700, 2)
    run = lambda x, name: dg.degraded(x, 1.5, r=0.3, profile=prof, s1_px=s1,  # noqa: E731
                                      donors=donors(), rng=dg.rng_for(0, 1.5, name))
    first = run(b, "seq2")
    a2 = a.copy()
    a2[-50:] += 1000.0
    run(a2, "seq1")
    second = run(b, "seq2")
    for x, y in zip(first, second):
        assert np.array_equal(x, y)


def test_fills_match_luna():
    pose = rigid_pose(400)
    miss = np.zeros((400, 7), dtype=bool)
    miss[10:12, 2] = True              # short: held
    miss[50:60, 3] = True              # long: missing, conf 0
    ab = np.zeros(400, dtype=bool)
    ab[200:205] = True
    coords, conf = dg.moseq_input(pose, np.full((400, 7), 0.9), miss, ab)
    assert np.isfinite(coords).all()
    assert (conf[50:60, 3] == 0).all() and (conf[10:12, 2] > 0).all()
    assert (conf[200:205] == 0).all()
    held, abst = dg.raw_input(pose, miss, ab)
    assert np.isfinite(held).all() and (abst == ab).all()


# --------------------------------------------------------------------------
# The label guard
# --------------------------------------------------------------------------

ALLOWED = {"vieb/calms/score.py", "scripts/calms_score.py", "scripts/calms_convert.py",
           "scripts/calms_probe.py"}
PATTERN = re.compile(r"calms\s+import\s+score|calms\.score|(CALMS|calms21)[^\n]*labels")


def test_only_the_scorer_reads_labels():
    offenders = []
    for p in glob.glob(os.path.join(REPO, "vieb", "**", "*.py"), recursive=True) + \
            glob.glob(os.path.join(REPO, "scripts", "*.py")):
        rel = os.path.relpath(p, REPO)
        if rel in ALLOWED:
            continue
        with open(p, encoding="utf-8") as fh:
            if PATTERN.search(fh.read()):
                offenders.append(rel)
    assert offenders == [], f"label store referenced outside the scorer: {offenders}"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_event_counts_iou():
    y = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 4 + [0] * 6)
    assert sc.event_counts(y, y, 1) == (2, 2, 2)
    yh = np.array([0] * 12 + [1] * 10 + [0] * 18)     # IoU 8/12 with bout 1
    assert sc.event_counts(y, yh, 1) == (1, 2, 1)
    yh2 = np.array([0] * 18 + [1] * 10 + [0] * 12)    # IoU 2/18
    assert sc.event_counts(y, yh2, 1) == (0, 2, 1)


def test_boundary_counts_tolerance():
    assert sc.boundary_counts([100, 200], [110, 260], 15) == (1, 2, 2)
    assert sc.boundary_counts([100], [99, 101], 15) == (1, 1, 2)


def test_true_boundaries_exclude_edges():
    y = np.array([1] * 5 + [0] * 5 + [1] * 5)
    assert list(sc.true_boundaries(y, 1)) == [5, 10]


def toy(n_seq=12, T=3000, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    y, st = {}, {}
    for i in range(n_seq):
        lab = np.zeros(T, dtype=np.int64)
        pos = 0
        while pos < T:
            d = int(rng.integers(20, 120))
            lab[pos:pos + d] = int(rng.random() < 0.3)
            pos += d
        y[f"v{i}"] = lab
        st[f"v{i}"] = (lab * 3 + rng.integers(0, 3, T)) if informative \
            else rng.integers(0, 6, T)
    return y, st


@pytest.mark.parametrize("informative,lo,hi", [(True, 0.9, 1.01), (False, -0.2, 0.2)])
def test_planted_toy_normalised_score(informative, lo, hi, monkeypatch):
    monkeypatch.setitem(sc.SETS, "toy", ("x", ("other", "groom")))
    y, st = toy(informative=informative)
    ids = sorted(y)
    mp, ev = ids[:6], ids[6:]
    per_seed = [sc.event_pass(st, y, mp, ev, "toy", ["groom"])["groom"]]

    def one(rng):
        ys = sc.shifted(y, ids, rng)
        return sc.f1(sc.event_pass(st, ys, mp, ev, "toy", ["groom"])["groom"].sum(0))

    null = sc.null_distribution(one, n=60)
    c = sc.cell(per_seed, null, 1.0, sc.boot_index(len(ev), 200))
    assert lo <= c["normalised"] <= hi
    assert c["recoverable"] == informative


def test_fit_map_is_many_to_one_and_unseen_is_other():
    m = sc.fit_map([np.array([0, 0, 1, 1, 2])], [np.array([1, 1, 0, 0, 1])], 2, 0)
    assert m == {0: 1, 1: 0, 2: 1}
    assert list(sc.apply_map(np.array([0, 5, -1]), m, 0)) == [1, 0, 0]


def C(rec, straddles=False):
    return {"recoverable": rec, "straddles": straddles}


def test_decision_rule():
    kw = dict(native=0.0, at=1.0, controls_fail=False, r_ge_1=False, groom_grid_limited=False)
    assert sc.decide({("M", "mount", 0.0): C(False)}, **kw)["outcome"] == "O1"
    assert sc.decide({("M", "mount", 0.0): C(False, True)}, **kw)["outcome"] == "INCONCLUSIVE"
    o2 = {("M", "mount", 0.0): C(True), ("M", "groom", 1.0): C(False)}
    assert sc.decide(o2, **kw)["outcome"] == "O2"
    o3 = {**o2, ("C", "groom", 1.5): C(True)}
    assert sc.decide(o3, **kw)["outcome"] == "O3"
    inc = {**o2, ("D", "groom", 1.0): C(False, True)}
    assert sc.decide(inc, **kw)["outcome"] == "INCONCLUSIVE"
    assert sc.decide(o3, **{**kw, "controls_fail": True})["outcome"] == "NOT_A_RESULT"
    assert sc.decide(o3, **{**kw, "r_ge_1": True})["outcome"] == "NOT_A_RESULT"
    assert sc.decide(o2, **{**kw, "groom_grid_limited": True})["outcome"] == "NOT_A_RESULT"


def test_breakdown_inside_or_bound():
    lv = [0, 0.25, 0.5, 1, 1.5, 2, 3]
    assert sc.breakdown(lv, [1, 1, 1, 0, 0, 0, 0])["level"] == 1
    assert sc.breakdown(lv, [1] * 7)["bound"] == "> 3x"
    assert "never" in sc.breakdown(lv, [0] * 7)["bound"]


def test_normalised_requires_ceiling_above_chance():
    with pytest.raises(AssertionError):
        sc.normalised(0.5, 0.6, 0.6)


# --------------------------------------------------------------------------
# Secondary arms and ablation (ABLATION_REGISTRATION.md)
# --------------------------------------------------------------------------

def test_arm_spec_round_trips_and_counts():
    from vieb.calms import arms as A
    for sp in A.all_specs():
        assert A.parse(sp.name) == sp
    assert A.parse("a-k5-m1_s2").input_arm == "m1_s2"
    assert A.parse("j1.5_s0").dropout is False and A.parse("d2_s0").jitter is False
    assert set(A.seeds("k", "M")) == {0, 1, 2} and set(A.seeds("m", "M")) == set(range(5))
    assert len(A.kpms_list()) == 45 + 87


def test_substitute_six_and_nonose():
    pose = rigid_pose(50)
    miss = np.zeros((50, 7), dtype=bool)
    miss[5, dg.LH] = True
    sc_ = np.full((50, 7), 0.9)
    sc_[7, dg.RE] = 0.2
    p6, m6, c6 = dg.substitute(pose, miss, sc_, "six")
    want = 0.5 * (0.5 * (pose[:, dg.LE] + pose[:, dg.RE]) + 0.5 * (pose[:, dg.LH] + pose[:, dg.RH]))
    assert np.allclose(p6[:, dg.CENTER], want) and m6[5, dg.CENTER] and c6[7, dg.CENTER] == 0.2
    pn, mn, cn = dg.substitute(pose, miss, sc_, "nonose")
    assert np.allclose(pn[:, dg.NOSE], 0.5 * (pose[:, dg.LE] + pose[:, dg.RE]))
    assert not mn[5, dg.NOSE] and cn[7, dg.NOSE] == 0.2
    p7, _m7, _c7 = dg.substitute(pose, miss, sc_, "seven")
    assert np.array_equal(p7, pose)


def test_smoothed_inputs_keep_long_gaps_missing():
    pose = rigid_pose(300) + np.random.default_rng(0).normal(0, 3, (300, 7, 2))
    miss = np.zeros((300, 7), dtype=bool)
    miss[100:110, 2] = True
    ab = np.zeros(300, dtype=bool)
    coords, conf = dg.moseq_input_smoothed(pose, np.full((300, 7), 0.9), miss, ab, 30.0)
    assert np.isfinite(coords).all() and (conf[100:110, 2] == 0).all()
    raw, _ = dg.raw_input(pose, miss, ab)
    sm, _ = dg.raw_input_smoothed(pose, miss, ab, 30.0)
    assert np.std(np.diff(sm[:, 0, 0])) < np.std(np.diff(raw[:, 0, 0]))


def test_decimate():
    a, b = dg.decimate(3, np.arange(10), np.arange(10) * 2)
    assert list(a) == [0, 3, 6, 9] and list(b) == [0, 6, 12, 18]


def test_abs_delta_lower():
    assert sc.abs_delta_lower(-0.1, 0.2) == 0.0
    assert sc.abs_delta_lower(0.05, 0.2) == 0.05
    assert sc.abs_delta_lower(-0.3, -0.1) == pytest.approx(0.1)
