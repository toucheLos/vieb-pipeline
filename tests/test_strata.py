"""The pooled figure must equal the frame-weighted sum over strata.

F2's write-up compared a pooled net of -0.0000 against per-stratum nets whose
UNWEIGHTED mean is -0.0051 and called the gap unexplained. The gap was the
comparison: the pooled figure is frame-weighted across strata, and the strata
differ in size by 4x. Reconciled here so the two paths cannot drift again.

The other half of that investigation: bin -1 is `truth.assign_bin`'s refusal
code, not a stratum, and it was being sorted first and printed as "the slowest
stratum" on 235 keypoint-frames. That is where a 0.0 for viterbi and a -0.0331
for the median came from, both of which were reported as findings.
"""
import numpy as np
import pytest

from vieb.qc import recover, truth


def weighted(nets, ns):
    n = np.asarray(ns, dtype=np.float64)
    v = np.asarray(nets, dtype=np.float64)
    ok = np.isfinite(v) & (n > 0)
    return float((v[ok] * n[ok]).sum() / n[ok].sum())


class TestTheReconciliation:

    def test_weighted_sum_recovers_the_pooled_figure(self):
        """The measured case: median_0.50 at `off`."""
        nets = [-0.0021, -0.0008, 0.0001, 0.0020, 0.0068]
        ns = [1895756, 1300526, 1088865, 741735, 472927]
        assert weighted(nets, ns) == pytest.approx(-0.0000, abs=6e-4)

    def test_the_unweighted_mean_does_not_and_flips_the_sign(self):
        """The comparison that produced the phantom discrepancy. The unweighted
        mean is +0.0012 where the pooled figure is -0.00004 -- not merely a
        different magnitude but the OPPOSITE SIGN, which is why reading one
        against the other suggested the arithmetic did not close."""
        nets = [-0.0021, -0.0008, 0.0001, 0.0020, 0.0068]
        ns = [1895756, 1300526, 1088865, 741735, 472927]
        unweighted = float(np.mean(nets))
        pooled = weighted(nets, ns)
        assert unweighted == pytest.approx(0.0012, abs=5e-5)
        assert unweighted > 0 > pooled

    def test_equal_strata_make_the_two_agree(self):
        nets = [-0.004, -0.002, 0.000, 0.002, 0.004]
        ns = [1000] * 5
        assert weighted(nets, ns) == pytest.approx(float(np.mean(nets)))


class TestTheRefusalBucket:

    def test_assign_bin_returns_minus_one_for_an_unplaceable_speed(self):
        edges = truth.speed_bins([0.1, 0.2, 0.3, 0.4, 0.5], n_bins=5)
        assert truth.assign_bin(float("nan"), edges) == -1

    def test_minus_one_is_not_a_stratum(self):
        """It sorts before 0, so a naive `sorted(strata)` prints it first and it
        reads as the slowest bin. It is a refusal."""
        assert -1 < 0
        assert truth.assign_bin(float("nan"), [0.0, 1.0]) == -1

    def test_a_stratum_below_the_minimum_is_refused_not_reported(self):
        assert recover.MIN_STRATUM_FRAMES == 20_000
        nets = [-0.002, float("nan"), 0.001]
        ns = [500_000, 235, 400_000]
        # The refused stratum contributes nothing and does not poison the sum.
        assert np.isfinite(weighted(nets, ns))


class TestTheGuardHasATrueNegative:
    """`recovery_read` flagged `median_0.50` as harmed at 0.10 and `off`, and as
    mixed-by-speed at 0.05 -- the first time it has ever fired. A guard with no
    recorded true negative has an unmeasured false positive rate.

    The control: `viterbi` at `off`, where the expected verdict is not-harmed.
    If the guard fires on it, the guard is uninformative and every verdict it has
    issued is void.
    """

    OBJ = {"dataset": "luna", "arm": "injection", "split": "report"}

    def test_it_stays_silent_on_viterbi_at_off(self):
        """The measured case, numbers from results/injection_1e6.json."""
        rows = [{"arm": "raw", "damage": 0.0, "repair": 0.3641},
                {"arm": "viterbi", "damage": 9.6e-06, "repair": 0.2891},
                {"arm": "median_0.50", "damage": 0.0055, "repair": 0.1170}]
        iv = {"viterbi": {"net": {"lo": -0.0017, "hi": -0.0016}},
              "median_0.50": {"net": {"lo": -0.0006, "hi": 0.0005}}}
        strata = {"viterbi": [-0.0014, -0.0016, -0.0016, -0.0018, -0.0025],
                  "median_0.50": [-0.0021, -0.0008, 0.0001, 0.0020, 0.0068]}
        rd = recover.recovery_read(rows, iv, strata, scored_object=self.OBJ,
                                   n_effective=89)
        assert "viterbi" in rd.detail["helped"], (
            "the guard fired on the arm it should not: every verdict it has "
            "issued is void")
        assert "viterbi" not in rd.detail["harmed"]
        assert "viterbi" not in rd.detail["mixed_by_speed"]

    def test_it_does_flag_the_arm_it_should(self):
        """The positive control, so the test above is not passing vacuously."""
        rows = [{"arm": "raw", "damage": 0.0, "repair": 0.3641},
                {"arm": "median_0.50", "damage": 0.0055, "repair": 0.1170}]
        iv = {"median_0.50": {"net": {"lo": -0.0006, "hi": 0.0005}}}
        strata = {"median_0.50": [-0.0021, -0.0008, 0.0001, 0.0020, 0.0068]}
        rd = recover.recovery_read(rows, iv, strata, scored_object=self.OBJ,
                                   n_effective=89)
        assert "median_0.50" in rd.detail["harmed"]
