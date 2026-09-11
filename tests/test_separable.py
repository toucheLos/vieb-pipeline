"""The atom check: corrected frames must not be findable in feature space.

The polarity is inverted from every other probe in this codebase — a high score
is the failure. A corrector that deposits mass on a constraint surface writes an
atom into an otherwise continuous distribution, k-means finds atoms, and it would
be reported as a behavioural state. This project has manufactured one state
signature already, from a persistence prior; this is the same failure one step
earlier.
"""
import numpy as np
import pytest

from vieb.audit import separable

OBJ = {"dataset": "luna", "arm": "disposition", "split": "report"}


def data(n=4000, k=8, *, frac=0.1, offset=0.0, seed=0, run=50):
    """`frac` of rows are 'corrected' and sit `offset` away in feature space."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, k))
    y = np.zeros(n, dtype=bool)
    y[: int(n * frac)] = True
    x[y] += offset
    groups = [f"r{i // run}" for i in range(n)]
    return x, y, groups


class TestItDetectsAnAtomWhenThereIsOne:

    def test_a_large_offset_is_separable(self):
        res = separable.separability(*data(offset=3.0))
        assert res["balanced_accuracy"] > 0.9
        rd = separable.separable_read(res, scored_object=OBJ, n_effective=89)
        assert rd.verdict == "FAIL" and "look like a behavioural state" in rd.reason

    def test_a_constraint_surface_is_separable_even_in_one_dimension(self):
        """The realistic shape: corrected rows pinned to one exact value in one
        feature, everything else continuous. That is the atom."""
        x, y, g = data(offset=0.0)
        x[y, 0] = 1.2345                      # every corrected row identical here
        res = separable.separability(x, y, g)
        assert res["balanced_accuracy"] > 0.8


class TestItPassesWhenThereIsNone:

    def test_no_offset_is_not_separable(self):
        res = separable.separability(*data(offset=0.0))
        assert res["balanced_accuracy"] <= separable.MAX_BALANCED_ACCURACY
        rd = separable.separable_read(res, scored_object=OBJ, n_effective=89)
        assert rd.verdict == "PASS"

    def test_the_baseline_is_a_real_half_not_the_prevalence(self):
        """With 1% corrected, a classifier predicting 'clean' always scores 99%
        plain accuracy. Balanced accuracy is what makes that read as chance."""
        # n large enough that 1% still clears MIN_MINORITY -- the probe refuses
        # below that, which is itself correct and is tested separately.
        res = separable.separability(*data(n=20000, frac=0.01, offset=0.0))
        assert 0.4 < res["balanced_accuracy"] < 0.6


class TestTheGroupedSplit:

    def test_temporal_autocorrelation_alone_does_not_make_it_separable(self):
        """Adjacent frames resemble each other. A row-wise split would put a
        run's frames on both sides and the probe would separate the classes from
        proximity rather than from the correction."""
        rng = np.random.default_rng(0)
        n, run = 4000, 50
        # A smooth random walk: neighbours are near-identical, classes are not.
        x = np.cumsum(rng.normal(0, 0.3, (n, 6)), axis=0)
        y = np.zeros(n, dtype=bool)
        for start in range(0, n, run * 4):      # whole runs are "corrected"
            y[start:start + 6] = True
        groups = [f"r{i // run}" for i in range(n)]
        res = separable.separability(x, y, groups)
        assert res["balanced_accuracy"] < 0.75


class TestRefusals:

    def test_too_few_corrected_rows_refuses(self):
        x, y, g = data(frac=0.001)
        res = separable.separability(x, y, g)
        assert res["balanced_accuracy"] is None
        rd = separable.separable_read(res, scored_object=OBJ, n_effective=89)
        assert rd.verdict == "INCONCLUSIVE"

    def test_an_unconverged_probe_is_refused_not_reported(self):
        res = {"balanced_accuracy": 0.55, "auc": 0.55, "converged": False,
               "n_corrected": 500}
        rd = separable.separable_read(res, scored_object=OBJ, n_effective=89)
        assert rd.verdict == "NOT_A_RESULT" and rd.degenerate

    def test_a_single_class_refuses(self):
        x = np.random.default_rng(0).normal(0, 1, (500, 4))
        res = separable.separability(x, np.zeros(500, bool),
                                     [f"r{i//50}" for i in range(500)])
        assert res["balanced_accuracy"] is None


# ---------------------------------------------------------------------------
# The grouping
# ---------------------------------------------------------------------------

def test_a_block_never_spans_a_recording() -> None:
    a = separable.blocks("rec_a", 100, fps=30.0)
    b = separable.blocks("rec_b", 100, fps=30.0)
    assert not (set(a) & set(b))


def test_a_corrected_frame_keeps_its_neighbours() -> None:
    # The whole point of blocks over runs: the frames either side of a corrected
    # one land in the same group, so the probe cannot win on temporal proximity.
    g = separable.blocks("r", 300, fps=30.0, seconds=3.0)
    t = 45
    assert g[t - 3] == g[t] == g[t + 3]


def test_the_block_length_is_the_one_asked_for() -> None:
    g = separable.blocks("r", 200, fps=30.0, seconds=3.0)
    assert len(set(g)) == 3          # 200 frames / 90 -> blocks 0, 1, 2
    assert g.count(g[0]) == 90
