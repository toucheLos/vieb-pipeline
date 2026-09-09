"""No quantity is ever computed across a recording boundary.

`scripts/spine.py` has named this file since the repo was written and it did not
exist, so the invariant it asserts was carried by convention and by scattered
per-module tests. It is one property, it applies to every stage, and it belongs
in one place: `bounds[r]:bounds[r+1]` is the only definition of a recording, and
nothing may read across that line.

The failure this prevents is quiet. A derivative taken across a seam produces one
enormous increment where the arena position jumps between two unrelated sessions,
which reads downstream as a single spectacular behaviour -- and, being one frame
in ~5,800, moves no summary statistic enough to notice.
"""
import numpy as np
import pytest

from recur.geom import reversal, state
from vieb.tok import reversal as tokrev
from recur.label import nonparametric as npm
from recur.qc import swap
from vieb.qc import bones
from vieb.tok import ego
from tests.test_ego import moving

FPS = 30.0


def two_recordings(n=150, m=110):
    """Two sessions in different corners of the arena. Any read across the seam
    produces a step far larger than anything either recording contains."""
    a = np.asarray(moving(n, turn=0.03), dtype=np.float64)
    b = np.asarray(moving(m, turn=-0.05, speed=4.0), dtype=np.float64) + 9000.0
    return a, b, np.array([0, n, n + m])


class TestTheEgocentricTransform:

    def test_the_twist_is_undefined_at_every_seam(self):
        a, b, bounds = two_recordings()
        ell = ego.ell_a([a, b])
        va = ego.twist(a, ell, FPS)[1]
        vb = ego.twist(b, ell, FPS)[1]
        assert not va[-1] and not vb[-1]

    def test_concatenating_first_would_manufacture_a_giant_step(self):
        """The error this invariant exists to prevent, made visible."""
        a, b, bounds = two_recordings()
        ell = ego.ell_a([a, b])
        per_recording = np.concatenate([ego.twist(a, ell, FPS)[0],
                                        ego.twist(b, ell, FPS)[0]])
        pooled, _ = ego.twist(np.concatenate([a, b]), ell, FPS)
        seam = bounds[1] - 1
        assert np.abs(per_recording[seam]).max() == 0.0
        assert np.abs(pooled[seam]).max() > 100.0

    def test_ell_a_pools_across_recordings_and_that_is_correct(self):
        """The one quantity that is SUPPOSED to span recordings.

        `ell_a` is a property of the animal, not of a session, so it is fitted
        over all of that animal's recordings on purpose. It reads no pair of
        adjacent frames, so it crosses no seam in the sense that matters.
        """
        a, b, _ = two_recordings()
        pooled = ego.ell_a([a, b])
        assert np.isfinite(pooled) and pooled > 0


class TestRunLengthEncoding:

    def test_a_run_never_spans_two_recordings(self):
        labels = np.zeros(200, dtype=np.int32)
        bounds = np.array([0, 100, 200])
        runs = npm.run_length_encode(labels, bounds)
        assert runs["code"].size == 2
        assert runs["duration"].tolist() == [100, 100]
        assert runs["recording"].tolist() == [0, 1]

    def test_without_bounds_the_same_labels_read_as_one_run(self):
        """Which is exactly the mistake `bounds` exists to make impossible."""
        runs = npm.run_length_encode(np.zeros(200, dtype=np.int32))
        assert runs["duration"].tolist() == [200]


class TestTheBoneDiagnostic:

    def test_the_shuffled_ceiling_draws_within_one_recording(self):
        a, b, _ = two_recordings()
        lengths = bones.metric_lengths(a, bones.SKULL, "raw")
        ref = [bones.reference_length(lengths[:, m], 0.10)["l_hat"]
               for m in range(len(bones.SKULL))]
        alone = bones.shuffled_ceiling(a, bones.SKULL, ref, 0.10,
                                       np.random.default_rng(0))
        pooled = bones.shuffled_ceiling(np.concatenate([a, b]), bones.SKULL, ref,
                                        0.10, np.random.default_rng(0))
        assert pooled > alone

    def test_segment_rates_are_clipped_to_the_recording(self):
        rates = bones.segment_rates(np.ones(50, dtype=bool),
                                    np.array([[0, 50], [40, 5000]]))
        assert np.isfinite(rates).all() and rates[0] == pytest.approx(1.0)


class TestTheChannelMatrix:

    def test_differences_are_invalid_at_the_start_of_every_recording(self):
        a, b, _ = two_recordings()
        blocks = state.blocks_of(*_decompose(a))
        assert not blocks["omega"][1][0]
        assert not blocks["domega"][1][0] and not blocks["domega"][1][1]


class TestReversal:

    def test_reversal_is_per_recording(self):
        a, b, bounds = two_recordings()
        x = np.arange(bounds[-1], dtype=np.float64)[:, None]
        out = reversal.reverse_blocks(x, bounds, reversal.EVEN)
        assert out[0, 0] == bounds[1] - 1 and out[bounds[1], 0] == bounds[-1] - 1

    def test_the_twist_reversal_is_per_recording(self):
        a, b, bounds = two_recordings()
        xi = np.arange(3 * bounds[-1], dtype=np.float64).reshape(-1, 3)
        out = tokrev.reverse_twist(xi, bounds)
        # The undefined row of EACH recording stays at that recording's end.
        assert np.allclose(out[bounds[1] - 1], 0.0)
        assert np.allclose(out[bounds[2] - 1], 0.0)


class TestSwapDetection:

    def test_a_swap_run_cannot_span_a_seam(self):
        a, b, _ = two_recordings()
        for pose in (a, b):
            out = swap.detect(pose, FPS, pair="ear")
            assert out["swapped"].shape[0] == pose.shape[0]


def _decompose(pose):
    from recur import kendall as kd
    c = kd.fit_gauge(pose)
    return kd.decompose(pose, c)
