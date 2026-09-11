r"""Leave-one-bone-out: the only check a wrong corrector cannot pass.

Distortion, high-frequency retention and held-out MDL all measure **magnitude**.
A corrector that moves a keypoint confidently in the wrong direction scores well
on every one of them -- it moves little, it deletes no fast movement, and it may
even predict better, because a plausible pose is easier to model than a broken
one. None of them can fail for the failure that matters.

And the obvious check is circular: a corrector that enforces skull constraints
will report near-zero skull violations by construction. Scoring it on the group
it constrains measures its own premise.

So: **constrain the skull, evaluate the trunk.** The trunk bones are never seen
by the corrector, they share exactly one keypoint with the skull group
(`nose`, via `nose-center`), and a correction that moves the nose towards its true
position should make that bone *more* plausible while a correction that merely
satisfies the skull triangle has no reason to.

The expected effect is **small** -- one shared bone out of six -- and that is
pre-registered. What is not permitted is for it to be absent or negative.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

from . import bones

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]

#: Below this relative fall in held-out violations, the correction is not
#: reaching the bones it never constrained.
MIN_RELATIVE_FALL = 0.0


def held_out_rate(pose: npt.ArrayLike, *, group: Sequence[tuple[int, int]],
                  pairs: Sequence[tuple[int, int]], keep: Sequence[int],
                  l_hat: npt.ArrayLike, eps: float) -> float:
    """Violation rate on a bone group, at a FIXED reference length.

    `l_hat` is passed in rather than refitted. Refitting on the corrected pose
    would let a corrector that shrinks the animal post a lower rate for having
    made it smaller -- the same trap the bakeoff avoids by refitting per arm, but
    inverted: here the yardstick must not move at all, because the two poses being
    compared are the same animal before and after.
    """
    lengths = bones.metric_lengths(pose, group, "raw", pairs=pairs, keep=keep)
    return float(bones.frame_mask(bones.violations(lengths, l_hat, eps)).mean())


def lobo_read(before: float, after: float, *, constrained: str, evaluated: str,
              scored_object: Detail, n_effective: int,
              min_fall: float = MIN_RELATIVE_FALL) -> Read:
    """Did violations fall on the group the corrector never saw?

    This is the gate. A `FAIL` stops the phase: it means the correction is
    satisfying its own constraint rather than moving keypoints towards the
    animal, and no distortion or retention number will show that.
    """
    detail: Detail = {"constrained": constrained, "evaluated": evaluated,
                      "rate_before": before, "rate_after": after,
                      "absolute_fall": before - after,
                      "relative_fall": (before - after) / before if before > 0
                      else float("nan"),
                      "min_relative_fall": min_fall}
    if not (np.isfinite(before) and np.isfinite(after)) or before <= 0:
        return Read("INCONCLUSIVE", scored_object,
                    f"no {evaluated} violations to evaluate against",
                    n_effective=n_effective, detail=detail)
    rel = (before - after) / before
    if rel > min_fall:
        return Read(
            "PASS", scored_object,
            f"constraining {constrained} lowered violations on {evaluated} -- "
            f"bones the corrector never saw -- from {before:.4%} to {after:.4%}, "
            f"a relative fall of {rel:.2%}. The correction reaches geometry it "
            f"was not fitted to, which is the only evidence here that it moves "
            f"keypoints towards the animal rather than towards its own "
            f"constraint surface",
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        f"constraining {constrained} did not lower violations on {evaluated}: "
        f"{before:.4%} to {after:.4%}, a relative change of {rel:+.2%}. The "
        f"corrector satisfies the bones it was given and does nothing for the "
        f"bones it was not, which is what a correction that moves points onto a "
        f"constraint surface rather than towards the animal looks like. "
        f"Distortion and retention cannot show this and MDL would not either",
        n_effective=n_effective, detail=detail)
