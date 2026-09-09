r"""Time reversal for the SE(2) twist, composed onto recur's reversal audit.

recur's `geom.reversal` owns the general machinery -- the parity constants, the
per-recording `reverse_blocks`, and seven checks on the chain rule and the delay
stack. This module adds the two facts that are specific to the egocentric
representation and belong to this repo, and it **composes** rather than edits:
`audit()` calls recur's, keeps every one of its checks, and appends three.

## Every twist component is odd, `v_x` included

The intuition that a forward velocity behaves like `centroid_speed` is wrong.
`centroid_speed` is a **magnitude**, and a magnitude kills the sign flip. `v_x`
is signed. Formally, if the step is :math:`g_t^{-1}g_{t+1} = \exp(\xi_t)` then
the reversed trajectory's step is the inverse group element,
:math:`g_{t+1}^{-1}g_t = \exp(-\xi_t)`, so the whole twist negates.

## And it shifts by one frame

This is the part that is easy to lose, and losing it is the same class of error
as forgetting to flip the lag order inside a delay-embedded row -- structural
rather than a sign, and therefore visible downstream.

A twist lives on the **interval** between two frames and is stored at the left
endpoint, so :math:`\xi_{T-1}` does not exist. Under :math:`t \to T-1-t` the
interval :math:`[t, t+1]` becomes :math:`[T-2-t, T-1-t]`, so

.. math:: \tilde\xi_k = -\xi_{T-2-k}, \quad k = 0 \ldots T-2

-- index :math:`T-2-k`, not :math:`T-1-k`. `naive_reverse_twist` drops the shift
so a test can show the difference is detectable rather than asserting the correct
one against itself.

The parity table is duplicated here as plain constants rather than imported from
`tok.ego`, so that nothing in this module depends on the channel layout; a test
asserts the two stay consistent with `ego.CHANNELS`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from recur.geom.reversal import EVEN, ODD, audit as recur_audit, reverse_blocks

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]

#: The egocentric representation: 14 body-frame coordinates, then the twist.
EGO_PARITY: dict[str, int] = {"s": EVEN, "v_x": ODD, "v_y": ODD, "omega": ODD}


def reverse_twist(xi: npt.ArrayLike, bounds: npt.ArrayLike) -> F64:
    """Reverse an SE(2) twist series: negate, AND shift by one frame."""
    x = np.asarray(xi, dtype=np.float64)
    out = np.zeros_like(x)
    b = np.asarray(bounds, dtype=np.int64)
    for r in range(b.size - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        if hi - lo < 2:
            continue
        out[lo:hi - 1] = -x[lo:hi - 1][::-1]
    return out


def naive_reverse_twist(xi: npt.ArrayLike, bounds: npt.ArrayLike) -> F64:
    """The WRONG version: negate and reverse, without the one-frame shift.

    It moves the undefined last row to the front and pairs every twist with the
    wrong two frames.
    """
    return np.asarray(
        reverse_blocks(np.asarray(xi, dtype=np.float64), bounds, parity=ODD),
        dtype=np.float64)


def audit(fps: float = 30.0, *, atol: float = 1e-9) -> Detail:
    """recur's reversal audit, plus three checks for the twist.

    Composed rather than edited: recur's seven checks run unchanged and their
    results are carried through, so a regression in either repo shows up here.
    """
    base = recur_audit(fps=fps, atol=atol)
    findings = list(base["checks"])

    def check(name: str, ok: bool, detail: str = "") -> None:
        findings.append({"check": name, "ok": bool(ok), "detail": detail})

    xi = np.stack([np.arange(6.0), 10 + np.arange(6.0), 20 + np.arange(6.0)], 1)
    xi[-1] = 0.0                                    # the undefined last row
    b6 = np.array([0, 6])
    rv = reverse_twist(xi, b6)
    check("the twist negates under reversal",
          bool(np.allclose(rv[:5], -xi[:5][::-1], atol=atol)),
          "every component, v_x included -- it is signed, not a magnitude")
    check("the reversed twist keeps its undefined row last",
          bool(np.allclose(rv[-1], 0.0, atol=atol)),
          "a twist stored at the left endpoint has no last row either way")
    check("NOT shifting the twist by one frame is detectable",
          not bool(np.allclose(rv, naive_reverse_twist(xi, b6), atol=1e-6)),
          "the naive version moves the undefined row to the front and pairs "
          "every twist with the wrong two frames")

    ok = all(f["ok"] for f in findings)
    return {"verdict": "PASS" if ok else "FAIL",
            "n_checks": len(findings),
            "n_failed": int(sum(not f["ok"] for f in findings)),
            "n_checks_inherited": int(base["n_checks"]),
            "inherited_verdict": base["verdict"],
            "checks": findings,
            "note": ("recur's reversal audit, run unchanged, plus three checks "
                     "for the SE(2) twist. The twist is odd in every component "
                     "and reversing it shifts by one frame, because it lives on "
                     "the interval between frames rather than on a frame.")}
