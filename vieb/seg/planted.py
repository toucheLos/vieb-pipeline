r"""A planted dose-response in ego space, so every null result says what it
would have detected.

`recur.recurrence.surrogate.make_template` and `plant` are the design this
follows and **not** the code it calls: both are hard-coupled to a `(T, K, 2)`
pose array -- `make_template` centres on `stack[:, :1, :, :]` and `plant` reads
`t, k, _ = a.shape` and applies a random heading. The ego representation is
`(T, C)`: egocentric shape plus SE(2) *increments*, with no absolute position
and no heading to randomise. So the three properties that matter are carried
over and the array handling is rewritten.

**The three properties, each with the failure it prevents:**

1. **The template is the average of real windows**, not an invented waveform.
   Ledger 23: a control once planted its structure in the band the pipeline
   deletes, and so measured the filter rather than the detector.
2. **Instances are stereotyped, not identical** -- a per-instance perturbation,
   because an exact repeat is recoverable by any method and would overstate the
   floor.
3. **The instance count is rounded with a coin, never with `round()`.** A
   recording holds few template-lengths, so integer rounding quantises the low
   end brutally: measured upstream, 0.25% rounded to zero instances while 0.5%
   and 1.0% both rounded to one, giving two cells that were meant to differ by
   a factor of two an identical realised occupancy. A dose-response whose dose
   does not move is not a dose-response.

**The realised fraction is what the floor is reported against.** The request is
not the object.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["OCCUPANCIES", "ego_plant", "ego_template"]

#: Q1's own ladder, including the 1% cell it reports a floor at.
OCCUPANCIES: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)


def ego_template(x: npt.ArrayLike, w: int, rng: np.random.Generator, *,
                 n_avg: int = 40, valid: npt.ArrayLike | None = None) -> F64:
    """A stereotyped motion template: the mean of `n_avg` real ego windows.

    No centring step, unlike the pose version. Pose windows had to be centred on
    their first frame or averaging would blur the motion away by averaging arena
    positions; ego windows carry no arena position, so centring would remove
    real shape instead.
    """
    a = np.asarray(x, dtype=np.float64)
    w = int(w)
    if a.shape[0] <= w:
        raise ValueError(f"cannot cut a {w}-frame template from {a.shape[0]} frames")
    ok = (np.ones(a.shape[0], dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool))
    # Only starts whose whole window is usable; otherwise the template averages
    # in the gap policy's interpolants.
    cum = np.concatenate([[0], np.cumsum((~ok).astype(np.int64))])
    starts = np.flatnonzero((cum[w:] - cum[:-w]) == 0)
    if starts.size < n_avg:
        raise ValueError(f"only {starts.size} clean windows of {w} frames; "
                         f"a template averaged over fewer than {n_avg} is one "
                         f"animal's particular movement, not a stereotype")
    idx = rng.choice(starts, size=int(n_avg), replace=False)
    stack = np.stack([a[int(i):int(i) + w] for i in idx])
    return np.asarray(stack.mean(axis=0), dtype=np.float64)


def ego_plant(x: npt.ArrayLike, template: npt.ArrayLike,
              rng: np.random.Generator, *, bounds: npt.ArrayLike,
              occupancy: float, jitter: float = 0.05, fade: int = 3,
              blocked: npt.ArrayLike | None = None) -> tuple[F64, BOOL, Detail]:
    """Inject `template` at `occupancy`, crossfaded, never across a seam.

    Placement is per recording, so no instance straddles a recording boundary --
    the invariant every stage in this project asserts by test.
    """
    a = np.array(np.asarray(x, dtype=np.float64), copy=True)
    tpl = np.asarray(template, dtype=np.float64)
    w = int(tpl.shape[0])
    b = np.asarray(bounds, dtype=np.int64)
    planted = np.zeros(a.shape[0], dtype=bool)
    bad = (np.zeros(a.shape[0], dtype=bool) if blocked is None
           else np.asarray(blocked, dtype=bool))
    cum = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
    scale = float(np.std(a, axis=0).mean())
    n_placed = 0
    for r in range(b.shape[0] - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        span = hi - lo
        if span <= w:
            continue
        want = float(occupancy) * span / w
        n = int(want) + int(rng.random() < (want - int(want)))
        for _ in range(n):
            placed = False
            for _try in range(40):
                s_ = int(rng.integers(lo, hi - w))
                if int(cum[s_ + w] - cum[s_]) != 0 or planted[s_:s_ + w].any():
                    continue
                inst = tpl + rng.normal(0.0, jitter * scale, size=tpl.shape)
                f = int(min(fade, w // 2))
                if f > 0:
                    ramp = np.linspace(0.0, 1.0, f + 2)[1:-1][:, None]
                    inst[:f] = (1 - ramp) * a[s_:s_ + f] + ramp * inst[:f]
                    inst[-f:] = ramp[::-1] * a[s_ + w - f:s_ + w] \
                        + (1 - ramp[::-1]) * inst[-f:]
                a[s_:s_ + w] = inst
                planted[s_:s_ + w] = True
                n_placed += 1
                placed = True
                break
            if not placed:
                continue
    meta: Detail = {"requested_occupancy": float(occupancy),
                    "realized_occupancy": float(planted.mean()),
                    "n_instances": int(n_placed), "template_frames": w,
                    "note": ("the floor is reported against the REALIZED "
                             "fraction; the request is not the object")}
    return a, planted, meta
