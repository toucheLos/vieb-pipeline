r"""Corruptions with known truth, drawn from what this corpus actually suffers.

## Why not Gaussian noise

A Gaussian perturbation benchmark would be easy to write and would settle
nothing, because **Gaussian noise is the thing a smoother is optimal against**.
Scoring a rolling median on additive white noise asks whether a median filter is
a median filter. The errors in this corpus are not that: they are teleports,
sustained parks, left/right swaps and dropouts, and each of them has a shape and
a rate that some earlier phase of this project already measured.

So every parameter below is a citation, not a choice:

| corruption | parameter | measured in |
|---|---|---|
| teleport | displacement quantiles, px | Phase D, `disposition.json` suspect moves |
| teleport | which keypoint | `CLEANING.md` per-keypoint Viterbi reassignment |
| teleport | 1-3 frames | Phase D's envelope, from the violating-run distribution |
| park | 4-30 frames, median ~9 | `CLEANING.md`, runs surviving `median_0.50` |
| swap | 1-8 frames, median ~3 | `recur/qc/swap.py`, `MAX_RUN_S = 0.25` |
| dropout | rate | `EFFECT.md`, 3.385% of keypoint-frames abandoned |

## The distinction the whole benchmark turns on

A **teleport** leaves and comes back. A **park** does not: the landmark sits off
the animal for a third of a second or more, and `CLEANING.md` already argues that
no temporal filter can reach it, because a parked landmark is temporally smooth
and satisfies both a median and a motion prior.

Phase E tried to test that with a smoothness statistic and failed, because a
15-frame median smooths *across* a 9-frame park and the statistic could not tell
that from a repair. Here the park has a **known true position**, so an arm that
smooths across it scores no repair at all. That is the same claim with an
instrument that can actually falsify it.

## Contract

Corruption happens **inside clean segments of one recording** and the arm is then
run over the whole recording, so it gets the temporal context it would have in
production. Nothing here crosses a seam: the function is given one recording.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

KINDS: tuple[str, ...] = ("teleport", "park", "swap", "dropout")

#: Phase D measured the displacement its projection applied to a suspect
#: keypoint, over 1,149 report recordings. These are that distribution's recorded
#: quantiles, in pixels, and `sample_displacement` interpolates its inverse CDF
#: rather than fitting a parametric family to them.
DISPLACEMENT_Q: tuple[tuple[float, float], ...] = (
    (0.10, 3.63), (0.25, 11.11), (0.50, 22.15), (0.75, 42.55),
    (0.90, 92.23), (0.99, 230.64), (1.00, 355.55),
)

#: Which landmark goes wrong, from `CLEANING.md`'s per-keypoint Viterbi
#: reassignment rates. Normalised at use. The centre is 24x safer than the nose
#: and a benchmark that corrupted them uniformly would be testing a corpus that
#: does not exist.
KEYPOINT_WEIGHT: tuple[float, ...] = (
    0.1385, 0.2555, 0.5315, 0.0224, 0.4212, 0.3872, 0.1881,
)

#: Per-corruption incidence, as a share of keypoint-frames inside the pool,
#: matched to the corpus-wide measured rates so the benchmark is neither harder
#: nor easier than the real thing. Pre-registered.
RATES: dict[str, float] = {
    "teleport": 0.012, "park": 0.004, "swap": 0.003, "dropout": 0.034,
}

TELEPORT_FRAMES = (1, 3)
PARK_FRAMES = (4, 30)
SWAP_FRAMES = (1, 8)
#: The two pairs `recur/qc/swap.py` finds: ears and hips. A swap is bilateral and
#: inventing a nose/tail swap would be inventing an error mode.
SWAP_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (4, 5))


def sample_displacement(rng: np.random.Generator, n: int) -> F64:
    """`n` teleport magnitudes in px, from Phase D's measured inverse CDF.

    Linear interpolation between the recorded quantiles, which is a distribution
    with the right median, the right tail and no parametric assumption. Values
    below the p10 are drawn down to a tenth of it rather than to zero, because a
    zero-magnitude teleport is not a corruption and would silently dilute the
    rate.
    """
    q = np.array([p for p, _ in DISPLACEMENT_Q], dtype=np.float64)
    v = np.array([x for _, x in DISPLACEMENT_Q], dtype=np.float64)
    u = rng.random(int(n))
    out = np.interp(u, q, v, left=v[0] * 0.1)
    return np.asarray(out, dtype=np.float64)


def sample_keypoint(rng: np.random.Generator, n: int,
                    weights: Sequence[float] = KEYPOINT_WEIGHT) -> I64:
    """Which landmark is corrupted, weighted by how often each really goes wrong."""
    w = np.asarray(weights, dtype=np.float64)
    return np.asarray(rng.choice(w.size, size=int(n), p=w / w.sum()),
                      dtype=np.int64)


def _realised(before: F64, after: F64, a: int, b: int,
              keypoints: Sequence[int], ell: float) -> float:
    """The error the corruption ACTUALLY creates, in body lengths.

    Not the displacement that was drawn. A park holds one position while the
    animal keeps moving, so its realised error grows across the window and is
    larger than the vector that placed it. Recording the drawn magnitude instead
    would make `raw` -- which repairs nothing -- score a 2.5% error reduction it
    did not achieve.
    """
    d = [float(np.linalg.norm(after[t, k] - before[t, k]))
         for k in keypoints for t in range(a, b)]
    d = [x for x in d if np.isfinite(x)]
    return float(np.mean(d) / ell) if d else float("nan")


def _place(rng: np.random.Generator, seg: Sequence[int], length: int,
           taken: BOOL) -> tuple[int, int] | None:
    """A free window of `length` frames inside one segment, or None.

    Corruptions never overlap. Two corruptions on the same keypoint-frame would
    make "the truth" ambiguous and the recovery unscoreable.
    """
    lo, hi = int(seg[0]), int(seg[1])
    if hi - lo < length:
        return None
    for _ in range(16):
        start = int(rng.integers(lo, hi - length + 1))
        if not taken[start:start + length].any():
            return start, start + length
    return None


def corrupt(pose: npt.ArrayLike, segs: npt.ArrayLike,
            rng: np.random.Generator, *, ell: float,
            rates: dict[str, float] | None = None) -> tuple[F64, Detail]:
    """``(corrupted, detail)`` for ONE recording. `pose` is not modified.

    `detail["events"]` lists every corruption with its kind, frames, keypoint and
    true magnitude, so recovery can be scored per kind. `detail["mask"]` is the
    ``(T, K)`` boolean of corrupted keypoint-frames -- the complement of it,
    inside the segments, is where **damage** is measured.

    Dropout is written as NaN rather than as a sentinel coordinate. shapeflow's
    own gap policy interpolates short gaps and abandons long ones, and an arm is
    entitled to do either; a `(0, 0)` sentinel would read downstream as the
    animal teleporting to the origin, which is a different corruption.
    """
    p = np.asarray(pose, dtype=np.float64)
    out = p.copy()
    t_n, k_n = p.shape[0], p.shape[1]
    r = dict(RATES if rates is None else rates)
    mask = np.zeros((t_n, k_n), dtype=bool)
    taken = np.zeros(t_n, dtype=bool)
    segs_a = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    events: list[Detail] = []
    if segs_a.size == 0 or not np.isfinite(ell) or ell <= 0:
        return out, {"events": [], "mask": mask, "n_pool_frames": 0,
                     "why": "no clean segment in this recording"}

    pool_frames = int((segs_a[:, 1] - segs_a[:, 0]).sum())
    # The rates are shares of KEYPOINT-frames, so the target is counted in
    # keypoint-frames and events are placed until it is met. Counting events
    # instead would make the realised rate depend on the span of each kind --
    # a park is up to 30 frames and a teleport up to 3, so the same event budget
    # would corrupt ten times as much of the pool for one as for the other.
    pool_kpf = pool_frames * k_n
    target = {k: r.get(k, 0.0) * pool_kpf for k in KINDS}
    got = {k: 0 for k in KINDS}
    order = list(KINDS)
    rng.shuffle(order)                      # so one kind cannot always win a slot

    for kind in order:
        span = {"teleport": TELEPORT_FRAMES, "park": PARK_FRAMES,
                "swap": SWAP_FRAMES, "dropout": TELEPORT_FRAMES}[kind]
        attempts = 0
        cap = int(target[kind]) * 8 + 64
        while got[kind] < target[kind] and attempts < cap:
            attempts += 1
            seg = segs_a[rng.integers(0, segs_a.shape[0])]
            length = int(rng.integers(span[0], span[1] + 1))
            win = _place(rng, seg, length, taken)
            if win is None:
                continue
            a, b = win
            if kind == "swap":
                i, j = SWAP_PAIRS[int(rng.integers(0, len(SWAP_PAIRS)))]
                tmp = out[a:b, i].copy()
                out[a:b, i] = out[a:b, j]
                out[a:b, j] = tmp
                mask[a:b, i] = True
                mask[a:b, j] = True
                mag = _realised(p, out, a, b, [int(i), int(j)], ell)
                events.append({"kind": kind, "a": a, "b": b,
                               "keypoints": [int(i), int(j)],
                               "magnitude_bl": mag})
                got[kind] += (b - a) * 2
            else:
                k = int(sample_keypoint(rng, 1)[0])
                if kind == "dropout":
                    out[a:b, k] = np.nan
                    mag = float("nan")
                else:
                    d = float(sample_displacement(rng, 1)[0])
                    theta = float(rng.random() * 2.0 * np.pi)
                    vec = np.array([np.cos(theta), np.sin(theta)]) * d
                    if kind == "teleport":
                        out[a:b, k] = p[a:b, k] + vec
                    else:
                        # A park holds ONE wrong position, it does not track the
                        # animal offset by a constant. That is what makes it
                        # temporally smooth and unreachable by a filter.
                        out[a:b, k] = p[a, k] + vec
                    mag = _realised(p, out, a, b, [k], ell)
                mask[a:b, k] = True
                events.append({"kind": kind, "a": a, "b": b,
                               "keypoints": [k], "magnitude_bl": mag})
                got[kind] += b - a
            taken[a:b] = True

    return out, {
        "events": events, "mask": mask,
        "n_pool_frames": pool_frames,
        "n_events": {k: int(sum(1 for e in events if e["kind"] == k))
                     for k in KINDS},
        "realised_rate": {
            k: float(sum((e["b"] - e["a"]) * len(e["keypoints"])
                         for e in events if e["kind"] == k)
                     / max(1, pool_frames * k_n))
            for k in KINDS},
        "requested_rate": r,
    }
