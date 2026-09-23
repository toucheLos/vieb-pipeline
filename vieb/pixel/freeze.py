"""An ezTrack-FAMILY freeze score, and the sweep that is published beside it.

`results/PIXEL_PREREGISTRATION.md` §1, §2, §4. **ezTrack is not installed.**
`freeze_mask` is a faithful reimplementation of `Measure_Freezing`
(`FreezeAnalysis/FreezeAnalysis_Functions.py:345`), read from source, including
its backward pass -- and `tests/test_pixel.py` checks it against a literal
line-by-line transcription of that loop rather than against an expectation.

## Why two of ezTrack's three defaults are not used -- M13

`FreezeThresh` is a **count of changed pixels**, so it scales with frame area.
ezTrack's freezing defaults were set on 320x240 videos; this corpus is 640x480,
**exactly 4x the pixels**, so 200 transplanted is 4x too strict. `MinDuration`
is in **frames**, and survives only because both corpora happen to run at 30 fps
-- it is carried in seconds here anyway, because M13 exists precisely because a
constant in samples changes meaning silently.

Both are kept as the `eztrack_default` arm of the sweep, labelled untransplanted,
so the size of the error is a published number rather than an argument.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from recur.read import Read
from recur.util import frames

B1 = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]

#: §4 headline: `FreezeThresh` as a percentile of the recording's OWN `Motion`.
#: An absolute pixel count cannot be shared across a corpus whose arena floor
#: alone spans 0-384 changed pixels per frame.
THRESH_PCT = 25.0
#: §2: registered in SECONDS, never in frames.
MIN_FREEZE_S = 0.5
#: §4, the registered grid. The headline is the centre of each.
CUTOFF_MULTIPLIERS = (1.0, 2.0, 4.0)
THRESH_PCTS = (10.0, 25.0, 40.0)
MIN_FREEZE_SECONDS = (0.25, 0.5, 1.0)
#: ezTrack's literal published defaults, reported as a fourth arm and NEVER
#: adopted: `mt_cutoff` in grey levels, `FreezeThresh` in changed pixels,
#: `MinDuration` in FRAMES.
EZTRACK_DEFAULTS = {"mt_cutoff": 10.0, "freeze_thresh": 200.0,
                    "min_duration_frames": 15}


def idx_list(mask: npt.ArrayLike) -> list[int]:
    """Indices of a boolean mask as plain `int`s, not numpy scalars."""
    return [int(v) for v in np.flatnonzero(np.asarray(mask, dtype=bool))]


def freeze_mask(motion: npt.ArrayLike, thresh: float,
                min_frames: int) -> B1:
    """``(T,)`` True where the animal is scored as freezing.

    Faithful to `Measure_Freezing`: frames below `thresh`, required to run for
    `min_frames` consecutively, then a backward pass that extends each accepted
    run back over its own ramp-up -- the frames that were already below
    threshold but had not yet accumulated the duration. Dropping that pass
    truncates every bout by `min_frames - 1` and would bias the score downward
    by a fixed amount that looks like a real effect.
    """
    m = np.asarray(motion, dtype=np.float64)
    n = int(m.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(int(min_frames), 1)
    below = m < float(thresh)
    # ezTrack's CumThresh, vectorised: the run length of `below` so far, reset
    # to 0 at every frame that is not below. Index 0 is always 0, as in source.
    # `CumThresh[0]` is 0 in the source whatever `Motion[0]` is, so the FIRST
    # frame of a series can never be scored freezing. Reproduced deliberately:
    # "fixing" it would shift every freeze fraction by one frame per recording,
    # in one direction, which is exactly the shape of a spurious effect.
    run = np.zeros(n, dtype=np.int64)
    c = 0
    for i in range(1, n):
        c = c + 1 if bool(below[i]) else 0
        run[i] = c
    acc = run >= k
    if not acc.any():
        return np.zeros(n, dtype=bool)
    # The backward pass. Each accepted frame carries its run back over the
    # `k - 1` frames that built it up.
    out = acc.copy()
    for j in idx_list(acc):
        out[max(0, j - (k - 1)):j] = True
    return np.asarray(out, dtype=bool)


def freeze_fraction(motion: npt.ArrayLike, thresh: float,
                    min_frames: int) -> float:
    """The fraction of frames scored freezing. A RATE, so session length cancels.

    §7: Context A sessions run ~17% longer than Context B. A count would carry
    that straight into the contrast; a fraction does not, and no count is
    published anywhere in this stage.
    """
    if not np.isfinite(thresh):
        return float("nan")
    f = freeze_mask(motion, thresh, min_frames)
    return float(f.mean()) if f.size else float("nan")


def thresh_of(motion: npt.ArrayLike, pct: float = THRESH_PCT) -> float:
    """§4: `FreezeThresh` as the `pct`-th percentile of this recording's Motion.

    **Returns NaN when the percentile is 0, and that is not an edge case.**
    `Measure_Freezing` tests `Motion < FreezeThresh` strictly, so a threshold of
    zero can never be met and the arm would silently report 0% freezing as
    though it had measured it. At a cutoff derived from the arena floor the
    Motion distribution carries a large atom at exactly zero -- most frames have
    no pixel changing by more than twice the arena's 99.99th percentile -- so a
    low percentile of it IS zero. Measured on this corpus: degenerate on 62% of
    recordings at the registered headline. See `DEVIATIONS.md` D17.
    """
    m = np.asarray(motion, dtype=np.float64)
    m = m[np.isfinite(m)]
    if not m.size:
        return float("nan")
    t = float(np.percentile(m, float(pct)))
    return t if t > 0 else float("nan")


def sweep_arms(fps: float) -> list[dict[str, Any]]:
    """The registered 27-point grid, plus ezTrack's untransplanted defaults.

    The grid is generated here so that the arms cannot drift between the floor
    stage and the freeze stage, and so `headline` is a property of the arm
    rather than a string matched later.
    """
    arms: list[dict[str, Any]] = []
    for mult in CUTOFF_MULTIPLIERS:
        for pct in THRESH_PCTS:
            for secs in MIN_FREEZE_SECONDS:
                arms.append({
                    "name": f"m{mult:g}_p{pct:g}_s{secs:g}",
                    "derived": True,
                    "cutoff_multiplier": float(mult),
                    "thresh_pct": float(pct),
                    "min_freeze_s": float(secs),
                    "min_frames": int(frames(secs, fps)),
                    "headline": (mult == 2.0 and pct == THRESH_PCT
                                 and secs == MIN_FREEZE_S)})
    arms.append({
        "name": "eztrack_default", "derived": False, "headline": False,
        "cutoff_multiplier": None,
        "mt_cutoff": EZTRACK_DEFAULTS["mt_cutoff"],
        "freeze_thresh": EZTRACK_DEFAULTS["freeze_thresh"],
        "min_frames": int(EZTRACK_DEFAULTS["min_duration_frames"]),
        "note": ("ezTrack's published defaults applied verbatim. "
                 "UNTRANSPLANTED: freeze_thresh is a pixel COUNT set on "
                 "320x240 video and this corpus is 640x480, 4x the pixels; "
                 "min_duration is in FRAMES. Reported to show the size of the "
                 "error, never adopted")})
    return arms


def floor_read(per_recording: list[dict[str, Any]], *,
               scored_object: dict[str, Any], n_effective: int) -> Read:
    """§3: does the arena floor stand, and does it differ between contexts?

    A floor that differs by context is not a defect -- §7 predicts it, because
    A and B are visually different arenas -- but it must be VISIBLE, because a
    sensor-level difference on the axis the contrast uses is the confound.
    """
    ok = [r for r in per_recording if r.get("usable")]
    if len(ok) < 2:
        return Read("NOT_A_RESULT", scored_object,
                    f"only {len(ok)} recordings cleared the arena-pixel "
                    f"minimum; there is no floor to report",
                    n_effective=n_effective)
    by = {c: np.asarray([r["mt_cutoff"] for r in ok
                         if r["context"] == c], dtype=np.float64)
          for c in ("A", "B")}
    have = {c: v for c, v in by.items() if v.size}
    span = (min(r["mt_cutoff"] for r in ok), max(r["mt_cutoff"] for r in ok))
    parts = [f"{c} median {np.median(v):.2f} mean {v.mean():.2f} (n={v.size})"
             for c, v in sorted(have.items())]
    # MEDIAN, not mean. The distribution is heavy-tailed -- a handful of
    # recordings sit an order of magnitude above the bulk -- and on this corpus
    # the mean gap (0.86) and the median gap (3.00) point to different
    # conclusions because the outliers are Context B while the bulk is Context A.
    # §7's refusal is a statement about the typical recording, so it takes the
    # robust statistic.
    gap = (abs(float(np.median(have["A"]) - np.median(have["B"])))
           if len(have) == 2 else float("nan"))
    gap_mean = (abs(float(have["A"].mean() - have["B"].mean()))
                if len(have) == 2 else float("nan"))
    return Read("PASS", scored_object,
                (f"the arena sets its own cutoff on all {len(ok)} recordings: "
                 f"mt_cutoff spans {span[0]:.2f} to {span[1]:.2f} grey levels "
                 f"against ezTrack's published default of "
                 f"{EZTRACK_DEFAULTS['mt_cutoff']:g}, by context "
                 + ", ".join(parts)
                 + (f", a MEDIAN difference of {gap:.2f} grey levels "
                    f"(mean difference {gap_mean:.2f}, pulled by a heavy tail)"
                    if np.isfinite(gap) else "")
                 + f". {len(per_recording) - len(ok)} recordings refused for "
                 f"too few arena pixel-pairs"),
                n_effective=n_effective,
                detail={"span": [float(span[0]), float(span[1])],
                        "context_gap_mean": float(gap_mean),
                        "by_context": {c: {"mean": float(v.mean()),
                                           "median": float(np.median(v)),
                                           "sd": float(v.std(ddof=1))
                                           if v.size > 1 else 0.0,
                                           "n": int(v.size)}
                                       for c, v in sorted(have.items())},
                        "context_gap": float(gap),
                        "n_usable": len(ok),
                        "n_refused": len(per_recording) - len(ok),
                        "eztrack_default": EZTRACK_DEFAULTS["mt_cutoff"]})
