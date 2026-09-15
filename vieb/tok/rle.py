r"""Step 3. Frames to runs, because frame mass and run mass are different things.

Run-length encoding is not a convenience here and it is not optional. The bias it
removes is measured and large: on the labeller this branch inherits, label 1 holds
**41.70% of frames but 10.07% of runs**, and label 0 holds **6.98% of frames but
17.66% of runs**. A model scored per frame is scored mostly on how long the animal
froze. Everything after this module operates on transitions.

## What this module is

A thin wrapper. `recur.label.nonparametric.run_length_encode` is already
seam-safe, and `recur.seq.motif.sequences` already splits on abstain. Neither is
reimplemented; a second implementation of "never cross a recording boundary" is a
second thing that can be wrong, and this project has paid for that before. What
is added here is the **distribution** and the **stop condition** -- the parts the
ladder needs and the two upstream functions do not provide.

## Abstain splits, and is never dropped

Dropping abstain runs turns *"did X, went unlabelled, did X again"* into `XX` --
a self-repeat that run-length encoding is supposed to make impossible. Measured
at **32.3%** of adjacent pairs on the 76%-coverage arm. The transition matrix has
a zero diagonal precisely because `AA` cannot occur, so every manufactured `XX`
is scored at expected count zero and sorts to the top of an effect-ranked table.

## The stop condition, and what it retires

If the median run is three frames or fewer -- a tenth of a second -- the alphabet
is cutting the trajectory faster than the animal changes what it is doing, and
the runs are quantization noise rather than behaviour. Earlier code on this
project produced median runs of 0.10 s on windows that shared 89 of their 90
input frames.

That retires **the alphabet**, not the sweep. It is a statement about one point
on the N grid, and a coarser N on the same grid may not have it. Only if every N
trips it is it a finding about the representation.

## No minimum-duration rule, ever

Not a stickiness prior, not a merge of short runs, not a median filter on the
symbol stream. Every one of those imports persistence, and importing persistence
is what retracted the dwell result: kappa = 1e6 manufactured non-geometric dwell
on white noise. Flicker is a measurement, and the hazard model is where it gets
interpreted.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
from recur.label import nonparametric as npm
from recur.label.nonparametric import ABSTAIN
from recur.read import Read
from recur.seq import motif
from recur.util import describe
from vieb.qc.runlen import BUCKETS

I64 = npt.NDArray[np.int64]
I32 = npt.NDArray[np.int32]
Detail = dict[str, Any]
Runs = dict[str, Any]

#: The typed boundary, declared once. recur predates annotation discipline and
#: `mypy.ini` follows its imports silently, so its fully unannotated functions
#: read as untyped here. Named casts rather than per-call-site ignores, so the
#: surface this module trusts from recur is visible in one place.
_run_length_encode = cast("Callable[..., Runs]", npm.run_length_encode)
_sequences = cast("Callable[..., dict[Any, list[I64]]]", motif.sequences)

__all__ = ["ABSTAIN", "BUCKETS", "MEDIAN_RUN_MIN", "distribution", "encode",
           "frame_versus_run_mass", "self_transitions", "sequences",
           "runlen_read"]

#: Frames. At or below this the alphabet is fitting quantization noise. Three
#: frames is 0.1 s at 30 fps, inherited from the brief and not selected against
#: any outcome on this corpus.
MEDIAN_RUN_MIN = 3


def encode(labels: npt.ArrayLike, bounds: npt.ArrayLike | None = None) -> Runs:
    """`(code, duration, recording)` per run. Delegated, never reimplemented."""
    return dict(_run_length_encode(labels, bounds))


def sequences(runs: Mapping[str, Any], group_of_recording: Sequence[Any],
              *, gap: int | None = ABSTAIN) -> dict[Any, list[I64]]:
    """Collapsed sequences per group, split at seams and at abstain gaps.

    `gap=None` is available only so a caller with no abstain bin can show that
    splitting changes nothing for it. It is not an option for this branch.
    """
    return dict(_sequences(runs, group_of_recording, gap=gap))


def self_transitions(runs: Mapping[str, Any]) -> int:
    """Adjacent runs sharing a code within one recording. Must be zero.

    Zero by construction after `run_length_encode`, which is exactly why it is
    worth counting: a non-zero here means a caller has concatenated across a
    seam or spliced out an abstain run, and both failures are invisible in the
    downstream statistics they corrupt.
    """
    code = np.asarray(runs["code"], dtype=np.int64)
    rec = np.asarray(runs["recording"], dtype=np.int64)
    if code.size < 2:
        return 0
    same = (code[1:] == code[:-1]) & (rec[1:] == rec[:-1])
    return int(same.sum())


def distribution(runs: Mapping[str, Any], fps: float, *,
                 exclude_abstain: bool = True) -> Detail:
    """Run-length distribution in frames and in seconds.

    Abstain runs are excluded from the behavioural distribution by default and
    reported separately. Folding them in would let a long tracking dropout read
    as a long behaviour, which is the one reading this whole abstain path exists
    to prevent.
    """
    code = np.asarray(runs["code"], dtype=np.int64)
    dur = np.asarray(runs["duration"], dtype=np.int64)
    is_ab = code == ABSTAIN
    keep = ~is_ab if exclude_abstain else np.ones_like(is_ab)
    d = dur[keep].astype(np.float64)
    out: Detail = {
        "n_runs": int(keep.sum()),
        "n_abstain_runs": int(is_ab.sum()),
        "abstain_run_share": float(is_ab.mean()) if is_ab.size else float("nan"),
        "abstain_frame_share": (float(dur[is_ab].sum() / dur.sum())
                                if dur.sum() else float("nan")),
        "frames": describe(d),
        "seconds": describe(d / float(fps)),
        "self_transitions": self_transitions(runs),
        "fps": float(fps),
    }
    out["median_frames"] = float(out["frames"].get("median", float("nan")))
    out["median_seconds"] = float(out["seconds"].get("median", float("nan")))
    # Share of RUNS and share of FRAMES at or below each bucket. They are
    # different numbers and the second is the one that says whether the stream
    # is mostly noise: half the runs being one frame long is survivable if those
    # runs hold two percent of the frames, and fatal if they hold half.
    # Thresholds inherited from `vieb.qc.runlen.BUCKETS` rather than chosen here.
    total_f = float(d.sum())
    out["buckets"] = [
        {"at_most": int(b),
         "share_runs": float((d <= b).mean()) if d.size else float("nan"),
         "share_frames": (float(d[d <= b].sum() / total_f)
                          if total_f > 0 else float("nan"))}
        for b in BUCKETS]
    return out


def frame_versus_run_mass(runs: Mapping[str, Any], n_states: int) -> Detail:
    """Per-symbol share of frames against share of runs, and the worst gap.

    The number that makes RLE non-optional. A symbol holding a large share of
    frames and a small share of runs is a long dwell; the reverse is a flicker.
    Scoring per frame rewards a model for predicting the first.
    """
    code = np.asarray(runs["code"], dtype=np.int64)
    dur = np.asarray(runs["duration"], dtype=np.int64)
    keep = code != ABSTAIN
    c, d = code[keep], dur[keep]
    frames = np.bincount(c, weights=d.astype(np.float64),
                         minlength=int(n_states))
    counts = np.bincount(c, minlength=int(n_states)).astype(np.float64)
    fs = frames / frames.sum() if frames.sum() else frames
    rs = counts / counts.sum() if counts.sum() else counts
    gap = fs - rs
    worst = int(np.argmax(np.abs(gap))) if gap.size else -1
    return {
        "n_states": int(n_states),
        "frame_share": fs.tolist(),
        "run_share": rs.tolist(),
        "worst_symbol": worst,
        "worst_frame_share": float(fs[worst]) if worst >= 0 else float("nan"),
        "worst_run_share": float(rs[worst]) if worst >= 0 else float("nan"),
        "max_abs_gap": float(np.abs(gap).max()) if gap.size else float("nan"),
    }


def runlen_read(dist: Mapping[str, Any], *, scored_object: Detail,
                n_effective: int, minimum: int = MEDIAN_RUN_MIN) -> Read:
    """Is this alphabet cutting behaviour, or cutting noise?

    A `FAIL` here retires **this alphabet**, not the sweep, and the reason says
    so -- a coarser N on the same grid may not trip it. Only every N tripping it
    would be a finding about the representation rather than about the grid.
    """
    med = float(dist["median_frames"])
    sec = float(dist["median_seconds"])
    detail: Detail = {k: dist[k] for k in
                      ("n_runs", "n_abstain_runs", "abstain_run_share",
                       "abstain_frame_share", "median_frames",
                       "median_seconds", "self_transitions")}
    detail["p75_frames"] = dist["frames"].get("p75")
    detail["p99_frames"] = dist["frames"].get("p99")

    if int(dist["self_transitions"]):
        return Read("FAIL", scored_object,
                    f"{int(dist['self_transitions'])} adjacent runs share a "
                    f"code within one recording. That is impossible after "
                    f"run-length encoding, so a caller has concatenated across "
                    f"a seam or spliced out an abstain run",
                    n_effective=n_effective, degenerate=True, detail=detail)
    if not np.isfinite(med):
        return Read("NOT_A_RESULT", scored_object,
                    "no non-abstain runs: there is no run-length distribution "
                    "to report on this alphabet",
                    n_effective=n_effective, degenerate=True, detail=detail)
    tail = (f"median run is {med:.1f} frames ({sec:.3f} s), p75 "
            f"{float(dist['frames'].get('p75', float('nan'))):.1f}, over "
            f"{int(dist['n_runs']):,} runs, with "
            f"{float(dist['abstain_frame_share']):.2%} of frames in abstain runs")
    if med <= minimum:
        return Read("FAIL", scored_object,
                    f"this alphabet is cutting the trajectory faster than the "
                    f"animal changes what it is doing -- {tail}, at or below "
                    f"the {minimum}-frame floor. The runs are quantization "
                    f"noise rather than behaviour. This retires THIS alphabet "
                    f"and not the sweep: a coarser N on the same grid may not "
                    f"trip it, and only every N tripping it would be a finding "
                    f"about the representation",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"the run-length distribution is usable: {tail}, above the "
                f"{minimum}-frame floor, and no adjacent runs share a code",
                n_effective=n_effective, detail=detail)
