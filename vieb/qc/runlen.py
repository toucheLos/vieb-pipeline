r"""How long is a violating run, and where does the violating mass live?

This had never been measured. `runs_of` was used to *build* Phase D's envelope
and to count abstain runs, but no script in this pipeline ever recorded the
distribution of violating run lengths -- so every run-length figure in every
published document here was a hardcoded literal in a Markdown generator or prose
in a docstring, traceable to nothing.

That matters because the figures were load-bearing. Phase D's envelope
(`MAX_CORRECT_FRAMES = 3`) was registered against "median violating run is 1-2
frames, p75 = 3"; the argument for abstention being the outer envelope was made
with "a run of 9 frames is 300 ms"; and the injection benchmark's
`PARK_FRAMES = (4, 30)` cites "runs surviving `median_0.50`". None of those three
numbers could be reproduced.

## The one that was measured on the wrong thing

The 9-frame figure is **post-`median_0.50`, on the eight worst recordings**, and
`CLEANING.md` labels it as such. But the corrector runs on the array *before* any
smoother, so a residual distribution was being used to justify the shape of an
operator applied to a pre-filter distribution.

And the direction of the error is knowable in advance, which is why this module
reports per arm rather than pooled: a 15-frame median cannot change a run of 8
frames or longer, and annihilates any run of 7 or fewer. So filtering **truncates
the distribution below 8 and lengthens nothing**, and a median rising from 2 to 9
is the arithmetic of that truncation rather than a property of the errors. The
`<= 7` bucket below exists to make that testable instead of argued.

## Count and mass are different questions and disagree here

Both are reported because they support opposite designs:

* **By count**, the median run length calibrates a correction threshold: if most
  runs are 1-2 frames, most *events* are glitches and minimal projection is the
  right operator.
* **By mass**, the share of violating keypoint-frames sitting in long runs decides
  whether abstention is needed at all: a threshold that reaches most runs may
  still reach a minority of the violating frames.

A distribution that is short by count and long by mass -- few long runs carrying
most of the frames -- justifies correction *and* abstention together, which is the
envelope Phase D built. A distribution that is short by both inverts that, and
minimal projection alone would be the right answer.

Nothing in this module decides which. It measures.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.qc.swap import runs_of
from recur.read import Read

from vieb.checks import assert_share, assert_unit

F64 = npt.NDArray[np.float64]
BOOL = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

#: Mass buckets, in frames. `3` is Phase D's envelope; `7` is `median_0.50`'s
#: cliff -- a 15-frame median is unchanged unless more than 7 of its samples are
#: corrupted, so a filter can reach the `<= 7` buckets and nothing above them.
BUCKETS: tuple[int, ...] = (1, 3, 7)

#: Below this many violating runs in a recording the quantiles are noise, and the
#: recording contributes its counts to the pooled totals but not its own summary.
MIN_RUNS = 5


def run_lengths(mask: npt.ArrayLike) -> I64:
    """``(n,)`` lengths of the contiguous True runs in one recording's mask.

    Seam-safe by contract: `runs_of` takes one recording and has no seam
    argument, so a run can never span two recordings.
    """
    runs = runs_of(np.asarray(mask, dtype=bool))
    if runs.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(runs[:, 1] - runs[:, 0], dtype=np.int64)


def distribution(lengths: npt.ArrayLike, *,
                 buckets: Sequence[int] = BUCKETS) -> Detail:
    """Count and mass summaries of a set of run lengths.

    `mass` is in violating keypoint-frames, i.e. weighted by run length -- the
    quantity that decides whether a correction threshold reaches the errors, as
    opposed to whether it reaches the events.
    """
    v = np.asarray(lengths, dtype=np.int64)
    v = v[v > 0]
    n_runs = int(v.size)
    total = int(v.sum())
    out: Detail = {
        "n_runs": n_runs,
        "n_violating_frames": total,
        "median_run": float(np.median(v)) if n_runs else float("nan"),
        "p75_run": float(np.percentile(v, 75)) if n_runs else float("nan"),
        "p90_run": float(np.percentile(v, 90)) if n_runs else float("nan"),
        "max_run": int(v.max()) if n_runs else 0,
        "mean_run": float(v.mean()) if n_runs else float("nan"),
    }
    for b in buckets:
        le = v <= int(b)
        out[f"frac_runs_le_{b}"] = float(le.mean()) if n_runs else float("nan")
        out[f"frac_mass_le_{b}"] = (float(v[le].sum() / total)
                                    if total else float("nan"))
    top = int(buckets[-1])
    out[f"frac_mass_gt_{top}"] = (float(v[v > top].sum() / total)
                                  if total else float("nan"))
    if total:
        # The mass at or below the widest bucket and the mass above it are a
        # partition of the violating frames, so they must come to exactly 1.
        # Nothing checked this before, and the same class of arithmetic slip
        # put 1.115 into the MDL duration charge.
        assert_unit(np.array([out[f"frac_mass_le_{top}"]
                              + out[f"frac_mass_gt_{top}"]]),
                    name="runlen mass at-or-below plus above the widest bucket")
        assert_share([out[f"frac_runs_le_{b}"] for b in buckets],
                     name="runlen frac_runs_le")
    return out


def pool(per_recording: Sequence[Mapping[str, Any]]) -> Detail:
    """Pool per-recording length arrays into one distribution.

    Pooling the *lengths* rather than averaging the per-recording medians,
    because a median of medians is not a median and the recordings differ by an
    order of magnitude in how many runs they contain.
    """
    lengths = np.concatenate(
        [np.asarray(r["lengths"], dtype=np.int64) for r in per_recording]
        or [np.zeros(0, dtype=np.int64)])
    return distribution(lengths)


def envelope_read(dist: Mapping[str, Any], *, scored_object: Detail,
                  n_effective: int, max_correct: int = 3) -> Read:
    """Does the measured distribution support the envelope Phase D registered?

    Phase D registered `MAX_CORRECT_FRAMES = 3` against "median violating run is
    1-2 frames, p75 = 3", and argued for abstention as the outer envelope on the
    grounds that the mass sits in long runs. Both halves are checked here, and
    they can disagree -- which is the informative outcome, not a failure.
    """
    median = float(dist.get("median_run", float("nan")))
    p75 = float(dist.get("p75_run", float("nan")))
    mass_inside = float(dist.get(f"frac_mass_le_{max_correct}", float("nan")))
    detail: Detail = {**dict(dist), "max_correct_frames": max_correct}

    if not np.isfinite(median) or not np.isfinite(mass_inside):
        return Read("NOT_A_RESULT", scored_object,
                    "no violating runs were found, so there is no distribution "
                    "to calibrate an envelope against",
                    n_effective=n_effective, degenerate=True, detail=detail)

    threshold_ok = p75 <= float(max_correct) + 1e-9
    if threshold_ok and mass_inside < 0.5:
        return Read(
            "PASS", scored_object,
            f"violating runs are short by count and long by mass: median "
            f"{median:.0f} frames and p75 {p75:.0f}, so a {max_correct}-frame "
            f"envelope reaches most runs, but those runs carry only "
            f"{mass_inside:.1%} of violating keypoint-frames. Correction and "
            f"abstention are both needed, which is the envelope Phase D built, "
            f"and this is the first measurement that supports it",
            n_effective=n_effective, detail=detail)
    if threshold_ok:
        return Read(
            "FAIL", scored_object,
            f"violating runs are short by count AND by mass: median {median:.0f} "
            f"frames, p75 {p75:.0f}, and runs of {max_correct} frames or fewer "
            f"carry {mass_inside:.1%} of violating keypoint-frames. Most of the "
            f"error is reachable by minimal projection, so abstention as the "
            f"outer envelope was calibrated against a truncation artifact rather "
            f"than against the error distribution",
            n_effective=n_effective, detail=detail)
    return Read(
        "GRID_LIMITED", scored_object,
        f"p75 is {p75:.0f} frames, not {max_correct}, so the registered envelope "
        f"threshold does not match the measured distribution. Runs of "
        f"{max_correct} frames or fewer carry {mass_inside:.1%} of violating "
        f"keypoint-frames. The threshold was registered against an unsourced "
        f"figure and this is the first measurement of the quantity it names",
        n_effective=n_effective, detail=detail)


def truncation_read(raw: Mapping[str, Any], filtered: Mapping[str, Any], *,
                    scored_object: Detail, n_effective: int,
                    cliff: int = 7) -> Read:
    """Is the filtered distribution the raw one truncated below the cliff?

    The claim under test is that a 15-frame median lengthens nothing and simply
    deletes every run at or below its cliff. If so, the filtered distribution's
    mass above the cliff should match the raw distribution's mass above the
    cliff, in absolute frames -- the survivors are untouched, and only the short
    runs are gone.
    """
    key = f"frac_mass_gt_{cliff}"
    raw_total = float(raw.get("n_violating_frames", 0.0))
    filt_total = float(filtered.get("n_violating_frames", 0.0))
    raw_above = raw_total * float(raw.get(key, float("nan")))
    filt_above = filt_total * float(filtered.get(key, float("nan")))
    detail: Detail = {
        "cliff_frames": cliff,
        "raw_mass_above_cliff": raw_above,
        "filtered_mass_above_cliff": filt_above,
        "ratio": (filt_above / raw_above) if raw_above > 0 else float("nan"),
        "raw_median_run": raw.get("median_run"),
        "filtered_median_run": filtered.get("median_run"),
    }
    if not np.isfinite(raw_above) or raw_above <= 0:
        return Read("NOT_A_RESULT", scored_object,
                    "the raw distribution has no mass above the cliff, so there "
                    "is nothing for truncation to preserve",
                    n_effective=n_effective, degenerate=True, detail=detail)
    ratio = filt_above / raw_above
    if 0.75 <= ratio <= 1.25:
        return Read(
            "PASS", scored_object,
            f"the filter preserved {ratio:.2f} of the violating mass in runs "
            f"longer than {cliff} frames while the median run length went from "
            f"{raw.get('median_run'):.0f} to {filtered.get('median_run'):.0f}. "
            f"The rise in median length is the arithmetic of deleting every run "
            f"at or below the cliff, not a property of the errors: nothing "
            f"lengthened",
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        f"the filter changed the violating mass above its own {cliff}-frame "
        f"cliff by a factor of {ratio:.2f}, which pure truncation cannot do. "
        f"The rise in median run length is therefore not only a truncation "
        f"artifact and the filter is doing something to the long runs as well",
        n_effective=n_effective, detail=detail)
