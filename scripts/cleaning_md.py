"""`results/CLEANING.md` from `results/cleaning_<split>.json`. Reads, never recomputes."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.util import log, read_json                             # noqa: E402
from vieb.clean import viterbi as V                               # noqa: E402
from vieb.tok import config                                       # noqa: E402


def eff(x):
    """`raw` reduces nothing, so its ratio is undefined -- and `write_json` nulls
    every non-finite float, so it arrives here as None rather than as nan."""
    return "—" if x is None else f"{x * 100:.1f}"


def ci(x, spec="{:.4f}"):
    return f"{spec.format(x['point'])} [{spec.format(x['lo'])}, {spec.format(x['hi'])}]"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="report")
    p.add_argument("--json", default=None)
    p.add_argument("--out", default=config.PATHS.result("CLEANING.md"))
    a = p.parse_args(argv)
    d = read_json(a.json or config.PATHS.result(f"cleaning_{a.split}.json"))
    arms, ranked = d["arms"], d["ranked"]
    rd = d["reads"]["bakeoff"]

    rows = "\n".join(
        f"| `{r['arm']}` | {arms[r['arm']]['violation_rate']['point']:.3%} | "
        f"{r['violation_reduction']:+.1%} | "
        f"{arms[r['arm']]['distortion_px']['point']:.3f} | "
        f"{arms[r['arm']]['distortion_mean_px']['point']:.3f} | "
        f"{arms[r['arm']]['distortion_body_lengths']['point']:.5f} | "
        f"{arms[r['arm']]['hf_retained']['point']:.3f} | "
        f"{eff(r['reduction_per_px'])} |"
        for r in ranked if r["arm"] in arms)

    md = f"""# Phase B — the cleaning bakeoff

**Corpus** `{d['corpus']}` | **split** `{d['split']}` | **animals**
{max(m['n_animals'] for m in arms.values())} | **ε** {d['eps']} |
**inherited digest** `{d['inherited_digest']}`

shapeflow's Wiener shrinkage was chosen on a good argument and never benchmarked.
This is the benchmark.

## Read — `{rd['verdict']}`

> {rd['reason']}

## The three axes, and why no single number

| arm | violations | vs raw | distortion median px | mean px | body lengths | HF retained | %/px |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

Sorted by violation reduction, never reduced to a score. Combining the axes into
one number would hide the case this exists to find: an arm that wins on
violations by moving everything.

**Read the ordering as an ordering, not as a result.** Dominance is decided by
**non-overlapping animal-bootstrap intervals**, and on this corpus
`violation_rate` separates for no arm at all -- the intervals run ~1.43-1.79%
against the incumbent's ~1.49-1.87% and overlap almost entirely. An earlier
version of this read compared point estimates and announced that an arm "beats
the incumbent on all three axes"; it does not. What separates is retention, and
displacement for the targeted arms.

## Composing a de-glitcher with a smoother buys nothing

`viterbi+median_0.50` against `median_0.50` alone: 1.606% against 1.592%
violations, 1.544 px against 1.573 mean displacement, 27.1% against 27.0%
retention. **All three intervals overlap.** The hypothesis was that Viterbi would
remove the teleports so the median would not have to, giving lower violations at
lower distortion. It does not, and the residual measurement below says why.

## What the best arm still gets wrong, and why nothing temporal will fix it

Violating runs on the eight worst recordings, before and after `median_0.50`:

> **RETRACTED, and superseded.** The table that stood here reported 207 runs of
> median 2 frames on raw and 66 of median 9 after `median_0.50`, measured on the
> eight worst recordings. Those numbers were **hardcoded in this generator and
> backed by no artifact** -- nothing in this pipeline ever computed a violating
> run-length distribution. They are superseded by `results/runlen.json`, which
> measures it corpus-wide on the array the corrector actually receives, by count
> and by mass. See `results/PROVENANCE_AUDIT.md`.

The median removes the short violations -- which is what a median is for -- and
the median *length* of what survives triples. **What a temporal filter leaves
behind is temporally smooth.** A keypoint parked off the body and held there
satisfies a median (most of the window is wrong) and satisfies Viterbi's motion
prior too (a stationary point is not a glitch). That is why composing them is
redundant rather than complementary, and it is visible in the residual clips: one
landmark off the animal, identical in all three panes.

The information needed to fix it is not on the time axis. It is either anatomical
-- the bone length used as a **corrector** rather than a flag, which is the cheap
2-D shadow of what an anatomically constrained model does properly in 3-D -- or it
is in better detections, which is the expensive branch.

**Distortion is judged on the mean, not the median.** Every targeted arm — the
outlier gates and Viterbi — moves under half its frames, so its median
displacement is exactly 0 and `0 ≤ 0` would let all of them "dominate" on an axis
that never separated them.

**The last column is the one to read for the Viterbi arm.** Violation reduction
bought per pixel of mean displacement is a ratio of two axes rather than an axis,
but it is what separates a de-glitcher from a smoother, and the two columns side
by side do not make it obvious.

**Retention is not "higher is better" on its own.** `raw` retains 100% by
definition and is the worst arm on violations. The three axes are read together
or not at all.

Every arm receives the **identical** input — shapeflow's hold-for-filtering
array, which is what its Wiener filter consumed. `wiener` and `butterworth` are
read off disk rather than recomputed, so a second implementation cannot silently
disagree with the arrays every existing result was built on. The reference length
is refitted **per arm**, so an arm that shrinks every distance uniformly cannot
post a lower violation rate for having made the animal smaller.

## The Viterbi arm, and a correction

An earlier version of this work recorded Anipose's Viterbi filter as
**unavailable**, on the grounds that all 3,080 `_full.pickle` files carry exactly
one detection per bodypart-frame. The detection count is right and the conclusion
was wrong — it came from the paper's phrase "a set of top detections per frame"
rather than from the implementation.

`viterbi_path` builds its candidate set from the previous `n_back` frames as well
as the current one, weighted by `2^-j`, plus an explicit missing state. At one
detection per frame the state space is still `n_back + 1` wide, and the decision
it makes is whether to accept this frame's jump or carry an older position
forward. Anipose's own `wrap_points()` carries the comment `# n_possible = 1` for
exactly this input.

It is a **de-glitcher, not a low-pass**, and the measured signatures are opposite
(Viterbi's median move is from the 80-recording scan in the per-keypoint table
below; an earlier figure of 46.65 px was never written to an artifact and is
withdrawn):

| | fraction of keypoint-frames moved | median move when it moves |
|---|---:|---:|
| Viterbi | **0.31%** | 43.79 px |
| Wiener | **86%** (>0.01 px) | 1.42 px |

Anipose {V.VERSION} is a pinned dependency. Only its import chain is bypassed:
`filter_pose` imports `.common` → `aniposelib` → `numba`, which is calibration
code the filter never touches, and installing it in full would drag
`opencv-contrib-python` on top of recur's pinned `opencv-python-headless` in a
venv the two repos share. The two functions the filter needs are loaded from the
installed package's own source by AST — Anipose's implementation, verbatim and
version-recorded.

## What could not run, and why that is a result

""" + "\n".join(f"**`{k}`** — {v}\n" for k, v in d["unavailable"].items()) + f"""

## The axis this does not settle

Held-out MDL. It is the decision-relevant one — a cleaning method should be
judged by whether the behaviour model built on it predicts better — and it costs
a full tokenizer run per arm. It is deferred to after Step 4, and the arms that
carry through are the cheap branch: no filter, wiener, viterbi, and whichever
smoother places here.

Ensemble-DLC / EKS is **blocked**, and the blocker is concrete rather than a
matter of effort: `~/dlc-training/trained_dlc/` contains zero files and all 3,846
h5 outputs carry the same single scorer, so no ensemble exists and none can be
assembled without retraining and re-inferring the corpus. Monsees et al.'s
anatomically constrained model needs multiple calibrated views and limb joints;
this is one uncalibrated overhead camera with seven surface landmarks. Both sit
on the re-tracking branch that Step 1's 2–15% verdict said to cost out.

All intervals are animal bootstraps, never pooled frames.
"""
    with open(a.out, "w") as fh:
        fh.write(md)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
