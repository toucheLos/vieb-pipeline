# Pre-registration — human boundary annotation, and the inter-rater ceiling

Registered before `scripts/annot_sample.py`, `vieb/seg/annot.py` or the
annotation tool exist, and **before the sample is drawn**. Source array
`work/clean/<rid>.npz` via `spine.clean`, the **`raw`** arm
(`held_array(pose_unfiltered, missing)`). Digest `198eb14ff258c7f6`.

## 1. Why a human, and why this question and not the obvious one

Step B scored the detector against a planted probe and failed at 12.8%. Step 0
(`PROBE_AUDIT.md`) then measured the probe itself: planted edges sit at the
**60th percentile** of the criterion's own scalar against real firings' **96th**,
and only **19.87%** of planted-edge frames clear their own recording's threshold
at all. The probe is a much harder target than a real boundary, so the 12.8% is
part probe and part criterion and no further statistic separates them.

Ground truth that can be wrong in the other direction is required, and the only
source of it is a person watching a mouse.

**The question put to raters is deliberately not "what is this behaviour".**
Humans disagree about what a behaviour *is* and agree far better about *when
something changed*. That asymmetry is the one pre-registerable prediction this
programme makes about human perception, and asking the easier question is what
makes the answer usable. A rater is given one instruction and no vocabulary:

> **Mark the frame where the animal starts doing something different.**

No labels, no categories, no list to choose from.

## 2. The sample, fixed before it is drawn

**Ten minutes: 60 clips × 10 s, one clip per recording, one recording per
animal.**

**Drawn from the 60 `tune` animals.** Step 2 only *evaluates* a frozen detector,
so any split would serve there — but **Step 3 selects its model class by
human-boundary recall**, which is fitting, and boundaries chosen against report
animals would contaminate Step 5's coverage number on the report split. The
existing three-way split (tune 60 / fit 149 / report 89, seed 0, keyed on
`animal_tag`, never resampled) exists for exactly this. One clip per tune animal
lands the sample at 60 by construction.

**Stratified across arena-position deciles: 6 clips per decile.** Violation rate
rises **3.7× monotone** from the middle of the animal's own cloud to its edge
(`CONCENTRATION.md:75-88`), so a uniform sample under-represents exactly the hard
regime. Edgeness is `concentration.edgeness(held[:, CENTRE])` — ‖(p − median)/IQR‖
in IQR units, per recording, derived from the animal's own centroid cloud because
shapeflow ships no arena definition. A clip's decile is the decile of its
**median** edgeness, taken against the corpus-wide decile edges computed over the
sampled recordings.

**Excluded from the draw:** any window whose abstain fraction exceeds **0.10**.
A rater cannot mark a change they cannot see, and a clip of lost tracking
measures the tracker. Any window crossing a recording boundary — never compute
across a seam.

Seed 0, `seeds.stable_seed`, as everywhere else.

## 3. The clips

**Plain video. No skeleton, no overlay, no marks.** The instruction is about the
animal, not the tracker, and an overlay invites raters to mark tracking failures
instead of behaviour. Per-clip flagged-frame density **is recorded in the
manifest and is not shown**, so boundaries piling up on flagged frames can be
detected afterwards rather than caused beforehand.

**All-intra encoding** (`-g 1 -bf 0`). The finest tolerance band is ±2 frames,
67 ms at 30 fps. Browser seeking is frame-accurate only on keyframes, so an
inter-frame GOP would put the instrument's measurement error at the same scale as
the thing it measures.

Fixed crop from `recur.render.video.crop_box`, which is deliberately a **fixed**
box on the clip's mean centroid: a box that follows the animal stabilises it and
deletes the locomotion.

## 4. Raters, identity, and independence

**Two or three raters.** One rater yields no ceiling and the stage refuses rather
than reporting a detector score against nothing —
`recur/scripts/score_ratings.py:327` already refuses this and the refusal is
inherited, not re-implemented.

**Rater identity is carried automatically** by the tool, not typed in.
**Naivety is per rater per clip** and is derived from which shards a rater has
already produced, following `recur/scripts/raters.py`, never self-reported.

**The page never renders another rater's marks.** Raters cannot converge by
watching each other.

**Independence is a commit protocol.** Each rater's marks are read back, written
to `results/annot/<rater>.json`, and **committed alone** before the next rater's
are opened. That is what makes "committed before either saw the other's"
checkable after the fact rather than asserted.

**"No change in this clip" is an answer and is recorded as one**, distinct from
"not yet rated". If raters find no boundaries where the detector finds none, that
is the continuum result; an instrument that cannot tell an empty clip from an
unrated one throws away the finding it was built to get.

## 5. The ceiling, computed before any detector is scored

**Tolerance-windowed boundary F1 between raters at ±2, ±5, ±10 frames**, with
**greedy one-to-one matching, nearest first**.

One-to-one is not a detail. `scripts/breaks.py:165 _agreement` already does
tolerance matching, but counts unmatched hits — two peaks inside one tolerance
window both score — which inflates recall exactly where a detector is noisiest.
The new matcher refuses to reuse a matched boundary.

F1 is symmetric between two raters, so the pair statistic needs no designated
truth. Cohen's κ is **not** computed: it is undefined for event sets, and there
is no category to agree about.

Aggregated with `recur.boot.animal_interval(how="mean")`, 2000 replicates,
clustered on **animal**, `n_effective` in animals. Clip-level intervals may be
printed beside it only to show how much narrower the wrong method looks.

**This is the benchmark, and it is not 1.0.** A detector matching rater–rater
agreement is at ceiling, not failing.

## 6. The reading, fixed in advance

| outcome | reading |
|---|---|
| raters agree with each other well above chance | the ceiling stands and Step 2 may be scored against it |
| **raters agree poorly with each other** | the question is **ill-posed at this tracking quality**. Report and stop. No detector is scored, because there is nothing to score it against |
| raters agree, and mark **near-zero boundaries** | the continuum reading is supported by human perception, independently of any detector |

"Poorly" is fixed now as **the ±5-frame inter-rater F1 interval including the
chance F1** — the F1 two raters would reach by placing the same number of marks
uniformly at random in the same clips, computed per clip under the same matcher
and reported beside the observed value. That chance level is computed **before**
the observed F1 is read.

## 7. What this stage may not do

It may not adjust the tolerance bands, the clip length, the sample size or the
matcher after seeing an agreement number. It may not drop a rater. It may not
re-draw the sample. Any of those turns a benchmark into a fitted parameter, and
the detector scored against it would be scored against itself.

## 8. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only. Ranked
on the effect-CI lower bound, never on a p-value. The **`raw`** arm asserted by
name — no Wiener array enters this stage, and a low-pass filter manufactures
exactly the smoothness whose breaks are at issue. Windows in seconds, converted
once via `recur.util.frames`. Never compute across a recording boundary.
`mypy --strict` on new modules.
