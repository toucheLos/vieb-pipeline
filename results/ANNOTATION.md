# Step 1b — the inter-rater ceiling does not stand

Registered in `results/ANNOTATION_PREREGISTRATION.md`, committed before the
sample was drawn. Two raters, `hem` and `ce`, each rated **58 of 58** clips.
Source array `work/clean/<rid>.npz` via `spine.clean`, the **`raw`** arm
(`held_array(pose_unfiltered, missing)`), `tune` split, 58 animals.
Digest `198eb14ff258c7f6`.

## The verdict

> **FAIL at ±2, ±5 and ±10 frames.** The ±5 interval includes its own chance
> level, which is the condition §6 fixed in advance as "raters agree poorly".

§6 names what follows, and it is not negotiable after the fact:

> the question is **ill-posed at this tracking quality**. Report and stop. No
> detector is scored, because there is nothing to score it against.

**Step 2 is blocked for this round.** The frozen detector is not scored against
these marks, at any tolerance, in any form. `scripts/annot_ceiling.py` exits
non-zero on this data and that is the correct behaviour.

## What the number is made of, before the number

`annot.prf` scores two empty sets as **F1 = 1.0** — deliberately, because two
raters who both say "nothing changed" agree — and `chance_f1` does the same. A
both-empty clip therefore contributes the maximum to the observed value **and**
to chance, and cannot separate them. Quoting a headline F1 without saying how
much of it is that overstates agreement.

| class | n clips | contributes |
|---|---:|---|
| both raters called it empty | **14** | 1.0 to observed *and* to chance |
| exactly one rater marked | **24** | 0.0 |
| **both raters marked** | **20** | the only clips where the matcher does work |

14 of 58 is **0.241**, which is essentially the whole ±2 chance level of 0.2510.

On the 20 informative clips alone:

| tol | F1, both-marked only | chance |
|---|---|---|
| **±2** | **0.0583 [0.0000, 0.1500]** | 0.0280 |
| ±5 | 0.1833 [0.0500, 0.3500] | 0.0692 |
| ±10 | 0.2417 [0.1000, 0.4000] | 0.1150 |

## The registered ceiling

Animal-level bootstrap, 2,000 replicates, `how="mean"`, n = 58 animals. Chance
is the F1 two raters reach placing the same mark counts uniformly at random in
the same clips, under the same one-to-one matcher, computed before the observed
value was read.

| tol | inter-rater F1 | chance F1 | verdict |
|---|---|---|---|
| **±2** | **0.2615 [0.1551, 0.3736]** | 0.2510 | **FAIL** |
| **±5** | **0.3046 [0.1954, 0.4224]** | 0.2652 | **FAIL** |
| **±10** | **0.3247 [0.2155, 0.4425]** | 0.2810 | **FAIL** |

Every interval contains its own chance level. The decomposition above is why
0.2615 is not four times better than 0.0583 — it is the same measurement with
14 uninformative clips left in, as the registration specified.

## Coarse agreement, and no fine agreement — descriptive

Read off **one** nearest-neighbour distribution (each of `ce`'s marks to `hem`'s
nearest, signed), not a second matching pass: §7 forbids scoring at a tolerance
chosen after an agreement number has been seen, and a nearest-neighbour distance
is not a score.

| | share of marks |
|---|---|
| within **one second** of the other rater's nearest | **0.853 [0.750, 0.964]** |
| within **two frames** | **0.088 [0.000, 0.233]** |

**The raters are looking at the same events and cannot place them.** That is a
different failure from raters looking at different events, and the registered F1
cannot tell them apart. It is also the single most useful thing this round
produced.

## Rate, against the detector — descriptive, and not a score

No recall and no precision was computed. These are two marginals side by side.

| | boundaries per second |
|---|---|
| **humans**, 58 clips, 116 clip-ratings | **0.109 [0.087, 0.134]** — one per 9.2 s |
| **detector of record**, `shape`, deriv 0.133 s, degree 3 | **0.443** (`BREAKS.md`) |

**The detector cuts about four times more often than a person marks a change** —
roughly 3.3× to 5.1× taking the human interval. Two caveats that bound it:

* **The two are not the same construct.** A human mark is a perceived change; a
  detector boundary is a threshold crossing on an acceleration mismatch. Nothing
  here says one should equal the other.
* **Different populations.** 0.443/s is corpus-wide over 298 animals; 0.109/s is
  58 ten-second clips from `tune` animals, stratified by arena decile.

> **A correction, recorded rather than quietly fixed.** An earlier reading of
> this comparison put the detector at 1.07/s and the gap at "an order of
> magnitude", by inverting the **median** segment duration of 0.933 s. That is
> not a rate: the duration distribution is heavily right-skewed (mean 2.171 s
> against median 0.933 s), so the reciprocal of its median overstates the rate
> by more than a factor of two. `BREAKS.md` already carried the directly
> measured rate and it is the one used here.

## What this result is, and what it is not

**It is a result about timescale and modality.** Raters were asked to resolve
**67 ms** — ±2 frames at 30 fps — from an overhead view of an animal whose
relevant movement is a few pixels. The ceiling failed because that is not a
question human vision answers, and the coarse/fine split above is the direct
evidence for that reading.

**It is not a verdict on the continuum question.** Nothing here licenses any
claim about whether the detector's boundaries are real, whether the 98.1%
unassigned mass is behaviour or noise, or whether a vocabulary exists. Human
annotation was asked to serve as an adjudicator for that question and this round
establishes that it cannot, at this timescale, on this footage.

**It is not a verdict on the raters.** The instrument behaved: each rater met the
clips in an order seeded from their own name, and only **2 of 58** clips sat in
the same position for both against a chance expectation of about 1. Marks spread
across clip position in both raters. "Something happened" tracks arena position
in the expected direction — both-empty clips sit at mean decile **2.79**, toward
the centre of the animal's own cloud; both-marked clips at **5.15**, toward the
wall.

## The protocol gap, which the registration did not close

The marker permitted marking **while paused** (`currentTime`) and **while
playing** (`rvfc`) and specified neither. `hem` paused on all 58 clips; `ce`
marked 34 while playing.

| how the mark was placed | n marks | median offset |
|---|---:|---:|
| playing (`rvfc`) | 27 | **+9 frames** (300 ms) |
| paused (`currentTime`) | 7 | **−12 frames** |

A mark placed while playing carries visual reaction time **and** a `currentTime`
that is stale by up to a frame interval. A mark placed while paused carries
neither.

**This is not the explanation.** The samples are small, the choice is perfectly
confounded with rater identity, and correcting a +9 median would not rescue ±2
agreement — the spread is ±15 frames and spread is what kills it. It is recorded
because an unregistered free choice inside an instrument is a variable, and a
variable nobody registered is a confound. `METHODS_FINDINGS.md` M10.

## What is owed

**A round with a ceiling that holds, if boundary annotation is attempted again.**
It would need its own registration, new raters (the clips are not burned — only
these two raters' marks are), the paused-versus-playing choice forced, and
tolerance bands derived from *measured* human precision rather than from the
detector's ±2 convention. Using this round to design the next is legitimate;
re-scoring this round's marks at a band chosen now is not.

**It is not the next thing.** The coarse/fine split says the fine scale is not a
question human vision answers, so a second round at the same timescale would buy
another failed ceiling. The open question — whether the detector's boundaries sit
above the tracking noise floor — is answerable without humans, by measuring the
floor.

## Provenance

`results/annot_ceiling.json` carries every number above, the full pair-row table
at all three tolerances, the offset distribution, and both raters' orders.
`results/annot/annot_hem.json` and `results/annot/annot_ce.json` were each
**committed alone**, in export order, before this document existed.

Independence is architectural rather than a protocol promise checked after the
fact: the marker keeps each rater's marks in their own browser's `localStorage`
under a per-rater key, has no shared store, and never renders another rater's
marks. Cross-reading was structurally impossible.

The four descriptive reads — the decomposition at each tolerance, the offset
distribution, and the frame-source split — return **`NOT_A_RESULT`** by
construction and carry no verdict. They describe the registered ceiling. They do
not restate or replace it.
