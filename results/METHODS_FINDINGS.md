# Methods findings — instruments this project found to be broken

Not deviations. A deviation is a departure from a registered plan;
`DEVIATIONS.md` holds those. These are **instruments that returned a confident
answer while measuring nothing**, and the rule each one leaves behind. They
transfer beyond this project.

---

## M1 — a probe with no negative control is not a probe

**What happened.** Step 2's separability precondition asked whether a linear
probe could tell corpus windows from surrogate windows, on the rule that a null
a probe can pick out is a null that differs from the data in ways unrelated to
the question. It ran on stacked windows, grouped by recording block, held out by
group, and it **passed all four nulls**.

It also returned **`white` — i.i.d. Gaussian noise — as inseparable from mouse
trajectories at AUC 0.496**, which is chance.

**Why it was blind.** `separability` fits a logistic regression, which is linear
in its features. The difference between a smooth trajectory and a rough one lives
in the **second moment of the increments** — a quadratic function of the raw
window that no linear model can form from raw frames. The probe had no access to
the only quantity that mattered.

**What it would have licensed.** Four PASSes, and the entire boundary-rate gate
read as valid. Adding per-channel `log` mean-squared-first-difference features
flipped **all four nulls to FAIL** — `ou` 0.999, `white` 0.989, `phase` 0.800,
`var5` 0.801 — and the gate became uninterpretable in both directions.

**Nothing but the control could have caught it.** Every other null's PASS was
plausible. Review would not have found it; the four numbers looked like a result.

> **RULE.** Every probe must include a null whose answer is known in advance and
> which it is **expected to fail**. A probe that cannot fail its own negative
> control has not been shown to measure anything, and its PASSes are not
> evidence.

A corollary, from the same episode: **check that the probe's features can
express the quantity in question.** "Linear model, quadratic property" is not an
exotic failure — it is the default outcome of feeding raw values to a linear
classifier and asking about variance.

---

## M2 — a precondition can be contradictory with the question it guards

**What happened.** That same precondition, once sharpened, could not be satisfied
by any admissible null. It demands a null **indistinguishable from the data**;
the hypothesis says the data is **piecewise smooth**; a null built to have no
boundaries differs from piecewise-smooth data in its roughness distribution
*precisely because it has none*. Once the probe could see roughness, no
boundary-free null could pass.

**Rule.** Before registering a precondition, ask what would have to be true of an
object that satisfies it. If the answer is "it contains the structure under
test", the precondition is not a guard, it is a contradiction. Recorded in full
as `DEVIATIONS.md` D7.

---

## M3 — a validity gate must not be satisfied by moving only the failing arms

**What happened.** Step 2's PCA validity read (`pca_read`) failed at `d = 192`
for the null arms of one channel group (Spearman ρ 0.977–0.988 against a 0.99
requirement) while the corpus arm passed. Its own FAIL text says *"raise the
component count."*

Raising it **only where it failed** would have embedded the corpus in 192
dimensions and its null in 384. The statistic's threshold θ is taken from the
**null's** own quantile, so two arms in differently-shaped spaces make θ
incomparable for reasons that have nothing to do with the effect.

**Resolution.** Every arm at full dimension — a rotation, which preserves
distances exactly.

> **RULE.** When a validity check fails for one arm of a comparison, the remedy
> applies to **every** arm. A per-arm fix silently changes what is being
> compared.

---

## M4 — name the suspect, then verify it, before believing the diagnosis

**What happened.** `exactness_read` failed on two arms and its text names TF32
as the first suspect — reasonable, since an A100 runs float32 matmuls at 10
mantissa bits by default. TF32 was disabled explicitly
(`allow_tf32 = False`, `matmul_precision = "highest"`) and **the disagreement did
not move by a single digit**: 1.8658e-04 before and after.

The real cause was elsewhere: a near-tie in the **within-animal** nearest
neighbour, whose index the search never returns, so `index_agreement` — which
compares only `i_cross` — could not register it.

**Rule.** A named first suspect in an error message is a hypothesis, not a
diagnosis. Change the suspected cause and check the number moves. And when a
gate covers two quantities, report it **split by quantity**, so a reader can see
which one each downstream number depends on — here `d_cross`, which agreed to
7.5e-06 at worst across all eighteen arms.

---

## M5 — a control that returns nothing is usually testing the control

**What happened, twice, in one ladder.** The planted dose-response floor
recovered nothing at any occupancy, then recovered it backwards.

1. **Planted into the corpus.** The background already scored +4% against the
   null, so every planted cell measured the corpus's own recurrence while the
   planting slowly overwrote it. The ladder ran **monotonically downward** in
   dose. A floor must sit on a background with none of the signal in it.
2. **A per-animal template.** The template was built inside the per-animal loop,
   so every animal received a *different* stereotype — and the statistic is
   **cross-animal** recurrence. The planted signal was invisible to the
   measurement by construction.

Neither was visible in the code. Both were obvious in the output: a
dose-response that decreases with dose, and one that is flat at zero.

> **RULE.** Read a control's *shape*, not just its verdict. A dose-response that
> does not respond to dose is a broken instrument, and it is broken in a way
> that a PASS/FAIL summary hides.

A third item from the same ladder, which is a limit rather than a bug: only
**2.3%** of planted instances became their own segment. The ladder therefore
measures **detector and matcher jointly**, and reports a floor for *planted
smooth blocks* rather than a general sensitivity bound. A control imported from a
window pipeline does not automatically mean the same thing in a boundary
pipeline.

---

## M6 — an interval that excludes its own estimate is the estimator, not the data

Recorded here because it cost five attempts. A symbol-homogeneity statistic was
given a confidence interval by bootstrap, by a reflected basic interval, by
redrawing the null per replicate, and by subsampling — and **four of them
produced an interval that excluded the point estimate they were built around**,
with the gap *growing* with N. The cause was clustering no item-level null
reproduces: a duplicated or omitted animal sends all of its runs to the same side
of the split. Leave-one-animal-out jackknife was the one that worked.

> **RULE.** An interval that does not contain its own estimate is not a tight
> interval. Stop and find the unit of resampling.

---

## M7 — a centred crop is evidence about the tracker, never about location

**What happened.** Ten segments from the widest clump were rendered as contact
sheets — first, middle and last frame, skeleton drawn, cropped 320 px on the
animal — to settle whether a 3.7×-slower state was immobility or a flat-line
tracking artifact. The sheet answered that: the skeleton sits on a plainly
visible animal in every frame, and the animal does not move, including across a
65-second segment.

**And I read a second thing off it that was not there.** Every row looked like an
animal pressed against the arena wall. That mattered: `CONCENTRATION.md` records
tracking failure rising **3.7× monotone** from arena centre to wall, so a
wall-enriched clump would be finding the arena rather than behaviour.

Measured, paired within animal, the clump sits at edgeness **1.69** against
**3.61** for the same animals' other segments — difference **−1.92 [−2.95,
−0.83]**. It is **less than half as far out**. The visual read was not merely
unsupported; it was backwards.

**Why the sheet cannot answer it.** The crop is centred on the animal. The wall
therefore fills the frame whenever the animal is anywhere near it, and it fills
the frame identically at one body-length away as at contact. The image contains
no arena-scale reference, so there is nothing in it from which proximity could be
judged — the information was cropped out before the scorer ever saw it.

> **RULE.** Separate what a rendering can show from what it cannot. A crop
> centred on the subject supports claims about the subject — is the skeleton on
> the animal, is the animal moving — and supports **no** claim about where the
> subject is, because the framing removed the reference frame. Before reading
> anything off an image, ask which measurement the image is standing in for, and
> whether the image could have come out differently had that measurement been
> different.

This is the second time in this programme that the by-eye scorer was corrected by
a number, and the first is recorded in `ADJUDICATION.md`: two frames read as
"rearing against a wall" were teleports, caught by a nine-frame contact sheet. In
that case the sheet was the correction. Here the sheet was the error, and the
measurement was the correction — so "look at the data" is not a rule that
supersedes measurement, only one that complements it.

---

## M8 — normalise in exactly one place, and make the shard say which

**What happened, twice in one session.** M-class failures are supposed to be
learned once. This one recurred within hours of being written down.

**First**, `BEHAVIOUR.md` compared a per-frame RMS in standardised channel units
against a 560-dimensional Euclidean norm already divided by its ambient scale,
and reported their ratio as though it meant something. Withdrawn as
`DEVIATIONS.md` D8; corrected, the two quantities are near-identical rather than
4× apart.

**Then**, in the very stage registered to fix that, `seg_dist.py` divided each
distance by its arm's ambient scale **when writing the shard** — and
`paired_excess`, which exists to do exactly that, divided again. So θ became a
quantile of `d/scale²` while the clump graph compared `d/scale`.

**What it looked like.** Not an error. A clean, plausible, publishable result:
**zero clumps in every arm, 100.00% unassigned**, under both new metrics, with
the nulls at zero too. Read at face value it said *the clumps were entirely an
artifact of time-normalisation* — a strong finding, consistent with the
hypothesis under test, and wrong.

**What caught it.** Not review, and not the verdict, which was internally
coherent. A single diagnostic: θ = 0.0395 against a corpus 1st-percentile
nearest-neighbour distance of 0.1415. **The threshold sat below anything the
data attained**, which no property of a metric explains and only a units error
does.

> **RULE.** A normalisation is applied in **one** place, and the artifact
> records which. A quantity written to disk is raw or normalised, never
> "probably normalised" — and a function whose whole purpose is to normalise
> must be handed raw input, or it will normalise twice and say nothing.

**The general form, and why it is worth a numbered entry.** A pipeline that
divides by a scale in two places produces numbers that are *self-consistent*
within each stage and wrong between them. Nothing crashes; every assertion
passes; the verdict reads fluently. The check that catches it is not a test of
the result but a **sanity check on the threshold against the data it is applied
to** — is this cut placed where any of the distribution actually is?

Add that check wherever a threshold from one distribution is applied to another.

## M9 — a blind panel leaks through whatever the render adds, not only through what it shows

**What happened.** `island_clips.py` renders the island's 361 segments against a
duration-matched control from the continuum, blind, with the arm held in a
separate key. It draws the skeleton and marks **flagged frames** with a red
border, so a viewer can see tracker failure rather than rating it as behaviour.

Each clip is rendered at its segment's own extent **plus half a second either
side** — padding, so a viewer sees the animal enter the state rather than opening
mid-state. The flag mask was applied across the whole rendered range, padding
included.

Measured on the first render:

| arm | clips carrying a flagged frame | flagged frames |
|---|---:|---:|
| control | **64 of 361** | 253 |
| island | **2 of 361** | 2 |

Inside the segments themselves, both arms measure **exactly zero** — a selected
segment has `abstain_frac == 0` and the abstain mask contains the bone-violation
mask. **Every one of those 255 frames lay in the padding.**

So a red border in the lead-in was an **18% cue for "control"** in a panel whose
entire purpose is that a viewer cannot tell the arms apart. Nothing about the
units differed. The *render* differed, because the island's segments happen to
sit in cleaner surrounding video than their duration-matched partners do.

**The rule.** A blind is a property of everything the viewer receives, not only
of the thing being compared. Anything a render **adds** around the unit — padding,
overlays, captions, a poster, a filename — is part of the stimulus and has to be
checked for class information as hard as the unit is. It is not enough that the
quantity under test was matched.

**What it cost, and what it would have cost.** Caught before publication by
counting the marks per arm rather than in total; the fix marks only inside the
segment, and both arms are now zero. Had the total alone been reported — "255
flagged frames across 722 clips" — the asymmetry would have been invisible and
the panel would have shipped with a working cue in it.

**The second-order point.** The corrected overlay never fires. That is worth
saying out loud on the page rather than leaving as an absence, because "no island
clip contains a flagged frame" reads as a finding about freezing and is nothing
of the kind: it is a property of the selection, true of the control too.

## M10 — a free choice inside an instrument is a variable, and an unregistered variable is a confound

**What happened.** The boundary marker let a rater place a mark two ways: with
the video **paused**, reading the frame off `currentTime`, or **while it
played**, reading it off `requestVideoFrameCallback`. The tool offered both, said
nothing about which to use, and recorded which was used per clip — that last part
only because the field was cheap to store, not because anyone had registered it
as a variable.

Two raters did two different things. `hem` paused on all 58 clips. `ce` marked
34 of 58 while the video was playing.

Measured against the other rater's nearest mark:

| how the mark was placed | n marks | median offset |
|---|---:|---:|
| playing (`rvfc`) | 27 | **+9 frames** (300 ms) |
| paused (`currentTime`) | 7 | **−12 frames** |

A mark placed while playing carries visual reaction time **and** a `currentTime`
that is stale by up to a frame interval. A mark placed while paused carries
neither. The finest registered tolerance band was **±2 frames**.

**The rule.** Every degree of freedom an instrument leaves to its operator is a
variable in the experiment, whether or not the registration names it. Registering
the *measurement* is not enough; the registration has to fix the **procedure**,
or the procedure becomes a per-operator choice that is then confounded with
operator identity. The fix is to remove the choice — force a pause on mark — or
to measure the lag and correct it, and either is a new registration.

**What it cost, and what it did not.** It cost nothing here, which is the only
reason it is a methods finding rather than a retraction. The ceiling failed at
±2, ±5 **and** ±10, and a median offset of +9 frames cannot explain a failure at
±10; the spread is ±15 frames and spread is what kills it. The samples are also
small and the choice is perfectly confounded with rater identity, so nothing
here separates "marking while playing is late" from "`ce` marks late".

**Why it is recorded anyway.** Had the ceiling *passed* at ±10 and failed at ±2,
this would have been the first suspect, and the data to check it would have
existed only by luck. The general form is worth more than this instance: an
instrument that permits two methods measures both, and reports their mixture as
one number.

**The near miss.** `frame_source` was stored as a convenience. Had it not been,
the two raters would have differed by an unrecoverable 300 ms and the difference
would have been invisible — indistinguishable from raters who simply disagree.
Recording *how* a measurement was taken, not only what it was, is what made this
finding possible at all.
