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

## M11 — match a control on what defines the object, at the level it is defined

**What happened.** `CONTEXT.md` found the island's occupancy differs by context.
`BEHAVIOUR.md` found the island **3.7× slower** than its animals' other
segments. The obvious worry is that the first is just the second, so
`CONTEXT_CONTROLS_PREREGISTRATION.md` §4 registered a stillness arm and fixed
its threshold by matching:

> θ_still is the speed at which the `stillness` arm's corpus-wide **occupancy**
> equals clump 0's corpus-wide occupancy.

That rule is defensible on its face — it makes the two arms equal on rate, so
they differ only in *how* frames were selected — and it is not tunable, which
was the point. It produced:

| | |
|---|---:|
| θ_still | **0.000199 body lengths/s** |
| island frame speed, median | 0.0279 — **140× higher** |
| **share of island frames the arm selected** | **0.0227** |

**The control selected 2.3% of the thing it was controlling for**, and about 39%
of what it did select was frames at *exactly* zero speed — a held pose, which is
a tracking dropout and not a slow animal.

**Why the rule failed.** The island is defined by a **segment-level** statistic:
3.7× is a ratio of *segment mean* speeds. Occupancy is a **frame-level**
marginal that the island merely happens to have. A slow segment still contains
fast frames, so matching on frame share reached into the extreme tail of the
frame speed distribution instead of selecting frames like the island's. The two
arms ended up near-disjoint.

**The consequence, and why it is not a small one.** The residual came back
excluding zero — `PASS`, "clump membership adds to stillness" — which reads
exactly like "the detector is not a freeze scorer". It is not that. Two
near-disjoint sets do not explain each other, and a residual saying so is a
statement about the arms. **The registered question was not answered, and the
number produced looks like an answer.**

**The rule.** When a control is matched to an object, match it on the statistic
that **defines** the object, at the **level** the definition lives. Matching on
a downstream marginal the object happens to share is not the same thing and can
select a disjoint population while satisfying the match exactly. Here the
correct arm draws **segments** matched on joint (log duration, log mean speed) —
which `ISLAND_LOOK.md` already specifies for its own `matched` arm.

**What it cost.** Nothing published, because `controls.overlap_read` was added
before the result was written up and names the arm vacuous inside the result
itself. It cost one registered arm and a re-registration.

**The relationship to M5.** M5 is *a control that returns nothing is usually
testing the control*. This is its converse and it is the more dangerous of the
two: **a control that returns everything is usually testing the control as
well**, and it arrives wearing a `PASS`. A null result invites scrutiny. A
positive one does not.

**The general precondition this leaves behind.** Any matched control should
report the **overlap** between what it selected and what it controls for, as a
precondition, before its verdict is read. An overlap near zero means the
comparison is vacuous however clean its interval looks.

## M12 — a registered threshold must be calibrated against the thing it replaces

**What happened.** `TRENDFILTER_PREREGISTRATION.md` §3 fixed the rule for
choosing the new detector's one parameter: take the smallest setting whose
jitter-only boundary rate is **below 5%** of the corpus rate. The 5% was
invented there and named as invented, which is the repo's convention for a
constant with no external source.

**What §3 never did was ask what the incumbent scores on it.**

| | jitter share |
|---|---:|
| the detector being replaced | **0.5475** |
| the challenger, best point on the grid | **0.0704** |
| **the registered bar** | **0.05** |

The frozen detector misses the bar by a factor of **11**. The challenger comes
within 1.4× of it and is **7.8× better than the incumbent** — and is refused.

**The rule.** A threshold registered in advance is only a fair test if it is
**calibrated against the object it is meant to improve on**. An absolute
constant chosen for its roundness can be simultaneously too strict for anything
that exists and too loose to be meaningful, and there is no way to tell which
from the constant alone. Before registering a bar, measure the incumbent on it
and state the margin being demanded — *"beat 0.5475 by 2×"* is a test; *"be
below 0.05"* is a number.

**What it cost.** The stage refuses. That cost is partly notional here, because
the solver had also not converged anywhere near the bar, so no measurement
existed to admit — but the two failures are independent, and had the solver
worked the bar alone would have thrown away a 7.8× improvement.

**Why the bar was not moved.** Moving it after seeing the challenger land at
0.0704 is choosing a threshold to admit a result, which is the failure every
registration in this programme exists to prevent. It stands, the stage refuses,
and the next attempt gets a re-registration with a relative bar.

**The relationship to M11.** M11 is about matching a **control** on the wrong
statistic; this is about setting a **threshold** with no reference point. Both
are the same underlying error — **a comparison specified without checking what
it is a comparison to** — arriving once on the control side and once on the
criterion side.

## M13 — a threshold defined in samples is a different threshold at a different sampling rate

**What happened.** Stage 2 re-ran both detectors at 15 fps to ask which
boundaries survive. Everything temporal in this programme derives from
`recur.util.frames(seconds, fps)` at use time, which is exactly the discipline
that should make a frame-rate sweep meaningful: pass a different `fps` and every
window re-derives in *seconds*.

One quantity does not.

```
breaks.identifiability_floor(degree=3, k=3) = k * (degree + 1) = 12   # SAMPLES
breaks.min_segment_frames = identifiability_floor + 2 * guard
```

| | 30 fps | 15 fps |
|---|---:|---:|
| `min_gap` | 16 frames | 14 frames |
| **refractory period** | **0.533 s** | **0.933 s** |

**The detector's refractory period is 1.75× longer at half the frame rate**, so
the 15 fps arm is mechanically forbidden from placing boundaries the 30 fps arm
places. Measured, the boundary rate fell to **0.627×** and 1/1.75 = 0.571 —
the floor alone predicts nearly the whole drop.

**The rule.** A parameter expressed in samples is a *time* parameter in
disguise, and it silently changes meaning whenever the sampling rate does.
Before any robustness test that varies the rate, every constant has to be
audited for its units — the ones in seconds are safe, the ones in samples are
the test's confound. The tell is that the quantity has a sound justification in
sample terms: `k·(degree+1)` is samples-per-parameter, which is exactly right
as an identifiability floor and exactly wrong as a refractory period, and it is
serving as both.

**What it cost.** Stage 2's headline. Configuration survival came out at
**0.2604**, and that number cannot be read as evidence about whether the
boundaries are real, because most of the loss is the floor. The registration
forbids changing the floor after seeing the number, so the test is reported as
**not settled** rather than as a failure, and a corrected version is owed.

**The relationship to M12.** M12 is a threshold set without reference to what it
is being compared *to*; this is a threshold whose *units* change what it means
between two arms of one comparison. Both are the same underlying failure —
**a comparison whose two sides are not the same measurement** — and both were
invisible until something was varied that had never been varied before.
