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
