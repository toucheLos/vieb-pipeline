# Step 2 — segments recur across animals, and the boundaries are why

Registered in `results/SEGRECUR_PREREGISTRATION.md`, committed at `190539a`
before any bank or search existed. Source array named in every number:
**`work/ego/raw__bodylen__*.npz`**, the F3-carried `raw` arm, asserted by name
and never Wiener. 89 report animals, 298-animal bank, `degree = 3`,
`deriv_sec = 0.133`, `k_mad = 3.0`, distance **time-normalised to 40 points**
(the registered primary). Digest `198eb14ff258c7f6`.

## The gate: PASS on all three channel groups

The registered question is not "do segments beat a surrogate" — a bank of any
units cut from a real mouse beats a surrogate — but **did the boundaries
contribute**. So the comparator is the identical statistic recomputed with
**length-matched random windows in the same space**, lengths drawn from the same
arm's own clean-segment durations and the count matched.

| group | segments vs `microstate` | length-matched windows | **gate: the difference** |
|---|---:|---:|---|
| **shape** | **+5.9301%** [+4.6077%, +7.1842%] | +0.7369% [+0.4687%, +1.0055%] | **+5.1931% [+3.9588%, +6.4495%]** |
| **both** | **+4.0483%** [+3.0711%, +4.9983%] | +0.3640% [+0.1455%, +0.5679%] | **+3.6844% [+2.7225%, +4.6385%]** |
| twist | +0.1170% [−0.1853%, +0.4266%] | −0.5352% [−0.7945%, −0.3079%] | **+0.6522% [+0.3360%, +0.9853%]** |

Animal-level bootstrap on the per-animal difference, n = 89, 2,000 replicates,
paired within animal.

**On shape, segmenting multiplies the cross-animal recurrence excess by 8×** —
from +0.74% for random windows of the same lengths to +5.93% for the intervals
between detected breaks. That is the boundaries doing work, measured against a
control that differs from them in exactly one way: where the edges fall.

## The disagreement between channel groups is the second finding

It was registered that the three groups run separately and are never pooled,
because their boundary sets at ±2 frames give Jaccard 0.443 (both|shape), 0.436
(both|twist) and **0.226 (shape|twist)** — largely different events. They
disagree, and the disagreement is informative:

| group | do segments recur above the null? | do the boundaries contribute? |
|---|---|---|
| shape | **yes**, `PASS` vs both nulls | **yes** |
| both | **yes**, `PASS` vs both nulls | **yes** |
| **twist** | **no** — `FAIL`, +0.117% [−0.185%, +0.427%], interval spans zero | yes, +0.652% |

**Twist segments do not recur.** The gate difference is positive only because
length-matched *windows* in the twist channels recur **less** than their
surrogate (−0.535%), so the boundaries improve on a negative baseline without
reaching zero. Whatever recurs across animals in this corpus is **egocentric
shape, not body-frame rotation**. The `both` group sits between its two halves
and below `shape` alone, which says the three twist channels dilute the signal
rather than adding to it.

## Why twist's number should not be read even as a null result

Its **length-match control is the worst in the set**: nearest-neighbour partners
share duration far more than chance, ratio **0.494** (mean absolute log-length
gap 0.451 against 0.914 shuffled), where shape is 0.922 and both 0.889. In three
dimensions warped to 40 points, "same duration" is most of what a twist segment
is, so a twist match is substantially a length match. `partition.py:20-37`
records a rater scoring +0.183 — a third of the real effect — purely by calling
the six longest clips "same".

Twist is also where `frac_within_closer` collapses to 0.012 (against 0.330 on
both, 0.344 on shape): almost nothing is closer within an animal than across,
which is what a space carrying little individual structure looks like.

## The controls, each able to have undercut the result

**Bank size.** Nearest-neighbour distance falls as a bank grows, and because θ
comes from the null's own quantile, a smaller null bank inflates the excess in
the direction that flatters the corpus. `microstate` produced 25% fewer segments
than the corpus on `shape`. **Declared post-hoc**, added after seeing the gap:
every arm re-run with its bank cut to the smallest in the group.

| group | gate as registered | gate at matched bank | matched size |
|---|---|---|---:|
| shape | +5.1931% [+3.9588%, +6.4495%] | **+4.8805% [+3.7053%, +6.0849%]** | 161,271 |
| both | +3.6844% [+2.7225%, +4.6385%] | **+3.4459% [+2.5487%, +4.3277%]** | 153,960 |
| twist | +0.6522% [+0.3360%, +0.9853%] | **+0.6448% [+0.3258%, +0.9812%]** | 221,838 |

The confound is real and small — it moves the shape gate by 0.31 points, `both`
by 0.24 and twist by 0.007, every interval still excluding zero. The result does
not rest on it.

**Duration matching**, reported beside the comparison and never gating it, as
registered. KS on log duration: 0.031–0.110 against `microstate0`, 0.084–0.125
against `microstate`; median duration ratios 0.78–1.08. The nulls' segments are
close in length to the corpus's, `microstate` running slightly longer.

**Length matching inside the matching.** Ratios 0.889 (both) and 0.922 (shape) —
nearest neighbours share duration slightly more than chance, but not nearly
enough to account for a 5-point excess. Twist at 0.494 is the exception, treated
above.

**Exactness, split by quantity because the statistic reads one of them.**
`d_cross` — the only quantity `paired_excess` reads — agrees with the float64
CPU reference to **7.5e-06** at worst across all 18 arms, against a 1e-04
tolerance. Two window arms (`shape|microstate0`, `twist|microstate`) drift on
`d_within` to 4.9e-04 and carry a joint `FAIL` from `exactness_read`, which gates
on both. TF32 was disabled explicitly and the drift did not move, so TF32 was
never the cause; it is a near-tie in the **within-animal** neighbour, which the
search never returns an index for, so `index_agreement` (which compares only
`i_cross`) cannot see it. `d_within` is a reported diagnostic and already carries
its registered upper-bound limitation. **No quoted number depends on it.**

**PCA validity.** At `d = 192` the `both` group's null arms failed `pca_read`
(Spearman ρ 0.977–0.988 against a 0.99 requirement) while the corpus passed.
Raising only the failing arms would have embedded corpus and null in
differently-shaped spaces, and θ comes from the null's quantile, so that would
make the threshold incomparable for reasons unrelated to recurrence. **Every arm
is now embedded at its full dimension** — 680 (both), 560 (shape), 120 (twist) —
a rotation that preserves distances exactly. Q1 reduced because its bank held
7.45M windows; these hold ~200,000 and the reduction bought nothing.

## The abstain exclusion is enormous, and it was registered as an exclusion

| | segments | abstain-adjacent | selected |
|---|---:|---:|---:|
| both | 817,228 | 354,193 | **192,357** |
| shape | 847,485 | 361,565 | **215,242** |
| twist | 847,849 | 355,321 | **221,850** |

**Roughly 43% of all segments touch an abstain block or sit beside one**, and
they are excluded rather than flagged. Abstain is 7.03% of frames arriving in
short runs, so it fragments the stream far more than it covers it, and a segment
whose boundary was manufactured by a dropout is not a behavioural unit. Only
about a quarter of raw segments survive into the bank.

## The planted floor: 2% occupancy, and the bug it caught

Planted into **`microstate`** and scored against unplanted `microstate`, so the
background carries no recurrence of its own. Reported on the **realised**
fraction.

| requested | realised | excess | |
|---|---:|---|---|
| 0.25% | 0.00250 | +0.0225% [−0.0060%, +0.0633%] | — |
| 0.5% | 0.00499 | +0.0288% [−0.0032%, +0.0667%] | — |
| 1% | 0.01001 | +0.0203% [−0.0277%, +0.0677%] | — |
| **2%** | 0.02001 | +0.1214% [+0.0164%, +0.2495%] | **recovered** |
| 5% | 0.04998 | +0.2106% [+0.0708%, +0.3719%] | recovered |
| 10% | 0.09994 | +0.3153% [+0.0947%, +0.5752%] | recovered |

**The floor is 2% realised occupancy**, against Q1's 0.25% for fixed windows. So
the segment pipeline is roughly an order of magnitude **less** sensitive to a
rare planted motif than the windowed one, and `twist`'s null result must be read
with that in mind: a twist vocabulary occupying under 2% of frames would not have
been detected.

**Two bugs in this ladder were caught by it returning nothing**, and both are
recorded because the corrected number depends on them:

1. **Planted into the corpus.** The first ladder ran +5.93% at 0.25% falling
   monotonically to +2.51% at 10% — backwards. The corpus background already
   scores +4% against this null, so every cell measured the corpus's own
   recurrence while the planting slowly overwrote it. Q1 plants into its
   surrogate (`ar-planted_wiener` scored against `ar_wiener`); this now does too.
2. **A per-animal template.** The template was built inside the per-animal loop,
   so every animal received a *different* stereotype. The statistic is
   **cross-animal** recurrence, so that plants a signal the measurement cannot
   see; the ladder recovered nothing at any occupancy. One template, averaged
   over three donor animals, is now planted corpus-wide.

**And a diagnosis the floor forces, which qualifies it.** Only **2.3%** of
planted instances become their own segment; 66.7% are merely overlapped by a
segment spanning them (2,920 instances, six animals, 10% occupancy). So the
ladder measures **detector and matcher jointly, and the detector is the
bottleneck** — the planted template is the mean of 40 real windows and therefore
very smooth, crossfaded at its seams, so it creates no acceleration
discontinuity for a break detector to find. The 2% figure is a floor for
*planted smooth blocks*, not a general sensitivity bound, and it is a weaker
statement than Q1's 0.25% in a way that is partly an artifact of importing a
window-pipeline control into a boundary pipeline.

## What this licenses

**Segments recur across animals, and the boundaries are the reason.** On shape,
the excess is 8× what length-matched random windows in the same space achieve,
against two dwell-matched nulls, with the bank-size and length-matching controls
each moving it by a fraction of a point and the exactness gate passing on the
quantity the statistic reads.

**It does not license a vocabulary.** Recurrence says the units are real; it
says nothing about whether they are discrete. That is Step 3, which can return
"continuum" — and a continuum is also a result.

It does not license reading `twist`, in either direction.

## One number this is deliberately not read against

Q1's **+1.639%** is the `wiener` cell (`Q1_BANKED.md`). This arm runs on `raw`,
where the same file's `unfiltered` cell is **−0.1991% [−0.4636%, +0.0418%]** —
no excess over VAR(5). Reading a raw-arm segment result against a Wiener-arm
window result would compare two things that differ in two ways at once, which is
why the comparator was registered in the identical ego space instead. The
windowed control here (+0.74% on shape, +0.36% on both) is that comparator, and
it is the number the gate is measured against.
