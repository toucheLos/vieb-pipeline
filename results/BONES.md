# Step 1 — Bone-length violations

**Corpus** `luna` | **recordings** 3,846 |
**frames** 22,355,989 | **animals** 298 |
**inherited digest** `fe62fec33da4485a`

Primary cell: **unfiltered** pose, **raw**
metric, **skull** group, **ε = 0.1**.
Unfiltered because it is the measurement and the array shapeflow's own bone_flagged was computed on, which is what makes the 2x2 like-for-like; skull because trunk bones flex.

## Reads

| read | verdict | n |
|---|---|---:|
| `r2_join` | `PASS` | 298 |
| `shuffled_ceiling` | `PASS` | 1192 |
| `skull_raw` | `GRID_LIMITED` [between the exclude and correct thresholds] | 298 |
| `skull_scalefree` | `GRID_LIMITED` [between the exclude and correct thresholds] | 298 |
| `trunk_raw` | `NOT_A_RESULT` | 298 |
| `trunk_scalefree` | `NOT_A_RESULT` | 298 |

> **skull_raw** — skull violations affect 2.134% of frames at eps = 0.1, inside the 2%-15% band. Too many to exclude and few enough to correct: proceed with correction, and cost out an ensemble-DLC path before anything is published on this feature space. The curve DECAYS SMOOTHLY -- the relative drops increase monotonically across the sweep (34%, 37%, 44%, 54%), which is one population thinning out -- so no threshold on this grid is principled and the rate has to be read as a curve rather than at a chosen eps.

> **r2_join** — per-animal Spearman rho between a segment's bone-violation rate and its ExBias reconstruction R^2 is -0.104 [-0.115, -0.094] (segment duration held fixed), animal bootstrap, interval excluding zero and |rho| above 0.1. The R^2 floor is partly tracking, so the violation rate is measuring something the segmenter also sees

> **shuffled_ceiling** — the observed rate is 2.1341% against a shuffled-keypoint ceiling of 79.5914%, a margin of 37.3x. Destroying the temporal association between landmarks -- within the same recording, so arena, animal and camera are held fixed -- makes the violation far more common, which is what licenses reading the observed rate as geometry

## The branch

Skull violations are **2.134%** of frames at ε = 0.1, inside the
2%–15% band. **Correct rather than exclude**, and cost out an ensemble-DLC path
before anything is published on this feature space. They are not concentrated in
a few recordings — 1,897 of
3,846 recordings sit above 1% — so exclusion would cost half
the corpus, which is what puts this in the correction branch rather than the
exclusion one. The median recording is 0.968% and the 95th
percentile is 8.675%; the worst 25 run
17.882%–52.726%.

## The ε curve

Unfiltered pose. **The curve is the deliverable, not any single cell.**

| ε | skull raw | skull scale-free | trunk raw | trunk scale-free | shuffled ceiling |
|---|---:|---:|---:|---:|---:|
| 0.02 | 5.130% | 5.081% | 9.665% | 8.600% | 83.8% |
| 0.05 | 3.410% | 3.560% | 5.545% | 5.245% | 82.0% |
| 0.1 | 2.134% | 2.319% | 2.958% | 3.128% | 79.6% |
| 0.2 | 1.203% | 1.313% | 1.420% | 1.690% | 75.8% |
| 0.5 | 0.549% | 0.556% | 0.477% | 0.617% | 66.8% |

**The curve decays smoothly — there is no elbow, and that is a result rather than a missing one.** The relative drops run 34%, 37%, 44%, 54% across the sweep, increasing monotonically, so the rate falls away faster and faster rather than falling off at one place. This is one population thinning out, not two with a gap between them.

So **no ε on this grid is a principled cut**, and the rate has to be read as a curve. ε = 0.1 is reported as the primary cell because the brief names it, not because the data prefers it — at ε = 0.20 the same corpus reads 1.203% and would fall in the *exclude* branch rather than the *correct* one. A reader who needs a single number needs to know the branch moves with a threshold nobody can justify from the shape.

Every observed rate sits far below the **shuffled-keypoint ceiling** — keypoints
drawn from random frames *within the same recording*, so arena, animal and camera
are held fixed. At ε = 0.1 the margin is
37×, measured on
1192 recordings. A diagnostic that fired as often on shuffled
landmarks as on real ones would be reporting its own threshold.

## Excess length is not perspective

This was the open question the second metric exists to settle, and it settles
against the hypothesis.

| | ε = 0.1 |
|---|---:|
| raw pixel lengths | 2.134% |
| per-frame common scale removed | 2.319% |

Removing the common scale does **not** reduce the violation rate — it raises it
slightly, at every ε on the sweep. A mouse rearing or moving nearer the lens
multiplies every distance by one factor, which the scale-free metric cancels
exactly; if rearing were generating these violations the second row would be far
smaller than the first. It is not, so the raw sweep can be read at face value.

The small *increase* is mechanical rather than mysterious: dividing by a frame's
own median length makes an over-long bone on an otherwise foreshortened frame
look relatively longer still, so the scale-free arm is marginally the more
sensitive detector.

## The second opinion — 2×2 against shapeflow's `bone_flagged`

shapeflow's gate flags 4.556% of frames on a different
principle: symmetric in the deviation, scale-free by construction, learned rather
than anatomical, and requiring three of 21 pairs to break at once. Reported as a
confusion rather than as two rates, because the disagreement is what carries
information.

| cell | raw | scale-free | what it is |
|---|---:|---:|---|
| both | 0.856% | 1.127% | agreed tracking failure — two unrelated criteria on one frame |
| new only | 1.278% | 1.192% | excess length the existing gate misses |
| `bone_flagged` only | 3.700% | 3.429% | relative geometry broken with no excess length — swaps and shortenings, which a one-sided test is blind to by construction |
| Jaccard | 0.147 | 0.196 | |
| P(`bone_flagged` \| new) | 0.401 | 0.486 | |

`new_only` barely moves between the metrics (1.278% →
1.192%) while `both` rises
(0.856% → 1.127%), which is the same verdict
the previous section reached by a different route: the excess survives
common-scale removal, so it is geometry and not perspective.

**1.278% of frames carry excess
skull length that `bone_flagged` does not see, on frames that were measured
rather than interpolated or filled.** That is the cell with no benign
explanation, and it is more than half the total. It is also exactly equal to
`new_only`, because a missing keypoint yields a non-finite length and a
non-finite length is never counted as a violation.

## What the filter absorbs

Every downstream feature — Q1's channels included — is built on the Wiener-shrunk
`pose`, not on the unfiltered array this diagnostic reads.

| ε | unfiltered | wiener | absorbed |
|---|---:|---:|---:|
| 0.02 | 5.130% | 4.973% | 3.0% |
| 0.05 | 3.410% | 2.936% | 13.9% |
| 0.1 | 2.134% | 1.648% | 22.8% |
| 0.2 | 1.203% | 0.840% | 30.2% |
| 0.5 | 0.549% | 0.312% | 43.2% |

The filter removes 23% of the ε = 0.1
violations, so **1.648% of frames in the feature space every
downstream stage consumes carry impossible skull geometry**. Shrinkage is not a
repair: it makes the excursion smaller without making the frame correct.

## The join with ExBias R², and the confound inside it

409,812 segments over 298 animals, joined on
recording id and frame range only — never on keypoint index, because ExBias
orders bodyparts alphabetically where shapeflow uses file order. Mean
`fit_r2` = 0.7023; 17.68% of segments
sit below 0.5 on raw R² and 24.58% on adjusted.

| decile | R² range | segments | mean violation rate | segments with any | mean duration |
|---:|---|---:|---:|---:|---:|
| 1 | -25.860 – 0.392 | 40,982 | 3.285% | 16.68% | 4.00 s |
| 2 | 0.392 – 0.526 | 40,981 | 4.127% | 18.98% | 3.30 s |
| 3 | 0.526 – 0.620 | 40,981 | 4.779% | 19.55% | 2.63 s |
| 4 | 0.620 – 0.692 | 40,981 | 5.035% | 18.87% | 2.09 s |
| 5 | 0.692 – 0.753 | 40,981 | 5.228% | 17.74% | 1.71 s |
| 6 | 0.753 – 0.804 | 40,981 | 5.281% | 16.38% | 1.37 s |
| 7 | 0.804 – 0.849 | 40,981 | 5.272% | 15.25% | 1.06 s |
| 8 | 0.849 – 0.889 | 40,981 | 5.069% | 13.23% | 0.83 s |
| 9 | 0.889 – 0.927 | 40,981 | 4.451% | 11.01% | 0.65 s |
| 10 | 0.927 – 0.997 | 40,982 | 3.799% | 8.60% | 0.53 s |

**Read the duration column before either of the others.** Mean segment duration
falls from 4.00 s in the worst-fitting decile to
0.53 s in the best — a factor of
7.6 — and
ρ(duration, R²) = **-0.538 [-0.545, -0.531]**. ExBias fits a cubic over each segment, so a
short segment fits well close to by construction, and its R² is substantially a
readout of its own length.

That confound sits on **both** sides of the join, because the violation rate's
denominator *is* the segment's length. It does not inflate the association here;
it suppresses it:

| statistic | ρ |
|---|---|
| violation rate vs R², marginal | -0.054 [-0.062, -0.046] |
| violation rate vs R², **segment duration held fixed** | **-0.104 [-0.115, -0.094]** |
| any violation (binary) vs R², marginal | -0.067 [-0.076, -0.059] |

The controlled coefficient is roughly double the marginal one at every ε on the
sweep, and it is the number the verdict is read on. Taken marginally the join
would have read `GRID_LIMITED` — "real but too small to attribute the R² floor to
tracking". Controlled, it clears the 0.1 threshold and reads `PASS`.

So: **segments that reconstruct badly do carry more bone violations, once you
stop comparing four-second segments against half-second ones.** The premise
behind this gate survives. Note what it does not say — ρ = -0.104
leaves the great majority of the R² floor unexplained by tracking, and the honest
description of ExBias's mean 0.70 is that it is mostly about how long a segment
is and how well a cubic spans it.

## Attribution

Share of violating bone-frames by landmark, ε = 0.1, skull:

| landmark | share |
|---|---:|
| nose | 0.363 |
| right_ear | 0.323 |
| left_ear | 0.314 |

The three skull landmarks carry the violations near-evenly
(0.314–0.363), which is
what a triangle of three mutually-constraining bones produces when no single
landmark is the culprit. A single systematically mistracked keypoint would show
as one share near 0.5 and is not what this corpus has.

## By split

| split | median recording | p90 | max |
|---|---:|---:|---:|
| fit | 0.983% | 5.782% | 52.726% |
| report | 0.965% | 5.735% | 39.084% |
| tune | 0.983% | 5.825% | 25.241% |

The three splits are indistinguishable, so nothing about the violation rate is
going to separate `tune` from `report` when a threshold is chosen on the former.

## What this licenses, and what it does not

**Licensed.** Proceed to Step 2. The feature space is not predominantly tracking
failure: 2.134% of frames carry excess skull geometry, the rate is
37× below its own shuffled ceiling,
and it is associated with independent evidence of bad reconstruction once
duration is controlled.

**Not licensed.** Any claim that the corpus is clean. Half the recordings sit
above 1%, 1.648% of frames survive filtering into the feature space
with impossible geometry, and
1.278% of frames carry excess
length on measured frames that the inherited QC does not flag. The **artifact
ablation is therefore mandatory rather than a formality** at Step 4: score with
and without these frames and report the delta. If it is material, part of any win
was tracking.

**Carried forward.** The per-frame violation masks are in
`work/bones/<animal_tag>.npz` under `viol|<pose_arm>|<metric>|<eps>|<group>`,
bit-packed over the corpus in `bounds` order, for exactly that ablation.

## Bones

`SKULL` (primary): left_ear–right_ear, left_ear–nose, right_ear–nose.

`TRUNK` (reported separately, never pooled into the headline):
nose–center, center–left_hip, center–right_hip, center–tail_base, left_hip–tail_base, right_hip–tail_base.

Trunk rates run 2.958% at ε = 0.1,
above the skull's. Those bones flex, so a violation there is real posture at
least as often as it is tracking, and pooling the two would produce a headline
stronger than the evidence.
