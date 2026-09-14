# Re-registration — the injected park length distribution

Written and committed **before the re-run**. Phase F and F2 both stand at the old
distribution; this adds a corrected arm beside them and neither replaces the
other.

## The defect

`inject.PARK_FRAMES = (4, 30)`, uniform. Its docstring cites "`CLEANING.md`, runs
surviving `median_0.50`" — a **post-filter** distribution used to model a
**pre-filter** error.

`RUNLEN.md` established why that is circular: `median_0.50` deletes every
violating run of 7 frames or fewer and leaves the rest untouched, preserving
1.11× of the violating mass above its own cliff. So "runs surviving
`median_0.50`" is by construction the distribution of runs the median **cannot
fix**. Drawing injected parks from it guarantees they land on the wrong side of
the arm's only threshold.

## The measurement

Pre-filter, on `held_array(pose_unfiltered, missing)` — the array the arms
receive — over 40 seeded report recordings. Violating runs are split by which
continuity component dominates: `step > spike` is a **persistent shift**, which
is what a park is; `spike > step` is a one-frame excursion, which is a teleport
and is injected separately.

| | park-like (step > spike) | spike-like | injected (4, 30) |
|---|---:|---:|---:|
| runs | 532 | 354 | — |
| median | **2** | 1 | 17 |
| p75 | 4 | 2 | 24 |
| p90 | 9 | 4 | 27 |
| p99 | 30 | 14 | 30 |
| max | 54 | 43 | 30 |
| **share ≤ 7 frames** | **87.6%** | 95.8% | **14.8%** |
| mass ≤ 7 frames | 53.0% | 70.2% | ~3% |

**Real parks are 5.9× more likely to fall under the median's cliff than injected
ones.** The p99 and max agree well; it is the body of the distribution that is
wrong.

## Fixed in advance

| | value |
|---|---|
| **changed** | park length drawn by inverse-CDF interpolation over the measured quantiles (1, 2, 4, 9, 30, 54 at p0/p50/p75/p90/p99/max), replacing `uniform(4, 30)` |
| unchanged | corruption kinds, all four rates, seed (blake2b `stable_seed`), arms, `MIN_SEGMENT_FRAMES = 30`, confidence floor 0.60, bone check, common-denominator rule, strata definition, `MIN_STRATUM_FRAMES = 20,000`, animal bootstrap 2000 replicates |
| thresholds | all four: 0.02, 0.05, 0.10, off |
| reporting | old and new side by side; **both stand** |

A park is still a *held* position — one wrong location for the whole run — which
is what makes it temporally smooth and unreachable by a motion prior. Only its
length distribution changes.

## Predictions

1. **`median_0.50`'s park repair rises sharply**, from 10.5% toward the ~85% that
   its cliff implies once 87.6% of parks fall under it.
2. **`median_0.50`'s net improves at every threshold and every stratum.** Stated
   in advance so that an improvement is read as **the expected consequence of
   removing a known bias**, not as vindication of the arm.
3. **The improvement is about 0.0013 in net**, and this is a quantitative
   prediction rather than a direction. Parks are 23,408 of ~4.5M scored
   keypoint-frames (0.5%) at injected error 0.359 bl; taking repair from 10.5% to
   ~85% removes 0.267 bl on 0.5% of frames ≈ **0.0013**.
4. **`viterbi` and `disposition` move little.** Neither repairs parks at all
   (−0.0% and 0.0%), so a change in park length cannot help them; their nets
   should move only through the shared denominator.
5. **The per-stratum crossing survives.**

## The falsifier, stated as a magnitude

The crossing reverses if `median_0.50`'s net in the **fastest stratum at `off`**
falls below `viterbi`'s there.

| | value |
|---|---:|
| `median_0.50`, s5, `off` | **+0.0068** |
| `viterbi`, s5, `off` | **−0.0025** |
| **improvement required to reverse** | **≥ 0.0093** |

That is larger than the entire threshold sweep moved this arm (0.0087) and **7×
the predicted improvement** (0.0013). So prediction 5 is a real bet: if the
measured improvement is anywhere near 0.0093, prediction 3 was badly wrong and
the F2 conclusion does not survive.

A partial reversal is also defined: if s4's net (currently +0.0020) goes negative
while s5 stays positive, the crossing has **moved**, not reversed, and the
conclusion holds in weakened form with the crossing point named.

## What this may not do

* It may not re-score Q1.
* It may not restate Phase F or F2. Both stand at the old distribution.
* It may not change any parameter other than the park length distribution.
* It may not report a stratum below 20,000 scored keypoint-frames.
