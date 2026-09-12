# Phase E1 — does a frame agree with the frames either side of it?

89 report animals, 445 arm-animals, ε = 0.10, threshold 0.10 body lengths.
Configuration fixed in `CONTINUITY_PREREGISTRATION.md`, committed before the
corpus run. **The bakeoff verdict is not restated and Q1 is not re-scored.**

## The headline: the bone check is missing most of it

| | raw array, 6,678,488 report frames |
|---|---|
| frames with a continuity spike > 0.10 body lengths | **3.798%** |
| frames the bone check flags at ε = 0.10 | **1.122%** |
| **share of spiking frames the bone check also flags** | **8.2%** |
| share of bone-flagged frames that also spike | 27.9% |
| Jaccard | **0.068** |

|  | bone **YES** | bone **NO** |
|---|---:|---:|
| **spike YES** | 20,875 | **232,792** |
| **spike NO** | 54,062 | 6,370,759 |

Two nearly disjoint nets. That is the answer to why the clips still look
inconsistent: **a length test is structurally blind to a keypoint that slides
along a bone**, or to a skull triangle that drifts coherently, and those are most
of what a viewer sees. 232,792 frames carry a visible discontinuity that no bone
check at any ε could have found, because nothing about them changes a length.

This is reported and nothing downstream acts on it. Phase D is pre-registered and
committed; widening its suspect channel on an instrument built afterwards is the
contamination the pre-registration exists to prevent.

## The arms

`spike = min(r⁻, r⁺)` is a one-frame excursion; `step = |r⁻ − r⁺|` is a persistent
shift. Both in body lengths, `ℓ̄` = 114 px. `hf_retained` is carried from the
bakeoff because a filter can make a recording perfectly continuous by deleting the
movement.

| arm | spike | spike p99 | step | step p99 | **spiking** | hf retained | efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw` | 0.0046 | 0.1119 | 0.0043 | 0.2131 | 1.271% | 1.000 | — |
| `wiener` (incumbent) | 0.0029 | 0.0559 | 0.0021 | 0.0675 | 0.381% | 0.220 | 55 |
| `median_0.50` | **0.0013** | **0.0409** | 0.0014 | 0.0737 | **0.101%** | 0.270 | 83 |
| **`viterbi`** | 0.0045 | 0.0662 | 0.0042 | 0.1486 | 0.419% | **0.586** | **354** |
| `disposition` | 0.0046 | 0.1072 | 0.0043 | 0.1975 | 1.215% | 0.894 | 4 |

*Efficiency: percent of spiking frames removed per pixel of mean displacement,
using the displacement figures from `CLEANING.md`.*

**Anipose Viterbi is the efficiency outlier again, by more than before.** It
removes **67.0%** of spiking frames while moving 0.189 px on average — **354% per
pixel against the next best arm's 83%, a factor of 4.3**. It also retains 58.6% of
the power above f_c where the two smoothers retain 22% and 27%. On this axis it is
not close.

`median_0.50` reaches the lowest spike of any arm and the read returns PASS for
it, because it clears the incumbent's interval on spike **and** retains more
high-frequency power than the incumbent does. Both halves of that are true and
neither should be read alone: against **raw** it has deleted 73% of the power
above f_c.

## The six pre-registered predictions

| # | prediction | outcome | |
|---|---|---|---|
| 1 | Viterbi lowers spike most per pixel | **354 %/px** against 83 and 55 | **held** |
| 2 | no arm meaningfully lowers `step` | wiener −51%, median_0.50 −67%; tail −68%/−65% | **failed** |
| 3 | the disposition lowers spike on the **held-out** side | −4.4%, intervals overlap raw's | **not supported** |
| 4 | 2×2 weak: Jaccard < 0.20, `spike_only` > `spike_and_bone` | **0.068**; 232,792 against 20,875 | **held** |
| 5 | `median_0.50` lowers spike and hf together | spike lowest at 0.0013, hf 0.270 against raw's 1.000 | **held** |
| 6 | median spike ≤ 0.02 body lengths on every arm | 0.0013 – 0.0046 | **held** |

### Prediction 2 failed, and I am not withdrawing the claim it was a falsifier for

The registration said: *"If an arm does lower `step`, that claim is wrong and must
be withdrawn."* An arm did — two did, on both the median and the tail.

The claim in question is `CLEANING.md`'s: what a temporal filter leaves behind is
temporally smooth, and closing the gap needs an anatomical prior rather than
another filter. **I am not withdrawing it, and the reason is that the falsifier I
registered does not test it.** A landmark parked off the body for 9 frames and a
0.50 s median filter is 15 frames wide: the filter smooths *across* the park,
shrinking the `step` at entry and exit while the landmark stays exactly as far off
the animal as it was. `step` conflates "the landmark came back" with "the
transition got smoothed", and only the first would bear on the claim.

So the honest statement is that **I registered a bad falsifier**, not that the
claim survived a good one. That is worse than a prediction failing cleanly and it
is recorded here rather than reinterpreted quietly.

The test that does bear on the claim was already run and is unchanged: violating
runs on the eight worst recordings go from a median of 2 frames to **9 frames**
after `median_0.50`, with **96.1%** of surviving violating frames in runs longer
than 3 frames. A filter that fixed the sustained error would shorten those runs.
It lengthens them.

### Prediction 3 is not supported, and that is a finding about Phase D

The disposition was scored on the **held-out neighbour** — the one
`donor_frame` did not pick — because the corrector moves the suspect *onto* the
donor-side prediction and scoring it there would measure its own premise.

| | spiking frames |
|---|---:|
| raw | 1.271% |
| disposition, **donor side** (circular) | 1.133%, −10.9% |
| disposition, **held-out side** | 1.215%, **−4.4%** |

The improvement is in the right direction and **it does not separate**: the
held-out interval [0.934%, 1.543%] overlaps raw's [0.977%, 1.616%] almost
entirely, and this repo's rule since the bakeoff is non-overlapping intervals or
no claim.

The gap between the two sides is the substance. **Better than half of the
disposition's apparent continuity gain is the corrector agreeing with its own
predictor**, and nothing inside Phase D could have shown that — the leave-one-bone-out
gate tests geometry, not time. It does not overturn the gate, which passed on
bones the corrector never saw. It does say the corrector's effect on temporal
continuity is, at this domain, indistinguishable from nothing.

## Per keypoint

Median spike on the raw array, body lengths:

| keypoint | median spike |
|---|---:|
| nose | **0.0067** |
| tail_base | 0.0052 |
| left_hip / right_hip | 0.0047 |
| right_ear | 0.0044 |
| left_ear | 0.0042 |
| center | **0.0034** |

The nose is the least continuous landmark and the centre the most, by a factor of
two. That agrees with Phase D from a completely different direction: the nose is
the suspect on **61.6%** of corrections there, identified by bone length, and it
is the worst keypoint here, identified by time. Two disjoint instruments naming
the same landmark is the strongest evidence in this phase that both are measuring
something real.

**One caveat on attribution, registered in advance.** The fit for each keypoint
uses the other six, so a badly wrong keypoint inflates its neighbours' residuals
too — measured at about 2.5× on a synthetic 45 px displacement. `argmax` over
keypoints is sound; the absolute level is inflated, equally for every arm.

## What this does not settle

Held-out MDL, as ever. Continuity is a fourth view and the bakeoff chose on three;
`results/cleaning.json` is unchanged and `median_0.50` remains what those three
axes say it is. What this phase adds to that decision is that **Viterbi's case is
much stronger than the bakeoff showed** — it was already the efficiency outlier on
violations, and it is the efficiency outlier on continuity by a factor of 4.3
while retaining more than twice the high-frequency power of either smoother.
