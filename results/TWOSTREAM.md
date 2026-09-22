# Stage 1 — the dynamics stream cuts elsewhere, is cleaner at the wall, and is 97% noise

Registered in `results/TWOSTREAM_PREREGISTRATION.md`, committed before
`vieb/seg/twostream.py` or `scripts/twostream.py` existed. **`tune` split, 60
animals**, F3 `raw` arm. Digest `198eb14ff258c7f6`.

## The four registered predictions

| # | prediction | outcome |
|---|---|---|
| 1 | `dyn_only` above 10% of all boundaries | **PASS** — 0.216 [0.207, 0.225] |
| 2 | dynamics rate beats the incumbent's jitter share of 0.5475 | **FAIL** — 0.9665 |
| 3 | `body_extension` moves the configuration rate by under 20% | **PASS** — 0.2% |
| 4 | dynamics tracks the wall less than configuration does | **PASS** — 1.09× against 1.97× |

## The rates

| arm | boundaries/s |
|---|---|
| `config14` — frozen detector, unchanged | 0.4746 [0.4573, 0.4909] |
| `config15` — plus `body_extension` | 0.4735 [0.4563, 0.4897] |
| **`dyn`** — persistence contrast | **0.2356 [0.2306, 0.2404]** |
| **`dyn_noise`** — the same on measured-colour noise | **0.2277 [0.2252, 0.2303]** |

## What the dynamics stream gets right

**It cuts in different places.** 14,789 of 32,452 dynamics boundaries match a
configuration boundary within ±15 frames; **the remaining 21.6% [20.7%, 22.5%]
do not**. Configuration and dynamics are not the same cut, which is the premise
the whole two-stream architecture rests on and it survives.

**It is far less contaminated by arena position.** Boundary rate from arena
centre to wall:

| stream | wall / centre |
|---|---:|
| `config15` | **1.97×** |
| `dyn` | **1.09×** |
| *(tracking failure itself, `CONCENTRATION.md`)* | *3.7×* |

The configuration stream's rate nearly doubles at the wall, where tracking
failure rises 3.7×. **The persistence stream barely moves — 9%.** That is the
scale-free descriptor doing exactly what Stage 0 predicted it would: an
acceleration threshold reads the tracker, a pole radius largely does not.

## What kills it

> **96.65% of the dynamics stream's boundary rate is attributable to noise.**
> 0.2277/s on a constant pose carrying measured-colour tracking noise, against
> 0.2356/s on the corpus.

The intervals are just non-overlapping — [0.2306, 0.2404] against
[0.2252, 0.2303] — so it technically clears its own floor. **Clearing a floor
is not the test.** The frozen detector it was built to improve on sits at a
jitter share of **0.5475**. The dynamics stream is at **0.9665** — nearly twice
as noise-dominated as the thing it replaces.

**This is `METHODS_FINDINGS.md` M12 earning its keep in the other direction.**
M12 was written when the trend filter was refused by an absolute bar the
incumbent misses by 11×. Here the registration stated the incumbent's value in
advance, and it is what converts "clears its floor, PASS" — which a bar-free
reading would have given — into a clear failure. A gate without a reference
point is as likely to admit a bad result as to refuse a good one.

**And it reconciles predictions 1 and 2, which look contradictory.** The stream
finds 21.6% of boundaries that configuration does not *and* is 96.7%
noise-attributable. Both are true because **the boundaries it finds alone are
mostly the noise ones**. Being differently located is not the same as being
right.

## `body_extension` contributes almost nothing

Adding the stretch-attend channel moves the configuration rate from 0.4746/s to
**0.4735/s — 0.2%**. Prediction 3 passes, but it passes by the channel being
nearly inert rather than by being well-behaved. Whether it separates *sub-types*
inside a segment is a different question and is Stage 3's; this says only that
it does not change where the detector cuts.

## What this licenses

**Licenses:** the two streams cut in measurably different places (21.6% unique
at ±15), and a persistence descriptor is substantially less sensitive to arena
position than an acceleration threshold (1.09× against 1.97×). Both are real
properties of the instruments and neither depends on the dynamics stream being
any good.

**Does not license using the dynamics stream as a boundary source.** At a
jitter share of 0.9665 its output is almost entirely noise, and any segment set
built from it would be a segmentation of tracking error.

**Does not license** any statement about the corpus's own behaviour, or any
re-reading of `SEGRECUR.md`, `VOCAB.md` or `CONTEXT.md` — this is a different
detector making a new measurement.

## What follows, and what is therefore not run

**Stage 2's segment-recurrence arm on the dynamics stream is not run.** Running
dwell-matched recurrence on a boundary set that is 96.7% noise would measure
the noise's recurrence. That is a consequence of prediction 2 failing and it is
recorded rather than attempted.

**What is still worth running, and is:** the frame-rate subsampling test, which
asks whether boundaries are rhythm or artifact and applies to both streams; and
the `white` negative control, which a new detector re-incurs and does not
inherit.
