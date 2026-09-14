# H3 — 100 frames scored by eye, blind

100 frames: 25 from each cell of {worst-decile recordings, passing recordings} ×
{flagged by the mask, not flagged}. Rendered with the skeleton drawn and
**`flagged=False` for every frame**, filenames a shuffled index, the key sealed
until scores were supplied. Scored from 5×5 contact sheets.

## The result

| | occluded | visible but wrong | fine | unsure |
|---|---:|---:|---:|---:|
| passing, not flagged | **0** | 3 | 21 | 1 |
| passing, **flagged** | **0** | **8** | 13 | 4 |
| worst decile, not flagged | **0** | 4 | 20 | 1 |
| worst decile, **flagged** | **0** | **8** | 16 | 1 |
| **total** | **0** | **23** | **70** | **7** |

## Three readings, in order of how much they change

**1. Zero occluded, in 100 frames.** Not one frame showed an animal genuinely
hidden — no keypoint that a human could not have placed. The footage-limit story
as I told it does not survive at the frame level.

**2. Twenty-three frames visible but wrong.** The landmark is plainly there and
the network put it somewhere else. By the read-out agreed in advance, *substantial
visible-but-wrong means network failure and the ensemble decision reopens.*
**It reopens.**

**3. The mask is real but weak, so the third outcome is ruled out.** Flagged
frames score visible-but-wrong **16/50 (32%)** against unflagged **7/50 (14%)** —
2.3× enrichment, so the mask is not measuring the wrong thing. But **29 of 50
flagged frames look fine**, and 7 unflagged frames look wrong. It is a noisy
detector of a real signal, not a broken one.

**Worst-decile and passing recordings score identically per flagged frame** —
8/25 visible-but-wrong in both. The worst decile is not worse footage; it has
*more* flagged frames of the same kind. That is consistent with H1's
concentration being about rate rather than severity, and it is not what I
expected.

## A named failure mode

Six frames show the same thing: **a keypoint locked onto a bright object above
the arena**, with a long bone stretching from the animal up to it.

| frame | flagged | recording | rate |
|---|---:|---|---:|
| f029 | yes | `…Box_2_CFC_Day_2_(Context_C)_711` | 6.92% |
| f048 | yes | `…Box_3_CFD_Day_3_(Context_B)_9058` | 6.70% |
| f056 | yes | `…Box_3_CFD_Day_7_(Context_B)_234` | 32.18% |
| f062 | **no** | `…Box_2_CFD_Day_7_(Context_B)_9004` | 6.49% |
| f069 | yes | `…Box_3_CFD_Day_6_(Context_B)_608` | 10.39% |
| f078 | yes | `…Box_2_CFD_Day_5_(Context_B)_576` | 14.75% |

All six are from worst-decile recordings, across **three boxes, four days and two
contexts** — so it is not one session or one rig. This is a distractor failure:
a bright object in the upper field attracts a keypoint away from the animal. It
is exactly the kind of error that is **not occlusion**, that a human labeller
would never make, and that members trained on different splits would plausibly
disagree about — which is the case for an ensemble.

## What this does to H1's interpretation

H1 measured a real and clean effect: violation rate rises monotonically **3.7×**
from the middle of the arena to the edge. I read that as occlusion. **Zero
occluded frames says that reading was wrong.**

The effect is still there and still needs an explanation. The likeliest one,
visible in the sheets, is that the arena wall is a field of high-contrast
vertical bars, and an animal against them sits in front of a strong distractor
texture. That is a **contrast and distractor** problem, not a visibility one —
and unlike occlusion it is something better detections could fix.

H1's other findings stand: 350× overdispersion, half the failure in a tenth of
the recordings, and no effect of apparatus unit.

## What is weak here, stated plainly

**The scorer is the least reliable instrument in this pipeline.** It is a
language model reading 320-pixel grayscale crops, and its one previous attempt in
this programme was wrong — two frames read as "rearing against a wall" turned out
to be teleports, corrected by a contact sheet. That is why this was blinded and
why the sheets were used again.

**The blinding leaks, and the leak has a direction.** A bone violation *looks
like* a stretched skeleton, which is visible in the render. So the scorer can
partially infer the flag despite `flagged=False`, and that **inflates the
flagged-versus-unflagged contrast** (32% against 14%). The enrichment is real but
its size should not be quoted as a measurement.

Two findings do **not** depend on the leak: **zero occluded**, which is a
property of the footage rather than of the mask, and the **bright-object mode**,
which was identified from image content.

**No counterexample was recalled from memory.** I have not watched this footage
and had no specific recording in mind; anything offered as one would have been
invented. The frames here are a seeded sample, and the six bright-object frames
are named above so they can be checked individually.

## Consequence

`ENSEMBLE_PROPOSAL.md`'s conclusion — "indicated but marginal, and H1 suggests
its return would be further limited by the shape of the errors" — **is
withdrawn**. The shape argument rested on the errors being occlusions that every
network would share. They are not occlusions.

The costs in that document stand unchanged: no labeled set, no trained model, no
DLC environment, ~95 GPU-hours of inference, and a full downstream re-run. What
has changed is the expected return, upward.
