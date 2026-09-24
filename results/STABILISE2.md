# STABILISE 2: a better mask, still refused, and two more gates that do not inform

Registered in `STABILISE2_PREREGISTRATION.md`, committed alone as `7087eaf`
before any code. `scripts/stabilise2.py` (it reuses `scripts/stabilise.py` by
loading it), `jobs/stabilise2.slurm` (12 shards, all `COMPLETED`). Digest
**`198eb14ff258c7f6`**. Sample: the same 300 `fit` recordings, all scanned.
§7 holds: K reproduces the incumbent with **max |diff| 0.0**.

**Headline.** B₂ and P₂ are **`NOT_A_RESULT` again**: **298 of 300 recordings
(99.3%)** are refused. **No arm is eligible**, and the keypoint-free route
through a median background is closed for this corpus by two registered stages.

## 1. The masked background works, and it is still refused

The change did what it was for. B₂'s median mask covers **0.311 × bl²** at the
median recording, inside a mouse's 0.3–0.4 range, against **0.151** under
STABILISE's plain median. The mask-area refusal alone would reject **159 of
300 (53%)**, against STABILISE's 213 (71%). Better, but still far over the 20%
arm limit.

**What refuses almost everything is §1's undefined-pixel rule.** A pixel is
undefined when fewer than 10 of the 301 samples see it with the animal's box
elsewhere, and a recording is refused past 1% undefined. At the median
recording **7.5%** of pixels are undefined (10th–90th percentile 2.3%–20.2%,
maximum 44%), and **294 of 300** exceed 1%. A 1-body-length box around an
animal that holds one place for most of a session covers that place in nearly
every sample. The same behaviour that put the animal into STABILISE's
background leaves B₂ with no view of the floor beneath it.

**Both refusal rules independently exceed the arm limit.** The verdict does not
hinge on the 1% choice. Relaxing it would still leave 53% refused on mask area
alone.

## 2. What K's gates show, now that two of them are readable

| gate | K |
|---|---|
| 1. duplicate frames | **FAIL**: 23,910 of 23,910 non-zero, as in STABILISE |
| 2. jitter reference | +0.340 [+0.285, +0.392], as in STABILISE |
| 3. immobility floor, 26 animals at ≥ 20 frames | FAIL: 3.99 × 10⁵ × arena noise, **not interpretable** (§3) |
| 4. oracle ratio, 30 animals | FAIL: **1.221 [1.162, 1.298]**, bar 1.25, **weakly informative** (§3) |

## 3. Two defects in this registration, recorded and not corrected (`DEVIATIONS.md` D24)

**Gate 3 is ill-conditioned.** It averages per-frame ratios of in-disc |Δ| to
that frame's arena |Δ|. On genuinely immobile frames the arena term is close to
zero, so single frames produce ratios in the millions and the mean follows
them. The statistic measures how near zero the denominator gets, not
registration. A ratio of means, or a difference in grey levels, would have
been well posed. That was not what was registered.

**Gate 4's calibration did not transfer from the synthetic scene.** On
`tests/test_register.py`'s scene, jittered K sat at 4.7–13.8× the oracle. On
the real pastes the oracle's own residual is about an order of magnitude larger
(about 6 grey levels on the smoke recording, against about 0.5 synthetic).
Hard paste edges and real floor and fur texture dominate the resampling cost,
and jittered K comes out at only **1.22×**. The bar still ranks correctly, but
its margin over the jitter regime collapsed from about 4× to about 1×. One
smoke observation also showed a **lower** tail the one-sided bar cannot see:
P₂ at 0.29× the oracle, possible only if its disc is off the moving animal.
A future version needs a two-sided bar and a calibration on real pastes.

## 4. What this licenses

**License:** on this corpus a median background, plain or keypoint-masked, does
not yield a mask stable enough for registration. The plain median contains the
animal. The masked median cannot see the floor where the animal rests.

**License:** STABILISE's incumbent measurements stand, reproduced exactly
here.

**Do not license** any claim about keypoint-free registration itself. No
keypoint-free arm has reached a verdict in either stage.

**What is owed** is a mask that does not come from a background at all: a
segmenter such as SAM, prompted without keypoints, with its own refusal rule.
That needs a checkpoint and GPU time, and its own registration. The gate 3 and
gate 4 fixes in §3 belong in it.
