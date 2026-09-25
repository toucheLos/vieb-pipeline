# STABILISE 4: still frames are solved; moving frames are not

Registered in `STABILISE4_PREREGISTRATION.md` (`c28dcda`), changing only
STABILISE 3's three defects. `vieb/pixel/sam.py` (`prompt_mode="keypoint"`),
`scripts/stabilise4.py`, `jobs/stabilise4.slurm` (12 A100 shards, all
`COMPLETED`, all 300 recordings). The checkpoint hash was verified in every
shard. §7 holds: **max |diff| 0.0**.

**Headline. SP: `FAIL`.** It passes gates 0–3 and fails gate 4. **The
keypoint-free registration is exact and jitter-immune on a still animal, and it
does not follow a moving one.** SP is not eligible for a grooming gate v2.

| gate | SP | K (incumbent) |
|---|---|---|
| 0. disc on the animal, lower bound ≥ 90% | **PASS** 93.7% [92.3, 95.0] | 91.9% [90.5, 93.3] |
| 1. duplicate frames exactly zero | **PASS** 0 of 22,950 | FAIL 100% of 23,910 |
| 2. planted σ = 2 px jitter response in [0.90, 1.10] | **PASS** 1.000× [0.999, 1.000] | positive control 11.79× [8.93, 15.14] |
| 3. ratio to identity on immobile frames, in [0.80, 1.25] | **PASS** 1.038 [1.016, 1.063], 24 animals | 5.22 [3.11, 8.26] |
| 4. ratio to true-transform oracle on moving plants, in [0.80, 1.25] | **FAIL** 17.05 [15.34, 18.74] | 3.93 [3.14, 4.87] |
| refused recordings (limit 20%) | 56 / 300 (18.7%) | 0 |

## 1. The fix to STABILISE 3 worked

Prompting SAM from the padded keypoint box at every keyframe removed the drift.
The disc is on the animal on 93.7% of keyframes, against STABILISE 3's
66.9%, and refusals fell from 144 to **56** recordings, under the limit. On the
smoke recording, frame refusals went from 37.5% to 0.1%.

**The planted-jitter test (gate 2) is decisive where the correlational one was
not (D25).** Two pixels of i.i.d. jitter, in every keypoint input including
SAM's prompts, multiply the incumbent's head energy on still windows by
**11.8×**. The same jitter moves SP's by **0.0%** (1.000 [0.999, 1.000]).
**Jitter does not reach SP**, which settles D25's question for this arm by
construction rather than by correlation. It also sharpens D21: jitter at the
level DLC produces multiplies the incumbent's still-window head energy by an
order of magnitude.

## 2. Why SP fails on moving animals: ECC settles between floor and animal

> **Corrected after inspecting the evidence images.** An earlier version of
> this section said ECC "locks onto the floor". The plant images
> (`results/stabilise4/plant_*.png`) and a per-pair check show that is too
> strong. ECC does move with the animal, but only part of the way.

Gate 4 puts SP at **17×** a perfect registration on real planted trajectories,
worse than the jittered incumbent. Per recording the ratio runs 1.7× to 45×
(5th–95th percentile). SP's energy also tracks keypoint speed at **+0.918**,
above the incumbent's +0.815.

**Per pair, on the two inspected plants** (descriptive,
`scripts/stabilise_figures.py` and a scratch check; 113 pairs). SP's disc sits
1.6–7.9 px from the true skull (median), so placement is not the problem. Inside
the 0.2 bl disc, SP leaves **67–82%** of the unregistered difference at every
motion size, slow, mid or fast. The oracle leaves **9–25%**. ECC never snaps to
the identity (0 of 113 pairs). **SP registers partially, at every speed**:
the static bar floor, which fills most of the padded box, pulls the fit toward
zero, and the animal pulls it toward its motion. On real fast frames the same
compromise shows as a recovered shift of a median **8.8%** of the keypoint
displacement (first 59 recordings). A full-frame difference image can hide
this: the animal can look dark while the head disc keeps most of its
difference, because the colour scale saturates.

**That is the right answer for a still animal**, which is why SP passes gates
1–3 exactly, and the wrong one for a moving animal. Amendment 1 (D22) chose
unmasked ECC because, on the *synthetic* scene, a mask-multiplied crop did
worse. That scene's background was smooth noise, not a bar grid. **The choice
did not transfer to this corpus.**

## 2a. The most-refused recording: high contrast, not an attached object

`results/stabilise4/masks_20250227_Box_2_CFD_Day_6_(Context_B)_9025.png`, the
recording STABILISE 4 refused most, shows SAM's mask taking in a **bright white
region beside the head**, and sometimes the tail. **The investigator's reading
(2026-09-25) is that this recording's video is simply very high contrast**:
blown-out highlights next to the animal, not a tag or object attached to it.
That is recorded as their reading and not independently verified. On that
reading it is an imaging property of this recording, not a confound tied to
some animals, and it enters the analysis only through §1's refusals.

## 3. Gate 5

No arm detects the planted local motion above the null rate at any amplitude
up to 4 px. For SP this follows from §2: on moving plants, residual body motion
swamps a 0.15 bl patch.

## 4. What this licenses

**License:** SAM ViT-B prompted from the padded keypoint box at every keyframe
gives a mask that is on the animal (≥ 92% of keyframes), with 18.7% of
recordings refused.

**License:** SP's registration is **exact on byte-identical frames**, **within
4% of perfect on genuinely still frames**, and **immune to planted keypoint
jitter** (1.000×). For still-body analyses, which is where the grooming
question lives, that is the property D21 showed the incumbent lacks.

**License:** planted σ = 2 px jitter multiplies the incumbent's still-window
head energy 11.8× [8.9, 15.1].

**Do not license** SP for any window in which the body moves: it leaves about
two-thirds or more of the in-disc difference unregistered at every speed.

## 5. What is owed

**ECC restricted to the animal:** SAM's mask as ECC's `inputMask`, dilated
slightly, so the static floor cannot win. A new registration would need gate 4
to pass on moving plants while gates 0–3 still hold. The alternative is to
scope any SP-based stage to still windows, where gates 1–3 already pass, with
that scope registered in advance, and to accept that SP is FAIL overall.
