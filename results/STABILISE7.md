# STABILISE 7: on the animal's pixels SM6 passes the still gate and fails the moving one; the line stops

Registered in `STABILISE7_PREREGISTRATION.md` (`7e996ce`), after the
incumbent-only calibration (`f9d6a6c`). `scripts/stabilise7.py`,
`jobs/stabilise7.slurm` (12 A100 shards, all `COMPLETED`, all 300
recordings). §7 holds: **max |diff| 0.0**. SAM's accepted keyframe masks are
saved with every record.

**Headline. SM6: `FAIL`, on gate 4 alone.** On the animal's own pixels it
passes every other gate: still frames, duplicates, planted jitter, the head
disc on the animal, and body motion against the keypoints. On moving planted
trajectories it is **2.129× [1.934, 2.346]** a perfect registration, against a
bar of 1.60, which is where a registration with *perfect* keypoints sits
(1.43). **By the stop rule fixed in advance, the registration line ends
here.** No stage 8 is registered.

| gate (animal pixels for 2–5) | K | SP | **SM6** |
|---|---|---|---|
| 0. disc on the animal, ≥ 90% | PASS 91.9% | PASS 93.7% | **PASS 93.7% [92.3, 95.0]** |
| 1. duplicate frames exactly zero | FAIL 100% | PASS 0 / 22,950 | **PASS 0 / 22,932** |
| 2. planted jitter, [0.90, 1.10] | control 9.90× [7.71, 12.27] | PASS 1.000 | **PASS 1.000 [0.999, 1.000]** |
| 3. still frames vs identity, [0.80, 1.25] | FAIL 1.721 | PASS 1.011 [1.002, 1.020] | **PASS 1.028 [1.003, 1.060]** |
| 4. moving plant vs oracle, [0.80, 1.60] | FAIL 8.264 | FAIL 38.95 | **FAIL 2.129 [1.934, 2.346]** |
| 6. chained keypoint agreement, β and f in [0.90, 1.10] | not scored | FAIL 0.166 … 0.562 | **PASS 0.952 … 1.067** |

## 1. STABILISE 6's reading is confirmed

On the floor-including 0.6 bl disc, SM6's still-frame ratio was 2.279. On the
animal's own pixels it is **1.028**. The failure was the bar floor inside the
disc, moved by SM6's sub-pixel fits to the animal (`STABILISE6.md` §2). **On a
still animal, SM6's registration is within 3% of doing nothing, which is the
correct answer.**

K's gate-3 value here (1.721) is higher than the calibration's (1.487). The
calibration sampled the first 60 immobile pairs per recording; this run uses
all of them. The bar was set from the calibration and is unchanged.

## 2. Where SM6 falls short: moving animals

On the planted moving trajectories, measured on the animal's pixels in a 0.2
bl head disc, SM6 leaves **2.1×** the residual of the true transform. A
registration with perfect keypoints leaves 1.43×. **SM6 is about 1.5× worse
than a perfect-pose registration on a moving head.** Over 10-frame chains on
real video it tracks the body's bulk motion correctly (0.95–1.07), so the
shortfall is in fine alignment of the head region during motion, not in
following the animal. This stage does not isolate its cause. Plausible
contributors are the rigid model on a bending body, ECC fitted over the whole
eroded body rather than the head, and the mask carried from a keyframe up to
one frame away. None was tested, and none is claimed.

## 3. The registration line, concluded (STABILISE 1–7)

Seven registered stages asked whether this corpus's head region can be aligned
without keypoints well enough to carry a crop feature.

* **What is established.** The incumbent pose-driven crop reports motion on
  every byte-identical frame pair, and 2 px of keypoint jitter multiplies its
  still-window head energy about 10–12×. On the animal's pixels, SM6 (SAM
  prompted from the keypoint box, ECC on SAM's eroded masks, from the better
  of two starts) is **exact on duplicates, immune to keypoint jitter, within 3%
  of perfect on a still animal, and follows the body on real video**.
* **What is not.** SM6 does not reach perfect-pose quality on a moving head
  (2.1× against 1.43×). **Under these gates, keypoint-free alignment of the
  head region is not solved for moving animals.** That is the recorded
  conclusion, as the stop rule required.
* **What the stop rule does not decide.** Every failure SM6 has left is on
  **moving** frames. A grooming measure is by definition a *still-body*
  question. Whether a grooming stage restricted to still windows could use
  SM6, where it passes every gate that applies, is a **different claim**. It
  would need its own registration, and the decision to make it belongs to the
  investigator, not to this line. It would still be precision-only: there is
  no grooming ground truth (`REVIEW.md`).
