# Registration — STABILISE 5: ECC on the animal only, judged against the keypoints on real motion

Committed **alone, before any code for it exists**, and before any masked
alignment has been run on a sample frame. Inherited digest
**`198eb14ff258c7f6`**.

`STABILISE4.md`: SP passed every still-frame gate (0–3) and failed the moving
plant (gate 4, 17× the oracle). The mechanism, corrected after inspecting the
images (§2 there): **unmasked ECC in SAM's padded box settles between the
static bar floor and the moving mouse**. It leaves 67–82% of the in-disc
difference at every speed, and recovers a median 8.8% of the keypoint
displacement on real fast frames. This stage changes that one thing, and adds
the real-motion test the investigator proposed. **Everything else is
`STABILISE4_PREREGISTRATION.md` unchanged:** the checkpoint and its hash, SAM
prompted from the padded keypoint box at every keyframe, the keyframe stride,
the IoU, area and centroid-in-box refusals, the bracketing refusal, the 10% and
20% limits, gates 0, 1, 3 and 4 with their calibrated [0.80, 1.25] bands, the
planted-jitter gate 2 with K as its positive control, gate 5 as a curve, the
sample and the prohibitions.

---

## 1. The new arm: SM, ECC restricted to the animal

* **Mask at *t−1*:** SAM's mask from the nearest accepted keyframe, translated
  by SAM's own interpolated centroid shift from that keyframe to *t−1*, then
  dilated by **0.15 body lengths**. It is image-derived throughout, with no
  keypoint involved. The dilation covers interpolation error over at most one
  frame and keeps a thin band of floor at the body's edge.
* **Initial warp:** a pure translation equal to SAM's interpolated centroid
  shift from *t−1* to *t*, with no rotation. It replaces the unmasked phase
  correlation, which is what the floor pulled toward zero.
* **Refinement:** `cv2.findTransformECC`, `MOTION_EUCLIDEAN`, the same criteria
  and `gaussFiltSize` as Amendment 1, over SAM's padded box, **with the mask above
  as `inputMask`**. Failure makes the pair NaN, as before.
* **Everything downstream is SP's:** differencing in *t−1*'s coordinates, the
  window-median disc, the identity reference, the snap tolerance.

**SP is carried unchanged** as the comparison arm, so this stage re-measures
STABILISE 4's failure beside its proposed fix. **K** is the incumbent, with
§7's exact check.

## 2. The new gate: agreement with the keypoints on real moving frames (gate 6)

The planted sequences of gate 4 are artificial: a pasted animal over a masked
median background. **Gate 6 asks the same question on real video**, using the
DLC keypoints as the reference for where the body went. That is the kind of
thing keypoints are good at even where they fail on fine head motion.

* **Population:** per recording, frame pairs (*t−1*, *t*) whose egocentric
  keypoint speed at *t* is above the recording's **75th percentile**; whose
  `CENTER` keypoint is present and **not interpolated** in both frames
  (`spine.clean`); and whose arm warp exists and is not refused.
* **Displacements:** `d_kp = CENTER(t) − CENTER(t−1)`, and `d_arm = W·c − c`,
  where `c = CENTER(t−1)` and `W` is the arm's full-frame warp carrying *t−1*'s
  coordinates into *t*'s.
* **Statistic:** per animal, pooled over its recordings' pairs,
  **f = Σ|d_arm|² / Σ (d_kp · d_arm)**. This is the inverse of the regression of
  `d_kp` on `d_arm`, so zero-mean keypoint error, independent of the arm, sits
  in the **dependent** variable and does not attenuate the estimate. Gate 2
  shows planted keypoint jitter does not reach SP (1.000×), so the independence
  is measured, not assumed. f ≈ 1 means the arm follows the body. An arm that
  recovers a fraction *q* of the motion gives f ≈ *q*. STABILISE 4's 8.8%
  predicts SP at about 0.1.
* **PASS iff** the animal-level interval (2,000 replicates, seed 0) lies within
  **[0.90, 1.10]**, over ≥ 20 animals with ≥ 200 pairs each.
* **K is not scored on gate 6.** Its warp is built from the keypoints, so its
  agreement with them is circular. It is reported for completeness and not read.
* **A stated limit:** gate 6 measures agreement with DLC's `CENTER` trajectory.
  If DLC were itself biased on fast frames (lag, or motion blur pulling the
  point), a perfect registration would score away from 1. The cleaned pose is
  the programme's reference for bulk motion, and that dependence is written
  down here rather than discovered later.

## 3. Verdicts and compute

**SM is PASS iff gates 0–4 and 6 all pass**; FAIL if any fails; `NOT_A_RESULT`
under the refusals. SP and K are reported. SM's masked ECC also runs inside the
planted-jitter windows (gate 2) and the planted sequences (gates 4 and 5).
Compute: 12 A100 shards, as STABILISE 4, plus one extra ECC per frame pair on
CPU. **Nothing is read as grooming**, and nothing as shock versus no-shock.
There is no grooming ground truth for this corpus (`REVIEW.md`), so even a PASS
here would lead only to a precision-scored grooming stage, never a recall one.

## 4. Incumbents (M12)

| quantity | incumbent | source |
|---|---|---|
| SP, gate 4 (moving plant) | 17.05 [15.34, 18.74] | `STABILISE4.md` |
| SP, recovered shift on real fast frames | median 8.8% of keypoint displacement | `STABILISE4.md` §2 |
| SP, gates 0–3 | PASS, PASS, PASS, PASS | `stabilise4.json` |
| K, gate 4 | 3.93 [3.14, 4.87] | `stabilise4.json` |
