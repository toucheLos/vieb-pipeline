# REPAIR — the pose, repaired with SAM as a second detector

Registration: `REPAIR_PREREGISTRATION.md`. Result: `results/repair.json`. Report
split: 89 animals, 6.68 M frames, animal-level bootstrap (2,000 replicates).

## Verdict: PASS. The behaviour stage uses `repaired`.

| gate | statistic | result | bar | |
|---|---|---|---|---|
| 1. skull jitter | repaired − raw | −0.0063 [−0.0079, −0.0048] | upper < 0 | PASS |
| | repaired − disposition | −0.0047 [−0.0057, −0.0038] | upper < 0 | PASS |
| 2. bone violations | repaired − disposition | +0.0002 [−0.0003, +0.0007] | upper ≤ 0.001 | PASS |
| 3. HF retention | repaired | 0.314 [0.296, 0.332] | lower ≥ 0.220 | PASS |
| 4. distortion (px) | repaired | 1.14 [1.01, 1.28] | upper ≤ 1.555 | PASS |

Per arm (point [95%]):

| arm | skull jitter (/bl) | violation rate | HF retained | distortion (px) |
|---|---|---|---|---|
| raw | 0.0351 [0.0309, 0.0398] | 0.0178 [0.0163, 0.0195] | 1 | 0 |
| disposition | 0.0335 [0.0298, 0.0375] | 0.0142 [0.0127, 0.0158] | 0.897 | 0.05 |
| repaired (veto) | **0.0289 [0.0260, 0.0319]** | 0.0144 [0.0128, 0.0160] | 0.314 | 1.14 |
| no-veto fallback | 0.0291 [0.0261, 0.0322] | 0.0149 [0.0132, 0.0166] | 0.320 | 1.14 |

**What the repair buys.** Skull jitter falls by 18% against raw, and the
violation rate by a fifth, to disposition's level. It keeps 31% of the
high-frequency power, the bakeoff's `median_0.25` figure (32%), against
Wiener's 22%: less smoothing than the pose shapeflow shipped.

**Almost all of it is the filter, not the veto.** The veto and no-veto arms
differ by less than their intervals on every metric. The mask veto removes
**0.07%** of masked keypoint-frames, as registered: 0.3 bl catches only
keypoints off the animal.

## Reported, not gated

* **A carried mask on 96.2% of frames**: 98.9% in context A, 91.5% in B.
* **Abstained frames: 1.3%**, 0.7% in A and 1.7% in B. **3.4%** of
  keypoint-frames stay missing (runs longer than 3 frames).
* 0.48 M report frames come from recordings whose ID carries no context
  label. They are repaired like the rest and marked `?` downstream.
* **The veto by context is flagged.** It removes 0.036% of masked
  keypoint-frames in A and 0.115% in B, a ratio of **3.2**, above the
  registered flag of 2. The **nose** takes 72% of all vetoes (22,277 of
  30,757). Two readings fit, and neither is settled here:
  - DLC places the nose off the animal more often in B (the wall points seen
    in `AGREEMENT.md`);
  - SAM clips the head more often in B, by more than 0.3 bl
    (`POINT_PROMPTS_GUARDED.md`: B clips about 2× as often).

  Either way, the absolute rate is about one keypoint-frame in a thousand. It
  cannot drive the behaviour stage, whose context comparisons are gated
  separately (`STATES_PREREGISTRATION.md` §2).
