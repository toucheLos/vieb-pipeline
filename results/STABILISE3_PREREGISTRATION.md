# Registration — STABILISE 3: a segmenter mask, and gates calibrated on real data

Committed **alone, before any code for the SAM arms exists**, and before SAM has
segmented any frame of the sample for this stage. (A 90-frame throughput check
on one recording, not scored, is reported in §6.) Inherited digest
**`198eb14ff258c7f6`**.

Registered because two stages refused every keypoint-free arm built on a median
background (`STABILISE.md`, `STABILISE2.md`). The animal rests in one place, so
a plain median contains it and a masked median never sees the floor beneath it.
**This stage takes the mask from a segmenter, with no background model at
all.** Everything not changed below is inherited from STABILISE (with its
Amendment 1) and STABILISE 2, including the §10.6 scope lock: a PASS licenses
eligibility, nothing more.

---

## 1. The mask: SAM ViT-B, on every third frame, prompted by its own last mask

* **Model:** `segment-anything` 1.0, `vit_b`, checkpoint
  `sam_vit_b_01ec64.pth` (375,042,383 bytes; SHA-256
  `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`; MD5
  prefix `01ec64` matches the filename). Held outside the repository at
  `~/models/sam/`, not vendored. `SamPredictor`, one box prompt,
  `multimask_output=False`, `inference_mode`, fp32, one A100 per shard.
* **Keyframes:** every frame with `t % 3 == 0`.
* **Prompt:** the previous **accepted** keyframe's mask bounding box, padded by
  0.25 body lengths. A **seed** is needed at the first keyframe, and after any
  keyframe the online rule refuses. It is the keypoint bounding box at that
  frame, padded by 0.25 body lengths. This is §0's **locate**: it tells SAM which
  object to cut, at most once per refused stretch. The mask SAM returns is drawn
  from image edges, not from the box.
* **Online refusal:** SAM's predicted IoU **< 0.80** refuses the keyframe and
  forces a re-seed at the next one.
* **Post-hoc refusal:** area outside **0.5–2×** the recording's median accepted
  keyframe area.
* **Between keyframes:** mask centroid, unwrapped angle (STABILISE §1 B.4
  continuity) and bounding box are **linearly interpolated** between the two
  bracketing accepted keyframes. A frame is refused if either bracket is.
* **Recording and arm refusal:** unchanged. More than 10% of frames refused
  refuses the recording; more than 20% of recordings refused makes the arm
  `NOT_A_RESULT`.

## 2. The arms

| arm | frames registered by | disc placed by |
|---|---|---|
| **K** | the incumbent, unchanged; §7's exact-reproduction check applies | per-frame pose, unchanged |
| **SB** | the moment warp of STABILISE's B, on SAM's interpolated mask pose | window-median skull / hip centroid in SAM's mask frame |
| **SP** | Amendment 1's phase-correlation-initialised ECC, in SAM's interpolated box at *t−1* | as SB, carried into *t−1* by SAM's mask pose |

## 3. Gates

**Gate 0 (new): the disc is on the animal.** On accepted keyframes, the head
disc centre (image coordinates, as placed for SP) must lie **inside SAM's
mask**. Per animal, the share of keyframes; animal-level bootstrap. **PASS iff
the lower bound ≥ 0.90.** This guards the lower tail STABILISE 2 exposed (D24).

**Gates 1 and 2:** inherited unchanged (exact zero on duplicate frames; jitter
decoupling on speed-selected still windows, upper bound < 0.10 and below K's
lower bound).

**Gate 3, re-posed as a ratio of means against the identity.** On census
immobile frames the animal is still, so **no transform at all is the perfect
registration**. Per animal: the sum of the arm's in-disc |Δ| over immobile
frames, divided by the sum of the **identity** |Δ| (`|b − b_old|`, unwarped)
in the same image-coordinate disc. **PASS iff the animal-level interval lies
within [0.80, 1.25].** The floor is ≥ 20 immobile frames per animal and ≥ 20
animals, as in STABILISE 2.

**Gate 4, at an interior disc, two-sided.** The same planted trajectory and
oracle as STABILISE 2. The source frame's paste is cut with **SAM's** mask
(dilated 0.1 bl), over STABILISE 2's masked-median raw background. The disc is
**0.2 body lengths** at the source skull centroid, inside the body, where paste
edges cannot enter. Statistic: per recording, the arm's median in-disc |Δ| over
the 60 pairs divided by the oracle's. Mean per animal. **PASS iff the
animal-level interval lies within [0.80, 1.25].**

**Both bars are calibrated on this sample before this registration, using
only the incumbent** (`scripts/stabilise3_calibrate.py`,
`results/stabilise3_calibration.json`, commit `6857460`):

| | perfect registration (K, perfect poses) | incumbent (K, real jitter) |
|---|---|---|
| gate 4, 0.2 bl disc | **0.955 [0.945, 0.965]** | **3.093 [2.466, 3.861]** |
| gate 4, 0.6 bl disc (STABILISE 2's) | 0.996 [0.993, 1.001] | 1.221 [1.162, 1.298] |
| gate 3, ratio of means to identity | 1 by construction | **4.431 [2.885, 6.609]**, 25 animals |

[0.80, 1.25] contains the perfect registration and excludes the incumbent's
whole interval on both gates. Its lower edge catches a disc that has left the
moving animal (STABILISE 2's P₂ at 0.29× on its smoke recording).

**Gate 5:** a curve, no verdict, exactly as STABILISE 2 §2 (moving plants,
per-arm and per-region null, head minus hip), at the 0.6 bl discs. The planted
frames are keyframed and segmented like any other.

**Verdict:** an arm is **PASS** iff gates 0–4 all pass, and **FAIL** if any
fails. Refusals are `NOT_A_RESULT`, as before.

## 4. Sample, compute, prohibitions

The same **300 `fit` recordings**, with no new draw. 12 shards, one A100
each. **No tuning:** the keyframe stride, both paddings, the 0.80 IoU and
0.5–2× area rules, the interpolation, the 0.2 bl gate-4 disc and both bars are
fixed here. A parameter that turns out to matter is swept and published in an
amendment committed alone, never picked. **Nothing is read as grooming**, and
nothing as shock versus no-shock.

## 5. Incumbents (M12)

| quantity | incumbent | source |
|---|---|---|
| K, duplicate frames non-zero | 100% of 23,910 | `STABILISE.md` |
| K, jitter → head energy, still windows | +0.340 [+0.285, +0.392] | `STABILISE.md` |
| median-background refusals | 213 / 300 (plain); 298 / 300 (masked) | `STABILISE.md`, `STABILISE2.md` |
| gate 4 and gate 3 calibration | table above | `stabilise3_calibration.json` |

## 6. Disclosed before the run

A throughput check segmented the first 90 frames of one sample recording
(`20251117_Box_2_CFD_Day_3_(Context_A)_104`), prompted as in §1 but on every
frame, and scored nothing. It ran at 145 ms/frame on an A100. Mask area was
0.43–0.62 × bl² and the median predicted IoU 0.96. It informed the compute
estimate and the 0.80 IoU floor's plausibility, and no gate quantity.
