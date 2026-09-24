# Registration — STABILISE 4: prompt SAM from the located animal, and plant the jitter

Committed **alone, before any code for it exists**, and before SAM has been
prompted from a keypoint box at every keyframe on any sample frame. Inherited
digest **`198eb14ff258c7f6`**.

`STABILISE3.md` found three defects, and this stage changes **only those
three**. Everything else is `STABILISE3_PREREGISTRATION.md` **unchanged**: the
checkpoint (SHA-256 `ec2df627…eb8c912`), the keyframe stride of 3, the 0.25 bl
padding, the 0.80 predicted-IoU floor, the 0.5–2× area rule, the bracketing
refusal, the 10% and 20% refusal limits, gate 0 (≥ 90%), gate 1, gate 3 and
gate 4 with their calibrated [0.80, 1.25] bands, gate 5 as a curve, the sample
and the prohibitions. The refusal parameters are **deliberately not revisited**
after STABILISE 3's refusal rates were seen.

---

## 1. Change 1: SAM is prompted from the located animal at every keyframe

**Every keyframe's prompt is the frame's keypoint bounding box, padded by 0.25
body lengths.** There is no propagation from the last mask. STABILISE 3's
drift (`STABILISE3.md` §1, the published overlay) came from that propagation:
once a padded box touched the arena wall, SAM cut the wall and the design had
nothing to bring it back.

**Why this stays inside §0.** The box tells SAM *which object* to cut. The mask
is drawn from image edges, so a few pixels of box jitter cannot move its
boundary the way a pose-driven warp moves a whole crop. Whether that holds is
measured, not assumed, by §3.

**One added keyframe refusal:** the mask centroid must lie **inside the padded
keypoint box** it was prompted with. That is a located sanity check against a
mask of some other object. Gate 0 remains the independent test.

## 2. Change 2: one candidate arm, SP

**SB is dropped.** Registration by interpolated pose moves the crop between
byte-identical frames (STABILISE 3 gate 1: 97.4% non-zero). **SP** registers
each pair by Amendment 1's phase-correlation-initialised ECC in SAM's
interpolated box at *t−1*. It is the only candidate here. SAM's pose is still
interpolated between keyframes, but only to **place** the box and the disc.
**K** is carried as the incumbent, and §7's exact check applies.

## 3. Change 3: gate 2 becomes a planted-jitter test

D25: skull-bone variance may rise when the head really moves, so its
correlation with energy cannot tell leakage from co-movement. **The
replacement plants the jitter, so its size and timing are known and cannot
co-vary with behaviour.**

* **Population:** per recording, the first **10** 2-second windows below the
  recording's 25th percentile of keypoint speed. These are the still windows,
  by speed alone, as in STABILISE §3.
* **The plant:** i.i.d. Gaussian noise, **σ = 2 px**, added to **every**
  keypoint on every frame of those windows (seed 0). It enters **every place
  keypoints enter**: SAM's prompts, SP's disc placement, and K's warp.
* **The measurement:** each window is processed on its own, twice, once with
  the original pose and once with the jittered pose. Each run gets its own SAM
  keyframe track and its own `scan_track`. The response is **R = Σ head energy
  (jittered) / Σ head energy (original)**, per animal, at 0.6 bl, with windows
  that either run refused excluded from both sums.
* **Positive control (K):** if K's animal-level lower bound for R is **below
  1.25**, the plant is too weak to test anything. Gate 2 is then
  `NOT_A_RESULT` for every arm.
* **PASS iff** SP's animal-level interval for R lies within **[0.90, 1.10]**,
  and K's control is satisfied. ≥ 20 animals, as before.

## 4. Verdicts and compute

SP is **PASS** iff gates 0–4 all pass, **FAIL** if any fails, and
`NOT_A_RESULT` under the refusals. K is reported as the incumbent. The compute
is 12 A100 shards, as STABILISE 3, plus the jitter plant: 10 windows × 60
frames × 2 runs per recording. **Nothing is read as grooming**, and nothing as
shock versus no-shock.

## 5. Incumbents (M12)

| quantity | incumbent | source |
|---|---|---|
| on-animal share, SAM propagated from its own box | 66.9% [58.5, 75.0] | `STABILISE3.md` |
| recordings refused, same | 144 / 300 | `STABILISE3.md` |
| SP gate 1 | 0 of 15,444 non-zero | `STABILISE3.md` |
| K gate 3 / gate 4 | 3.54 [2.36, 5.02] / 3.93 [3.14, 4.87] | `STABILISE3.md` |
