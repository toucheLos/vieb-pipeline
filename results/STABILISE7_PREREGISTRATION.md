# Registration — STABILISE 7: score the gates on the animal's own pixels

Committed **alone, before the stage-7 run**, and after the incumbent-only
calibration (`results/stabilise7_calibration.json`). Inherited digest
**`198eb14ff258c7f6`**.

`STABILISE6.md` §2: SM6 passes gates 0, 1, 2 and 6 and fails gates 3 and 4.
Its gate-3 ratio grows with the disc (1.75 / 2.28 / 2.84 at 0.4 / 0.6 / 0.8
bl) while SP's stays flat. The reading is that the disc reaches the bar floor,
which SM6's sub-pixel fits to the animal move. **A grooming measure concerns
the animal's pixels, so this stage scores the gates there.** It is also the
direct test of that reading: if the reading is right, SM6's still-frame ratio
on animal pixels falls toward 1; if not, it does not, and the reading is
withdrawn.

**Everything not listed below is `STABILISE6_PREREGISTRATION.md` unchanged:**
the SAM track and its refusals, SM6 (masked ECC from the better of two starts),
SP and K as carried arms, §7's exact check, gate 0, gate 1, gate 6 (chained,
k = 10), the sample and the prohibitions. STABILISE 5's arm SM is not carried.

---

## 1. The region: disc ∩ the animal

"On the animal's pixels" means the disc **intersected with SAM's mask, eroded
0.04 bl** (STABILISE 5 Amendment 1), for the frame the difference is taken in:
*t−1*'s image coordinates for SP, SM6 and the identity, and carried into K's
body frame by K's own warp for K. For gate 4's oracle it is the source frame's
eroded SAM mask. A pair whose intersection is empty is not scored.
Implementation: `register.scan_track(region_fn=…)`, keys ending `|a`, and
`stabilise._oracle(region=…)`, both committed at `34b6b56` before this
registration.

## 2. The gates on animal pixels, and their bars

* **Gate 2** (planted σ = 2 px jitter, K as the positive control): the energy
  is taken on animal pixels. The bar is unchanged, [0.90, 1.10].
* **Gate 3** (immobile frames against the identity, 0.6 bl disc ∩ animal):
  **PASS iff the animal-level interval lies within [0.80, 1.25]**, unchanged.
  The calibration puts K at 1.487 [1.286, 1.719], so the bar still excludes
  the incumbent, **by a narrow margin** (0.036 at K's lower bound). That margin
  is stated here and not widened.
* **Gate 4** (moving plant against the oracle, 0.2 bl disc ∩ animal): **PASS iff
  the animal-level interval lies within [0.80, 1.60].** The upper bar rises
  from 1.25 because the calibration shows that **even a registration with
  perfect keypoints** sits at **1.433 [1.304, 1.570]** of the oracle on animal
  pixels, where the oracle's own residual is only 0.18 grey levels and mask
  edges and interpolation dominate. A keypoint-free alignment is not required
  to beat perfect poses. The jittered incumbent sits at 8.264 [6.435, 10.323],
  far outside.
* **Gate 5:** a curve, no verdict, at 0.6 bl ∩ animal.

**SM6 is PASS iff gates 0, 1, 2, 3, 4 and 6 all pass.**

## 3. Masks are saved

Every accepted keyframe's SAM mask is saved, bit-packed with its offset, so
any later re-scoring of this sample needs no GPU.

## 4. The stop rule, fixed before the run

This is the **last stage of the registration line** unless SM6 passes.

* **If SM6 passes every gate**, it becomes eligible for a separately registered
  grooming stage. With no grooming ground truth (`REVIEW.md`), that stage can
  only ever be scored for **precision**, by blind panels with mixed controls.
* **If SM6 fails gate 3 or gate 4**, the line stops. The recorded conclusion is
  that keypoint-free alignment of this corpus's head region is not solved by
  SAM + ECC under these gates, with the failing gate named. No stage 8 is
  registered to rescue it.

**Nothing here is read as grooming**, and nothing as shock versus no-shock.

## 5. Incumbents (M12)

| quantity | incumbent | source |
|---|---|---|
| SM6, gate 3 at the 0.6 bl disc (floor included) | 2.279 [1.844, 2.773] | `STABILISE6.md` |
| SM6, gate 4 at the 0.2 bl disc (floor included) | 1.305 [1.266, 1.344] | `STABILISE6.md` |
| K on animal pixels, gate 3 / gate 4 | 1.487 [1.286, 1.719] / 8.264 [6.435, 10.323] | `stabilise7_calibration.json` |
| K0 (perfect poses) on animal pixels, gate 4 | 1.433 [1.304, 1.570] | `stabilise7_calibration.json` |
