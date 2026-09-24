# Registration — STABILISE 2: a background the animal is not in, and gates that can inform

Committed **alone, before any code for it exists**, and before any frame has
been registered against the background it specifies. Inherited digest
**`198eb14ff258c7f6`**.

This is a **new instrument**, registered because `STABILISE.md` refused both
keypoint-free arms. It is not an amendment of STABILISE: that stage's verdicts
stand as published. Everything not changed below is inherited from
`STABILISE_PREREGISTRATION.md` **with its Amendment 1**, including the scope
lock (§10.6: a PASS licenses eligibility only).

---

## 0. Why, from the published record only

* **The background contained the animal.** `STABILISE.md` §2: B's median mask
  covered 0.151 × bl² at the median recording, against a mouse outline of
  roughly 0.3–0.4 × bl², and 213 of 300 recordings were refused (Spearman
  −0.575 between mask size and refusal). A 101-frame median keeps an animal
  that holds one place.
* **Gate 4 could not be met by any registration** (D23): its yardstick was the
  arena's sensor noise, 50× below the resampling residual of a perfect
  registration.
* **Gate 5's null did not match its plants** (D23).
* **Gate 3's per-animal floor was set without counts** (D23).

**Arm S (SAM) is not carried forward.** As STABILISE §1 wrote it, S's frames
are refused on *B's* mask area, so S would have inherited B's refusal whatever
its own mask did. A SAM arm needs its own registration, with its own refusal.

## 1. The one change to the arms: B's background

**B₂'s background is the per-pixel median over 301 evenly spaced frames, taking
at each pixel only the samples in which that pixel lies OUTSIDE the animal's
exclusion box.** The box is `motion.exclusion_mask`, dilated by
`DILATE_BODY_LENGTHS = 1.0`, the arena definition `PIXEL_PREREGISTRATION.md`
already uses. A sample with an undefined box (too few keypoints) contributes no
pixel at all, as in the arena floor.

* **Why this does not break §0's principle.** Keypoints decide *which samples*
  enter a single static image per recording. They **locate** where the animal
  is not. That image is the same for every frame, so frame-to-frame keypoint
  jitter cannot reach any registered frame through it.
* **Undefined pixels.** A pixel with fewer than **10** usable samples takes the
  unmasked median of all 301. A recording in which more than **1%** of pixels
  are undefined is **refused** for B₂ and P₂.
* **Everything else in B₂ is B:** the blur, the `m2` cutoff, the opening, the
  largest component, the moments, continuity, the warp, the window-median disc,
  and the 0.5–2× area refusal (now against B₂'s own mask).
* **P₂ is P** (Amendment 1's phase-correlation-initialised ECC), placed and
  refused by B₂'s mask, exactly as P was by B's.
* **K is K, unchanged**, and §7's incumbent check applies again: max |diff| ≤
  1e-9 against the stored grooming series, compared as float32, or the stage
  stops.

## 2. Gates

**Gate 1 (duplicate frames)** and **gate 2 (jitter decoupling)** are inherited
**unchanged**: population, statistic, bars and the 20-animal / 30-window floors.

**Gate 3 (immobility floor):** statistic and bar unchanged (upper bound ≤ 1.10×
the arena's own noise). The per-animal minimum changes from 50 immobile frames
to **20**. That floor is derived from the census counts STABILISE produced,
which are **arm-independent** (they depend on speed and the pixel pilot's
arrays, never on a registration): 27 of 30 animals reach 20 frames, against 15
at 50. The 20-animal floor is unchanged.

**Gate 4 (planted trajectory), rebased on a perfect registration.** Same paste,
same real trajectory and the same jittered pose track for K (STABILISE §5). The
yardstick is now an **oracle**: each planted frame is carried back to the
source frame by the **true** inverse transform, consecutive frames are
differenced, and the discs are placed at the source frame's skull and hip
centroids. The oracle residual is what resampling alone costs.

> **Statistic:** per recording, the arm's median in-disc |Δ| over the 60 planted
> pairs divided by the oracle's. Mean per animal, animal-level bootstrap.
> **PASS iff the upper bound ≤ 1.25.**

**The bar is calibrated before the run, on synthetic ground truth**
(`tests/test_register.py`'s scene, three speeds, head disc at 0.6 bl):

| arm | ratio to oracle |
|---|---|
| K, perfect keypoints | 1.00 |
| B, clean background | 1.02–1.05 |
| P | 1.06–1.11 |
| K, keypoint jitter σ = 1.5 px | 4.67–7.67 |
| K, jitter σ = 3 px | 8.29–13.79 |

1.25 clears every working registration there with margin and sits far below
the jitter regime. This is M12 applied *before* the result: the bar is set
against what a perfect registration achieves.

**Gate 5 (planted local motion): a curve, no verdict, with a matched null.**
Only the **moving** condition is scored. In the still condition B₂ and P₂ are
identically zero between unplanted frames, which is what broke STABILISE's
null. Per arm and **per region**, the detection threshold is the 95th
percentile of `peak_excess` over that arm's and that region's **own unplanted
moving windows**. Published per (frequency, amplitude): head recall, hip
recall, and **head minus hip**. A registration that isolates local motion
shows head ≫ hip. One whose mask moments follow the patch shows them equal,
which is STABILISE's pattern, now readable.

## 3. Verdicts, sample, prohibitions

Inherited from STABILISE §8–§10, with the gates above. An arm is **PASS** iff
gates 1–4 pass. The sample is the same **300 `fit` recordings**, with no new
draw. **No tuning:** 301 samples, the 10-sample and 1% undefined-pixel rules,
the 20-frame floor and the 1.25 bar are fixed here. A parameter that turns out
to matter is swept and published in an amendment committed alone, never
picked. **Nothing is read as grooming**, and nothing as shock versus no-shock.

## 4. Incumbents (M12)

| quantity | incumbent | source |
|---|---|---|
| B's refusal, 101-frame median | 213 of 300 recordings | `STABILISE.md` |
| B's median mask | 0.151 × bl² | `STABILISE.md` §2 |
| K, duplicate frames non-zero | 100% of 23,910 | `STABILISE.md` |
| K, jitter → head energy on still windows | +0.340 [+0.285, +0.392] | `STABILISE.md` |
