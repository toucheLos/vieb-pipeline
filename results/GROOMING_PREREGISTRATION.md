# Registration — the 3–8 Hz grooming gate

Committed **alone, before `scripts/grooming_gate.py` or `vieb/pixel/head.py`
exist** and before any head-region pixel has been read. Inherited digest
**`198eb14ff258c7f6`**.

This is the **make-or-break** question for fine behaviour in VIEB. It decides
whether the video route is open at all, and §10 fixes the wording of the
negative outcome **before** the answer, because that wording is the deliverable
if the gate fails.

---

## 0. The question, and why keypoints cannot answer it

There are **no paw keypoints**. The LUNA skeleton is
`LEFT_EAR, RIGHT_EAR, NOSE, CENTER, LEFT_HIP, RIGHT_HIP, TAIL_BASE` — seven
points, none on a forelimb. Grooming is precisely the behaviour where keypoints
and pixels must disagree: the **body reads as still** while **pixels near the
head move**.

**The keypoint verdict is already in, and it is the incumbent (M12).**
`DYNAMICS.md`: `pole_radius` clears at 88.7% against 5% chance, but
`pole_frequency_hz` clears at **2.1% — below chance — and nothing above 2 Hz
clears at all**. So on keypoints, structure above 2 Hz is not recoverable. The
question here is whether **pixels** recover what keypoints could not.

**Sampling is tight and that is stated now, not after.** At 30 fps the Nyquist
limit is 15 Hz, so 8 Hz is **3.75 samples per cycle**. The band is legal but the
upper edge is the fragile one, and any result that depends on the 6–8 Hz
sub-band alone is reported as such rather than as "3–8 Hz".

## 1. The signal: head-region motion energy

Per frame, from the same sequential decode `vieb/pixel/motion.py` already does:

* **Head region** = a disc centred on the centroid of the **skull triangle**
  (`LEFT_EAR`, `RIGHT_EAR`, `NOSE`) — the same three points `bones.SKULL` uses —
  of radius **0.6 × the recording's median body length** (nose-to-tail_base).
  0.6 rather than 0.5 because the forepaws during a grooming stroke travel
  **in front of** the nose, and a disc that stops at the nose excludes the
  motion under test.
* **Energy** = the **mean |Δ|** inside the disc, after `GaussianBlur(SIGMA=1)`.
  Mean, not a thresholded count: a count above a cutoff is a coarse quantisation
  and the statistic in §4 is spectral.
* **Noise subtraction:** the same frame's **arena** mean |Δ| is subtracted, so
  the series is head motion **above that recording's own floor**
  (`PIXEL_NOISEFLOOR.md`: the floor spans 6.50–128.50 grey levels of derived
  cutoff across recordings, a factor of 20, so a global correction would be
  meaningless).
* Frames whose skull mask is undefined (fewer than 3 locatable keypoints, or
  fewer than 2 of the skull three) contribute **NaN**, and a window containing
  any NaN is dropped.

## 2. Candidates are DETECTED, and the obvious test on them is circular

There are no grooming labels. Candidates are windows of **`WIN_S = 2.0 s`**
(60 frames, non-overlapping) satisfying, **within recording**:

* mean egocentric keypoint speed **below that recording's 25th percentile**, and
* mean head-region motion energy **above that recording's 75th percentile**.

> **Selecting on high head-region motion and then asking whether head-region
> motion clears a floor guarantees the answer.** The selection is on the very
> quantity that would be under test. Two fixes, both applied, and neither
> optional.

## 3. Fix 1 — the clips are looked at

The claim "these are grooming" is a claim about **content**, not about a
threshold, and no statistic in this document establishes it.

**30 candidate windows are rendered as egocentric crops and confirmed by eye**,
drawn at seed 0 across at least 15 distinct animals, and **the confirmation rate
is published as the detector's own precision.** Alongside them, **15
speed-matched non-selected windows** are rendered and shuffled into the same
panel under opaque ids, so the confirmation is not made knowing which arm a clip
came from; the key is written beside the clips and is **not published**.

**If fewer than 50% of candidates are confirmed as grooming**, the detector does
not detect grooming and §4's statistic is **not read** — whatever it says. The
gate would then be measuring an unnamed behaviour.

## 4. Fix 2 — the statistic is a spectral PEAK, not an amplitude

Amplitude is what selection used, so amplitude cannot be the test. **Shape can.**

Per window, the multitaper PSD (`descriptors.multitaper_psd`, `NW = 3`,
5 tapers — 1.5 Hz resolution over a 2 s window, so the 5 Hz-wide band spans
about three resolution elements). Then:

> **The statistic is `peak_excess`: log-power in 3–8 Hz minus a smooth
> background fitted to `log P` as a linear function of `log f` over
> 0.5–15 Hz *excluding* 3–8 Hz.**

**Why this and not band share.** A band share (3–8 Hz power over total power) is
scale-free but not shape-free: selecting windows with more broadband motion can
move it. A **peak above an interpolated background** is a statement about the
spectrum's *shape at that frequency*, and it is invariant to any multiplicative
rescaling of the whole spectrum — which is exactly what selecting on amplitude
does. A real stroke rhythm is a narrow peak; motion-scaled noise is broad and
leaves no excess over its own background.

**Band share is computed too, and reported as a secondary**, so the two can be
seen to agree or not.

## 5. The control: speed-matched, and matched on what defines it — M11

Non-selected windows from the same recordings, matched to candidates on **mean
egocentric keypoint speed**, via `controls.matched_partners`, within animal,
without replacement, **balance precondition |SMD| < 0.10**. **If balance fails,
the stage refuses; re-drawing for balance is forbidden.**

**Head-region amplitude is deliberately NOT matched**, and this is the one place
this registration departs from M11's usual reading. Amplitude is half the
selection rule, so matching on it leaves no contrast at all. The protection
against that omission is §4: the statistic is invariant to the rescaling that
amplitude selection performs. **To make the omission checkable rather than
argued, `peak_excess` is additionally reported per decile of head-region
amplitude.** If the excess rises monotonically with amplitude, the invariance
claim is false and the result is `INCONCLUSIVE`.

## 6. Gates, refusals and incumbents

**PASS** iff **all** of:

1. §3's confirmation rate **≥ 50%**;
2. candidate `peak_excess` exceeds the speed-matched control's, with
   **non-overlapping animal-level intervals** (2,000 replicates, `how="mean"`);
3. the verdict is stable across §7's sweep — otherwise **`GRID_LIMITED`**;
4. no monotone rise of `peak_excess` across amplitude deciles — otherwise
   **`INCONCLUSIVE`**.

**Incumbents, fixed here (M12):**

| quantity | incumbent | source |
|---|---|---|
| keypoint recovery above 2 Hz | **nothing clears**; `pole_frequency_hz` at 2.1%, below 5% chance | `DYNAMICS.md` |
| a pixel measure's context effect | B − A −0.1264 [−0.1706, −0.0851] | `PIXEL_PILOT.md` |
| `peak_excess` for a flat-background window | **0 by construction** — the background is fitted to the same spectrum | this document |

**Refusals.** Fewer than **20 animals** contributing at least one matched pair →
`NOT_A_RESULT`. Fewer than **500** candidate windows → `NOT_A_RESULT`. Balance
failure → refuse. Any stratum below the standing **20,000** scored-frame floor is
not reported.

## 7. The sweep

| parameter | grid |
|---|---|
| head disc radius, body lengths | 0.4, **0.6**, 0.8 |
| window length, s | 1.5, **2.0**, 3.0 |
| band | **3–8 Hz**, and 3–6 / 6–8 reported separately because of §0's sampling limit |
| speed percentile for "still" | 10, **25**, 40 |

Headline is the emphasised centre. **A verdict that holds only at the centre is
`GRID_LIMITED`, not `PASS`.**

## 8. The sample

The **300 recordings** of `PIXEL_PREREGISTRATION.md` Amendment 1 — 30 `fit`
animals, 10 per box, all ten (context, day) cells each. Already registered,
already balanced, already scanned once. **No new draw**, and no `report` animal
is touched.

## 9. Prohibitions

1. **No corpus pass**, and no widening of the sample to reach a verdict.
2. **No `report` animals.**
3. **No tuning.** The grid in §7 is registered; the headline is its centre.
4. **No reading §4 if §3's confirmation rate is below 50%.**
5. **No substituting band share for `peak_excess`** if the latter is
   unfavourable. Both are published; `peak_excess` is the gate.
6. **No claim about grooming specifically** unless §3 confirms grooming. If the
   confirmed content is some other held-still-with-head-motion behaviour, that
   is what gets named.
7. **Nothing is described as shock versus no-shock.**

## 10. The negative outcome, worded now

If the gate does not pass, the published claim is **exactly**:

> Fine limb behaviour in this corpus is **unrecoverable by keypoints and by
> motion-energy features at this resolution and compression**.

**Not "by pixels".** Frame differencing inside a crop is **one** pixel method,
and the stronger phrasing would foreclose methods that were never tried here —
optical flow, learned features, segmentation-derived contours. The narrower claim
still makes the recording case concrete, which is the only thing the stronger one
was buying, and it is the claim this stage's evidence actually supports.

**The recording case either way** — a side camera, a higher frame rate, and
paw-visible views — does not depend on the outcome and is being made to Luna
now, not after.

---

# Amendment 1 — §1's head disc measures TRANSLATION, so the signal is stabilised

**Committed alone. Recorded as `DEVIATIONS.md` D20.** Made on the evidence of a
diagnostic about the **instrument**, before the gate had produced any verdict —
`peak_excess` had not been compared between arms at the time this was written,
because §2 yielded **no arms to compare**.

**What §1 specified, and what it turned out to measure.** §1 put the head disc at
the skull centroid **in image coordinates**. When the animal translates, the
whole scene slides underneath that disc, and the resulting |Δ| is large — so the
disc reports **body movement**, which is the one thing a still-body detector must
exclude.

Measured over 25 recordings and 2,408 windows:

| | |
|---|---|
| corr(head energy, keypoint speed), radius 0.4 bl | **+0.911** |
| corr(head energy, keypoint speed), radius **0.6 bl** (headline) | **+0.926** |
| corr(head energy, keypoint speed), radius 0.8 bl | **+0.936** |
| head energy SD remaining after removing speed | 35.9% |
| **candidate windows found by §2** | **0 of 2,408** |

**The correlation rises with the disc radius, and that is the mechanism, not a
coincidence** — a larger disc catches more sliding scene. A disc measuring motion
*relative to the body* could not behave that way.

**The consequence for §2 is arithmetic, not judgement.** §2 asks for windows in
the bottom 25% of speed **and** the top 25% of head energy. At r = +0.93 those
sets are nearly disjoint, and the observed intersection is **empty**. §6's
refusal at fewer than 500 candidates fires, and **that refusal is reported as the
registered outcome** in `GROOMING.md` — it is not skipped over.

## What changes

**A second signal is added. Nothing is removed and nothing is re-chosen.**

> **`energy_ego`**: consecutive frames are first **registered on the animal** —
> rotated by the body-axis heading (`ego.heading`, `NOSE → TAIL_BASE`) and
> translated so `CENTER` is fixed — and only then differenced, inside the same
> skull disc. Body translation and body rotation are removed by construction, so
> what remains is motion **relative to the animal's own frame**, which is what
> "still body, moving head" means and what §1 should have specified.

This is the **egocentric crop** the approved plan named; §1 implemented a disc in
image coordinates instead, and that was the error.

**Both signals are computed in one pass and both are reported.** The registered
image-coordinate signal is run on all 300 recordings and its refusal published;
`energy_ego` is run beside it. Everything else in this registration is
**unchanged and applies to both**: §2's two conditions and percentiles, §3's
eye-confirmation and its 50% bar, §4's `peak_excess` (with D19's corrections),
§5's speed-matched control and amplitude-decile check, §6's gates and refusals,
§7's sweep, §8's sample, §9's prohibitions and §10's negative wording.

**Why this is not estimator-shopping.** §9.4 forbids tuning, and choosing a
statistic after seeing which one the data favours is exactly what D17 refused to
do for the freeze score. The distinction here is that **no verdict existed to
shop for**: the registered signal produced zero candidates, so `peak_excess` was
never evaluated on either arm, and the defect is established by a correlation
with **keypoint speed** — a quantity entirely outside the gate. If `energy_ego`
also refuses or fails, that is reported with the same weight.

**A stabilised difference has its own failure mode, stated now.** Registering on
a noisy pose injects motion of its own: a keypoint jitter of one pixel rotates
and shifts the whole crop, and that appears as apparent motion everywhere,
including the head disc. So `energy_ego` is reported **with its own control** —
the identical stabilised difference measured in a disc on the **hindquarters**
(`LEFT_HIP`, `RIGHT_HIP`, `TAIL_BASE`), where no grooming stroke occurs and
which therefore carries registration noise and nothing else. **A 3–8 Hz excess
that appears equally at the hips is registration noise, not behaviour**, and the
gate fails whatever the head disc shows.
