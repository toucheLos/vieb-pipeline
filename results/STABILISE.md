# STABILISE: the keypoint-free arms are refused, and the incumbent is measured

Registered in `STABILISE_PREREGISTRATION.md`, committed alone as `39efb12`, and
amended alone as `eefb78c` (Amendment 1, `DEVIATIONS.md` D22) before any real
frame was registered. `vieb/pixel/register.py`, `scripts/stabilise.py`,
`jobs/stabilise.slurm` (12 shards, all `COMPLETED`). Digest
**`198eb14ff258c7f6`**. Sample: the 300 registered `fit` recordings, 30
animals. All 300 were scanned.

**Headline.** Neither keypoint-free arm reaches a verdict. **B and P are both
`NOT_A_RESULT`**: B's background mask is refused on **213 of 300 recordings
(71.0%)**, against a registered arm-level limit of 20%. P uses B's mask to
place its box, so it inherits the same refusal. **No arm is eligible** for a
grooming gate v2.

**What the stage does establish is about the incumbent.** It is measured on
populations that no pixel energy selected, so it confirms D21 without D21's
selection problem:

| | incumbent K | source |
|---|---|---|
| duplicate video frames with **non-zero** reported head motion | **23,910 of 23,910 (100%)** | gate 1 |
| skull jitter → head energy, speed held fixed, **speed-selected still windows** | **+0.340 [+0.285, +0.392]**, 30 animals | gate 2 reference |

**Every duplicate frame reads as motion under the incumbent warp.** The image
content is byte-identical, and the pose-driven crop still moves. That is D21's
mechanism measured directly, with no selection involved. The jitter coupling on
an unselected still population (+0.340) is *stronger* than D21's +0.276, not
weaker. On the non-refused recordings, B and P give exactly zero on every
duplicate frame (**0 of 3,048** values non-zero), as a registration that reads
only pixels must.

---

## 1. §7 held, after it caught a real bug

K reproduces the stored grooming-run `energy_ego__*` series **exactly: maximum
absolute difference 0.0** at all three radii over 300 recordings (compared as
float32, the precision the grooming run stored). The incumbent is also
deterministic on its own: two fresh reruns and the stored run agree to 0.0.

**The first smoke run failed §7**, and the failure was the code's. In OpenCV
5.0.0, `cv2.phaseCorrelate` writes its Hann window **into its inputs in place**
whenever a crop is already a DFT-optimal size. P's crops were views of the
frame, so the frame the *next* pair was differenced against was silently
altered. On the smoke recording, 12 of 6,302 arena values were off by up to
0.864 grey levels. The failure was deterministic, tracked P being on (not
threading), and reproduced on synthetic crops at every DFT-optimal size tested.
`rigid_between` now hands OpenCV copies. `tests/test_register.py` pins it at
three sizes, and the test fails on the old code. **No gate was read before the
fix**; the smoke run's file was deleted and the full array run afterwards.

## 2. Why B's mask is refused: the animal is in its own background

§1 refuses a frame when B's mask area falls outside 0.5–2× the recording's
median. At the median recording, **22.3%** of frames are refused (10th–90th
percentile: 1.3%–52.1%). Two-thirds of refused frames are **above 2×**
(294,693 of 436,019).

**The diagnostic, descriptive and not gated.** A mouse's outline covers roughly
0.3–0.4 × body length² from overhead. B's **median** mask covers **0.151 ×
bl²** at the median recording, and **0.068** at the 25th percentile. So in most
recordings the typical mask captures a *fraction* of the animal. The recordings
where it captures least are the ones refused most: rank correlation **−0.575**
between median mask area and refusal rate.

The reading consistent with both: **§1 B.1's 101-frame median background
contains the animal.** In a contextual-fear test the animal holds one place for
long stretches. The median then keeps it, and while it stays there only its
moving edges differ from "background". The mask is small at rest and whole
when it walks, which is exactly the bimodality the refusal cuts. **B's
background model fails on the behaviour this corpus exists to measure.** This
is a property of the registered method on this corpus, and it is reported as
such, not tuned. §10.2 forbids choosing a different background after seeing
this.

## 3. The gates that could be read

| gate | K | B | P |
|---|---|---|---|
| 1. duplicate frames exactly zero | **FAIL**: 100% non-zero (registered expectation) | **PASS**: 0 of 3,048 | **PASS**: 0 of 3,048 |
| 2. jitter decoupling | reference: +0.340 [+0.285, +0.392] | `NOT_A_RESULT`: 19 animals, against 20 | `NOT_A_RESULT`: 19 animals |
| 3. immobility floor | `NOT_A_RESULT`: 15 animals with ≥ 50 immobile frames | `NOT_A_RESULT`: 3 animals | `NOT_A_RESULT`: 3 animals |
| 4. planted trajectory ≤ 1.10× floor | **FAIL**: 1016× | **FAIL**: 1726× | **FAIL**: 1041× |
| **arm** | **FAIL** | **`NOT_A_RESULT`** (71% refused) | **`NOT_A_RESULT`** (71% refused) |

**Gate 2 for B and P is not reported.** It missed the registered animal floor
by one animal, and a refused value is not published. This follows the precedent
of `PIXEL_PILOT.md` Q1.

**Gate 3 refuses even the incumbent.** Genuine immobility at the strictest
margin is rare enough that only 15 of 30 animals reach 50 frames. The census
counts frames *pooled over* recordings, and this gate needs them per animal.
That is a sample-size defect in the registration's §8 floor, not a finding
about any arm.

## 4. Two defects in the registration, measured rather than argued

**Gate 4's bar cannot be met by any registration, so gate 4 carries no
information (`DEVIATIONS.md` D23).** The bar compares in-disc |Δ| on a
*moving, interpolated* paste against the real arena's median |Δ|. Across
recordings its median is **0.012 grey levels** (5th–95th percentile 0.003–1.29)
on these blurred, compressed, static floors. On
`tests/test_register.py`'s synthetic scene, a registration handed **perfect**
keypoints still leaves about **0.6 grey levels** inside the disc: that is
resampling the texture, not registration error. That is about 50× the bar at the median recording, before any error at
all, and every arm lands at 10³×, K included. The bar was
set without calibrating it against what a perfect registration achieves. That
is `METHODS_FINDINGS.md` **M12** again, and it is recorded rather than
corrected. The ordering across arms (P 1041×, K 1016×, B 1726×) is not read:
the gate's scale is set by interpolation, not by the property it was meant to
test.

**Gate 5's null is not comparable to its planted arms for B and P (D23).**
Unplanted *still* windows are exactly zero for B and P (identical images), so
their spectrum is undefined and the detection threshold comes from *moving*
windows alone. Two further signs that the curve does not measure local-motion
sensitivity:

* head and hip recall are nearly equal in every cell (for example B, still,
  6 Hz, 0.25 px: head 0.94, hip 0.95);
* recall hardly moves between 0.25 px and 4 px.

The pattern fits the planted patch nudging B's mask moments, so the *whole*
crop oscillates at the planted frequency. That is a registration artefact
reaching the hip disc too, not a head signal. **No curve from gate 5 is
published as a sensitivity.** The table is in `stabilise.json` for completeness.

## 5. Reported, never gated: coupling with keypoint speed

Median within-recording correlation of 2 s window energy with keypoint speed,
0.6 body lengths, all recordings (refused frames excluded):

| | head | hip |
|---|---|---|
| K | +0.815 | +0.858 |
| B | +0.563 | +0.560 |
| P | +0.853 | +0.840 |

K reproduces `GROOMING.md`'s +0.815 / +0.858 exactly, as §7 requires. B's
lower coupling is not a success criterion (§8). On most recordings it is also
computed on the partial mask that §2 diagnoses.

## 6. What this licenses

**License:** on this corpus, the incumbent pose-driven crop reports motion on
every byte-identical frame pair, and skull jitter predicts its head energy at
+0.340 on speed-selected still windows. D21's mechanism is now measured without
selection.

**License:** a median-background mask does not isolate the animal in this
corpus. 71% of recordings fail its own area stability, and the typical mask
covers a fraction of the body.

**Do not license** any claim that keypoint-free registration fails or succeeds
here. Neither arm reached a verdict, and gate 4 could not have given one.

**Do not license** reading gate 5 as a sensitivity curve.

**Nothing is described as grooming**, and nothing as shock versus no-shock.

## 7. What is owed

* **A mask that does not come from a median background.** The registration's
  optional arm S (SAM, prompted without keypoints) is the registered candidate,
  and it needs a checkpoint and a GPU job. A different background model would
  be a new instrument, needing its own registration with §2 as its motivation.
* **A gate 4 whose bar is calibrated against a perfect registration**, for
  example residual relative to the same paste registered with its *true*
  transform. That too belongs in a new registration, not in an amendment made
  after the result.
* **Gate 3 needs its animal floor re-derived** from the per-animal immobile
  counts reported here, before it is used again.

## 8. Provenance

| number | source |
|---|---|
| 213 of 300 refused, 71.0% | `stabilise.json` `reads.B.detail`; per-recording `nan_frac` in `work/stabilise/rec` |
| 23,910 duplicate values, 100% non-zero | `reads.K.gate1` |
| +0.340 [+0.285, +0.392] | `reads.K.gate2`, 2,000 animal-bootstrap replicates, seed 0 |
| median mask 0.151 × bl², Spearman −0.575 | `mask_area`, `body_length_px`, `nan_frac` in `work/stabilise/rec` (descriptive) |
| §7 max \|diff\| 0.0 | `reads.k_check` |
| phaseCorrelate in-place write | `tests/test_register.py::test_rigid_between_never_writes_into_the_frame` |
