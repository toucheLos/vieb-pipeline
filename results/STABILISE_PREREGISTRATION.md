# Registration — crop registration that reads no keypoints

Committed **alone, before `vieb/pixel/register.py` or `scripts/stabilise.py`
exist**, and before any frame has been registered by a keypoint-free method.
Inherited digest **`198eb14ff258c7f6`**.

This stage builds **an instrument, not a detector**. It asks no question about
grooming or any other behaviour. Its only output is a verdict on whether each
registration method is fit to carry a crop feature. `GROOMING_PREREGISTRATION.md`
§9.4 and `DEVIATIONS.md` D21 forbid choosing a replacement grooming detector
after watching the incumbent fail, and nothing here does. An arm that passes
becomes *eligible* for a separately registered gate. No more than that.

---

## 0. Why: the pose → warp → apparent-motion path

D21 measured it three ways. The incumbent egocentric crop (`head._ego_warp`,
GROOMING Amendment 1) is rotated and translated **per frame from DLC pose**.
Skull keypoints jitter by about 3.2 px on a physically rigid triangle, so the
crop moves frame to frame. Differencing then reads that movement as motion
inside the animal. Candidate windows carried **3.44×** the skull jitter of
speed-matched partners, and jitter predicted head energy at **+0.276** with
speed held fixed.

`REVIEW.md` records the general form. **Any** crop feature aligned per frame on
pose inherits pose jitter: motion energy, band concentration, and the frozen
DINOv3/V-JEPA embeddings the external review proposes. This stage is therefore
the prerequisite for every masked-crop feature, not only for grooming.

**The principle, fixed now.** Keypoints may **locate** a region. They may not
**register** a frame. Where an arm needs to know where the head is, it uses the
**window-median** skull position in its own body frame: one value per window,
which frame-to-frame jitter cannot move.

## 1. The arms

All arms are computed in **one sequential decode per recording**, sharing the
decode and the blur (`motion.SIGMA`) exactly as `head.scan_head` does. Each arm
produces a registered difference image per frame pair. From that image it
reports the mean |Δ| in a **head disc** and in a **hip disc** (the hindquarter
control from GROOMING Amendment 1), at radii 0.4, **0.6** and 0.8 body lengths.
The centre, 0.6, is the headline.

| arm | frames registered by | disc placed by |
|---|---|---|
| **K**, incumbent (M12) | `head._ego_warp`: per-frame pose heading and `CENTER` | per-frame pose, `head._fixed_disc`, **unchanged** |
| **B**, background mask | the animal mask's centroid and second-moment orientation | window-median skull / hip centroid in the mask frame |
| **P**, phase correlation | Fourier–Mellin phase correlation of frame *t* onto *t−1* | as B, carried into frame *t−1* by B's mask pose |
| **S**, SAM mask (optional) | as B, with the mask from `segment-anything` | as B |

**K is the incumbent exactly as it ran.** It is not repaired or re-parameterised.
Its per-recording output must reproduce the stored `energy_ego__*` series of the
grooming run on the same recordings (§7). If it does not, the stage stops before
any gate is read.

**B, specified completely.**

1. **Background:** the per-pixel median of **101 frames evenly spaced** over the
   recording, after the same blur.
2. **Foreground:** |frame − background| **>** the recording's own derived cutoff,
   `cutoff_values["m2"]` in `work/pixel/rec/*.npz` (`PIXEL_NOISEFLOOR.md`'s
   2 × p99.99 rule). Nothing new is chosen.
3. **Mask:** one 3 × 3 binary opening, then the **largest 8-connected
   component**.
4. **Pose:** centroid, and the major-axis angle from the second central moments.
   The major axis has a 180° ambiguity. It is resolved by **continuity**: each
   frame takes whichever of θ and θ + 180° is nearer the previous frame's angle.
   No head/tail sign is ever needed, because only the inter-frame change enters
   the difference.
5. **Warp:** the incumbent's convention, with the mask centroid at the image
   centre and the major axis horizontal. `cv2.warpAffine`, `INTER_LINEAR`,
   constant border 0.

**The tail is in the mask, and that is a known risk, stated now.** A sweeping
tail moves the centroid and the moment axis. This is not patched in advance.
§4's immobility gate and §5's plant exist to measure what it costs.

**P, specified completely.**

1. **Region:** B's mask bounding box at *t−1*, padded by 0.25 body lengths, with
   a Hann window. The same box is used for frame *t*.
2. **Rotation:** log-polar resampling of the FFT magnitudes, then
   `cv2.phaseCorrelate`.
3. **Translation:** undo the rotation, then `cv2.phaseCorrelate` again.
4. **Snap:** |shift| < 1e-3 px and |rotation| < 1e-3° are set to the identity.
   This is a floating-point tolerance, not a tuning parameter. Without it, two
   identical frames can return a 1e-12 px shift and interpolation would invent a
   difference.
5. **Difference:** frame *t* is warped onto *t−1* and the pair is differenced in
   *t−1*'s coordinates.

P uses B's mask to place its box and its disc. **It never uses B's orientation to
register.**

**S is optional and non-gating.** No SAM checkpoint is present on disk at
registration. S runs only if one is obtained, on its own GPU job, prompted by
**B's centroid, never a keypoint**. Otherwise S is `NOT_A_RESULT` for want of a
checkpoint, reported as such, and nothing else waits on it.

**Per-frame refusals, all arms except K.** A frame is NaN when B's mask area
falls outside **0.5–2×** the recording's median mask area. A recording is refused
when more than **10%** of its frames are NaN. An arm with more than **20%** of
recordings refused is `NOT_A_RESULT`.

## 2. Gate 1: duplicate frames (sharp, a construction check)

On the census's **duplicate video frames** (`identical` in `work/pixel/rec`,
`PIXEL_PILOT.md` §6), the two images are identical. Any registration that
depends only on the images must give an **in-disc |Δ| of exactly zero
(≤ 1e-6 grey levels)**.

**PASS iff this holds on 100% of duplicate frames.** One violation is a bug, not
a statistic.

**The expected outcome for K is registered here, before it is measured.** K fails
wherever DLC gave the duplicate pair different poses. The **share of duplicate
frames on which K reports non-zero energy** is published as a direct measurement
of D21's mechanism: image content identical, reported motion non-zero.

## 3. Gate 2: jitter decoupling (the D21 path)

**Population: every 2 s window whose mean egocentric keypoint speed is below its
recording's 25th percentile.** These are the still windows, selected on speed
alone. **No arm's energy enters the selection**, which removes the reporting-back
the D21 partial had to correct for.

**Statistic:** per animal, the Pearson correlation between head energy and skull
jitter (`scripts/grooming_jitter.py::skull_jitter`, within-window SD), after
**rank-residualising both on speed**, which is D21's construction. The mean
across animals is taken, with an animal-level bootstrap interval (2,000
replicates, seed 0, `how="mean"`).

**PASS iff** the arm's **upper bound < 0.10 and** lies below K's **lower bound**
on the same population. K's value on this population is computed here, not
assumed. The +0.276 was measured on a selected population, and it is quoted only
as the motivation.

## 4. Gate 3: the immobility floor

**Population:** census frames that are **genuine immobility** at the strictest
margin, `TRI_MARGINS = 0`. These are zero-ego-speed frames that are not
duplicates and in which the animal region changed by no more than the arena's
own noise predicts. Behaviour in them is, by the census's own definition, nil.

**Statistic:** per frame, the arm's in-disc registered |Δ| **divided by** the
arena mean |Δ| of the same frame pair. This is the mean per animal, with an
animal-level bootstrap as in §3.

**PASS iff** the upper bound **≤ 1.10**: registered motion inside a still animal
is within 10% of the arena's sensor noise. K's value is reported beside it.
This is the general gate. It catches **any** registration noise, including the
tail and mask-edge noise B is exposed to, not only D21's.

## 5. Gate 4: registration fidelity, planted

**Construction, per recording.**

1. Take one genuinely-immobile frame (§4's population, first eligible at seed 0).
2. Cut the animal out with B's mask, dilated by 0.1 body lengths.
3. Paste it onto the recording's median background.
4. Over **60 frames**, move the paste along a **real** rigid trajectory: the
   inter-frame translation and heading of the recording's first 2 s window above
   the 75th speed percentile, taken from keypoints. Keypoints here only
   **generate** the motion; they do not register it.

**K gets a pose track with realistic jitter.** K receives the pasted frame's pose,
carried along the same trajectory, **plus** the real per-frame DLC deviation of a
still window (pose minus window-median pose, in the body frame). Without that
jitter K would be handed perfect keypoints, and the test would be empty for the
incumbent.

**Statistic:** in-disc registered |Δ| per frame, divided by the recording's
median arena |Δ| from the real video. The synthetic background is static, so the
real floor is the yardstick. Mean per animal, animal-level bootstrap.

**PASS iff** the upper bound **≤ 1.10**.

## 6. Gate 5: local motion, planted (a curve, not a gate)

On §5's paste, first held **still** and then carried along the **moving**
trajectory, a patch of radius **0.15 body lengths** at the head-disc centre is
displaced sinusoidally.

| factor | levels |
|---|---|
| frequency | 4 and 6 Hz |
| amplitude | 0.25, 0.5, 1, 2 and 4 px |

**Detection:** a 2 s window's head `peak_excess` (`head.peak_excess`, D19's
corrections) exceeds the 95th percentile of the same arm's **unplanted** windows
from §5. The hip disc is scored identically as the control.

**Published:** the recall-vs-amplitude curve per arm, as in `PLANT.md`, with head
and hip side by side. **No verdict rests on it.** It tells a later gate what
amplitude an arm can see. That gate must register its own bar.

## 7. The incumbent check, before any gate

On the first shard's recordings, K's `energy_ego__<r>` is compared with the stored
`work/grooming/rec` series at every radius. **Maximum absolute difference ≤ 1e-9
grey levels, or the stage stops.** That confirms that K is the incumbent and not
a re-implementation of it.

## 8. Verdicts

Per arm:

- **`PASS`** iff gates 1–4 all pass.
- **`FAIL`** if any of gates 1–4 fails, with every gate's number published
  regardless.
- **`NOT_A_RESULT`** under §1's refusals. Also if fewer than **20 animals**
  contribute to any of gates 2–4, or an animal has fewer than **30** still
  windows (gate 2) or **50** immobile frames (gate 3). Such an animal is dropped,
  not padded.

**Incumbents, fixed here (M12):**

| quantity | incumbent | source |
|---|---|---|
| jitter → head energy, speed held fixed | +0.276 (selected population; motivation only) | D21 |
| head energy ~ keypoint speed, stabilised, 0.6 bl | +0.815 | `GROOMING.md` |
| hip energy ~ keypoint speed, stabilised, 0.6 bl | +0.858 | `GROOMING.md` |
| arena cutoff range | 6.50–128.50 grey levels | `PIXEL_NOISEFLOOR.md` |

**Reported, never gated:** each arm's coupling with keypoint speed at every
radius, and the head–hip coupling gap. A registration that removes jitter should
not be expected to remove genuine body-relative motion, so a low coupling is not
a success criterion.

## 9. The sample

The **300 recordings** of `PIXEL_PREREGISTRATION.md` Amendment 1
(`work/pixel/manifest.json`): 30 `fit` animals, 10 per box. It is the grooming
gate's sample, so K is comparable to its stored run. **No new draw.**

## 10. Prohibitions

1. **No `report` animals.** No widening of the sample to reach a verdict.
2. **No tuning.** The mask threshold is the already-derived `m2` cutoff. The
   opening, the 101-frame median, the padding, the snap tolerance and every bar
   above are fixed here. A parameter that turns out to matter is **swept and
   published** as its own amendment, committed alone, and never picked.
3. **No reading any spectrum as a behaviour.** Gate 5's `peak_excess` measures a
   planted sinusoid and nothing else.
4. **No claim about grooming**, and nothing described as shock versus no-shock.
5. **K is not repaired inside this stage.** Smoothing the pose that feeds K is a
   different instrument. D21 already stated its limit: selection is a
   within-recording percentile, so uniform smoothing moves values and threshold
   together.
6. **A `PASS` licenses only eligibility.** It licenses using that arm's
   registration in a later, separately registered stage. It licenses no claim
   about what the registered crops contain.
