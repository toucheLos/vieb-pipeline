# Registration — pixel motion, the arena noise floor, and the freeze score

Committed **alone, before any of `vieb/pixel/` exists** and before a single video
frame has been decoded for this stage. Inherited digest **`198eb14ff258c7f6`**.

This registers a **pilot**, not a corpus pass: **60 recordings**, and every
number it produces is a pilot number and is labelled as one.

---

## 0. What Step 1 changed about this stage, stated before any pixel is read

`FIT_REPLICATION.md` landed first: the island **recurs** in 149 held-out animals
at the same prevalence (0.541% against 0.561%) but its **context asymmetry does
not** (B − A **−0.00216 [−0.00536, +0.00123]**, p = 0.2020, against the
published −0.00811 [−0.01520, −0.00243], at 99.8% power on the published point).

The approved plan posed two questions here. **One of them is no longer askable
on `fit` and saying so now is the point of this section.**

> "Does the island carry context information **beyond** freezing?" is a test on
> the island's context contrast. On `fit` that contrast is already
> indistinguishable from zero. A residual after removing freezing would
> therefore be null **whatever the truth is**, and a test that cannot fail
> informatively is not a test. **It is not registered in that form.**

So the two questions are re-posed, both answerable, both open:

| | question | why it survives |
|---|---|---|
| **Q1, primary** | Does island occupancy track the freeze score **within animal**? | Convergent validity. Needs **no** context contrast, so Step 1 does not touch it. |
| **Q2** | Does the **freeze score itself** show an A-versus-B contrast on `fit`? | Genuinely open, and it *interprets* Step 1 either way. |

**Q2's two outcomes are both informative, which is why it replaces the original.**

| Q2 result | reading |
|---|---|
| freeze score **does** separate A from B on `fit` | a standard pixel measure finds a context effect where the island does not. **The island is the worse instrument**, and that is a finding about the island, not about the animals |
| freeze score **does not** separate them either | the context effect is **`report`-specific on two independent measurement channels**. This is the strongest available statement that `CONTEXT.md`'s headline was sample-specific |

The original "beyond freezing" question is **not abandoned** — it is the right
question on `report`, where a non-null incumbent exists. It is **not registered
here**, because running it would spend `report` on a post-hoc analysis, and
`report` is spent once. It returns only in its own registration, if at all.

---

## 1. The algorithm is ezTrack's, read from source, and it is named a reimplementation

ezTrack is **not installed**. Its source was fetched from
`github.com/DeniseCaiLab/ezTrack` (`master`) and read. What follows is **verified
against `FreezeAnalysis/FreezeAnalysis_Functions.py`**, not remembered:

**`Measure_Motion` (line 196)**, per frame pair:
1. `cv2.cvtColor(..., COLOR_BGR2GRAY)`
2. optional downsample (`dsmpl`, default **1** = none), optional crop
3. `cv2.GaussianBlur(frame, (0,0), SIGMA)` with **`SIGMA = 1`**
4. `frame_dif = np.absolute(frame_new - frame_old)` on floats
5. `frame_cut = frame_dif > mt_cutoff`
6. **`Motion[x] = frame_cut.sum()`** — *a count of changed pixels*, never a sum
   of magnitudes. This distinction decides everything in §2.

**`Measure_Freezing` (line 345):** frames with `Motion < FreezeThresh`, required
to run for `MinDuration` consecutive frames, plus a **backward pass** that
extends each accepted run back over its own ramp-up. Returned ×100 as a percent.

**`Calibrate` (line 1077):** on an **animal-free** video, sample `cal_pix`
random pixels across `cal_frms` frames and set
**`mt_cutoff = 2 × percentile(|Δ|, 99.99)`**.

**Published defaults, verified in the shipped notebooks:** `mt_cutoff = 10`,
`FreezeThresh = 200`, `MinDuration = 15`, `SIGMA = 1`, `cal_pix = 10000`,
`cal_frms = 300`, `dsmpl = 1`.

It remains a **reimplementation**, and every result says so.

## 2. Two of the three defaults cannot be transplanted, and this is measured — M13

| parameter | unit | ezTrack's videos | ours | transplants? |
|---|---|---|---|---|
| `SIGMA` | px | 1 | 1 | **yes** — a blur radius in pixels, and both are on the same sensor scale |
| `mt_cutoff` | **grey levels** | 10 | — | resolution-independent, but sensor- and compression-dependent. **ezTrack itself says derive it**, so we derive it |
| `FreezeThresh` | **changed pixels / frame** | 200 | — | **NO** |
| `MinDuration` | **frames** | 15 | — | **only by coincidence** |

**`FreezeThresh` is a pixel count and therefore scales with frame area.**
ezTrack's freezing defaults were set on `PracticeVideos/Freezing_animal.mpg` at
**320 × 480/2 = 320 × 240 = 76,800 px**. Our corpus is **640 × 480 =
307,200 px** — **exactly 4×**. The same physical motion produces ~4× the changed
pixels, so **`FreezeThresh = 200` transplanted is 4× too strict**, and its
area-scaled equivalent is **800**. Neither number is adopted: §4 derives it.

**`MinDuration = 15` is in frames.** ezTrack's videos and ours are both 30 fps,
so 15 frames is 0.5 s in both and the number happens to survive. **It is
registered in seconds anyway** — `MIN_FREEZE_S = 0.5` — because M13 exists
precisely because a constant in samples silently changes meaning, and `VALIDATION.md`
already recorded one such confound in this repo.

## 3. The arena noise floor — `PIXEL_NOISEFLOOR.md`

**Named distinctly on purpose.** "Noise floor" already means the DLC keypoint
jitter floor in body lengths (`NOISEFLOOR.md`, `vieb/seg/noise.py`). That floor
is synthesised from skull-bone residuals and is **a different physical quantity**
— this one is sensor read noise, H.264 quantisation and lighting flicker. The
two are never compared and never share a document.

**We have no animal-free video, so the arena supplies the calibration instead.**
Per recording:

* the animal's exclusion mask is the **bounding box of that frame's keypoints**
  from `spine.clean(rid)["pose"]` (pixel coordinates, confirmed to index the
  frame directly: 17.4 px blob-vs-pose against 145/103 px for the flipped and
  swapped hypotheses), **dilated by 1.0 × the recording's median body length in
  pixels** — scale-free, so it adapts to box and camera distance;
* a pixel counts as arena only if it is outside the mask in **both** frames of
  the pair;
* a frame with **fewer than 3 locatable keypoints** contributes **no** arena
  pixels — an undefined mask cannot be trusted to exclude the animal;
* |Δ| is measured **after** `GaussianBlur(SIGMA=1)`, exactly as `Measure_Motion`
  does, or the floor would not be on the same scale as the thing it thresholds.

**The cutoff is then ezTrack's own rule applied to this sample:**

> **`mt_cutoff(rid) = 2 × percentile(arena |Δ|, 99.99)`**, per recording.

**Registered in advance:** the floor is reported **broken down by context**, and
the A-versus-B difference in the floor is reported **whether or not it is
convenient**. It is expected to be non-zero — see §7.

**Refusal.** A recording contributing fewer than **200,000** arena pixel-pairs
is dropped from the pilot and counted in the report. Below that the 99.99th
percentile is estimated from fewer than 20 order statistics and is not a
percentile.

## 4. The freeze score, and its sensitivity is published beside its value

`FreezeThresh` is derived, not adopted: for each recording it is set to the
**p-th percentile of that recording's own `Motion` series**, with **`p = 25`**
registered as the headline. An absolute pixel count cannot be shared across a
corpus whose arena floor alone spans **0 to 384 changed pixels per frame** —
a range that straddles ezTrack's own `FreezeThresh = 200` entirely.

**The sweep is registered as part of the result, not as a robustness footnote.**
Every published freeze number is accompanied by the full grid:

| parameter | grid |
|---|---|
| `mt_cutoff` multiplier on the 99.99 percentile | **1, 2 (headline), 4** |
| `FreezeThresh` percentile `p` | **10, 25 (headline), 40** |
| `MIN_FREEZE_S` | **0.25, 0.5 (headline), 1.0** |
| ezTrack's literal defaults | `mt_cutoff = 10`, `FreezeThresh = 200`, `MinDuration = 15` — reported as a **fourth arm**, labelled *untransplanted*, so the size of the M13 error is visible rather than argued |

**If Q1's or Q2's verdict flips anywhere on this 27-point grid, the headline is
`GRID_LIMITED`, not `PASS`.** A verdict that survives only at the registered
centre is a tuned verdict, and this repo does not tune.

## 5. The gates, with the incumbent's value on each — M12

**Q1 (convergent validity).** Spearman ρ between per-(animal, day) island
occupancy and per-(animal, day) freeze fraction, **within animal**, animal-level
bootstrap, 2,000 replicates.

* `PASS` iff the CI lower bound **exceeds 0**.
* **Incumbent:** there is none. No pixel measure has ever been computed in this
  repo, so the honest incumbent is **ρ = 0**, and it is stated as such rather
  than dressed up.
* **Refusal:** fewer than **20** animals with ≥ 2 usable cells → `NOT_A_RESULT`.

**Q2 (freeze score across contexts).** The identical paired machinery as
`CONTEXT.md` and `FIT_REPLICATION.md`: `cell_occupancy` → `aligned_deltas` →
`animal_interval` → `pair_flip_null`, days 3–7, B − A, on the **`fit`** split.

* **Incumbents, fixed here:** the island on `fit`, **−0.00216 [−0.00536,
  +0.00123]**, p = 0.2020; the island on `report`, **−0.00811 [−0.01520,
  −0.00243]**, p = 0.0050; the duration-and-speed-matched residual on `report`,
  **−0.00685 [−0.01371, −0.00147]**.
* **MDE is computed and reported before the verdict**, and a null is read only
  if the MDE is below a freeze-fraction difference of **0.02**. Otherwise the
  MDE *is* the result.

**Neither question is framed as shock versus no-shock.**

## 6. The held-pose trichotomy, and the correction it forces

Over 441,706 zero-ego-speed frames, **80.3%** have **no missing and no
interpolated keypoint**, so `CONTEXT_CONTROLS.md` describing them as "a tracking
dropout" is wrong for four-fifths of them. The pixel pass settles it:

| pixel \|Δ\| at a zero-ego-speed frame | reading |
|---|---|
| **exactly zero** across the whole frame | a **duplicate video frame** — an encoder artefact, not behaviour |
| at or above the recording's own arena floor | **tracking dropout**: the animal moved, the keypoints did not |
| **nonzero but below** the floor | **genuine immobility** |

**Registered before the answer:** the correction lands in **both**
`CONTEXT.md` and `CONTEXT_CONTROLS.md` **whichever way the three proportions
fall**, including the outcome where the original description turns out right.

## 7. Every cross-context pixel number carries the arena confound

**Context A and Context B are visually different arenas.** Two independent
measurements, neither of which assumes anything about behaviour:

| measurement | A vs B | n |
|---|---|---|
| bits per frame, paired, days 3–7 | **1.320×** | 1,476 cells, 298 animals |
| keypoint speed, paired, days 3–7 | **0.758×** | same |
| median-background \|Δ\|, same animal, same day, same box | **57.7 – 62.0** | 3 boxes |
| *same-context across days, baseline* | *5.4 – 10.2* | |

**The first two point opposite ways, and that is the proof.** The animal moves
**24% less** in A while A frames cost **32% more** to encode; H.264 bitrate rises
with temporal complexity, so motion predicts the opposite sign and cannot be the
cause. Mean brightness is nearly identical (117.4 against 115.2), so it is
**spatial pattern** — a floor insert or a wall pattern.

**Consequence, registered:** a pixel measure compared across contexts is
confounded **on exactly the axis Q2 uses**. So Q2's result is reported **with the
per-context arena floor beside it**, and if the floors themselves differ between
A and B by more than the freeze difference under test, **Q2 is reported
`INCONCLUSIVE` regardless of its interval**. A sensor-level difference that is
larger than the effect is not a behavioural result.

Also on the record and documented nowhere in any of the three repos: **A sessions
run ~17% longer than B** (6,302 against 5,392 frames). Occupancy and freeze
*fraction* are rates, so this cancels; any count would not, and no count is
published.

## 8. The pilot's sample

**60 recordings**, not 50: the design cells are 3 boxes × {A, B} × days 3–7 =
**30**, and 60 is **exactly 2 per cell** where 50 is not exactly anything.
Balance is the reason the pilot exists, and an imbalanced balanced-design is
worse than 10 extra videos. Drawn at **seed 0**, from the **`fit`** split only.

Videos are **read-only** and read **sequentially**: a random seek costs 5–36 ms
against 0.5–0.65 ms for a sequential frame, measured.

## 9. Prohibitions

1. **No corpus pass.** If the pilot's numbers argue for one, it gets its own
   registration.
2. **No re-reading of `report`.** Q2 runs on `fit`. `report`'s values appear only
   as fixed incumbents quoted from committed documents.
3. **No re-derivation of the island**, its θ, or its scale.
4. **No tuning.** The grid in §4 is registered; the headline is the registered
   centre; a verdict that holds only at the centre is `GRID_LIMITED`.
5. **No adoption of ezTrack's `FreezeThresh` or `MinDuration`** as if they
   transplanted. They appear only as the labelled fourth arm of §4.
6. **SAM is off the critical path.** `segment-anything` 1.0 and torchvision
   0.28.0+cu130 are installed and pinned in **both** `requirements.txt` files,
   both `--no-deps` so neither could replace torch 2.13.0+cu130 in a venv that is
   a symlink shared with `~/recur`. **No result registered here depends on SAM**,
   and nothing is refused if its checkpoint is absent.
7. **Nothing is described as shock versus no-shock.**

## 10. The negative outcome, stated precisely in advance

If the 3–8 Hz grooming gate (its own step, registered separately, sketched here
so the wording is fixed before the answer) does not clear its speed-matched
control:

> The claim is that fine limb behaviour is **unrecoverable by keypoints and by
> motion-energy features at this resolution and compression** — *not* "by
> pixels". Frame differencing inside a crop is **one** pixel method, and the
> stronger phrasing would foreclose methods that were never tried. The narrower
> claim still makes the recording case concrete, which is the only thing the
> stronger one was buying.

The gate itself: candidates are **detected** (low keypoint speed **and** high
head-region pixel motion), **~30 clips are rendered for confirmation by eye** and
the confirmation rate is published as the detector's own precision, and the test
is on whether **3–8 Hz power is *concentrated* relative to broadband** against
**speed-matched non-selected clips** — never on amplitude, which is the quantity
the selection already used and would therefore guarantee its own answer.
