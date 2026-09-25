# The external review, reconciled against what is measured

An external review of the programme ("VIEB Re-examined: What Holds, What Breaks,
and the Revised Video → Behaviour Pipeline") was written against an **older
state of this repository**. Its §5 steps 1, 2 and 4 have since been run, and two
of them came out differently from what it assumed. This file puts each claim
next to the record, so the review can be cited only where the record supports
it.

**Status vocabulary.**

| status | meaning |
|---|---|
| `MEASURED-AGREES` | a registered stage here measured it, and the measurement agrees |
| `MEASURED-DISAGREES` | a registered stage here measured it, and the measurement contradicts it |
| `NOT-LICENSED` | it goes beyond what this repository's own licences permit |
| `ALREADY-RETIRED` | the review recommends stopping something that was already stopped |
| `OPEN` | nothing here has tested it |
| `OUT-OF-SCOPE` | other domains (BJJ, rat gait, faces); this repository holds none of their data |

---

## 1. Where the review is behind the record

**These four change what the review's own plan should do next.**

| review claim | status | record |
|---|---|---|
| §5 step 1: "island replication … gate: A > B sign replicates with subject-level CI excluding 0 → keep as positive control" | **`MEASURED-DISAGREES`** | `FIT_REPLICATION.md`: on 149 held-out animals, B − A = **−0.00216 [−0.00536, +0.00123]**, p = 0.2020, at **99.8% power** against the published estimate. Verdict `FAIL`. The island's *prevalence* replicates (0.541% vs 0.561%); its context effect does not. |
| §2.6: the island "is almost certainly the long, clean tail of freezing … has passed a positive control" | **`NOT-LICENSED`** (in kind); the positive-control reading is **`MEASURED-DISAGREES`** | `VALIDATION.md`: a **unitary slow state**, less bimodal than its duration-matched partners on both channels. The same file forbids calling it freezing without the context-to-shock mapping and an ethogram, and this repository holds neither. As a positive control it fails, because it no longer carries the context contrast. |
| §5 step 2: "if the island adds no context information beyond the freeze score … label it 'freezing-tip' and move on" | **`MEASURED`**, the branch is reached | `PIXEL_PILOT.md` Q2: on the same fit animals, an ezTrack-family freeze score gives B − A = **−0.1264 [−0.1706, −0.0851]**, p = 0.0005, and survives arena-noise subtraction. §0's registered reading is that the island is the worse instrument. The magnitude is **not** established: the threshold is untransplanted from 320 × 240. |
| §5 step 4 / §2.9: a "3–8 Hz band-concentration gate on egocentric crops is a reasonable cheap first pass" | **`MEASURED`**: refused, for a cause the review does not name | `GROOMING.md`, D20, D21. `NOT_A_RESULT` at all 54 arms, and precision **0/30** by eye. The egocentric crop is **registered using keypoints**, so DLC skull jitter (**3.44×** in candidates vs speed-matched controls) rotates the crop, and the rotation reads as head motion. The detector selects still animals with noisy tracking. |

**A failure mode the review's stage table lacks, and must carry:**

> **Keypoint-driven registration injects apparent motion.** Any masked or
> egocentric *crop* feature (motion energy, band concentration, DINOv3/V-JEPA
> embeddings) that is aligned per frame on pose inherits pose jitter as image
> motion. Keypoints may *locate* a region; they may not *register* the frame.
> Gate: a rigid-bone jitter partial correlation, speed held fixed (D21: +0.276
> for the incumbent warp).

The next registered stage, `STABILISE_PREREGISTRATION.md`, tested exactly this (`STABILISE.md`). The incumbent reports motion on 100% of byte-identical frame pairs. The keypoint-free arms were refused, because a median-background mask cannot isolate an animal that stays in one place. That is direct evidence for the review's A1 mask-first recommendation (SAM or a learned segmenter) over background subtraction.

## 2. Where the review agrees with what is already measured

| review claim | status | record |
|---|---|---|
| §2.10: held-pose excess in A "could partly be encoder or duplicate artefacts; your trichotomy is exactly the right test" | `MEASURED-AGREES` (partly) | `PIXEL_PILOT.md`: of 43,206 zero-ego-speed frames, **14.2% are duplicate video frames, 9.6% tracking dropouts and 76.1% genuine immobility**. `CONTEXT.md` and `CONTEXT_CONTROLS.md` carry the correction; their intervals are unchanged. |
| §2.10: per-recording, per-context pixel floors (ezTrack's 2 × p99.99 rule) | `MEASURED-AGREES` | `PIXEL_NOISEFLOOR.md`: self-calibrated from 1.0–1.5 billion arena pixel-pairs per recording. The cutoff spans **6.50–128.50** grey levels, so a global threshold is indefensible. The arena floor is B − A **−1.296 [−2.020, −0.505]**: A is noisier. |
| §2.10: A sessions run ~17% longer, so time-match | `MEASURED-AGREES` for occupancy, by construction | `FIT_REPLICATION.md` §4: occupancy is a rate, so session length cancels. The **image** difference does not cancel, and it remains the live confound. No context result here is time-*windowed*; if a future statistic is a count rather than a rate, it must be. |
| §2.7: noise floors are the finding for the C² detector | `MEASURED-AGREES` | `NOISEFLOOR.md` (floor 0.2599/s, 54.8% of the corpus rate), `PLANT.md` (50% order-2 recall at ≈0.99 body lengths). |
| §2.9: grooming is not recoverable from these 7 keypoints | `MEASURED-AGREES` | `DYNAMICS.md`: nothing above 2 Hz clears the descriptor floor. |
| §2.6: the island is homogeneous rather than a mixture | `MEASURED-AGREES` | `VALIDATION.md`: it does not split on `body_extension` or `pole_radius`. |

## 3. What the review recommends stopping, already stopped

| review recommendation | status | record |
|---|---|---|
| retire rate-distortion / IB knees | `ALREADY-RETIRED` upstream | retired in `~/recur`, before the repo split; not recorded or re-run here |
| KPMS only as a named baseline | `ALREADY-RETIRED` upstream | the KPMS audit is `~/recur`'s; not recorded or re-run here |
| no new alphabets on the 17-dim stream | `ALREADY-RETIRED` | stages 3, 4 and 7: median run 1.0 frame in all 8 cells; the surrogate falsifier fired |
| drop ℓ1 trend filtering rather than debug ADMM | `ALREADY-RETIRED` | `TRENDFILTER.md`: `GRID_LIMITED`, refused |
| promote the 148× hazard fall to a headline; engine must be semi-Markov | `MEASURED-AGREES` | `FRAILTY.md`: 0.756 [0.742, 0.769] of the log fall survives speed conditioning. `FALSIFIER.md` and `DWELL.md`: the information is in how long, not what next. |

## 4. Open: untested here

| review claim | what testing it would need |
|---|---|
| §2.1: max-entropy microstates on delay embeddings, transfer-operator spectrum, metastable sets only if the gap beats surrogates | its own registration; partition fitted on tune only |
| §2.3: "the smoother is part of the measurement and must be inside the null"; EKS | consistent with the Q1 banking (`Q1_BANKED.md`: the unfiltered cell is −0.199%). A rerun of Q1 with the smoother applied to every surrogate has not been done. |
| §2.6: prediction-error / BOCPD boundaries, recall of smooth regime changes vs the C² detector at matched false-positive rate | its own registration; `PLANT.md`'s battery is the comparison |
| §2.8: human boundary study redone at ≥5 raters, coarse and fine grain, continuous playback, matched boundary rate | `ANNOTATION.md`'s own diagnosis already points this way, for a sharper reason than rater count: **0.853** of marks fall within one second of the other rater's nearest, but only **0.088** fall within two frames. The raters see the same events and cannot place them at 67 ms. So the redo should score at second-scale tolerance, not ±2 frames. One correction to the review: the detector's rate is **0.443/s** (`BREAKS.md`), not "about 1/s". The gap to humans' 0.109/s is ~4×. |
| §2.10: immobility-leak control (masked crops decode context at chance during true immobility); background-swap invariance | needs masks that do not depend on pose; blocked behind `STABILISE` |
| §2.11 / §2.12: DINOv3 / V-JEPA 2 crop embeddings, time-only CEBRA, all behind the identity-leak probe | blocked behind `STABILISE`, and behind the immobility-leak control |
| §2.9: denser pose (SuperAnimal TopViewMouse, Lightning Pose ×5 + EKS, variance as validity mask) | GPU and ~300 labels; not started |
| §2.9 / §5 step 4: validate grooming "against a small labelled set (a few hundred bouts)" | **no grooming ground truth exists** (confirmed by the investigator, 2026-09-25). Until one is made, a grooming detector can be scored for **precision only**, by blind panels with mixed controls (`GROOMING_PREREGISTRATION.md` §3). **Recall cannot be measured.** Any claim about how much grooming occurs is out of reach. |

## 5. Out of scope

§2.13 (BJJ), §2.14 (rat gait contact channel) and §2.15 (faces) concern data
this repository does not hold. They are noted here and not assessed.

---

**The review's literature claims are external and unverified here.** This covers
SAM 3, DINOv3 and V-JEPA 2 figures, LookAgain, EASE and STREAMER, Costa et al.,
Geuther et al., Gruene et al., the ViCoS licence, Harmony4D, Hi4D, Monsees et
al. and Dewberry et al. Wherever one is used, it is cited as *the review's
claim*, never as a finding of this programme. The same goes for its GPU-hour and
person-time figures, which the review itself flags as ±3×.
