# STEP 4 machinery (C1–C4) on the tune split: the semi-Markov part is dropped, and the stationary control FAILS

**Registration:** `REGISTRATION.md` (7d26641), with Amendment 1 (00c48c7, D33) and
Amendment 2 (1196e8b, D34). Both amendments were committed alone, after smoke tests
and before any registered fit.

**Data:** `hsmm_emissions.json`, `hsmm_select.json`, `hsmm_final.json`,
`hsmm_controls.json`, and the pre-run diagnostic `hsmm_floor_diagnostic.json`.
Everything uses the tune split (60 animals, 770 recordings).

## Verdict

**Per the registration, the model is not used for learning fits.**

| registered item | result | |
|---|---|---|
| HSMM vs plain HMM (held-out, 5-fold animal CV) | log-normal best; gain over geometric **+0.0015 nats/frame [−0.0003, +0.0034]**, n = 60 animals | **semi-Markov part DROPPED** (lower bound ≤ 0) |
| jitter-only stationary control | mean P(freeze) **0.994**; **min 0.860** (one recording of 50: `…CFC_Day_0_(Context_A)_602`); bar: mean ≥ 0.95 **and** every recording ≥ 0.90 | **FAIL** |
| planted bouts | frame accuracy **0.957** (bar 0.90; min 0.950); one source skipped for having no qualifying move snippet | duration recovery **NOT_A_RESULT**: the Weibull recovery is undefined once the semi-Markov part is dropped |

**Which control fails, and how.** The stationary control fails on **one** recording
of 50; the other 49 read ≥ 0.95. The registered bar is per-recording, so this is a
FAIL, and nothing was adjusted after seeing it. Why that recording falls short is
not yet examined. Whether to amend, investigate, or accept the failure is the
human's decision.

## The emission had to be re-anchored twice before any fit

| freeze emission | real frames below the jitter-only surrogate's 5th percentile |
|---|---|
| as registered: motionless pose + Step 1 jitter at 1.0× | **72%** (diagnostic, 30 recordings) |
| Amendment 1: skull-estimator level on still frames | about 50% (60-recording smoke) |
| **Amendment 2: frame-to-frame increment on still frames** | **2.3%** (all 770 tune recordings) |

**Still-frame jitter is sub-pixel.** The increment SD of the skull bone lengths
between consecutive still frames is **0.0021 BL**, measured on 342 recordings with
≥ 200 still pairs, so the operator level is **m_inc = 0.032** of Step 1's 1.0×. The
level-based estimator on the same frames reads 0.036 BL, and the whole-recording
level is 0.0618 BL.
- **The tracker is far steadier on still frames than its whole-recording noise
  calibration says.**
- That bears on Step 1's "Luna level" and on Paper 1's σ̄ (`GAPS.md` G14,
  `paper1/GAPS.md` E9).

## What the fitted model (plain HMM, after the drop) says on tune

**These are descriptive.** The model failed a registered control.

- **Mean bout durations:** freeze **0.37 s**, move **1.38 s**, tracking_lost
  **2.06 s**. The median decoded freeze bout is **0.2 s**.
- **Freeze fraction on CFD days:** Context A **0.212** (unflagged frames 0.216);
  Context B **0.211** (unflagged 0.225); 295 recordings each.
- **The prior-imprint control.** Frame-shuffled emissions decode to a median freeze
  bout of **0.1 s** (10th–90th percentile 0.03–0.23 s), against **0.2 s**
  (0.07–1.0 s) for the real decode, KS = 0.38. The circular shift reproduces the
  real decode (KS < 0.001). So the decoded dwell is not the prior's doing alone.
- **Flicker:** the raw per-frame argmax switches **5.0 times per second**; the
  smoothed posterior **1.1 per second**.

**Reading.** A frame-level freeze state anchored to measured still-frame jitter
gives **sub-second "freeze" bouts**. Fear-conditioning freezing is usually scored
in bouts of seconds. Whether this model's "freeze" is the same quantity is not
established here, and the model has failed a registered control in any case.
**No Step 4 learning quantity is computed.**

## Not run (registered for C5, blocked)

- shuffled session order and time-reversed sessions;
- the stationary surrogate with learning;
- planted learning;
- the MDE.

All of these wait for the lab's shock protocol and session types (`PROTOCOL.md`).

## Diagnosis of the stationary failure (2026-10-20; descriptive, after the result)

**The failing recording's surrogate is degenerate, through a construction bug in
the registered surrogate.**
- "The recording's median pose" is taken **keypoint by keypoint in image
  coordinates over the whole recording**.
- As the animal turns around the arena, the median nose and the median tail_base
  collapse toward the same point.
- For the failing recording (`…_602`, P(freeze) 0.860), the median pose's
  nose–tail_base distance is **0.024 BL**. Across the 50 stationary recordings the
  median is 0.748 BL (minimum among the rest 0.091), and P(freeze) correlates with
  this ratio (r = 0.41).
- With the heading axis near zero length, the noise swings the heading, the
  egocentric frame rotates, and the pose-change rate r rises: its median bin is
  15, against 9 typically.

**The same median poses built the freeze emission histogram** (`prep`, pass 3), so
the defect is not confined to the control. The registered result stands as
reported. A corrected construction needs an amendment made after seeing the
result, and that is the human's decision.
