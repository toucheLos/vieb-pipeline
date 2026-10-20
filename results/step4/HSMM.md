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

---

## Run 2: after Amendment 3 (real still pose), post-result and registered as such

The original run (above, and `results/step4/run1/`) stays on record. **Every bar is
unchanged.**

| registered item | run 1 (median pose) | **run 2 (real still pose)** |
|---|---|---|
| real frames below the surrogate's 5th percentile | 2.3% | **2.0%** |
| HSMM vs HMM gain (nats/frame) | +0.0015 [−0.0003, +0.0034]: dropped | **+0.0015 [−0.0007, +0.0038]: dropped** |
| jitter-only stationary (mean / min) | 0.994 / 0.860: **FAIL** | **0.998 / 0.996: PASS** |
| planted bouts: accuracy (bar 0.90) | 0.957 | **0.963** |
| planted bouts: duration recovery | NOT_A_RESULT (geometric) | **NOT_A_RESULT (geometric)** |

**What the fitted plain HMM says.** These are descriptive, and the model is not used
for learning fits.
- **Mean bout durations:** freeze 0.31 s, move 1.21 s, lost 0.66 s. The median
  decoded freeze bout is 0.17 s.
- **Freeze fraction on tune CFD days:** Context A 0.191, Context B 0.184.
- **Prior imprint:** frame-shuffled emissions decode to a median of 0.07 s, against
  0.17 s for the real decode (KS 0.35). The circular shift matches the real decode.
- **Flicker:** 4.7 switches per second raw, 1.3 smoothed.

**Status against C4.** The stationary control **passes**. The planted control's
duration half is **NOT_A_RESULT**, because explicit durations were dropped, so the
planted control has **not passed** as registered. The model is therefore still not
cleared for learning fits, and C5 is blocked on the lab in any case.

**Reading.** With the semi-Markov part dropped twice, a plain HMM with
noise-anchored emissions decodes **sub-second "freeze" bouts** on about 19% of
frames. Whether these correspond to scored fear freezing, which is usually counted
in bouts of seconds, is open. **No learning quantity is computed.**

---

## Amendment 4 planted control and the known-answer check (both registered; tune split)

**Data:** `hsmm_planted_bouts.json` (Amendment 4, D36, post-result) and
`hsmm_known_answer.json` (`KNOWN_ANSWER.md`, 29e84cc).

### Planted control (amended): PASS

| item | result | bar |
|---|---|---|
| frame accuracy | **0.963** (min 0.957) | ≥ 0.90 |
| recall, planted freeze bouts ≥ 1 s (IoU ≥ 0.5) | **0.985** [0.981, 0.989], n = 1,632 | ≥ 0.80 |
| recall, planted freeze bouts ≥ 2 s | **0.991** [0.986, 0.996], n = 1,111 | ≥ 0.80 |

Beside the bars:
- precision is 0.991 (≥ 1 s) and 0.997 (≥ 2 s);
- a matched decoded bout has 0.94–0.95 of the planted duration (median);
- 4% of planted bouts are fragmented.

**The decoder recovers long freeze bouts when the input contains them.** The
sub-second bouts on real data therefore do not come from the decoder.

### Known-answer check: row 4, FAIL

There are 60 paired animals and 590 CFD recordings on Days 3–7, and every
animal-day has both contexts. No pose/video frame-count mismatches.

**Fraction of frames in freeze bouts, by bout length.** Each cell is the paired
A − B with its animal-bootstrap 95% interval.

| bout length | HSMM A | HSMM B | **HSMM A − B** | pixel A | pixel B | **pixel A − B** |
|---|---|---|---|---|---|---|
| < 1 s | 0.139 | 0.115 | **+0.024 [+0.007, +0.041]** effect | 0.031 | 0.032 | −0.002 [−0.005, +0.002] |
| 1–2 s | 0.041 | 0.044 | −0.003 [−0.013, +0.007] | 0.048 | 0.046 | +0.002 [−0.004, +0.009] |
| ≥ 2 s | 0.014 | 0.028 | **−0.014 [−0.022, −0.007] reverse** | 0.669 | 0.479 | **+0.190 [+0.154, +0.226]** |
| total | 0.193 | 0.187 | +0.007 [−0.024, +0.037] | 0.748 | 0.557 | **+0.191 [+0.157, +0.225]** |

- **The pixel anchor holds on tune.** The pixel score gives Context A **+19.1
  points** more freezing, and nearly all of that difference is in bouts ≥ 2 s.
- **The decoded freeze runs the other way in the long bins.** It shows *less*
  long-bout freeze in A, and the A excess sits in sub-second bouts.
- **That is registered row 4: "decoded long freeze runs against both signals; the
  freeze state is not freezing."** The model stays blocked.

**Per-frame agreement within each context** (Cohen's κ with the pixel score; A / B):

| decoded freeze frames | all frames | unflagged frames |
|---|---|---|
| all | 0.137 / 0.216 | 0.138 / 0.227 |
| in decoded bouts ≥ 1 s | 0.036 / 0.066 | 0.036 / 0.069 |
| in decoded bouts ≥ 2 s | 0.008 / 0.023 | 0.008 / 0.024 |

Within either context, the two signals hardly agree on which frames are freezing.

**Centroid speed (descriptive):**
- the median per-animal A/B ratio is **0.51** [0.40, 0.66];
- median speed is 0.033 BL/s in A and 0.068 BL/s in B.

### What this says (descriptive, after the result)

- **The keypoint signal carries the context difference.** Raw centroid speed is
  halved in A, in the same direction as the pixel score. So the failure is not
  "the keypoints miss freezing".
- **The freeze state is too strict.** It is anchored to still-frame tracker jitter
  (Amendment 2: m_inc = 0.032 of Step 1's level, 0.0021 BL increments). Only frames
  as motionless as tracker noise qualify.
- A freezing mouse breathes and sways. Those frames sit above that floor, so they go
  to "move", and long freezing is cut into sub-second pieces whenever motion briefly
  reaches the floor.
- On this reading, the "freeze" state measures **tracker-floor stillness**, not
  freezing. That fits the planted control passing: the planted freeze was exactly a
  rigid pose plus jitter.
- **Any fix is a registered change to the freeze state's definition** (for example,
  an emission anchored to scored freezing rather than tracker jitter). Choosing one
  is the human's decision. **No learning quantity is computed.**
