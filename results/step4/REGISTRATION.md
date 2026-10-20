# Registration — STEP 4 machinery: a freeze / move hidden semi-Markov model

Committed **alone**, before any HSMM code exists. It covers C1–C4 of the Step 4
revision prompt (2026-10-17), and all of it runs on the **tune** split (60 animals)
and on surrogates. **C5, the learning fits on the fit split, is blocked** on the shock
protocol (`PROTOCOL.md`, items 1, 2 and 4). It is named here and will be registered
in full before it runs.

**Purpose.** Return P(freeze at t) per frame, in place of a threshold-plus-minimum-
duration rule, with explicit dwell durations and emissions anchored to the measured
tracking-noise floor rather than to behaviour labels.

**Conventions:**
- every claim is a `recur.read.Read`;
- intervals are animal bootstraps (2,000 replicates, seed 0);
- every temporal parameter is in seconds and is converted to frames once, through
  `recur.util.frames`;
- nothing is computed across a recording boundary;
- `mypy --strict` on the new modules.

## C1. The core: domain-agnostic (`vieb/hsmm/core.py`; no mouse imports)

- **States** for the mouse model: `freeze`, `move`, `tracking_lost`. The core takes
  any S.
- **Pieces:**
  - emission log-likelihoods `log e_s(x_t)`, a (T, S) array;
  - a hazard `h_s(τ)`, where τ is the time already spent in the state, in frames;
  - an exit kernel `P(j | s)`, with zero diagonal. Registered simplification: the
    kernel does **not** depend on τ.
- **Forward (filtering).** With α_t(s, τ) the probability of being in s with elapsed
  time τ, given x_1..t:

      stay:  α_t(s, τ)  ∝ e_s(x_t) · α_{t-1}(s, τ-1) · (1 − h_s(τ-1))
      enter: α_t(j, 1)  ∝ e_j(x_t) · Σ_s Σ_τ α_{t-1}(s, τ) · h_s(τ) · P(j | s)
      P(s_t | x_1..t) = Σ_τ α_t(s, τ)

- **The cap.** τ is capped at **D_max = 10 s**. Beyond the cap the hazard is held at
  h_s(D_max) (a constant-hazard tail), so no probability mass is lost.
- **Backward.** The backward pass gives smoothed posteriors P(s_t | x_1..T).
- **Initial distribution:** τ = 1, states uniform. Each recording restarts from it.
- **Checks:**
  - normalisation is asserted at every step (|Σ − 1| < 1e-9);
  - the computation is in log or scaled space;
  - **one recording per sequence**: a batched implementation pads and masks, and a
    seam-invariance test perturbs one recording and asserts that another changes by
    zero bits.
- **Duration families:**
  - *geometric*: constant hazard, so the model is an HMM;
  - *Weibull* (shape k, scale λ in s);
  - *log-normal* (μ, σ of log seconds).

  The hazard is h(τ) = P(D = τ | D ≥ τ) for the discretised family, with frames as
  the time unit.
- **Tests:**
  - brute-force enumeration over segmentations on tiny T;
  - the geometric HSMM equals the HMM (`vieb/moseq/forward.py` `forward`) to 1e-10;
  - posteriors sum to 1;
  - seam invariance;
  - planted-bout recovery on synthetic emissions.

## C2. Emissions: from the measured noise floor, not from labels (`vieb/hsmm/mouse.py`)

**Signal, per frame, from the `raw` arm.** The arm is
`held_array(pose_unfiltered, missing)` from `spine.clean`. ℓ_a is the animal's
median nose–tail_base length over its usable frames (`ego.ell_a`). Window
**w = 0.2 s**: displacement over [t − w/2, t + w/2], clipped at recording edges.
- `v_t`: centroid speed. The centroid is the mean of the 7 keypoints; speed is
  |Δcentroid| / w / ℓ_a, in body lengths per second.
- `r_t`: pose-change rate. The mean over keypoints of
  |Δ egocentric coordinate| / w, with `ego.egocentric(pose, ℓ_a)`, in body lengths
  per second.

**The emission variable** is x_t = (log₁₀ v_t, log₁₀ r_t), binned on a fixed
**40 × 40 grid from −3 to 1.5** (values outside are clipped). It also carries the
**tracking flag** f_t = `bone_flagged | missing.any | interpolated.any` (the
committed flag used by recur's triage).

**No whole-frame pixel features.** Contexts A and B are different images.

**p(x | freeze): the distribution jitter alone produces.**
- **Rendering.** Each tune recording's median pose, held motionless and scaled to
  ℓ = 113.72 px, has Luna's measured tracking noise added: Step 1's operator at
  1.0×, i.e. `degrade.jitter_noise` with `added_bl(1, 0, profile, g)` and
  `noise.json`'s s1 and gain.
- **Same computation.** The result goes through the identical signal computation.
- **Size.** One surrogate per tune recording, at that recording's length.
- **Smoothing of the histogram.** A pseudocount of 1e-6 per bin, then normalised.
- **It is not fitted to behaviour.**

**p(x | move): nonparametric.**
- **Data.** Fitted on the tune split's unflagged frames.
- **Model.** A two-component mixture, π · p_freeze + (1 − π) · h_move. **p_freeze
  is held fixed**, and h_move is a free 40 × 40 histogram.
- **Fit.** EM to a relative log-likelihood change < 1e-8 or 500 iterations, with
  pseudocount 1e-6. It uses no temporal information.

**tracking_lost:**
- P(f = 1 | tracking_lost) = 0.99.
- P(f = 1 | freeze) = P(f = 1 | move) = the tune split's flag rate.
- The (v, r) emission under `tracking_lost` is the tune split's marginal histogram,
  which is uninformative.

**Combined emission.**
log e_s(x_t) = log p_s(bin(v_t, r_t)) + log P(f_t | s).

## C3. Durations: a prior, estimated only from registered fits

1. **Fit.** The parameters are fitted by **maximising the HSMM marginal
   likelihood** (the forward pass) with scipy `minimize` (Nelder–Mead,
   maxiter 400):
   - per state: the family's two parameters (one for geometric);
   - the exit kernel: one free probability per state.

   **They are never estimated from decoded sequences.** For tractability the fit
   uses **120 recordings** drawn with `random.Random(0)` from the training animals of
   each fold (two per animal where available). Evaluation uses **all** recordings of
   the held-out animals.
2. **Family.** Weibull against log-normal by **held-out log-likelihood** under
   **5-fold animal cross-validation on tune** (animals permuted with seed 0). The
   statistic is the mean per-frame log-likelihood per held-out animal, paired across
   models. The family with the higher mean wins, and a tie goes to Weibull.
3. **Against geometric dwell** (a plain HMM), the same way. The gain is the chosen
   family minus geometric, in nats per frame per animal, with an animal bootstrap.
   **The semi-Markov part is kept iff the lower 95% bound of the gain is > 0.**
   Otherwise it is dropped, and the report says so.
4. **The prior-imprint control.**
   - The emissions are **shuffled across frames within each recording**: no temporal
     structure, the same marginals. They are decoded with the fitted model, and the
     decoded freeze-bout durations are reported beside the real decode: median,
     10th/90th percentiles, and the KS distance.
   - The share of the real decoded dwell that the prior reproduces from shuffled
     input is reported as the ratio of median bout durations.
   - **The literal circular shift** of the whole emission sequence is reported too, as
     a sanity row. It should match the real decode.
5. **Flicker.** The switch rate (per second) of the raw per-frame emission argmax is
   reported beside the switch rate of the smoothed posterior argmax.
6. **No states beyond freeze / move / tracking_lost** without a registered result
   from a representation that shows them.

## C4. Controls: each a Read, each must pass before any learning fit

| control | construction | pass (registered) |
|---|---|---|
| **jitter-only stationary** | 50 tune recordings (seed 0): median pose, motionless, plus Luna jitter at 1.0×, at the recording's length | mean smoothed P(freeze) ≥ **0.95**, and ≥ 0.90 in every recording |
| **planted bouts** | 30 synthetic sessions of 5 min. Freeze bouts are a motionless median pose plus Luna jitter. Move bouts are real tune pose snippets whose windowed centroid speed exceeds **0.5 BL/s throughout** (chosen by the signal, never by the model), carrying their own real jitter. Bout durations are drawn from Weibull(k = 1.5, λ = 3 s) for freeze and Weibull(k = 1.2, λ = 2 s) for move. | (a) frame accuracy of smoothed P(freeze) > 0.5 against the truth is **≥ 0.90**; (b) refitting the durations on the planted sessions recovers each true (k, λ) **inside its 95% bootstrap interval over sessions** |
| **session order and time reversal** | implemented and unit-tested now; **run with C5** (they are learning-fit controls) | as registered for C5 |
| **missingness** | decoded freeze fraction per context (A / B, tune CFD days), with and without flagged frames | descriptive; reported per context |

**If a control fails, the model is not used for learning fits**, and the failure is
reported as the result.

## C5 (named, blocked)

Step 4 as this model: hazard parameters vary by session and day, fitted
hierarchically by animal on **fit**. The MDE comes first. The controls are the
shuffled session order, time-reversed sessions, a stationary surrogate and planted
learning. Uncertainty comes from an animal bootstrap that refits all parameters. It
**waits for** the shock protocol and session types from the lab, and is registered
in full before it runs.

---

# Amendment 1: the freeze emission is anchored to the still-frame noise floor

**Committed alone, recorded as `DEVIATIONS.md` D33.** Made after a smoke test and a
pre-run diagnostic (`hsmm_floor_diagnostic.json`, 4c8b255). It precedes any
registered fit, control or decode. The user chose this option over three others.

**The defect.** The registered p(x | freeze), a motionless pose plus Luna's
measured jitter at the Step 1 level, sits **above** the stillest real frames.
- The real pose-change rate has its 5th percentile at 0.010 BL/s. The surrogate's
  median is 0.362 (shapeflow's native noise gives 0.129).
- Half of the real frames lie below the surrogate's own 5th percentile.
- EM gives the freeze component about zero weight.
- So the tracker is steadier on still frames than its whole-recording noise
  calibration, and a stationary noise floor cannot represent stillness.

**Amended C2, freeze emission.** The jitter level is re-measured **on still frames
only**, with the same skull-bone estimator:
1. **Still frames:** tune-split frames that are unflagged and lie in the **lowest
   decile (10%) of the registered centroid speed v_t**, with the decile pooled over
   all unflagged tune frames.
2. **σ̄_still:** `vieb.seg.jitter.residuals` on the `raw` arm, with every non-still
   frame marked unusable, so the per-recording reference length is the median over
   that recording's still frames. Then `jitter.calibrate` on DLC confidence, and the
   count-weighted mean (`scripts/plant.py sigma_bar`'s statistic).
3. **The freeze surrogate** is as registered, a motionless median pose plus Step 1's
   jitter operator, but at level **m_still = σ̄_still / σ̄_L** instead of 1.0:
   `added_bl(m_still, 0, profile, g)`. This keeps Luna's measured colour,
   common-mode share and per-keypoint profile.
4. The rest of C2 is unchanged: p(x | move) by EM with the new freeze density held
   fixed; tracking_lost; the grid.

**Reported beside it (descriptive):** σ̄_still against σ̄_L; and the share of real
unflagged frames below the new surrogate's 5th percentile, for comparison with the
diagnostic's 0.72.

**What this concedes.** "Still" is chosen by centroid speed, which is behaviour,
so the anchor is no longer behaviour-free. The emission is still a **noise model**
measured on the rigid skull, not a fit to posture change. The jitter-only control
becomes "a motionless animal with still-frame jitter must read freeze".

**Unchanged:** C1, C3 (duration fits, family choice, HSMM against HMM, prior
imprint, flicker), C4's bars, and C5.

---

# Amendment 2: the still-frame jitter is measured from frame-to-frame increments

**Committed alone, recorded as `DEVIATIONS.md` D34.** It supersedes Amendment 1's
estimator; the still-frame selection is kept. Made after Amendment 1's
**60-recording smoke**, which is not a registered run and from which no result is
reported. No registered fit, control or decode exists. The user chose this option.

**Why Amendment 1 was not enough.** On still frames this tracker's frame-to-frame
jitter is sub-pixel: in about 10% of consecutive frame pairs every keypoint moves
less than 0.5 px. The skull-bone estimator measures each bone's deviation from a
**reference length**, so it also absorbs real differences between still bouts,
such as head posture. On the smoke it read 0.0254 BL, and half of the real frames
still lay below the surrogate's 5th percentile. The emission signal (v, r) is built
from displacements over a short window, so its noise floor has to be measured the
same way, from increments.

**Amended C2, freeze emission.**
1. **Still frames:** unchanged from Amendment 1. Unflagged tune frames in the
   lowest decile of the registered centroid speed v_t, with the decile pooled.
2. **The real increment statistic.** For each recording, the three skull bone
   lengths are taken on `pose_unfiltered`, in BL (÷ ℓ_a). Increments are
   ΔL = L_{t+1} − L_t over **consecutive pairs where both frames are still**, with
   no missing or interpolated endpoint. s_real is the SD of ΔL pooled over bones and
   recordings, the **median over recordings** of each recording's SD (recordings
   with fewer than 200 pairs are skipped).
3. **Calibration through the same statistic.** s_op(1) is the same increment SD
   applied to Step 1's jitter operator at level 1. That means
   `added_bl(1, 0, profile, g)` noise on the rigid calibration pose (Luna's median
   skull geometry, as in Step 1), 300,000 frames, seed 0.
   **m_inc = s_real / s_op(1)**. The operator is linear in its level, so this keeps
   Luna's measured colour, common mode and per-keypoint profile.
4. **The freeze surrogate** is a motionless median pose plus the operator at
   **m_inc**. The rest of C2 is unchanged.

**Reported beside it (descriptive):** s_real, s_op(1), m_inc; Amendment 1's
level-based σ_still, for comparison; and the share of real unflagged frames below
the new surrogate's 5th percentile.

**Unchanged:** C1, C3, C4's bars (the stationary and planted controls use m_inc
jitter), and C5.

---

# Amendment 3 (post-result): the motionless pose is a real still pose, not a coordinate-wise median

**Committed alone, recorded as `DEVIATIONS.md` D35.** **Decided after the registered
result was seen.** The result was: semi-Markov part dropped, stationary control FAIL
(`HSMM.md`, b72d152 / 3efa462). That result stays on record unchanged. The user chose
this amendment.

**The defect.** "The recording's median pose" (C2, C4) was a keypoint-by-keypoint
median in image coordinates over the whole recording. As the animal turns around the
arena it collapses: the failing recording's median pose has nose–tail_base 0.024 BL,
and the typical one 0.748 BL. A degenerate heading axis lets the noise rotate the
egocentric frame, which inflates the pose-change rate. The same poses built the freeze
emission.

**Amended construction**, wherever C2–C4 uses "the median pose": the freeze surrogate,
the stationary control, and the planted freeze bouts.
- The motionless pose is the **real pose at the recording's stillest frame**: the
  unflagged, fully usable frame with the smallest registered centroid speed v_t.
  Ties are broken by the earliest frame.
- It is a real mouse's geometry. It carries that one frame's tracking error, which is
  fixed and adds no motion.
- It is scaled to ℓ = 113.72 px as before.

**Re-run.** Prep pass 3 (the freeze surrogate) and the EM move fit, then the 5-fold
fits, select, final, the controls, the planted refits and the report. **Every bar,
threshold, family rule and seed is unchanged**, and so is Amendment 2's noise level
(m_inc).

**Reporting.** The original FAIL and the re-run are reported side by side. The re-run
is labelled post-result.

---

# Amendment 4 (post-result): the planted control tests what a plain HMM can do

**Committed alone, recorded as `DEVIATIONS.md` D36.** **Decided after the run-2
result was seen** (`HSMM.md`, run 2): the semi-Markov part was dropped, so the planted
control's half (b), recovering the planted Weibull (k, λ), is `NOT_A_RESULT` by
construction. A plain HMM has geometric dwell times and cannot reproduce a planted
non-geometric dwell shape. The user directed this amendment. The run-1 and run-2
results stay on record.

**The model is accepted as a plain HMM, but not cleared.** Dropping the dwell-time
model is a result in its own right: it gained nothing over a plain HMM.

**The amended planted control.** Same 30 planted sessions (`work/hsmm/planted.npz`,
run 2), same final model, same decode (smoothed posterior, argmax).
- **(a) Frame accuracy** of smoothed P(freeze) > 0.5 against the truth is **≥ 0.90**.
  Unchanged; run 2 already reads 0.963.
- **(b′) Bout recall.** For planted freeze bouts with duration **≥ 1 s**, and separately
  **≥ 2 s**, the fraction matched by a decoded freeze bout must be **≥ 0.80** in each
  bin.
  - *Matched* means one-to-one at **IoU ≥ 0.5**, Step 1's event rule (`vieb/calms/score.py`
    matching), applied per session.
  - Recall is pooled over sessions. The 95% interval is a bootstrap over the 30
    sessions (2,000 replicates) and is reported, not used as the bar.
- **Reported beside it, not a bar:**
  - the precision of decoded freeze bouts in each bin;
  - the median ratio of decoded to planted duration for matched bouts;
  - the number of decoded fragments per planted bout.

**Passing (a) and (b′) does not clear the model for learning fits.** The known-answer
check (`KNOWN_ANSWER.md`, registered separately) must also be run first, and C5 stays
blocked on the lab's protocol.
