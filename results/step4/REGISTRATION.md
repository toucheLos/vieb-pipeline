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
