# Registration — MOSEQ: Keypoint-MoSeq run as a test, with the answer fixed in advance

Committed **alone, before any MOSEQ code exists** and before any model is
fitted to the unsmoothed repaired pose. Inherited digest **`198eb14ff258c7f6`**.

**Why.** MoSeq always returns syllables. Its prior assumes a sequence of
discrete autoregressive regimes and fits the best such sequence whether or not
the regimes are real, so a count of 40–50 syllables proves nothing. `~/recur`'s
audit (`INVENTORY.md`, `LEDGER.md` §33, §39, §48) showed this directly:
- on white noise, MoSeq returned 32 syllables with a median dwell of 12 frames,
  the corpus's own figure, and non-geometric dwell produced by the κ = 1e6
  stickiness prior;
- its grammar test passed on white noise;
- 2.82% of frames sat in tracker-driven syllables.

MoSeq does not escape the separation and timescale-gap problem STATES failed on
(`STATES.md`); it imposes discreteness through its prior. Its syllables already
are tokens. **The only question is whether they mean anything**, and it is asked
here as three registered gates plus a reported comparison.

---

## 1. Input: the repaired pose before smoothing

MoSeq has its own noise model, and a pre-smoothed pose would flatter it. Per
recording, REPAIR §1 steps 1–3 only:
1. the disposition pose (`work/disposition/<animal>.npz`);
2. the stored mask veto (`vetoed` in `work/repaired/<animal>.npz`);
3. missing = DLC's own missing flag ∪ vetoed. Runs of ≤ 3 frames are linearly
   held; longer runs, and runs touching either end, are NaN.

**No `median_0.25`.** The code path is shared with REPAIR: its output must
equal `rolling_median(unsmoothed)` (tested).

Confidences passed to kpms: DLC's likelihood where the keypoint is present, and
**0** where it is missing or vetoed. Seven keypoints (the tail tip is dropped,
as in every earlier fit), 30 fps.

**Surrogates**, built per recording from the same pose, never across a seam:
- `white`: `recur.recurrence.surrogate.white_surrogate`, i.i.d. Gaussian at
  each coordinate's own mean and SD (the audit's generator);
- `phase`: `recur.recurrence.surrogate.phase_surrogate`, common-phase
  randomisation of the flattened (T, 14) pose, which keeps every auto- and
  cross-spectrum.

Gaps are linearly filled for generation only. The real recording's NaN mask
and confidence trace are then re-imposed, so **all three arms have identical
missingness and confidence**.

## 2. Fitting (identical for every arm)

- keypoint_moseq 0.6.8 / jax_moseq 0.3.3 (`~/moseq/venv-moseq`), x64
  precision.
- The protocol of the labeller of record (`~/moseq/train_moseq.py`):
  - latent_dim 10, nlags 3;
  - AR-HMM at κ = 1e4 for 50 iterations;
  - then the full keypoint-SLDS at κ = 1e6 up to iteration 550;
  - anterior/posterior body parts as in `luna_demo/config.yml`.
- **κ is not tuned.** It is the labeller of record's value.
- **Fit set:** 100 recordings drawn with `random.Random(0)` from the **tune**
  split's recordings. It is the **same 100 recordings** for every arm, so only
  the signal differs.
- **Seeds (Gibbs/init):** real 0–4, white 0–2, phase 0–2, making 11 fits.
- **Apply:** each model to its own arm's **report** recordings (89 animals),
  `apply_model`, 20 iterations, giving one syllable per frame.

## 3. Gate 1: held-out likelihood beats the surrogates

**Latent trajectory.** Each report recording is projected into the model's own
latent space, deterministically:
- kpms's egocentric alignment (`align_egocentric`, anterior/posterior body
  parts);
- its centre embedding;
- x = pinv(C)(y − d), with C and d from the model's `Cd`.

A frame with any NaN keypoint breaks the sequence. Each unbroken run of at
least 30 frames is scored on its own, starting at its fourth frame.

**MoSeq's predictive likelihood.** The exact forward filter of the fitted
AR-HMM:
- Ab: AR(3) + bias per state; Q: noise covariance per state; π: the sticky
  transition matrix, all 100 states as fitted;
- initial distribution: π's stationary distribution;
- per frame, log p(xₜ | x₁:ₜ₋₁), in nats.

**Baseline.** One AR(3) + bias with full noise covariance, least squares on the
same arm's 100 fit recordings projected the same way. Per frame, log p(xₜ | xₜ₋₃:ₜ₋₁).

**gain = mean over frames of (MoSeq − baseline)**, per animal on report, with an
animal bootstrap of 2,000 replicates (seed 0). The bootstrap resamples animals
and recomputes the paired difference below on each replicate.

> **Gate 1 PASS iff the 95% lower bound of
> gain_real − max(gain_white, gain_phase) exceeds 0.05 nats/frame**
> (≈ 1.5 nats/s), using seed 0 of each arm. The other seeds' gains are
> reported.

A linear-Gaussian process (phase) or an i.i.d. one (white) is fitted exactly by
the baseline. Their gain measures what the switching prior buys on data with no
regimes. The mouse gain must beat that by the margin.

## 4. Gate 2: seed stability

Frame-level AMI (`sklearn.metrics.adjusted_mutual_info_score`) of the report
syllable labels, for each of the 10 pairs among real seeds 0–4.

> **Gate 2 PASS iff the median pairwise AMI ≥ 0.70.**

For reference, the bottleneck codes that disqualified the VIEB encoder scored
a median of 0.432 over 10 seed pairs. White and phase seed AMIs (3 pairs each)
are reported beside the real value.

## 5. Reported beside the real syllables, not gated

For every arm and seed, on report:
- syllables in use (> 0.5% of frames, the kpms convention) and syllables used
  at all;
- median dwell (frames) and var/mean² of dwell;
- the share of frames in the 10 most-used syllables.

**The white-noise count and dwell always sit next to the real ones.**

## 6. Verdict

- **PASS** iff gates 1 and 2 both pass.
- **FAIL** if either fails, and the failing gate is named.
- **NOT_A_RESULT** if any fit diverges (non-finite log-likelihood), or more
  than 5% of report frames cannot be scored in §3.

## 7. Context test: do syllables separate A from B beyond pixel freezing?

**Pixel freezing on report.**
- `vieb.pixel.motion.scan` on every report recording of days 3–7 in contexts A
  and B (the `pixel_pilot.py` shard code), giving Motion at `mt_cutoff = 10`
  grey levels.
- Per frame: `freeze.freeze_mask(motion, 800, min_duration = 15)`.
- **FreezeThresh = 800** is ezTrack's 200 scaled to this corpus's 4× frame
  area, the equivalent `PIXEL_PREREGISTRATION.md` §2 states. It is registered
  here because that pilot's own derivation left the threshold undefined on 62%
  of cells. **The freezing measure itself was not tuned to this test.**

**The test.**
- Unit: report recording, days 3–7, contexts A/B.
- Features:
  - **F** = freeze fraction;
  - **S** = CLR-transformed occupancy of the syllables in use (seed 0),
    computed on **non-freezing frames only**.
- Logistic regression (L2, C = 1, standardised features), context A vs B,
  **animal-grouped 5-fold CV** (seeded permutation of animals, seed 0).
- Scored as per-animal balanced accuracy with `boot.animal_interval`.

> **ADDS iff the 95% lower bound of BA(F + S) − BA(F) > 0.**
> Otherwise **REPRODUCES FREEZING**. BA(F) and BA(S) are reported alone.

**Leak guard.** S computed on **freezing frames only**, where behaviour is by
definition equivalent across contexts. If its BA lower bound exceeds **0.60**,
the syllables are reading the arena. ADDS is then reported as
**CONFOUNDED**, not as behaviour.

The context test runs whatever §6 says. **It is read as evidence about
behaviour only if MOSEQ is PASS**; otherwise it is labelled descriptive.

## 8. Prohibitions

- No κ, latent_dim, iteration count, margin or bar is changed after any fit
  exists.
- No report-split tuning.
- Surrogates pass through the identical pipeline, including their own fits.
- No syllable is named as a behaviour without its clips.
- Nothing is read as grooming.
- No shock-versus-no-shock framing.

---

# Amendment 1: what the 5% refusal counts, and abstained frames

**Committed alone, recorded as `DEVIATIONS.md` D30.** Made while the 11 fits
are queued or running, before any fit has finished and before any MOSEQ
result exists. The only data looked at are the input files' confidence
traces.

1. **§6's refusal rule was ambiguous, and read literally it refuses by
   construction.** §3 excludes, by design, frames with an unobserved keypoint,
   runs shorter than 30 frames, and the first 3 frames of every run. On the
   report input that is **7.0%** of 6,678,488 frames (4.66% have an unobserved
   keypoint). The exclusion is identical in all three arms, because they share
   one confidence trace. The rule was meant for a *projection or likelihood
   failure*. **Amended:** NOT_A_RESULT iff a fit diverges, or more than 5% of
   the frames §3 selects for scoring have a non-finite latent or
   log-likelihood. The share §3 excludes by design is reported beside the
   verdict.

2. **Frames disposition abstained on are missing on every keypoint** in the §1
   input: confidence 0, coordinates filled. §1 said "the disposition pose"
   without saying what happens on its abstained frames, which keep their
   bone-violating coordinates in `work/disposition`. `ego` already treats
   those frames as unusable, and MOSEQ follows it.

**Unchanged:** every bar, the fit protocol, the surrogates, gates 1 and 2, and
the context test.
