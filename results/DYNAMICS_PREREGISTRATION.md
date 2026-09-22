# Pre-registration — the noise colour, the descriptor floors, and the dynamics probe

Written and committed **before `vieb/seg/noise.py`, `vieb/seg/descriptors.py`,
`vieb/seg/dynplant.py` or `scripts/dynamics_floor.py` exist**, and before any
descriptor floor or recovery number is computed. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm; `shape` group; **`tune` split,
60 animals**. Digest `198eb14ff258c7f6`.

## 1. Why this stage exists

A next architecture proposes cutting where **either** the configuration or the
**dynamical regime** changes. Before any of it is built, one thing has to be
fixed, and it invalidates an assumption two published stages rest on.

`shapeflow/results/CALIBRATION.md` — a **frozen consumed artifact**,
`spine.json_("sf_calibration")`, inside digest `198eb14ff258c7f6` — measured the
tracking residual on 3,846 recordings and **17,160 quiet segments**, and its
whiteness read is **`FAIL`**:

> lag-1 autocorrelation **+0.5817**, decaying over **267 ms**. Cross-keypoint
> correlation +0.1180, common mode **0.2257** of residual power against
> **0.1429** for an exactly independent residual.

**`vieb/seg/jitter.py:draw` samples independent noise per frame.** It is white.
`NOISEFLOOR.md`'s floor of **0.2599/s** and `PLANT.md`'s recovery curve —
order-2 at **16 σ̄ ≈ 0.99 bl** — were both measured against noise of the wrong
colour.

A 267 ms correlation time is **≈3.7 Hz**, and the same artifact puts the
signal's usable band at **0.476–4.83 Hz** (`f_c = 4.833 Hz`, peak SNR 23.2×).
**The tracking noise occupies the same band as the only usable signal.** For an
amplitude-thresholding detector that is survivable; for a *dynamics* channel,
where slow coloured noise is indistinguishable by construction from a slow
behavioural rhythm, it is the central problem. This stage measures it before
anything is built on top.

## 2. The noise model, built from the frozen artifact and not re-derived

`calibration.json` publishes `psd_noise_pooled` on a 32-point grid to 14.76 Hz,
`per_keypoint_f_c`, and the residual `acf` at 11 lags. **The model is
synthesised from those numbers**, not re-estimated:

* **Spectral synthesis** against `psd_noise_pooled`, so the autocovariance is
  reproduced by Wiener–Khinchin rather than approximated by an AR(1) fit. The
  measured ACF is not AR(1): it falls 1.0 → 0.582 → 0.346 → 0.207 and then
  **flattens at ~0.15**, which a single pole cannot produce.
* **A common-mode component carrying `0.2257` of residual power**, shared across
  keypoints, with the remainder independent. That reproduces the measured
  cross-keypoint correlation instead of assuming independence.
* Units are **pixels**, as the calibration is. The existing `jitter.draw`
  returns body lengths and callers multiply by `ell`; the new generator returns
  pixels and is added to the pose directly. The contract difference is stated
  here so it cannot be mistaken for a bug.

**Registered acceptance for the model itself** — mechanical, and checked before
any descriptor floor is quoted:

> the synthesised noise reproduces **lag-1 ACF 0.5817 ± 0.05** and
> **correlation time 0.2667 s ± 1 frame**, and its PSD matches
> `psd_noise_pooled` within a factor of 1.25 across 0.476–14.76 Hz.

If it does not, **no descriptor floor is reported** and the model is the result.

## 3. Descriptors, and the one that is primary

Per window, per channel, on the 14 `shape` channels:

* **Local AR(2–4) poles → (frequency, damping).** Primary. Measured on this
  corpus, the high-frequency noise plateau is **182× lower in still windows than
  in moving ones** while the SNR profile is nearly unchanged — so any
  absolute-amplitude descriptor is dominated by the motion regime, and poles are
  the only proposed descriptor that means the same thing in a freeze and in a
  run. Fitted with `recur.recurrence.surrogate.fit_var`, **on live columns
  only**: the ego array is rank 11 + 3 and a 17-channel fit has a singular
  residual covariance.
* **Multitaper band power** via `scipy.signal.windows.dpss`, **normalised per
  window** so it is a shape and not an amplitude.
* **Motion energy**, reported as a covariate and never a boundary source.

## 4. The bands, and the null that is predicted

Registered bands, from the frozen `f_c`:

| band | Hz | registered expectation |
|---|---|---|
| **slow** | 0.5–2 | clears the floor |
| **mid** | 2–4 | clears the floor |
| **marginal** | 4–6 | may not clear; reported either way |
| **fast** | 6–15 | **predicted not to clear** |

**The fast band is swept anyway.** Excluding it would leave the null asserted;
sweeping it makes it a demonstrated negative result. The prediction is entered
here with its reason: the published movement/noise PSDs cross at **4.83 Hz** and
the SNR is **0.79 at 6.19 Hz** and **0.60 at 8.10 Hz**.

## 5. The dynamics planted probe

The analogue of `PLANT.md`, on the same harness. A dynamics detector exposing
`peaks_of(x, w, alpha, *, blocked=None) -> I64` drops into it unchanged —
`floor.peaks_of` and `trendfilter.peaks_of` already share that signature
deliberately — keeping the placement rule, the ±2 band, the per-recording chance
level and the `white` arm.

Three planted changes, each order-controlled in its own parameter:

1. **Δfrequency** — an AR-pole switch at constant damping and amplitude.
2. **Δdamping** — a pole-radius switch at constant frequency and amplitude.
3. **Δamplitude** — an oscillation-amplitude change at constant poles.

**Reported in physical units**: minimum detectable **ΔHz**, minimum oscillation
**amplitude in body lengths**, minimum **Δdamping** for 50% recall — the
dynamics analogue of `PLANT.md`'s 0.99 bl.

## 6. Predictions, fixed now

1. **The noise model passes §2's acceptance.** Mechanical; if it fails nothing
   else here means anything.
2. **Re-running `NOISEFLOOR.md`'s floor under coloured noise moves it by less
   than 25%.** `PLANT.md` measured near-invariance to amplitude (4× moved the
   rate 6.5%) and the threshold is scale-adaptive. *Falsifier: a larger move
   means the published 0.2599/s is colour-dependent and must be restated, and
   this stage says so rather than leaving it.*
3. **The AR-pole null under coloured noise concentrates below 4 Hz.** This is
   the threat stated as a prediction: 267 ms-correlated noise *is* a slow
   apparent rhythm. *Falsifier: a flat or high-frequency null would mean the
   colour does not masquerade as dynamics, and the channel is easier than
   feared.*
4. **Δfrequency recovery reaches 50% somewhere in 0.5–4 Hz and nowhere in
   6–15 Hz.**
5. **The `white` arm stays below the corpus arm**, as it does for the frozen
   detector. A new detector re-incurs `DETECTOR_PREREGISTRATION.md:84-90`; it is
   not inherited.

## 7. The gate

> If no plausible mouse rhythm clears its descriptor floor under **correctly
> coloured** noise, the dynamics channel **stops here** and the result is
> reported as *"footage lacks the signal"* for in-place rhythm.

Expected to pass for 0.5–4 Hz and fail above 6 Hz. Both outcomes are registered,
and a refusal is a complete finding.

## 8. What this stage may not do

* It may not **silently restate** `NOISEFLOOR.md` or `PLANT.md`. If prediction 2
  fails, the correction is a `DEVIATIONS.md` entry and an explicit amendment,
  not an edit.
* It may not **re-score Q1**, restate **Phase F**, or use any **Wiener** array.
* It may not **tune** the frozen detector, or change `f_c`, the band edges, the
  ±2 band or the probe ladder after seeing a recovery number.
* It may not report a stratum below **20,000 scored keypoint-frames**.
* It may not **compute across a recording boundary**.
* It may not describe any context contrast as **shock versus no-shock**.
  `recur/labels.py:41-43` refuses a `shocked` flag and the mapping is not in
  this repository; it is Context A versus Context B.

## 9. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only, 2,000
replicates, `how="mean"`. Ranked on the effect-CI lower bound, never a p-value.
Seed 0, `seeds.stable_seed`. `mypy --strict` on new modules. The **`raw`** arm
asserted by name.

**And the gate discipline this programme paid for.** `METHODS_FINDINGS.md`
**M12**: a registered threshold is only a fair test if it is calibrated against
the thing it replaces. **Every gate here states the incumbent's value on the
same quantity** — the frozen detector's jitter share **0.5475**, boundary rate
**0.443/s**, order-2 recovery floor **16 σ̄ ≈ 0.99 bl**.
