# Stage 0 — the noise has the wrong colour, the floor survives it, and only persistence clears

Registered in `results/DYNAMICS_PREREGISTRATION.md`, committed before
`vieb/seg/noise.py`, `descriptors.py`, `dynplant.py` or
`scripts/dynamics_floor.py` existed. **`tune` split, 60 animals**, F3 `raw` arm,
`shape` group. Digest `198eb14ff258c7f6`.

## The gate, in one line

> **The dynamics channel is viable as a *persistence* channel in the slow band,
> and as nothing else.** Pole radius separates corpus from noise decisively
> (88.7% of corpus windows above the null's 95th percentile, against the 5%
> chance gives). Pole **frequency** does not separate at all (2.1%). Nothing
> above 2 Hz clears.

## The noise model — `PASS`

Built from the frozen calibration rather than re-estimated: spectral synthesis
against `psd_noise_pooled`, plus a common-mode component carrying the measured
cross-keypoint correlation.

| | synthesised | measured | tolerance |
|---|---:|---:|---|
| lag-1 ACF | **0.5905** | 0.5817 | ±0.05 |
| correlation time | 0.2333 s | 0.2667 s | ±1 frame |
| PSD ratio | 0.994 | 1.0 | ×1.25 |

**White noise at the same variance gives lag-1 0.0.** That is the size of the
error this corrects.

> **The first run appeared to fail, and the comparison was what was broken.**
> Raw lag-1 came out 0.724 against 0.582. `shapeflow/calibrate.py:591
> residual_structure` detrends each 2.11 s segment *before* correlating, so the
> published +0.5817 is a **detrended** statistic; measuring a synthetic sample
> raw compares two different quantities. `noise.detrended_acf` applies the
> published recipe and the discrepancy vanishes.

## Registered prediction 2 — the published floor survives. `PASS`

The registration's one prediction that could damage a published number.

| noise colour | floor |
|---|---|
| **white** — what `NOISEFLOOR.md` used | **0.2599 [0.2553, 0.2645]** /s |
| **measured colour** | **0.2977 [0.2918, 0.3039]** /s |
| move | **14.6%**, against a registered 25% |

**`NOISEFLOOR.md`'s 0.2599/s stands**, and now for a measured reason rather than
an assumption — the threshold is scale-adaptive, so the rate is far more
sensitive to the rule than to the noise.

**One number does move and is restated here rather than left.** The corrected
floor is **62.7%** of the corpus's 0.4746/s, not the published 54.8%. The
separation still holds — corpus [0.4573, 0.4909] against [0.2918, 0.3039],
non-overlapping — so `NOISEFLOOR.md`'s verdict is unchanged and its margin is
tighter than published. Recorded in `DEVIATIONS.md` D14.

## The descriptor floor — what separates and what does not

Corpus windows against the same descriptors computed on measured-colour noise.
Under the null, 5% of windows should exceed the null's 95th percentile.

| descriptor | corpus median | null median | null p95 | **corpus > null p95** |
|---|---:|---:|---:|---:|
| **`pole_radius`** | **0.6051** | 0.4350 | 0.4899 | **0.887** |
| `band_slow` (0.5–2 Hz) | 0.5541 | 0.4669 | 0.4991 | **0.587** |
| `band_marginal` (4–6) | 0.0732 | 0.0840 | 0.0933 | 0.383 |
| `band_fast` (6–15) | 0.1554 | 0.2514 | 0.2721 | 0.273 |
| `band_mid` (2–4) | 0.1668 | 0.1983 | 0.2144 | 0.245 |
| **`pole_frequency_hz`** | 5.8748 | 8.4041 | 10.0851 | **0.021** |

**Persistence clears by 17.7×.** 88.7% against the 5% chance would give.

**Frequency does not clear at all** — 2.1%, *below* chance. And the corpus's
apparent pole frequency (5.87 Hz) is **lower** than the noise's (8.40 Hz), so
the two are not even ordered the way an "is there a rhythm" test assumes.

**Every band above 2 Hz has *less* relative power in the corpus than in the
noise.** The corpus's power is concentrated low; the tracking noise's is spread.
That is the fast-band null, demonstrated rather than asserted, on the corpus's
own descriptors.

### Registered prediction 3 — refuted, and usefully

I predicted the AR-pole null would concentrate **below 4 Hz**, on the grounds
that a 267 ms correlation time is ≈3.7 Hz. **It sits at 8.40 Hz.** The AR(4) fit
puts the correlated component into a *real* pole — drift — and leaves the
oscillatory pole to pick up the residual high-frequency content. So coloured
noise does not masquerade as a slow rhythm; it masquerades as a **fast, heavily
damped** one, which is easier to tell from behaviour than feared, and is exactly
why radius and not frequency is the discriminator.

## The dynamics probe — `f_c`-limited, and a localisation failure

240 instances per cell, planted into real `tune` ego slices, oscillation
spanning the whole slice so the parameter switch is the only change, scored at
the registered ±2 band against a per-recording chance level.

| plant | level | recall (±2) | chance | offset median | **within ±30** |
|---|---|---:|---:|---:|---:|
| Δfrequency | 0.25 Hz | 0.058 | 0.022 | +0 | 0.504 |
| Δfrequency | 1.0 | 0.050 | 0.018 | −4 | 0.609 |
| Δfrequency | 2.0 | 0.113 | 0.023 | −4 | 0.854 |
| **Δfrequency** | **4.0** | **0.325** | 0.042 | −2 | **0.991** |
| Δdamping → 0.50 | | 0.129 | 0.044 | +6 | 0.729 |
| Δdamping → 0.95 | (no change) | 0.037 | 0.024 | +2 | 0.513 |
| Δamplitude ×2 | | 0.104 | 0.024 | +2 | 0.571 |
| Δamplitude ×5 | | 0.200 | 0.033 | +2 | 0.781 |

**Registered prediction 4 is refuted.** Δfrequency recovery never reaches 50%
anywhere on the ladder; the best is **0.325 at a 4 Hz step**, which is a
*doubling* of the 2 Hz carrier and sits at the edge of the usable band
(`f_c = 4.833 Hz`).

**The instrument behaves correctly in every direction it was asked to.** Recall
is monotone in Δfrequency, monotone in |Δradius| (0.129 at the largest change
down to 0.037 at no change), and monotone in Δamplitude. Nothing is broken —
the task is hard.

> **And the failure is localisation, not detection.** At Δfrequency = 4 Hz,
> **99.1%** of detections land within ±30 frames and **32.5%** within ±2. The
> contrast rises tenfold at the boundary — 4.43 against a 0.46 baseline — but
> its peak is broad and biased 5–25 frames early, because a window straddling
> the change has an unstable AR fit.

**That is the third instrument in this programme to show the same shape.** Human
raters: 85.3% within 1 s, 8.8% within 2 frames. Cleaning arms: 0.89 at ±10,
0.76 at ±2. Now the dynamics channel: 0.99 at ±30, 0.33 at ±2. **Coarse
agreement with fine disagreement is a property of the question, not of any one
tool**, and the ±2 convention inherited from ExBias has now been missed by
everything that has ever been pointed at it.

## What this licenses, and the scope it fixes for Stage 1

**Licenses:** a dynamics channel built on **pole persistence in the slow band**.
`pole_radius` separates corpus from measured-colour noise on 88.7% of windows
and `band_slow` on 58.7%, both far above the 5% chance allows.

**Does not license** a frequency-based dynamics channel (2.1%, below chance), a
mid-band channel (24.5%), or anything in 4–15 Hz. **Grooming's 6–7 Hz stroke and
sniffing's 4–12 Hz are unrecoverable from this footage**, now demonstrated on
the corpus's own descriptors rather than argued from the literature.

**Does not license** boundary placement at ±2 frames from the dynamics channel.
Any Stage 1 detector must be scored at a band its own probe supports, and that
band must be registered before it is used.

## What is owed

1. **Registered prediction 5 was not tested.** The `white` arm was not run in
   the probe phase — the plant harness's negative control is still owed for this
   detector, and `DETECTOR_PREREGISTRATION.md:84-90` is not inherited.
2. **A tolerance band derived from measured instrument precision.** Three
   instruments have now missed ±2. Choosing a band now, after seeing these
   numbers, is what the annotation registration §7 forbids; it needs its own.
