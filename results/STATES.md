# STATES — behaviour as predictive states and a transfer operator

Registration: `STATES_PREREGISTRATION.md` with Amendment 1 (D28). Deviation:
D29. Result: `results/states.json`. Input: REPAIR's adopted `repaired` pose
(`REPAIR.md`) through `ego.transform` (17 channels), plus the 4-channel SAM mask
stream. Report split: 89 animals.

## Verdict: CONTINUUM as registered, and it is uninformative

**No state is named.** No gap ratio tᵢ/tᵢ₊₁ beats the surrogates, so M = 1 and
the registered verdict is CONTINUUM. **The planted-recovery check failed.**
A 2 s switch, planted at +1 SD in two channels of 20% of recordings, was
recovered at a balanced accuracy of **0.500** (bar 0.8). **This pipeline could
not have found metastable states even where they existed.** The CONTINUUM
verdict is therefore **not evidence that mouse behaviour here is a continuum.**
It is a null from an instrument that failed its own sensitivity check.

## What ran, as registered

| step | result |
|---|---|
| leak: context during immobility, pose stream | 0.510 [0.496, 0.526]: kept |
| leak: context during immobility, mask stream | 0.502 [0.500, 0.504]: kept |
| K\* (held-out R² of the features 0.5 s ahead) | **1**: R² 0.365, 0.366, 0.363, 0.359, 0.352, 0.339 for K = 1…32. Past frames add nothing to the prediction |
| τ\* | **64 (grid edge)**: t₂ never plateaued (137 → 1,505 frames), D29 |
| corpus t₂…t₄ at τ\* | 217, 199, 196 frames |
| gap ratios t₂/t₃ … t₁₀/t₁₁ | 1.09, 1.02, 1.00, 1.10, 1.07, 1.02, 1.05, 1.12, 1.01; lower bounds 1.00–1.01 |
| surrogates' upper bounds, t₂/t₃ | phase 2.53, var5 2.44, microstate 2.01, microstate0 2.49, white 3.07 |
| gate 1 | no ratio counts: **M = 1** |
| gate 2 (AMI) | not reached |
| gate 3 (planted switch) | **0.500: FAIL** |

The VAR(5) surrogate first crashed on non-stationary fits. It now falls back
to phase randomisation on 1–2% of recordings (D29).

## Why the instrument failed (post-hoc, not registered)

The two diagnostics are `scripts/diag/states_tau_floor.py` (output `results/states_diag.json`) and `scripts/diag/states_eigvec_locality.py`. **They explain the null; they do not
change the verdict.**

**1. τ\* landed on the estimation floor.** On white noise, t₂ = 3.8 τ at every
lag: a spurious constant |λ₂| ≈ 0.77. Against it:

| τ (frames) | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| corpus t₂ | 37.6 | 57.5 | 90.5 | 278.8 |
| white-noise t₂ | 3.8 | 15.3 | 61.3 | 244.7 |

The corpus has dynamics well above the floor at short lags, and none that
survive to 2 s. The registered "plateau" rule cannot plateau on a floor that
grows linearly with τ, so it fell to the grid edge, exactly where the floor
hides everything. The corpus gap ratios therefore sit where white noise's do.

**2. At short lags the spectrum is dense, with no gap.** At τ = 1 the leading
eigenvalues are 0.973, 0.964 and 0.962. The slow eigenvectors are spread over
well-visited microstates: the median microstate has 4,750 report visits, and
the five heaviest states carry ≤ 7% of each eigenvector. A 2-state PCCA+
confidently assigns only 14% of frames. There are many comparably slow
directions and no few dominant ones.

**3. The plant fails at every lag.** At τ = 1, 4, 16 and 64 the planted switch
is recovered at 0.500. A +1 SD offset in 2 of 21 channels, among many natural
slow modes of similar speed, is not what the top eigenvectors pick out.

## What this licenses

- **Nothing about states.** There are no clips, occupancy or dwell results,
  since none are licensed.
- **The corpus is not white noise at sub-second lags** (the table above),
  which the earlier tokenisers' 1-frame runs had left open. This is
  descriptive and post-hoc.
- The repaired pose and the mask stream pass both leak gates. They are the
  inputs for whatever comes next.
