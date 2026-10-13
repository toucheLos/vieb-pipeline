# Registration — STEP 1 secondary arms (§9) and the decision-importance ablation (§1.8)

Committed **alone**, after the human has seen the Step 1 verdict (O2, 37df6d6)
and before any secondary or ablation arm exists. Everything here is
**descriptive**: none of it changes the Step 1 outcome.

Inherited unchanged from `REGISTRATION.md` + Amendment 1:
- partitions, noise operators, g, r and the instruments' hyperparameters;
- scoring: map, event / boundary F1, 1,000 circular shifts with the map
  refitted, ceilings, sequence bootstrap (2,000, seed 0), the support rule,
  and the recoverable rule.

## A. Secondary arms (REGISTRATION §9)

| arm | what changes from the primary | levels | instruments, seeds |
|---|---|---|---|
| **j: jitter only** | no dropout | 1.0, 1.5, 2.0, 3.0. At 0.25 and 0.5 the jitter is below native, so those arms are identical to native, which the primary m0 already scored | D, C: 0–4; M: 0–2 |
| **d: dropout only** | native jitter, nothing added | 0.25, 0.5, 1.0, 1.5, 2.0, 3.0. At 0 the arm is native | D, C: 0–4; M: 0–2 |
| **k: 6 keypoints** | the neck is discarded; **`center`'s slot is the midpoint between the ear midpoint and the hip midpoint**; then the primary joint arm | all 7 | D, C: 0–4; M: 0–2 |

- **MoSeq runs 3 seeds here, not 5.** The decision is made here, before any
  of these fits exists, to bound GPU use at about 50 fits.
- Each arm gets **curves and a breakdown per behaviour × instrument**, one
  Read per cell, and `n_effective` = evaluation sequences.

## B. Ablation (§1.8)

Run on the **primary joint arm at m ∈ {0, 1.0}**, changing **one decision at a
time** from the primary configuration:

| decision | primary | variants | instruments |
|---|---|---|---|
| **smoothing** | none (D, C: the `raw` arm; M: `inputs.real`, unsmoothed) | **median 0.25 s**: `arms.rolling_median` on the held series, then long gaps restored to missing, exactly as `repair.repair` does. **Wiener: `NOT_A_RESULT`**: no committed implementation exists, only shapeflow's stored Luna arrays, and `vieb/clean/arms.py` refuses a second implementation (GAPS G11) | D, C, M |
| **window length** | `DERIV_SEC` 0.133 s | 0.067 s, 0.267 s (`scripts/breaks.py` `DERIV_SWEEP`, seconds) | D, and C through its segments. M: no window parameter, n/a |
| **keypoint set** | 7, neck as `center` | 6 (arm k above); **without nose**: the nose slot is filled by the ear midpoint, so the heading axis becomes ear-midpoint → tail_base | D, C, M |
| **frame rate** | 30 fps | 15 and 10 fps, by **decimation** (every 2nd or 3rd frame) of the degraded pose. Labels are decimated identically. Instruments receive the new fps, and every seconds → frames conversion goes through `recur.util.frames`. The boundary tolerance stays ±0.5 s | D, C, M |
| **MoSeq κ** | 1e6 | 1e5, one decade lower (the prompt's "fitted" κ does not exist, G4) | M |
| **representation** | D: channel group `both` (pose + velocity) | `shape` (pose only) | D. M models pose only, so this is n/a for M. C: already `shape` |

- **Smoothing is applied inside the nulls too.** The median variant is also
  run on the white and phase controls at m = 0, for every instrument. **If a
  control fires the recoverable rule under the variant, that variant's Δ is
  `NOT_A_RESULT`.**
- **Seeds:** D and C 0–4; M 0–2, so M's Δ is against the primary's seeds 0–2.
- **Effect:** Δ = normalised(variant) − normalised(primary) at the same level
  and on the same seeds. Each side keeps its own chance (1,000 shifts) and
  ceiling. The interval comes from a **paired** bootstrap over evaluation
  sequences, with the same index matrix on both sides.
- **Ranking:** by the **lower CI bound of |Δ|**, which is 0 if the CI contains 0,
  and min(|lo|, |hi|) otherwise.
  - The ranking is reported **per behaviour** (investigation, mount, groom)
    **and pooled**. The pooled value is the mean Δ over the three, with the
    T1 and G sequences resampled independently in each replicate.
- **Output:** a table of which decision matters most, for which behaviour, at
  which noise level.

## Not changed

The Step 1 outcome (O2), every primary number, and every bar.
