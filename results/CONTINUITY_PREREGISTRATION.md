# Pre-registration — the continuity residual

Written and committed **before the instrument was run on more than one
recording**. Configuration fixed below; predictions falsifiable; the reason this
document exists stated first.

## Why this is pre-registered

Phase D pre-registered a threshold, the code did not implement the configuration
that was registered, and **the registered prediction is the only thing that
caught it**. That is the argument for doing it again here, and it is worth more
than the cost.

This instrument is also more dangerous than Phase D's, in one specific way: it
produces a number for every arm, and "lower is better" is intuitive and **wrong**.
A filter can make a recording perfectly continuous by deleting all the movement.
Two of the five arms scored here can do exactly that. Registering the
over-smoothing signature in advance is what stops a flattened recording being
reported as a clean one.

## What prompted it

The comparison clips on the Atlas still look inconsistent, and the bone check
cannot explain why. A length test is blind to a keypoint that slides **along** a
bone and to a skull triangle that drifts coherently — and there is no measure
anywhere in this repo of whether a frame agrees with the frames either side of it.

Prototyped on one recording, `FC_Day_0_(Context_A)_308`, T = 6,303, ℓ = 107.7 px:

| | value |
|---|---|
| median spike residual | 0.0101 body lengths |
| p99 spike residual | 0.119 body lengths |
| **AUC(spike → bone flag)** | **0.597** |
| frames with a large spike and no bone flag | **3.776%** (against 1.666% bone-flagged) |

Those numbers are the reason for running it and are **not** a result. One
recording, one animal, no bootstrap.

## The configuration, fixed in advance

| | value | why this and not something tuned |
|---|---|---|
| predictor | similarity `x ↦ sRx + c` fitted on the **six non-suspect keypoints** | absorbs translation, rotation and apparent scale, so what is left is the keypoint moving relative to the body rather than the animal moving |
| fit | closed-form 2D Procrustes (Horn), batched | the same fit `disposition.predict` does by SVD, pinned to it by test at 1e-9. An SVD per (frame, keypoint) is minutes per recording |
| directions | both, `r⁻` from `t−1` and `r⁺` from `t+1` | one side alone cannot separate an excursion from a shift |
| `spike` | `min(r⁻, r⁺)` | a teleport leaves and returns, so both neighbours disagree |
| `step` | `\|r⁻ − r⁺\|` | a persistent shift disagrees with one side only |
| threshold | **0.10 body lengths** | just under the prototype's p99 of 0.119. Round, inherited, not selected against any outcome |
| units | body lengths, `ego.ell_a`, per animal | never per frame — that deletes rearing |
| seams | `r⁻[0]` and `r⁺[−1]` are NaN | takes one recording and indexes only inside it |
| arms | `raw`, `wiener`, `median_0.50`, `viterbi`, `disposition` | the cheap branch that carries through to MDL, plus the corrector |
| disposition scored on | the **held-out** neighbour, the one `donor_frame` did not pick | the corrector moves the suspect *to* the prediction from the donor side; scoring it there measures its own premise |
| bootstrap | animals, 2000 replicates, `boot.animal_interval` | frame-level resampling gave intervals ~19× too narrow |
| dominance | **non-overlapping** animal-bootstrap intervals | a point-estimate comparison crowned an arm on 0.029 px once already |

## Predictions

1. **Viterbi lowers `spike` most per pixel of displacement.** It is a
   path-selection de-glitcher and a one-frame excursion is precisely what it
   selects against. It already leads the bakeoff's efficiency axis at 16.9%
   violation reduction per pixel against the incumbent's 4.1%.
2. **No arm meaningfully lowers `step`.** This is the falsifier for a claim
   already standing in `CLEANING.md` — that what a temporal filter leaves behind
   is temporally smooth and needs an anatomical prior rather than another filter.
   **If an arm does lower `step`, that claim is wrong and must be withdrawn.**
3. **The disposition lowers `spike` on the held-out side.** If it improves only
   on the donor side it is fitting its own predictor rather than finding the
   animal — which is a finding about Phase D that nothing inside Phase D could
   have produced.
4. **The 2×2 against `bone_flagged` shows weak agreement.** Jaccard **under
   0.20**, and `spike_only` exceeds `spike_and_bone`. The prototype's
   3.776%-vs-1.666% reproduces corpus-wide to within a factor of two.
5. **`median_0.50` lowers `spike` and `hf_retained` together** — the
   over-smoothing signature. It is the arm most likely to look best on this axis
   for the worst reason, and saying so now is the point of saying it now.
6. **Effect size.** `spike` at or below **0.02 body lengths** at the median on
   every arm. This measure is dominated by ordinary small motion; the interesting
   mass is in the tail, not the centre.

## What this phase may not do

* It may not restate the bakeoff verdict. `results/cleaning.json` chose on three
  axes, it is committed, and continuity is reported **beside** it in its own file
  rather than folded into it.
* It may not change the Phase D corrector or widen its input. The ~3.8% of
  spike-flagged, bone-clean frames get **no disposition in this phase**. Phase D
  is pre-registered and committed; widening its suspect channel on an instrument
  built afterwards is exactly the contamination that pre-registration exists to
  prevent. The overlap is measured and reported, and that is all.
* It may not re-score Q1.
* It may not report a lower residual as an improvement without `hf_retained`
  beside it.

## One known property, recorded before it is discovered in a plot

The fit for keypoint `k` uses the other six, so a badly wrong keypoint
contaminates every other keypoint's prediction in that frame. Measured on a
synthetic 45 px nose displacement: the guilty keypoint reads 1.41 body lengths
and the innocent ones up to 0.57 — a margin of about **2.5×**, not a clean zero.

Attribution by `argmax` over keypoints is therefore sound, and the **absolute
level of the pooled residual is inflated**. The inflation is identical across
arms, so comparisons hold; the level is not the size of the error.
