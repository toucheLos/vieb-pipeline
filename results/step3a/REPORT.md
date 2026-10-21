# STEP 3A for grooming: the registered reading is "the representation is the problem", but the information is at frame level

**Registration:** `REGISTRATION.md` (629d3ba). This branch is a **post-result** switch
from 3B. **Data:** `probe.json` (10 cells, 200 refitted shifts per cell).

**Negative control.** White-noise features (R and L) give a normalised score of 0
and AUROC 0.48–0.58 everywhere. The control **does not fire**, so the readings
stand.

## 3A.1: the linear probe at native noise (m0)

Event F1 at IoU 0.5, probability > 0.5, normalised with a 95% CI. **Recoverable**
needs a lower bound > 0.25.

| label | R: ego representation | L: MoSeq latent | AUROC (R / L) | Step 1 instruments | **registered reading** |
|---|---|---|---|---|---|
| **groom** | +0.064 [+0.040, +0.096] | +0.091 [+0.055, +0.131] | **0.862 / 0.870** | none recoverable | **the representation is the problem** |
| investigation | +0.134 [+0.069, +0.204] | +0.033 [+0.012, +0.062] | 0.782 / 0.756 | D recovers it | instruments recover it |
| mount | 0.000 | 0.000 | **0.868 / 0.794** | none | the representation is the problem |
| attack | GRID_LIMITED | | | | |

**At Luna 1.0×**, groom falls to +0.028 (R) and +0.019 (L), with AUROC 0.84 and 0.80.

## 3A.2: localisation stage by stage (normalised event score)

| stage | groom m0 | groom 1.0× | investigation m0 |
|---|---|---|---|
| representation (R) | +0.064 | +0.028 | +0.134 |
| segmentation (D-segment means of R) | +0.064 [−0.002, +0.133] | +0.081 | +0.150 |
| embedding (C segments, PCA 20) | +0.055 [−0.003, +0.117] | +0.047 | +0.133 |
| clustering (Step 1, C / M) | 0 / +0.013 | 0 / 0 | 0 / +0.044 |

**No stage loses grooming relative to the representation.** It is below the bar at
the first stage, and stays below it.

## What the registered reading does not capture (stated, not used to rescore)

**Frame AUROC is 0.86 for groom and 0.87 for mount**, yet the event score is near
zero. A linear probe ranks groom and mount frames well, but at the registered
threshold (p > 0.5) and the registered event match (IoU ≥ 0.5) it produces few
matched bouts. Mount produces **none**: its probability rarely exceeds 0.5, because
the class is rare.

So the information is present **at frame level** in the representation. What fails
is turning it into bouts at this threshold and match. "The representation is the
problem" is the registered reading **under the registered metric**. A
threshold-free or bout-construction-aware probe could read differently, but that
would need its own registration.

**The branch stops here, as registered.** The fix and the full Step 1 re-run (3A.3)
come only after the human has seen this.

---

# Amendment 1 (post-result): bout-aware probe, scorer check, speed baseline

**Registration:** `REGISTRATION.md` Amendment 1 (9a77cc5). It was committed alone,
marked post-result, and directed by the user. **Data:** `bout_probe.json` (18 cells;
100 shifts per cell, with the probe and the readout refitted in each).

**Scope.** This probe is supervised and diagnostic. **It does not change the Step 1
verdict.**

## A1.1: the scorer passes; event F1 is fragile to flicker

- **Identity check: PASS.** Feeding the true labels in as predictions gives event F1
  = 1.0 and boundary F1 = 1.0 for every label on both the map and eval partitions.
  **The earlier 3A event verdicts stand.**
- **Planted flicker.** Each cell gives event F1 at IoU ≥ 0.5 on eval, raw → after
  each label's fitted R/native readout.

| frames flipped | groom | mount | investigation | attack |
|---|---|---|---|---|
| 1% | 0.148 → 0.832 | 0.108 → 0.877 | 0.425 → 0.589 | 0.163 → 0.854 |
| 5% | 0.021 → 0.842 | 0.014 → 0.877 | 0.105 → 0.588 | 0.027 → 0.854 |
| 10% | 0.006 → 0.776 | 0.004 → 0.877 | 0.042 → 0.590 | 0.010 → 0.860 |
| 20% | 0.002 → 0.560 | 0.001 → 0.874 | 0.014 → 0.569 | 0.002 → 0.857 |

- **Raw event F1 at IoU ≥ 0.5 collapses even when the frames are 99% correct.**
  An unsmoothed per-frame readout cannot score on this metric.
- Investigation's ceiling after the readout (about 0.59) comes from its fitted 1 s
  minimum bout, which removes short true bouts.
- **That accounts for mount's zero in 3A.1.** It was a readout artefact, not
  missing information.

## A1.2: frame information against a speed/stillness baseline (native, m0)

PR-AUC is given with a sequence-bootstrap 95% interval. Δ is paired against B, the
speed/stillness probe.

| label | prevalence | B: speed only | R: ego | L: MoSeq latent | **Δ R − B** | **Δ L − B** |
|---|---|---|---|---|---|---|
| **groom** | 0.086 | **0.445** [0.245, 0.552] | 0.293 [0.135, 0.535] | 0.342 [0.153, 0.581] | −0.151 [−0.198, +0.017] | −0.102 [−0.164, +0.060] |
| mount | 0.122 | 0.169 | 0.474 | 0.315 | **+0.305 [+0.135, +0.424]** | **+0.145 [+0.055, +0.233]** |
| investigation | 0.234 | 0.303 | 0.581 | 0.512 | **+0.278 [+0.209, +0.326]** | **+0.208 [+0.147, +0.260]** |
| attack | 0.048 | 0.529 | 0.464 | 0.454 | −0.065 [−0.107, −0.007] | −0.076 [−0.128, +0.008] |

AUROC for groom is B 0.886, R 0.862 and L 0.870. **Speed and stillness alone rank
grooming at least as well as the full pose representation.** The 0.86 AUROC that
looked like "the information is there" (3A.1) is what speed alone already gives.

**At Luna 1.0×:**
- groom B 0.317, R 0.248, L 0.219 (Δ L − B −0.098 [−0.143, −0.009]);
- investigation and mount stay above B. For mount L, Δ +0.048 [−0.029, +0.135].

## A1.3: event score with the readout fitted on map (normalised; recoverable needs lower bound > 0.25)

| label | arm | B | R | L |
|---|---|---|---|---|
| **groom** | native | +0.289 [+0.218, +0.347] | +0.191 [+0.108, +0.261] | +0.211 [+0.140, +0.278] |
| groom | 1.0× | +0.219 [+0.159, +0.267] | +0.157 [+0.084, +0.235] | +0.075 [+0.044, +0.105] |
| mount | native | +0.059 | **+0.276 [+0.145, +0.405]** | +0.207 [+0.093, +0.321] |
| investigation | native | +0.140 | **+0.291 [+0.150, +0.433]** | +0.185 [+0.054, +0.338] |

- **No cell is recoverable:** every lower bound is ≤ 0.25. Attack is too sparse to
  score (GRID_LIMITED), so its reading rests on frame metrics only.
- **The white-noise control is silent:** normalised score ≤ +0.012 and AUROC 0.48–0.65.
- **The fitted readout** lifts groom R from +0.064 (fixed 0.5) to +0.191, and mount R
  from 0 to +0.276. Most chosen readouts use the 1 s minimum bout and a low threshold.

## Registered readings

| label | native | 1.0× |
|---|---|---|
| **groom** | **the representation, under a linear probe, lacks grooming beyond speed** | same |
| mount | information at frame level beyond speed, but a linear probe cannot assemble it into bouts | same |
| investigation | information at frame level beyond speed, but a linear probe cannot assemble it into bouts | same |
| attack | lacks it beyond speed (frame metrics only; events GRID_LIMITED) | same |

**For grooming this sharpens 3A.1.**
- The pose representations carry no grooming information that speed and stillness
  do not already carry.
- Even a supervised, readout-fitted probe stays below the bar.
- **So 3A does not proceed to localise a loss for grooming**: no stage-wise loss
  exists to find.
- Whether a richer (non-linear, or keypoint-local) representation would carry it is
  outside this registration.

The Step 1 verdict (O2 → 3B) is unchanged.
