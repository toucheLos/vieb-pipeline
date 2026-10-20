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
