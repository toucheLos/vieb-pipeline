# Pre-registration — suspect-keypoint disposition

Written and committed **before the disposition code was run on any data**. The
configuration below is fixed; the predictions are falsifiable; and the reason this
document exists is stated first.

## Why this is pre-registered

Q1's headline numbers are **+1.639% [+1.205, +2.080]** at w = 0.4 s and
**+5.854% [+4.833, +6.815]** at w = 1.5 s, on 89 report animals.

A disposition that touches 0.4–1.8% of keypoint-frames can plausibly move a number
of that size. If the corrector is built and Q1 is then re-scored, **no analysis can
separate "the corrector improved the estimate" from "the corrector moved the
estimate"** — the choice of configuration would have been made with the answer
visible. That is not a hypothetical failure mode for this project; it is the
shape of error its ledger is mostly made of.

So: **Q1 is not re-scored in this phase, at all.** The disposition enters the
pipeline at Step 3 as one cleaning arm among several, and whenever Q1 is next
touched, both filtered and unfiltered headline numbers are reported regardless of
which is more favourable.

## The configuration, fixed in advance

| | value | why this and not something tuned |
|---|---|---|
| ε | 0.10 | Step 1's primary cell. The ε curve decays smoothly with no elbow, so no ε is a principled cut; 0.10 is inherited, not chosen here |
| group constrained | `SKULL` = ear–ear, ear–nose ×2 | near-rigid; trunk bones flex and their violations are real posture as often as error |
| group evaluated | `TRUNK` | never seen by the corrector — this is the gate |
| envelope | runs ≤ **3 frames** (100 ms) eligible for projection | empirical: median violating run is 1–2 frames, p75 = 3. Precedent `swap.MAX_RUN_S = 0.25` |
| longer runs | never corrected | a 300 ms run is the tracker on a wrong mode, not a perturbation |
| suspect | keypoint common to every violating bone in the frame | well-defined on 95.8% (worst) / 99.2% (ordinary) of violating frames |
| tie-break | lowest DLC confidence | needs monotonicity only; AUC 0.60–0.67 for predicting a violation |
| scale | per-frame median log length over rigid pairs **excluding those touching the suspect** | per-frame scale SD is 0.155 log units against 0.178 for the geometry itself |
| constraint | ℓ ≤ ℓ̂·(1+ε), **inequality** | projection can shorten a bone but never lengthen it |
| target | interior, toward the neighbour-predicted position | a boundary projection deposits frames on a codimension-1 surface |
| moved | the suspect keypoint only | everything else bit-identical |

## Predictions

Recorded now so that "it worked" cannot be decided afterwards.

1. **Domain.** Projection touches **0.3–0.6%** of keypoint-frames. Anything above
   1% means the envelope is not doing its job.
2. **The gate.** Trunk violations fall. The prediction is a **small** fall — the
   only trunk bone touching a skull keypoint is nose–center — and the effect is
   expected to be under 1 percentage point. **Falsifier: if trunk violations do
   not fall at all, or rise, the corrector is satisfying its own constraint and
   the phase stops.**
3. **Identity null.** On frames with no violation the corrector changes **exactly
   nothing** — bit-identical, not approximately. Any firing means suspect
   identification is keying on geometry rather than on error.
4. **Injection recovery.** For displacements injected within the envelope, the
   recovered direction correlates with the true direction at **> 0.8**. Magnitude
   recovery alone is not sufficient: a corrector that merely lands on the
   constraint surface recovers magnitude and not direction.
5. **The atom.** A grouped, class-balanced probe distinguishing corrected frames
   from clean ones in egocentric feature space scores **balanced accuracy ≤ 0.60**.
   Above that, the correction has written a detectable signature into the feature
   distribution, k-means will find it downstream, and it will look like a state —
   which is the failure this project has already produced once, through a
   persistence prior rather than a corrector.
6. **Effect size overall.** Small. The honest prior, from the measured run-length
   distribution, is that this changes little and that the sustained residual —
   which nothing temporal or geometric here can reach — is the larger share.

## What this phase may not do

* It may not re-score Q1.
* It may not tune ε, the envelope threshold, or the target rule against any
  outcome measured on `report`. Selection happens on `tune` if it happens at all.
* It may not report the bone-violation rate on the constrained group as evidence
  that the corrector works. A corrector that enforces skull constraints scoring
  well on skull violations is measuring its own premise.

## Reliability

Emitted per keypoint-frame in [0, 1], from confidence × geometric consistency, and
consumed at Step 4 as a **covariate**, never a likelihood weight — a weighted
likelihood is not a code length, and a model that can down-weight what it predicts
badly is not comparable across arms.
