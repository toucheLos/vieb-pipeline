# Phase D — suspect-keypoint disposition

89 report animals, 1,149 recordings, ε = 0.10. Configuration fixed in
`DISPOSITION_PREREGISTRATION.md`; the corpus ran twice and Amendment 1 of that
document says why. **Q1 was not re-scored.**

## The one-line result

The gate passes and the atom check fails.

Constraining the skull lowers violations on the **trunk — bones the corrector
never sees — from 2.4042% to 2.3219%**, a 3.42% relative fall on 89 animals. That
is the only evidence here that the correction moves keypoints towards the animal
rather than towards its own constraint surface, and it is the only check in this
phase that could have failed on its own terms.

A probe held out by three-second block still separates corrected frames from
clean ones at **0.680 balanced accuracy**, above the pre-registered 0.60. The
same probe on the array the corrector was *handed* — same rows, same labels, same
groups — scores **0.810**. So the correction moves those frames towards the clean
distribution without making them ordinary, and how much of the residue is a
signature rather than the selection is not resolved here.

## The six pre-registered predictions

| # | prediction | outcome | |
|---|---|---|---|
| 1 | domain 0.3–0.6% of keypoint-frames | **0.5043%** [0.4425, 0.5748] | **held** |
| 2 | trunk violations fall, by under 1 pp | 2.4042% → 2.3219%, **0.082 pp**, 3.42% relative | **held** |
| 3 | identity null: bit-identical on clean frames | asserted by test, all 241 green | **held** |
| 4 | injection direction cosine > 0.8 | > 0.8 at all four injected vectors | **held** |
| 5 | separability ≤ 0.60 balanced accuracy | **0.735** first run, **0.680** second | **failed, twice** |
| 6 | effect size small | 0.50% corrected, 1.32% abstained; median move **0.194 body lengths** | **held on mass, not on magnitude** |

Predictions 3 and 4 are asserted on synthetic data by the blocking tests, not
measured on the corpus. They are nulls, and a null that only ever ran on the
corpus could not have been checked against a known truth at all — but they are
weaker evidence than the corpus numbers beside them and should not be read as
equal to them.

Prediction 6 is the one worth arguing with. The **mass** is small exactly as
predicted. The **magnitude** is not: the median correction moves the suspect
**22.1 px, 0.194 body lengths**, and the 90th percentile moves it **0.807 body
lengths**. Within the ≤3-frame envelope the predictor is not nudging a slightly
noisy keypoint, it is saying the keypoint was most of a body away from where it
belonged. That is consistent with the teleports the Step 1 contact sheets showed,
and it is why "small effect" is only half true.

## Why the first run failed prediction 5, and what changed

Prediction 5 existed to catch a corrector depositing mass on a codimension-1
surface. It caught one. Reading `project` after the probe fired found three
departures from the configuration table, measured on 8 report animals:

| | first run | second run |
|---|---|---|
| corrections landing **exactly** on a constraint surface | **100.0%** | **13.2%** |
| constrained bones | all 21 pairs incident on the suspect | the **2 SKULL** pairs, and nothing else |
| trunk bone (2,3) in the constraint set | on **61.6%** of corrections (binding on 1.5%) | never |
| exits with a constraint still violated, reported as corrected | 0.5% | none — those frames abstain |
| median displacement | 11.81 px | **21.40 px** |

The table had fixed the target as "interior, toward the neighbour-predicted
position", giving the codimension-1 surface as the reason. The loop's only move
was `x = other + (x - other) * (limit / d)`, which sets the bone to *exactly*
`ℓ̂(1+ε)`. Every corrected frame landed on the surface. The second run replaces
the suspect with its position in the **nearest non-violating frame of the same
recording**, carried through the similarity transform fitted on the six
non-suspect keypoints — a map that multiplies every bone length by one scale, so
a feasible donor maps to a feasible position, continuously, on no surface.

**The gate's claim was also false in the first run, and the verdict survived it.**
A nose correction constrained trunk bone (2,3) directly. That path is bounded: a
binding trunk constraint covered 0.0059% of frames against a 2.40% trunk
violation rate, so at most ~0.25 of the 3.10 percentage points of relative fall,
under a tenth. The second run removes the path entirely and the fall gets
*larger*, 3.10% → **3.42%**. The gate was not being carried by the leak.

## Amendment 1's own predictions

Made with the first run's numbers visible and fenced off as such. Two did not hold.

| # | prediction | outcome | |
|---|---|---|---|
| 1 | boundary landings fall below 5% | **13.2%** | **failed** |
| 2 | separability falls below 0.735, not to 0.60 | 0.735 → **0.680**; before-probe 0.810 | held |
| 3 | displacement rises | 11.81 → **21.40 px**, 1.81× | held |
| 4 | the gate still passes, trunk genuinely unconstrained | **PASS**, fall rises to 3.42% | held |
| 5 | the domain shrinks slightly | it **rose**, 0.3906% → 0.5043% | **failed** |

Amendment prediction 1 is the substantive miss. 13.2% of corrections still land
on the surface, because the target is infeasible that often: the similarity is
fitted on the six non-suspect keypoints, but the constraint is evaluated against
the *observed* partner positions in the violating frame, and where those are
themselves off the mapped position does not satisfy it. A boundary landing on one
correction in eight is far better than eight in eight and is not nothing — this is
the most likely remaining source of the 0.680.

Amendment prediction 5 failed for a mechanism that never fired: **no frame was
abstained for want of a donor**, on any of 1,149 recordings. Violating runs
against a recording boundary are rarer than the prediction assumed. The domain
rose instead, because replacing the suspect rather than pulling it to the nearest
feasible point registers a move on frames the old rule left untouched. 0.5043% is
still inside the **original** pre-registered band of 0.3–0.6%.

## What the disposition emits

| | value |
|---|---|
| corrected | 0.5043% of frames [0.4425, 0.5748] |
| abstained | 1.3209% of frames [1.2024, 1.4425] |
| correctable before disposition | 0.5385% |
| abstain runs per recording | median 4, mean 7.4, p90 20.2 |
| abstained for non-convergence | 2,181 frames |
| abstained for want of a donor | 0 |
| abstained for want of a scale estimate | 0 |
| mean reliability | 0.6984 [0.6885, 0.7079] |

`reliability` enters Step 4 as a **covariate, never a likelihood weight**. A
weighted likelihood is not a code length — a model lowers its cost by
down-weighting what it predicts badly — and that destroys the cross-arm
comparability that charging duration identically exists to create.

## What may not be cited

On the **constrained** group, skull violations fall 1.7877% → 1.3235%. That
number is in `disposition.json` under `constrained_group_do_not_cite` and is
recorded so the circularity is visible. A corrector that enforces skull
constraints scoring well on skull violations is measuring its own premise.

## Where this leaves the phase

The gate is the check that could have failed and it passed twice, the second time
with the held-out group genuinely held out. The atom check failed twice and is
**not** resolved: 0.680 is above the limit, and the limit stands. Three readings
remain open and this phase does not choose between them.

1. The residue is the **selection**. These frames are picked for violating and
   read 0.810 before anything touches them; no corrector confined to one keypoint
   inside an ε-ball can make them ordinary. On this reading 0.60 was the wrong
   threshold rather than the corrector being wrong.
2. The residue is the **remaining atom** — the 13.2% still landing on the surface.
   Testable: re-run the probe on corrections that used the target only.
3. Correction is the wrong disposition at this ε and **down-weighting is the
   answer**, because it moves nothing and so can write no signature at all. The
   reliability channel is already emitted and already consumed as a covariate, so
   this costs nothing to adopt.

The decision belongs at Step 3, on the held-out MDL axis, where the disposition
is one cleaning arm among several and the question is which one predicts best.
Deciding it here, on a probe, would be choosing the vocabulary before measuring
what it buys — which is the thing this programme was set up to stop doing.
