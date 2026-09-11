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

---

# Amendment 1 — the implementation did not match the configuration above

Written **after** the first corpus run and **before** the second. Everything in
this section was decided with the first run's numbers visible, and it is fenced
off here for that reason: nothing below is a fresh pre-registered test, and the
second run's separability number must never be quoted as one.

## What the first run measured

| | result |
|---|---|
| leave-one-bone-out gate | **PASS** — trunk 2.4042% → 2.3298%, a 3.10% relative fall |
| domain | corrected **0.3906%**, abstained **1.2885%** |
| prediction 5, the atom | **FAILED** — balanced accuracy **0.735**, AUC 0.791, against a limit of 0.60 |

The same probe on the array the corrector was *handed*, with the same rows,
labels and groups, scores **0.839** (AUC 0.906). So the correction moved those
frames *towards* the clean distribution rather than away from it. That is not an
acquittal: the pre-registered threshold is 0.60 and the result is 0.735.

## Three departures from the configuration table, all in `project`

Found by reading the code after the probe fired, and measured on 8 report
animals (1,805 corrected frames):

1. **It lands on the boundary, not in the interior.** The table fixed the target
   as "interior, toward the neighbour-predicted position", with the reason
   "a boundary projection deposits frames on a codimension-1 surface". The loop's
   only move is `x = other + (x - other) * (limit / d)`, which places the bone at
   length *exactly* `ℓ̂(1+ε)`. **100.0%** of corrected frames land exactly on a
   constraint surface. This is structural, not a tail: it is the failure
   prediction 5 was written to catch, and prediction 5 caught it.
2. **It constrains all 21 pairs, not the SKULL.** The table fixed the constrained
   group as `SKULL`. `pairs_touching` returns every pair incident on the suspect,
   so a correction on the nose constrains the trunk bone (2,3) directly. The
   suspect is the nose on **61.6%** of corrections, and (2,3) is the *binding*
   surface on **1.5%** of them. The gate's own reason string says "bones the
   corrector never saw", and for that bone it was not true.
3. **It can exit with a constraint still violated** and still report the frame as
   corrected — **0.5%** of corrections. Non-convergence was neither recorded nor
   refused.

**What this does to the gate.** A binding trunk constraint on 1.5% of corrections
covers 0.0059% of frames, against a trunk violation rate of 2.40% — so the direct
path can account for at most ~0.25 percentage points of relative fall out of the
3.10% observed, under a tenth of it. The gate's verdict survives; its reason
string does not, and both the leak and this bound are now recorded in the result.

## The second run

The three departures are fixed so the code does what the table already said. That
is a correction towards the pre-registration, not away from it — but the
*mechanism* of the interior target was chosen after seeing the first probe, and
that is the contamination this amendment exists to record.

| | second run |
|---|---|
| constrained bones | the `SKULL` pairs touching the suspect, and nothing else. The trunk is then genuinely unseen |
| target | the suspect's position in the **nearest non-violating frame of the same recording**, carried through the similarity transform fitted on the six non-suspect keypoints |
| why that target is interior | a similarity map preserves every bone-length ratio, so a feasible donor frame maps to a feasible position. The landing is a continuous function of the donor's pose and of the fit, so it lies on no surface |
| donor | frame `a-1` or `b` of the violating run, whichever is nearer; **abstain** if the run touches a recording boundary. Never across a seam |
| fallback | if the mapped position is still infeasible it is projected, the frame is recorded as `landed_on_boundary`, and that count is reported |
| non-convergence | recorded per frame; such frames are **abstained**, not corrected |

### Predictions for the second run

1. **The atom.** Boundary landings fall from 100% to **under 5%** of corrections.
   This one is mechanical and close to a tautology; it is listed so the claim is
   checkable rather than asserted.
2. **Separability.** Falls below the 0.735 of the first run. It is **not**
   predicted to reach 0.60 — corrected frames are selected for violating and read
   0.839 before anything touches them, so a probe reading the selection has
   signal no corrector can remove. **The 0.60 limit is retained unchanged and the
   read will still be FAIL if it is not met**; what is added is the before-probe
   beside it, so a reader can see which part is selection and which is signature.
3. **Displacement rises.** Replacing a keypoint with its predicted position is a
   larger move than pulling it to the nearest feasible point. If median
   displacement does *not* rise, the predictor is returning something close to
   the observed position and is not doing the work claimed for it.
4. **The gate still passes**, and now means what it says. **Falsifier: if trunk
   violations stop falling once the trunk is genuinely unconstrained, the first
   run's PASS was carried by the leak and the phase stops.**
5. **The domain shrinks slightly** — runs touching a recording boundary have no
   donor and are abstained instead of corrected.
