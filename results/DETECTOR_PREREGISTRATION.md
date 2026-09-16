# Pre-registration — can the detector find what is demonstrably there?

Written and committed **before any diagnosis is run and before any parameter is
swept**. The gate is named here, and no parameter is tuned until it passes.

## 1. The number this stage exists for

`SEGRECUR.md` reports it as a diagnosis and not a result, which is what makes it
the bottleneck: of planted instances inserted into the null at 10% occupancy,

> **2.3% became their own segment. 66.7% were merely overlapped** by a segment
> that spanned them.

The detector misses **97.7% of events that are demonstrably present**. No matcher
improvement fixes that, and every coverage figure downstream — 1.86% of segments
in a clump, 852 mergeable adjacent pairs — is bounded by it. Raising the bank
before raising this buys more assignments under a detector that cannot see.

## 2. Diagnose before changing anything

The 2.3% is one number covering four different failures, which point at four
different fixes. Every planted instance is classified into exactly one bin, and
**the four are asserted to sum to the total**:

| bin | what it means | what it implicates |
|---|---|---|
| **1. no boundary at either edge** | the criterion cannot see the insertion | **the criterion** — a smooth crossfaded block may produce no acceleration break at all |
| **2. one edge only** | half-detected; the segment runs into its neighbour | the criterion, weakly — or an edge falling inside the guard band |
| **3. both edges present but merged** | NMS or the minimum-gap floor swallowed one | **the threshold and the refractory period** |
| **4. both edges present but split** | the instance became several segments | the threshold, in the other direction |

An edge counts as detected if a boundary falls within **±2 frames** of it — the
same tolerance `BREAKS.md` uses for its channel-group Jaccard, so the number is
comparable to one already published.

**If bin 1 dominates, no sweep of thresholds will help**, and that is a finding
about the criterion rather than about its parameters.

## 3. The sweep, in the order the evidence ranks it

**Window length and degree.** Currently `deriv_sec = 0.133`, `degree = 3`. The
shortest resolvable segment is bounded by the derivative half-width and the
degrees of freedom, so these are the parameters that set what can be seen at all.
`scripts/breaks.py` already carries `DERIV_SWEEP = (0.067, 0.133, 0.267)` and a
degree sweep; both are reused rather than redefined. **Reported per cell: the
smallest planted duration recovered with an interval excluding zero** — a
resolution, not only a rate.

**`shape` as its own detection cell.** `shape` and `twist` share only **23%** of
boundaries at ±2 frames, and `shape` is the arm whose segments recur. Detection
has only ever run on `both` as the primary; running it on `shape` directly is a
different detector, not a filter applied afterwards.

**Threshold and NMS.** `k_mad ≤ 2.0` stays excluded and the reason stays
registered: at `k = 1` the four nulls of the closed boundary-rate gate agreed to
**0.40%** at 53% of the NMS resolution ceiling, so rates there are a property of
the refractory period rather than of the signal.

## 4. The planted control, unchanged from the one that caught two bugs

One template for the whole corpus, averaged over three donor animals, planted
into **`microstate`** and scored against unplanted `microstate`. Both properties
are load-bearing and both were learned by the control returning nothing: a
per-animal template plants a signal a cross-animal statistic cannot see, and
planting into the corpus measures the corpus's own recurrence.

Instance counts are rounded **with a coin**, and everything is reported against
the **realised** occupancy.

## 5. The gate, fixed now

> **Planted-instance isolation ≥ 20% at the primary planted duration.**

A ~9× improvement on 2.3%. Registered rationale: below roughly this level the
detector loses most of what is demonstrably present, so a coverage figure
computed on its output is a statement about the detector and not about
behaviour. The number is named now precisely so it cannot be moved to whatever
the sweep happens to reach.

**No parameter is tuned until the gate passes**, and **a refusal is a correct
outcome**: if the best cell in the sweep cannot reach 20%, that is reported as
the answer and Step D does not run.

## 6. The negative control, required

`METHODS_FINDINGS.md` M1: **every probe needs a null it is expected to fail.**
Any detector cell that clears the gate is also run on **`white`** — i.i.d. noise
at each recording's own mean and SD — and must **not** show the same isolation
there. A detector that "recovers" planted instances in white noise is recovering
its own threshold.

## 7. What a PASS would and would not license

It would license re-running coverage on a detector that finds a fifth of what is
present instead of a fortieth.

It would **not** retroactively change any published number. `SEGRECUR.md` and
`VOCAB.md` stand on the detector they were computed with; a better detector makes
a **new** measurement, and comparing the two is a separate question with its own
registration.
