# Continuing the search for tokens — what the measurements say to do next

Written after Step 4 was refused. **Nothing here is a result**; it is a plan, and
every item is written to be registered before it runs. It is ordered by what the
evidence says the blocker is, not by what is interesting.

## The blocker, quantified

Step 4 needed a symbol stream. It got one in which 98% of tokens are a single
`unassigned` symbol, and **852 mergeable adjacent pairs** on `shape`.

| quantity | shape | both |
|---|---:|---:|
| segments (report split) | 64,341 | 57,406 |
| have any cross-animal neighbour within θ | 3.31% | 2.40% |
| sit in a clump of ≥ 20 | **1.86%** | 1.13% |
| adjacent pairs with **both** members assigned | **852** | 460 |

The gap between rows 2 and 3 is the informative one: **roughly half of the
segments that have a close partner never reach a group.** They form pairs and
short chains. So the problem is not that the matcher fails — Step 2's gate
passed on all three groups, by 8× over length-matched windows on `shape` — it is
that matches do not *accumulate* into groups large enough to name.

Two numbers say the same thing from the other side. Clump members sit at median
normalised distance **0.173** against θ = 0.190, where unassigned segments sit at
**0.361**: the clumps are the tail that just squeaks under the threshold. And
assignment is **38× clustered in time** — when one segment is in a clump its
neighbour usually is too — so what exists is real and simply too sparse.

---

## Lever 1 — run the open-end distance as a primary. Cheapest, and already owed.

`SEGRECUR_PREREGISTRATION.md` registered open-end alignment as the **secondary**
and it was never computed (`DEVIATIONS.md` D8). That is a debt and also an
opportunity, because four independent lines in this programme say **duration
carries the signal**: rung 2 failing on the data term alone, the 148× hazard fall
surviving speed conditioning at 0.756 [0.742, 0.769], the falsifier decomposition
where the corpus's whole advantage is dwell, and every alphabet under-resolved.

The registered primary **time-normalises, and therefore throws duration away.**
That is why one clump can pool 0.53 s with 65 s — a 122× range at distance zero.

**What to register.** Open-end as the primary, time-normalised as the secondary
— the mirror of what was registered before, and it must be registered *before*
seeing its clump counts, because the previous run's numbers are now known.

**Both outcomes are results.** Higher coverage → duration was the missing
constraint and the first pass was measuring tempo-invariance. Lower coverage →
the recurrence really is shape-at-any-tempo, and the 122× range is the finding
rather than an artifact.

## Lever 2 — make the boundaries reproducible. The measured bottleneck.

The planted floor produced the hardest number in the study: **only 2.3% of
planted instances became their own segment**, and 66.7% were merely overlapped
by a segment spanning them. The detector, not the matcher, is what loses a
planted stereotype.

Two candidates, each needing its own registration because each is a **new
estimator** rather than a new threshold:

* **Boundaries where channel groups agree.** `shape|twist` Jaccard is 0.226 at
  ±2 frames — largely different events. An intersection detector would fire less
  often and, if the agreement is meaningful, more reproducibly. It must be
  registered with its own null, because "where two detectors agree" is exactly
  the kind of construction that manufactures apparent structure.
* **A break criterion that is not second-derivative mismatch.** The planted
  template is the mean of 40 real windows — very smooth, crossfaded — so it
  creates no acceleration discontinuity. A detector that cannot see an inserted
  stereotype is telling us the criterion and the planted control disagree about
  what a boundary is; one of them should change, and the registration has to say
  which before running.

**Negative control, required** (`METHODS_FINDINGS.md` M1): any new detector is
run on `white` and must fail to find structure there. A detector without a null
it is known to fail is not a detector.

## Lever 3 — run `shape` as primary and retire `twist`.

Measured, not preferred:

| group | segments recur? | length-match control | within-vs-cross |
|---|---|---:|---:|
| **shape** | `PASS` vs both nulls, +5.93% | 0.922 | 0.344 |
| both | `PASS` vs both nulls, +4.05% | 0.889 | 0.330 |
| twist | **`FAIL`** — +0.117% [−0.185%, +0.427%] | **0.494** | 0.012 |

`both` sits **below** `shape` alone on every comparison, so the three twist
channels dilute rather than add. Twist's own length-match control says a twist
match is substantially a length match. Carrying all three groups triples the
compute to defend a channel set that does not recur.

**Keep `both` as a reported control, drop `twist` as a primary.** State it in the
registration so it is a decision and not a disappearance.

## Lever 4 — raise the bank, which raises a count rather than a rate.

Coverage is a rate; 852 is a count. The analysis is scored on 89 report animals
of 298, and the vocabulary stage reads **no effect** — it describes structure.

**What could be registered:** pooling `tune` and `fit` animals **for the
vocabulary stage only**, where no gated number is read, while every effect
estimate stays on `report`. That multiplies the pair count roughly threefold
without touching a gated result. It needs its own registration precisely because
"use more data where it is convenient" is how a split stops meaning anything, and
the registration has to name which numbers may and may not be computed on the
pooled set.

**What must not happen:** re-scoring Step 2 or Step 3 on a pooled bank. Those
verdicts stand on the split they were registered under.

---

## What is not the answer

**A different clustering algorithm.** ExBias's stack returned `n_states = 0` and
self-diagnoses why. The constraint here is not the clusterer; 98% of segments
have no close cross-animal partner at all, and no algorithm creates neighbours.

**Lowering `min_size` or widening θ.** Both are registered, and both were fixed
before the coverage number existed. Moving either now is choosing a parameter
against a known outcome.

**Reading the Step 2 boundary-rate gate.** It is closed, its separability
precondition was contradictory (`DEVIATIONS.md` D7), and neither its `FAIL` nor
its 93× excess is evidence in either direction.

**Any Wiener-arm array.** Q1's +1.639% is a `wiener` result; the same cell
unfiltered is −0.1991% [−0.4636%, +0.0418%]. This work runs on `raw`, where a
low-pass filter would manufacture the smoothness whose breaks the detector looks
for.

## Three rules this programme paid for, carried forward

From `METHODS_FINDINGS.md`, because each cost a wrong answer that looked right:

1. **Every probe needs a negative control it is expected to fail.** A
   separability probe passed four nulls while returning i.i.d. noise as
   inseparable from a mouse at AUC 0.496.
2. **When a validity check fails for one arm, the remedy applies to every arm.**
   Raising a PCA dimension only where it failed would have put corpus and null
   in differently-shaped spaces while the threshold came from the null.
3. **Read a control's shape, not just its verdict.** The planted ladder ran
   backwards, then flat, for two different reasons — both invisible in the code
   and obvious in the output.
