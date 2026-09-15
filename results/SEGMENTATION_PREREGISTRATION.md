# Pre-registration — are the boundaries real?

Written and committed **before any surrogate for this stage is generated and
before any corpus-versus-surrogate number exists**. `BREAKS.md` deliberately
reports a statistic and no verdict; this document fixes the verdict in advance.

## 1. The question, and why a rate alone cannot answer it

A changepoint detector fires on noise. `BREAKS.md` records 0.26–0.64
boundaries per second across eighteen cells, and **none of those numbers is
evidence of anything** without a null beside it: a detector run on a smooth,
structureless signal also produces boundaries, at some rate, with some duration
distribution.

So the question is not *how often does it fire* but *does it fire more on the
corpus than on a signal with no boundaries in it, and in places that are not
explained by something else*.

## 2. The nulls

Four families, all generated in the 17-dim egocentric space on the F3-carried
`raw` arm, per recording, never across a seam, through
`vieb/tok/surrogate.py` — the same generator the falsifier and the dwell arms
used.

| arm | what it holds fixed | role |
|---|---|---|
| **`phase`** | the exact power spectrum and cross-spectrum | **primary** — the tightest null. Same smoothness, so the same quantization-scale roughness, and no stereotyped sequence |
| `var5` | a VAR(5) fitted per recording | continuity with Q1's registered null |
| `ou` | a smooth, aperiodic flow with no discrete states | the continuum alternative |
| `white` | i.i.d. noise at each recording's own mean and SD | the negative control — a detector that cannot beat this is measuring nothing |

Every arm goes through the **identical** pipeline: same standardising SD, same
derivative window, same degree, same guard band, same non-maximum suppression,
same abstain mask carried over from the corpus. **Any preprocessing is
admissible precisely because the null goes through it too**; that is the
governing principle of this stage and the reason a smoother is not automatically
a confound.

## 3. The separability check runs first, on every null, before it calibrates anything

Stage A measured the corpus sitting **24× further from every surrogate than from
itself** while matching ambient scale to 2%. A null can look identically scaled
to the data and still be trivially separable from it in the basis the analysis
uses, and a null that is trivially separable constrains nothing.

`vieb/audit/separable.py` is the instrument. Three things are fixed here because
each would silently produce a meaningless PASS:

1. **Window features, not frames.** It is a linear probe, and `phase` matches
   every first- and second-order moment by construction, so a frame-level probe
   is at chance *by construction*. The 24× figure was measured in a **window**
   basis. Stacked windows go in.
2. **Group labels are class-prefixed** — `"{kind}:{recording}:{block}"` — or
   corpus and surrogate blocks from the same recording collide into one group
   and land on the same side of the held-out split.
3. **A separate `manifold_read`.** `separable_read`'s prose is about a corrector
   depositing mass on a constraint surface and would print a paragraph about
   k-means manufacturing a behavioural state.

**Polarity: high separability is FAILURE.** A null the probe can pick out is a
null that differs from the corpus in ways unrelated to boundaries, and a
boundary-rate gap against it means nothing. Registered limit: **balanced
accuracy ≤ 0.60**, inherited from `separable.py:MAX_BALANCED_ACCURACY` rather
than chosen here.

## 4. The threshold is a sweep, never a point

`k_mad ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0}`, fixed now. A single
threshold is a parameter chosen against an outcome, and ExBias's `k_mad = 3.0`
is adaptive to the recording but carries no stated false-positive rate.

The threshold is MAD-based and therefore **scale-adaptive per recording**, so the
same `k_mad` means "this far above this signal's own noise floor" for corpus and
surrogate alike. That is what makes the comparison at a common `k_mad` fair.

## 5. The gate, fixed in advance

Boundary rate per second, corpus against each surrogate, **animal-level bootstrap
at n = 89** on the per-animal difference.

| outcome | verdict | consequence |
|---|---|---|
| corpus rate exceeds every surrogate's, interval excluding zero, **across the whole sweep** | `PASS` | boundaries are not explained by a structureless signal. Step 3 runs. |
| corpus rate sits **inside** the surrogate range anywhere the comparison is powered | `FAIL` | **the segmentation route closes**, as cleanly as the falsifier closed the tokenizer, and that is the finding |
| corpus rate exceeds `white` only | `FAIL` | beating i.i.d. noise is not a result; every smooth signal does |

**No threshold on the size of the gap** — only on whether the interval excludes
zero and on the sweep being consistent. A gap size fixed now would be arbitrary
and one fixed later would be selected against the outcome.

## 6. Four things reported alongside, each able to invalidate a PASS

**The segment duration distribution.** Structured and heavy-tailed if boundaries
are behavioural; near-geometric if the detector is crossing a smooth flow. Its
**association with fit quality is reported beside it**, because ExBias measured
ρ(duration, R²) = **−0.538** on this corpus — fit quality there is largely a
length readout, and any statistic over segments is exposed to it.

**The length-matched random-interval control.** For each detected segment, a
random interval of the **same length from the same recording**, fitted the same
way, compared on adjusted R². ExBias's own figure is **65.9% ± 1.5%** (955
segments, 10 recordings, mean ΔR² +0.048) — "real and highly significant, but far
from the >90% a cleanly piecewise-smooth process would give." That is the number
to beat and the standard to be judged against.

**The boundary rate must not track anything else.** Session, animal, body size,
and arena position. The last specifically: violation rate rises **3.7× monotone**
from arena centre to wall (`CONCENTRATION.md`), so a boundary rate correlating
with wall proximity is finding occlusion, not behaviour. Measured with
`vieb/qc/concentration.py:edgeness`.

**Artifact ablation with a delta.** Boundary rate with and without flagged
frames, and the difference reported rather than the better number.

## 7. What a PASS would and would not license

It would license saying that the detector fires more on the corpus than on four
structureless signals put through the identical pipeline, and that the excess is
not explained by arena position, body size, animal or session.

It would **not** license "these are behaviours". The boundaries would be places
where the trajectory stops being smooth; whether those places are behaviourally
meaningful is Step 3's question and Step 4's after it. And the axiom itself —
*an action is a maximal interval on which the trajectory is smooth* — remains an
assumption with known failure modes: gradual transitions produce no kink,
continuous tremor produces no boundaries at all, and context-defined behaviours
are invisible to kinematics.
