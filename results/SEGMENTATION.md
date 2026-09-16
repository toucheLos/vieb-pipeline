# Step 2 — are the boundaries real? The gate returns **no verdict**, and why

Registered in `results/SEGMENTATION_PREREGISTRATION.md`, committed at `11de347`
before any surrogate for this stage was generated. Source array named in every
number below: **`work/ego/raw__bodylen__*.npz`** — the F3-carried `raw` arm,
`held_array(pose_unfiltered, missing)`, never Wiener. 89 report animals, 1,149
recordings. Cell `group=both, degree=3, deriv_sec=0.133`, `h = 4`, `guard = 1`,
`min_gap = 14` frames. Digest `198eb14ff258c7f6`.

## The headline, stated as it happened

Two registered checks ran. **Both failed, and the second one disqualifies the
first.**

1. **The gate** — the corpus boundary rate sits *inside* the surrogate range in
   9 of 24 structured cells, worst `ou` at `k = 1`, **−0.3518 [−0.3609,
   −0.3422]** boundaries/s. Registered verdict: `FAIL`, "the segmentation route
   closes".
2. **The separability precondition**, which the registration requires to run
   *first* and which licenses reading the gate at all — **`FAIL` on all four
   nulls**: `ou` 0.999, `white` 0.989, `phase` 0.800, `var5` 0.801 balanced
   accuracy, against a registered limit of 0.60.

The registration's own words, and `manifold_read`'s own reason string, say what
follows from a separability `FAIL`: the null *"differs from the corpus in ways
unrelated to boundaries, so a boundary-rate gap against it is uninterpretable and
is not reported as evidence."*

**That cuts both ways, and it has to.** If the gap is not evidence that the
boundaries are real, it is equally not evidence that they are not. So the `FAIL`
in (1) **cannot be read as "the route closes"**, any more than the corpus's 93×
excess at `k = 6` can be read as "the boundaries are behavioural".

> **Step 2 does not adjudicate whether the boundaries are real. The registration
> as written cannot adjudicate it**, for a reason given below that is a property
> of the design and not of the result.

Nothing downstream is licensed. Step 3 does not run on this.

## The gate, reported in full because it was registered

Boundaries per second, 89 report animals, animal-level bootstrap on the
per-animal difference, 2,000 replicates. The NMS resolution ceiling is
30/14 = **2.143 b/s**.

| `k_mad` | corpus | phase | var5 | ou | white | corpus − phase |
|---|---:|---:|---:|---:|---:|---|
| 1.0 | 0.7762 | 1.1252 | 1.1238 | 1.1280 | 1.1284 | −0.3490 [−0.3581, −0.3393] |
| 1.5 | 0.6518 | 0.9328 | 0.9349 | 0.9314 | 0.8953 | −0.2810 [−0.2898, −0.2717] |
| 2.0 | 0.5490 | 0.6870 | 0.6896 | 0.6713 | 0.5796 | −0.1380 [−0.1466, −0.1288] |
| 2.5 | 0.4669 | 0.4424 | 0.4437 | 0.4119 | 0.2991 | +0.0244 [+0.0152, +0.0345] |
| 3.0 | 0.4013 | 0.2531 | 0.2545 | 0.2183 | 0.1301 | +0.1483 [+0.1383, +0.1592] |
| 4.0 | 0.3069 | 0.0626 | 0.0640 | 0.0444 | 0.0202 | +0.2443 [+0.2336, +0.2552] |
| 5.0 | 0.2437 | 0.0125 | 0.0128 | 0.0066 | 0.0033 | +0.2312 [+0.2200, +0.2430] |
| 6.0 | 0.1989 | 0.0021 | 0.0023 | 0.0008 | 0.0007 | +0.1968 [+0.1852, +0.2088] |

The sweep is the whole point and it changes sign inside itself. A single
threshold would have licensed either conclusion depending on which one was
chosen, which is exactly why the registration fixed eight of them in advance.

**Two features of this table were not anticipated and are worth naming.**

**At `k = 1` the four nulls agree to 0.4%** — 1.1238 to 1.1284, four families
with nothing in common but their pipeline, at 53% of the NMS ceiling. That is the
detector saturating: at a low enough multiple of each signal's own noise floor,
every signal produces boundaries at the rate the refractory period permits, and
the number stops being about the signal. The spread between nulls only opens up
by `k = 6`, where the largest null rate is 3.3× the smallest.

**The corpus's `D` is heavy-tailed and the nulls' is not.** Across the sweep the
corpus rate falls **3.9×** (0.776 → 0.199) while `phase` falls **536×** and
`white` **1612×**. A signal whose second-derivative mismatch is mostly small with
rare large excursions is exactly what "smooth stretches punctuated by
discontinuities" would produce — and it is also, mechanically, what makes the
corpus cross *below* the nulls at low `k`, because a heavy tail inflates the MAD
that sets the threshold. The registered statistic compares rates at
self-standardised thresholds, and a heavy-tailed `D` is penalised by it at the
bottom of the sweep by construction. **This does not rescue the gate** — it is an
observation about the instrument, made after the result, and it is recorded here
rather than acted on.

## The separability precondition, and the negative control that earned its keep

The registration required the probe to run on **window** features, not frames,
because `phase` matches every first- and second-order moment and a frame-level
linear probe would be at chance *by construction*. It ran that way.

**The first run passed all four nulls — and the negative control proved the
instrument was blind.**

| null | raw window only | | + roughness | |
|---|---:|---:|---:|---:|
| | acc | AUC | acc | AUC |
| `phase` | 0.5167 | 0.5038 | **0.8000** | 0.8729 |
| `var5` | 0.5121 | 0.5006 | **0.8011** | 0.8746 |
| `ou` | 0.5583 | 0.4979 | **0.9991** | 0.9999 |
| `white` | 0.5232 | **0.4962** | **0.9892** | 0.9986 |

`white` is i.i.d. noise. It is obviously distinguishable from a mouse
trajectory, and the probe returned it as inseparable **at AUC 0.496 — chance**.
That is a fact about the instrument, not about the null: `separability` fits a
logistic regression, which is linear in its features, and the difference between
a smooth signal and a rough one lives in the **second** moment of the increments,
a quadratic function of the window that no linear model can form from raw frames.

So each window now carries, beside its raw frames, the per-channel
`log` mean-squared first difference — the smoothness the entire arm is about,
made visible to a linear model. `white` fails, as it must. And so does everything
else.

Both runs are on disk: `work/tok/seg_validate/_separability_rawonly.json`
preserves the blind result rather than deleting it, and
`work/tok/seg_validate/_separability.json` is the one that stands. Both probe
all **298** animals rather than the 89 report animals — the probe is a check on
how a null was *constructed*, not an effect estimate, so it is not scored on a
held-out split; that is a deviation from this repo's usual practice and is stated
rather than buried.

## What the probe is separating on — post-hoc, descriptive, no verdict attached

Written after the failure, and it does not reopen anything. There were two
candidates with opposite consequences.

**A roughness *level* difference would have been benign.** The threshold is
`median(D) + k·1.4826·MAD(D)` of each recording's own `D`, so a null that is
uniformly rougher or smoother is standardised back before a single boundary is
counted: the probe would see a difference the gate does not.

**It is not a level difference.** Per-channel mean `log` mean-squared increment,
and its spread across windows, averaged over the 17 ego channels:

| arm | mean log MSD | SD across windows |
|---|---:|---:|
| **corpus** | **−7.67** | **2.88** |
| `phase` | −5.45 | 1.34 |
| `var5` | −5.45 | 1.35 |
| `ou` | −3.04 | 1.39 |
| `white` | −3.35 | 0.95 |

`phase` preserves the corpus's power spectrum **exactly**, so it preserves the
mean squared increment exactly. It does not preserve the mean of the *log*, which
sits 2.22 nats lower on the corpus — a factor of 9.2 — in **15 of 17 channels**;
and the corpus's window-to-window spread is **2.15×** larger, in the same 15.

That is one statement, not two: **the corpus's roughness is concentrated.** Its
typical window is far smoother than any null's while its total power is
identical, which means the power lives in a small minority of windows. Smooth
stretches punctuated by rare rough moments — the piecewise-smooth axiom, visible
directly in the feature the probe uses, without any boundary having been detected.

(The third arm in that diagnostic, `roughness_centred`, is kept in the code and
in the JSON with a note saying it answers nothing: both classes are transformed
by the same affine constants and a logistic regression is affine-invariant, so it
is arithmetically identical to `roughness_only` and returned identical numbers to
four decimals. It was a null test that could not have failed. The table above is
a description, not a probe, and is what actually answers the question.)

## The consequence, which is about the design and not about the result

The precondition and the question are in tension, and the diagnostic shows why.

The precondition demands a null that **no probe can distinguish from the corpus**.
The hypothesis under test says the corpus is **piecewise smooth** — smooth
stretches separated by discontinuities. A null built to have no boundaries in it
therefore differs from the corpus in its roughness distribution *precisely
because* it has no boundaries. Give the probe a feature sharp enough to see
smoothness, and it separates them **on the thing under test**.

So, once the probe could see roughness at all, **no boundary-free null could have
passed**. A null that passed would be one whose roughness is as concentrated as
the corpus's — and a signal with concentrated roughness is a signal with
boundaries in it.

**The registered precondition is not satisfiable by any null this question
admits.** That is a defect in the registration, which I wrote, and it was not
visible in advance: the first run passed all four nulls and would have let the
gate proceed had the negative control not been there to expose the probe as
blind. The control is the reason this is known.

I am not amending the registration to route around it. The stage stands as
`FAIL`/`FAIL` with no downstream licence, and reopening requires a **new**
registration with a precondition that can distinguish "this null is off-manifold
in an irrelevant way" from "this null lacks the structure under test" — the
current one cannot, and neither can a threshold moved to fit.

## The four things registered alongside, reported because they were registered

Each of these is descriptive. None is evidence for or against the boundaries,
for the reason above.

**Segment duration** (clean segments, no abstained frame): median **0.496 s**,
mean 1.322 s, p25 0.117, p75 1.097, p90 3.030, p99 14.763, max 48.5 s. Heavily
right-skewed — sd 3.10 against a mean of 1.32 — but the p10 of 0.037 s says a
large mass sits at the resolution floor, which is the detector's refractory
period rather than a duration measurement.

**Fit quality**: adjusted R² median **0.456**, mean 0.426, p10 0.096, p90 0.705.
Adjusted, never raw: raw R² is a length filter in disguise, since its null
expectation rises with length while the adjusted version's is 0 at every length.

**The length-matched random-interval control**: **57.93%** over **164,726**
comparisons, mean ΔR²(adj) **+0.070**. ExBias's reference on this corpus is
**65.9% ± 1.5%**, which its authors described as *"real and highly significant,
but far from the >90% a cleanly piecewise-smooth process would give."* This is
**below that reference, not above it** — 57.9% against 65.9% — on 170× more
comparisons. A detector that beats a random interval of the same length 58% of
the time is barely beating it.

**Confounds.** Boundary rate against arena position: per-animal Spearman
ρ = **+0.264 [+0.202, +0.326]**, n = 89 — nonzero and the interval excludes zero.
Across edgeness deciles the rate is **+15.7%** at the wall against the centre
(0.381 → 0.441), non-monotone: it dips to 0.357 at decile 3, then rises
monotonically over the top five to 0.446 at decile 8 — a 1.25× spread end to
end. `CONCENTRATION.md`
records violation rate rising **3.7×** over the same axis, so this is a much
weaker gradient than the artifact's — but it is the same sign, and a boundary
rate that tracks wall proximity is partly finding occlusion. Body size across
animals: ρ = +0.335. By group, worst/best: animal 1.72× (0.301–0.518 b/s), date
2.10×, day 1.19×, context 1.14× — session and context are close to flat, animal
and date are not.

**Artifact ablation**: blocking abstained frames *lowers* the rate by a roughly
constant **+0.012 to +0.020 b/s** at every `k` (0.7762 blocked against 0.7956
unblocked at `k = 1`; 0.1989 against 0.2113 at `k = 6`). So roughly 2–6% of
unblocked boundaries are at abstain edges. The delta is reported, not the better
number.

## What stands from this stage

* The gate ran as registered on all 298 animals' shards and is scored on the 89
  report animals. Its numbers are in `results/seg_gate.json`.
* The separability probe reproduces **bit-for-bit** after the refactor that gave
  the diagnostic access to its window builder — verified, not assumed.
* **The negative control is the finding that transfers.** A probe that returns
  i.i.d. noise as inseparable from a mouse is blind, and nothing else in the
  first run would have said so. Every future separability check in this
  programme needs a null whose answer is known, and needs features that can form
  the quantity in question.
* The corpus's roughness is concentrated — 9.2× lower typical window energy than
  a spectrum-matched surrogate, 2.15× more spread — which is the piecewise-smooth
  signature observed directly, before any detector. It is the most encouraging
  number in this document and it is **not** a boundary measurement.

## What does not stand

* "The segmentation route closes." Not licensed. The gate's `FAIL` is
  disqualified by the same rule that would have disqualified a `PASS`.
* Any reading of the `k ≥ 2.5` excess as evidence that the boundaries are real.
* Step 3 and Step 4, which were gated behind a `PASS` that did not occur and
  could not have been believed if it had.
