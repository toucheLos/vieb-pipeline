# The first accuracy number — and it takes a body length to see an acceleration break

Registered in `results/PLANT_PREREGISTRATION.md`, committed before
`vieb/seg/plant.py` or `scripts/plant.py` existed. **`tune` split, 60 animals**,
**7,356 planted instances per cell**, F3 `raw` arm, `shape` group, detector
frozen at `deriv_sec = 0.133`, `degree = 3`, `k_mad = 3.0`. Construction
refined from §2 and recorded as `DEVIATIONS.md` **D12**.
Digest `198eb14ff258c7f6`.

## Why this is the first of its kind here

**Nothing in this repository was an accuracy.** Every earlier result is a
comparison against a null — recurrence excess, a context contrast, a rate
against a floor. Those establish *not nothing*. None is a recall. Both previous
attempts at ground truth failed diagnosably: Step B's planted template had no
acceleration discontinuity to find (`PROBE_AUDIT.md`), and human raters could
not resolve 67 ms from overhead (`ANNOTATION.md`).

This plants breaks whose **order** is controlled, at amplitudes measured in the
tracker's own noise, and asks what the detector recovers.

## The result

> **To recover an acceleration discontinuity — the only break its axiom claims —
> at 50%, the detector needs a displacement of 16 σ̄ ≈ 0.99 body lengths.**
> A position jump needs 4 σ̄ ≈ 0.25.

σ̄ = **0.061799** body lengths, the confidence-weighted mean per-keypoint jitter
from `NOISEFLOOR.md`'s skull-bone calibration. Amplitude is the instance's peak
displacement.

### Recall at the onset, ±2 frames, corpus arm

| order | 1 σ̄ | 2 σ̄ | 4 σ̄ | 8 σ̄ | 16 σ̄ | 32 σ̄ |
|---|---|---|---|---|---|---|
| **0** — position jumps | 0.213 | 0.360 | **0.606** | 0.862 | 0.955 | 0.990 |
| **1** — velocity jumps | 0.091 | 0.170 | 0.292 | 0.480 | **0.774** | 0.931 |
| **2** — acceleration jumps | 0.062 | 0.105 | 0.192 | 0.322 | **0.542** | 0.828 |

Chance is **0.0744**, from the detector's own firing rate in a ±2 window — and
it matches the 0.074 predicted in advance from the measured 0.443/s, which is a
check on the harness rather than a finding. At order 2 and 16 σ̄ the effect over
chance is **+0.4677 [+0.4111, +0.5252]**.

**At 1 σ̄ the detector is at chance for an acceleration break** — recall 0.062
against chance 0.0744. A break the size of the tracking noise is invisible to it.

## The five registered predictions

| # | prediction | outcome |
|---|---|---|
| 1 | order 0 above 0.8 at 32 σ̄ | **0.990 — held** |
| 2 | monotone in amplitude within each order | **held**, all three orders |
| 3 | ordered 0 ≥ 1 ≥ 2 at every amplitude | **held**, all six amplitudes |
| 4 | order 2 needs ≥1 more ladder rung than order 1 to reach 50% | **FAILED** |
| 5 | `white` recovery below corpus above 4 σ̄ | **held**, worst gap −0.0577 |

**Prediction 4 failed at the ladder's resolution, and prediction 3 says why
that is not the conclusion it looks like.** Both orders cross 50% at the 16 σ̄
rung, and the ladder doubles, so it cannot resolve a difference smaller than 2×.
Prediction 3 compares *every* cell and order 2 sits below order 1 at all six,
by 0.028 to 0.232 of recall. **The criterion is preferentially blind to the
smoother break; the registered test was simply too coarse to show it.**

Descriptive, and not registered — log-interpolating the crossings within the
ladder gives order 0 ≈ **3.0 σ̄**, order 1 ≈ **8.4 σ̄**, order 2 ≈ **14.0 σ̄**, so
order 2 needs about **1.7×** order 1's amplitude and **4.7×** order 0's. Those
three numbers are an interpolation between measured rungs, not measurements, and
nothing rests on them.

## The negative control, discharged at last

`DETECTOR_PREREGISTRATION.md:84-90` has required since Step B that any cell
clearing the gate be run on **`white`** and fail there. No cell ever cleared, so
**it had never run** (`DETECTOR.md:104-109`). It ran here, on every cell.

| order | 1 σ̄ | 4 σ̄ | 8 σ̄ | 16 σ̄ | 32 σ̄ |
|---|---|---|---|---|---|
| 0 | 0.021 | 0.057 | 0.259 | 0.732 | 0.932 |
| 1 | 0.021 | 0.022 | 0.033 | 0.120 | 0.525 |
| 2 | 0.020 | 0.021 | 0.023 | 0.036 | 0.155 |

**Corpus beats `white` at every cell above 4 σ̄**, worst gap −0.0577. The probe
measures the plant and not the detector's own threshold. **The debt is
discharged for these cells.** It is not discharged for any future detector.

Worth noting on its own: on i.i.d. noise the detector recovers an acceleration
break at **0.155** even at 32 σ̄, against 0.828 on real data. The corpus's
piecewise-smooth structure is most of what makes a break findable at all.

## One measure that did not work, reported rather than dropped

A secondary "recovered **anywhere** in the instance" was added to separate two
failures the ±2 test scores identically — *blind to the event* versus *found it,
missed the onset*. **It cannot do that job and should not be read.** The
instance runs 96 frames, so a detector at 0.443/s fires inside it by luck about
**76%** of the time; the observed values sit at 0.51–0.97 and are at or below
that chance level. The window is too long for the question. Separating
mislocation from blindness needs a scored window sized to the detector's own
refractory period, and that is a different registration.

## What this licenses

**Licenses:** the frozen detector recovers a planted acceleration discontinuity
above chance from about 4 σ̄ upward, reaching 50% at 16 σ̄ ≈ 0.99 body lengths,
on 60 `tune` animals and 7,356 instances per cell, with a negative control that
passes. **This is a recall, and it is the first one this programme has.**

**Does not license** any statement about the corpus's own boundaries. **A
planted discontinuity is not a behaviour.** Recall against a plant bounds the
*instrument*; it says nothing about how many real behavioural transitions are
that large, and this repository has no way to know.

**Does not re-score Step B.** `DETECTOR.md`'s 12.8% stands, on its own probe,
and is not withdrawn. What this adds is the missing axis: Step B's target was
both smooth *and* of unmeasured magnitude, and the two could not be separated.

**Does not license tuning.** A detector that needed a smaller amplitude would be
a different detector, and `DETECTOR_PREREGISTRATION.md:97-100` fixes that such a
thing makes a new measurement rather than correcting this one.

## What it means for the programme

A mouse is about 9 cm. **0.99 body lengths of peak displacement, delivered
inside a quarter second, is a very large movement** — far larger than the
crouch-to-still transitions the island is made of. Set beside `NOISEFLOOR.md`'s
finding that the detector sits at its own noise floor in the slowest 40% of
frames, the two agree: **this instrument is built for large, fast events.**

That is consistent with everything measured so far and it sharpens the
characterisation rather than overturning it. The detector finds real structure —
`STILLNESS_CONTROL.md` shows its boundary placement carries context information
that duration- and speed-matched segments do not — but its sensitivity to the
break its axiom is about is poor, and now quantified.
