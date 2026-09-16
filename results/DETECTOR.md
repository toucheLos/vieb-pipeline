# Step B — the criterion cannot see the insertion

Registered in `results/DETECTOR_PREREGISTRATION.md`, committed at `fe373da`
before any diagnosis was run and before any parameter was swept. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm; `shape` group; planted into
`microstate` and scored against it; 25 cells. Digest `198eb14ff258c7f6`.

## The verdict: FAIL, and the reason is the criterion

> **12.8%** of 5,914 planted instances became their own segment at the best cell
> in the sweep, against the registered **20%** target. The dominant failure is
> **`no_edge` at 66.2%**.

The gate was named before the sweep ran and it is not moved to meet the result.

## The diagnosis is what makes this decisive

The 2.3% in `SEGRECUR.md` was one number covering four different failures. Split
into the four bins registered in advance, at the detector of record:

| bin | share | what it implicates |
|---|---:|---|
| isolated | 4.3% | — |
| **no boundary at either edge** | **73.8%** | **the criterion** |
| one edge only | 21.9% | the criterion, weakly |
| both edges present but **merged** | **0.0%** | the refractory period — **exonerated** |
| both edges present but **split** | **0.0%** | the threshold — **exonerated** |

**`merged` and `split` are zero.** Non-maximum suppression is swallowing nothing
and the minimum-gap floor is swallowing nothing. Whatever is wrong, it is not
the threshold and it is not the refractory period — which is exactly the
distinction the four bins were registered to make, and which the single 2.3%
could not have shown.

Two-thirds to three-quarters of instances that are **demonstrably present**
produce **no boundary at all** at either edge.

## The sweep helps, and not enough

Over 3 derivative half-widths × 2 degrees × 4 planted durations:

| deriv_sec | degree | planted | isolated | no_edge | one_edge | merged |
|---:|---:|---:|---:|---:|---:|---:|
| **0.067** | **2** | **0.50 s** | **12.8%** | 66.2% | 20.9% | 0.0% |
| 0.067 | 3 | 0.50 s | 9.8% | 72.4% | 17.7% | 0.0% |
| 0.133 | 3 | 1.00 s | 9.4% | 70.4% | 20.1% | 0.0% |
| 0.067 | 2 | 1.00 s | 9.2% | 56.1% | 27.1% | 0.0% |
| 0.133 | 3 | 0.50 s | 3.7% | 78.1% | 18.2% | 0.0% |
| 0.133 | 3 | 0.25 s | 0.3% | 88.2% | 11.4% | 0.0% |

A shorter derivative window and a lower degree both help, in the direction
theory predicts — the shortest resolvable feature is bounded by the half-width
and the degrees of freedom. Isolation rises **5.6×**, from 2.3% to 12.8%.

**No cell reached the target at any planted duration**, so
`smallest_isolated_duration_s` is null for every cell. There is no parameter
setting at which this detector isolates a fifth of what is planted.

## What this means, stated at the right strength

The registered reading for a `no_edge`-dominant failure was fixed in advance:

> *"this is a statement about the acceleration-mismatch criterion and not about
> its parameters — no further threshold sweep is indicated."*

The criterion looks for a discontinuity in acceleration. A planted instance is
the mean of 40 real windows — very smooth — crossfaded into the stream over 3
frames. **It has no acceleration discontinuity to find.** The detector is not
malfunctioning; it is correctly reporting that nothing breaks there.

**So this result is as much about the control as about the detector**, and both
readings are live:

* **If the planted control is the right probe**, the criterion misses most
  behavioural transitions and the segmentation route needs a different criterion.
* **If real behavioural transitions are sharper than a crossfaded template
  average**, then the probe understates the detector and the 12.8% is a floor for
  *smooth planted blocks* rather than a bound on the detector's real sensitivity.

Nothing here distinguishes those, and inventing a sharper template after seeing
this result would be choosing a probe against an outcome. **What would
distinguish them is a planted instance whose insertion is a genuine C²
discontinuity** — and that needs its own registration, because "make the control
easier until the detector passes" is the failure mode this gate exists to
prevent.

## The negative control did not run, and why

`METHODS_FINDINGS.md` M1 requires any cell that clears the gate to also be run on
`white` and to fail there. **No cell cleared the gate**, so there is nothing
whose specificity needed checking. This is recorded rather than silently skipped:
the control is owed the moment any future detector passes.

## Consequence

**Step D does not run.** Coverage computed on a detector that isolates an eighth
of what is demonstrably present would be a statement about the detector. The
registration said so before the number existed.

`SEGRECUR.md` and `VOCAB.md` stand unchanged. They were computed with the
detector of record and a better detector would be a **new** measurement, not a
correction to theirs.
