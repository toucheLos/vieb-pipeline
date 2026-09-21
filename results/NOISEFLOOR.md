# The tracking-noise floor — the detector is above it when the animal moves, and on it when it does not

Registered in `results/NOISEFLOOR_PREREGISTRATION.md`, committed before
`vieb/seg/jitter.py`, `vieb/seg/floor.py` or `scripts/noise_floor.py` existed.
**`tune` split, 60 animals, 770 recordings** — not `fit`, not `report`, because
this floor is meant to select a parameter later and choosing a parameter against
a number is fitting. Source array `work/clean/<rid>.npz` via `spine.clean`, the
**`raw`** arm, through the standard ego transform. Detector frozen at
`deriv_sec = 0.133`, `degree = 3`, `k_mad = 3.0`, `shape` group.
Digest `198eb14ff258c7f6`.

## The result in one line

> **The frozen detector cuts 0.2599 boundaries per second on a constant pose
> carrying nothing but measured DLC jitter. The corpus cuts 0.4746. The floor is
> 54.8% of the rate.**

And it is not uniform. **34 of 75 joint cells are not distinguishable from that
floor**, and which ones is entirely predictable: **speed**.

## The calibration

**12,752,679 skull-bone residuals**, all **10 of 10** confidence bins filled.
Skull bones only — `bones.SKULL`, the rigid ear–ear–nose triangle, whose length
is physically constant so all of its variance is tracking noise. Trunk bones
excluded because they flex. Held and interpolated keypoint-frames excluded:
their residual is zero by construction, which would pull the estimate toward
zero exactly where tracking is worst.

Per-keypoint, per-axis noise amplitude in body lengths:

| DLC confidence | σ (body lengths) | keypoint-frames |
|---|---:|---:|
| < 0.20 | **0.27700** | 73,047 |
| 0.30–0.40 | 0.13222 | 842,703 |
| 0.50–0.60 | 0.06539 | 2,206,976 |
| 0.70–0.80 | 0.03885 | 3,035,214 |
| 0.90–0.95 | **0.02849** | 211,603 |

**Monotone across nine of ten bins, an 8.76× range.** The last bin (≥ 0.95,
n = 54,929) ticks back up to 0.03160 and is the thinnest; nothing is read into
it. Confidence is a real covariate on this corpus even though the confidence
*gate* refuses here for want of a bimodal mode — gate-refuses is not
covariate-useless.

## The three injection arms

| arm | boundaries/s | interval |
|---|---:|---|
| `static` — constant pose, no noise | **0.0000** | [0.0000, 0.0000] |
| `jitter ×0.5` | 0.2547 | [0.2504, 0.2588] |
| **`jitter ×1.0` — the floor** | **0.2599** | **[0.2554, 0.2644]** |
| `jitter ×2.0` | 0.2713 | [0.2660, 0.2753] |
| **corpus**, same animals, same detector | **0.4746** | [0.4573, 0.4909] |

**Prediction 1 held.** The constant-pose arm produced **exactly zero**
boundaries over 137,832 eligible seconds. The harness responds to variance and
not to itself, so the rest of the table means something.

**Prediction 2 held.** The detector fires on pure tracking noise at
**13.9% of the 1.875/s NMS resolution ceiling**. There is a floor.

**Prediction 3 held, and its magnitude is the more interesting number.**
The rate is monotone in amplitude, so it is reported as a PASS. But a **4×
range in injected noise moves the rate by 6.5%**:

> **The floor is very nearly amplitude-invariant, and that is structural.**
> `discontinuity` is scale-equivariant (asserted at `tests/test_breaks.py:76`)
> and `mad_threshold` is scale-adaptive — `median + k·MAD` of a scaled signal
> scales exactly — so multiplying the noise multiplies the threshold with it and
> the crossings barely move. Only the ego transform's nonlinearity breaks the
> invariance, which is why the trend exists at all.
>
> **Halving the tracking noise would lower the noise-driven boundary rate by
> about 2%.** This floor cannot be retracked away. It is a property of a
> `median + 3 MAD` rule with a 16-frame refractory period, not of DeepLabCut.

## Where the corpus sits on it

**Overall: above the floor.** 0.4746 [0.4573, 0.4909] against 0.2599
[0.2554, 0.2644], non-overlapping.

**By speed quintile — and this is the whole diagnosis:**

| speed quintile | boundaries/s | cells not separated from the floor |
|---|---:|---:|
| **q0 (slowest)** | **0.1851** | **13 of 15** |
| q1 | 0.2493 | 11 of 15 |
| q2 | 0.3759 | 8 of 15 |
| q3 | 0.6140 | 2 of 15 |
| q4 (fastest) | 0.9359 | **0 of 15** |

**The two slowest quintiles — 40% of all frames — sit at or below the floor of
0.2599.** The slowest quintile cuts at 0.1851/s, which is *less* than the
detector's own rate on pure noise.

**The registered cell fails, as predicted.** Still × centre-arena ×
low-confidence: corpus **0.3112 [0.2541, 0.3696]** against the floor
**0.2599 [0.2554, 0.2644]** — overlapping. Prediction 5 held.

> In the stillest, most central, least confident frames the detector is **not
> distinguishable from its own noise floor**. Boundaries found there are not
> evidence of behaviour.

## Why the joint grid was necessary, demonstrated

The marginals on their own would have supported the wrong story. Both look
dramatic:

| confidence quintile | boundaries/s | | arena decile | boundaries/s |
|---|---:|---|---|---:|
| q0 (lowest) | **0.8890** | | 0 (centre) | 0.3081 |
| q2 | 0.3897 | | 5 | 0.4372 |
| q4 (highest) | **0.2207** | | 9 (wall) | **0.8485** |

Rate falls **4×** as confidence rises and rises **2.8×** from arena centre to
wall. Read alone, either would say "the detector is cutting tracking noise".

**The joint grid says otherwise.** Within the slowest speed quintile, the two
cells that *do* clear the floor are the **lowest-confidence, non-central** ones —
exactly the rearing signature the registration named in advance as the
alternative reading, and exactly the cells a confidence marginal would have
mislabelled as noise. Everywhere else in the slow regime, across every
confidence level and every arena third, the detector is at its floor.

**So the axis is speed, not confidence and not the wall.** Confidence and arena
position are largely proxies for it: a fast mouse is a badly tracked mouse near
the wall.

## What this licenses

**Licenses:** the frozen detector's boundary rate is above its own measured
tracking-noise floor overall, on 60 `tune` animals, and is **not** above it in
the slow regime — 34 of 75 joint speed × confidence × arena cells, including the
registered still × centre × low-confidence cell, have intervals overlapping the
floor. Every boundary rate in this programme now has a floor to be read against,
and that floor is measured rather than asserted.

**Does not license** any claim about whether individual boundaries are correct.
A rate above a floor says the population is not all noise; it says nothing about
which members are.

**Does not license** re-reading any published result. `SEGRECUR.md` and
`VOCAB.md` stand on the detector they were computed with
(`DETECTOR_PREREGISTRATION.md:97-100`), and this is a `tune`-split measurement
that may not be quoted as a corpus rate.

## The consequence nobody will like

**The island is 3.7× slower than its animals' other segments**
(`BEHAVIOUR.md`, speed ratio 0.272 [0.223, 0.337]). That places it squarely in
the speed regime where this stage finds the detector sitting on its own noise
floor.

That is a caveat and not a refutation, and the distinction matters:

* This stage measures a **rate**, per frame, in a stratum. The island is a set
  of **segments** selected by the recurrence of their content, not by their
  boundary density.
* `CONTEXT_CONTROLS.md` shows the island's context effect survives a
  length-matched random-window control — boundary **placement** carries
  information that random placement does not, which a pure-noise process cannot
  produce.

Both are true at once: **boundary rate in slow frames is at the noise floor,
and boundary placement still carries context information.** Nothing here
resolves that, and it is the sharpest open question this programme has.

## Cross-arm concordance — §6, and the prediction is refuted

Registered prediction 4 said raw-vs-viterbi concordance at ±2 would be **below
0.5**: `viterbi` reassigns only 0.31% of keypoint-frames, so if that moved most
of the boundaries, the boundaries were sitting on noise — an inference needing
no floor, no human and no surrogate.

**It is 0.7559 [0.7239, 0.7878]. The prediction is refuted.** The verdict is
`INCONCLUSIVE`: between the registered 0.5 limit and the 0.9 robustness bar, the
boundaries are neither clearly noise nor clearly robust to the cleaner.

60 `tune` animals, 65,115 `raw` boundaries. **`wiener` and `butterworth`
excluded by name** — a low-pass filter manufactures the smoothness whose breaks
this detects, and both are `STORED_ARMS` that cannot be applied to an array in
any case. Matched with `annot.match` / `annot.prf`, greedy nearest-first and
one-to-one; `scripts/breaks.py:165 _agreement` is not used because it counts
unmatched hits.

| pair | ±2 | ±5 | ±10 |
|---|---|---|---|
| `raw` vs **`disposition`** | **0.9542** [0.9467, 0.9612] | 0.9698 | 0.9819 |
| `raw` vs **`viterbi`** | **0.7559** [0.7239, 0.7878] | 0.8325 | 0.8927 |
| `viterbi` vs `disposition` | 0.7564 [0.7257, 0.7872] | 0.8343 | 0.8944 |

**`raw` boundaries surviving *both* other arms at ±2: 0.7212 [0.6871, 0.7553].**

### What it says, at the strength it earns

**Roughly three-quarters of boundaries are arm-invariant, and a quarter are
not.** That is neither the wholesale collapse the prediction described nor
robustness. A quarter of a detector's output moving when 0.31% of the data
changes is a real amount of instability, and it is now measured rather than
suspected.

**The disagreement is mostly placement, not existence.** Concordance rises from
0.756 at ±2 to 0.893 at ±10, so most arm-to-arm disagreement is a boundary
shifting by a few frames rather than appearing or vanishing. That is the same
shape the human raters showed — 8.8% within two frames against 85.3% within a
second — at a much finer scale. **Coarse agreement with fine disagreement is
turning up in every instrument this programme points at the question.**

**Edit magnitude matters, not edit count.** `disposition` touches ~0.5% of
frames and moves boundaries almost not at all (0.954); `viterbi` reassigns 0.31%
and moves a quarter of them. The difference is what each edit does: disposition
corrects short violation runs by geometric projection, in moves measured at
0.00004 body lengths of damage, while viterbi relocates a keypoint by a **median
46.65 px** (`CLEANING.md`). **Boundary stability tracks how far a cleaner moves
a keypoint, not how often.** A cleaning arm is not "gentle" because it edits
rarely.

**Which arm is the odd one out is answered.** `viterbi` vs `disposition` (0.7564)
is indistinguishable from `raw` vs `viterbi` (0.7559), while `raw` vs
`disposition` is 0.9542. `raw` and `disposition` produce nearly the same
boundaries and `viterbi` produces different ones — so the instability is a
property of Viterbi path selection, not a symmetric disagreement among three
equal arms.

## What is owed

1. **The order-controlled C² plant** (§1e of the plan, `DETECTOR.md:98-102`).
   The floor fixes the amplitude such a plant must stand above, which is what it
   was needed for.
3. **The registered negative control is still not discharged.**
   `DETECTOR_PREREGISTRATION.md:84-90` requires any cell clearing the 20%
   isolation gate to be run on `white` and fail there. None has cleared, so it
   has never run, and measuring a noise floor is a different test.
