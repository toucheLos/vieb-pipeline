# F2 — the injection benchmark in the fast regime

89 report animals, 298 shards per arm, four pool thresholds, four arms.
Registered in `STRATIFIED_PREREGISTRATION.md` before the sweep ran. Corrected
after the F3 audit; the corrections are in `DEVIATIONS.md` and are named below.

## The headline: `median_0.50`'s win is confined to slow frames

Per-stratum net error against known truth, slowest → fastest, five strata, all of
them above the registered 20,000-keypoint-frame minimum (smallest is 472k).
**Negative is better; positive means the arm left the data further from the truth
than it found it.**

| threshold | arm | pooled | s1 | s2 | s3 | s4 | s5 |
|---|---|---:|---:|---:|---:|---:|---:|
| **0.02** | `median_0.50` | −0.0026 | −0.0029 | −0.0028 | −0.0022 | −0.0023 | −0.0018 |
| | `viterbi` | −0.0016 | −0.0014 | −0.0017 | −0.0016 | −0.0019 | −0.0018 |
| | `disposition` | −0.0013 | −0.0011 | −0.0014 | −0.0013 | −0.0014 | −0.0015 |
| **0.05** | `median_0.50` | −0.0007 | −0.0022 | −0.0012 | −0.0005 | **+0.0004** | **+0.0041** |
| | `viterbi` | −0.0016 | −0.0014 | −0.0015 | −0.0017 | −0.0018 | −0.0025 |
| | `disposition` | −0.0012 | −0.0011 | −0.0011 | −0.0013 | −0.0015 | −0.0017 |
| **0.10** | `median_0.50` | −0.0001 | −0.0021 | −0.0009 | +0.0000 | **+0.0017** | **+0.0068** |
| | `viterbi` | −0.0016 | −0.0014 | −0.0016 | −0.0016 | −0.0018 | −0.0025 |
| | `disposition` | −0.0012 | −0.0010 | −0.0012 | −0.0013 | −0.0014 | −0.0017 |
| **off** | `median_0.50` | −0.0000 | −0.0021 | −0.0008 | +0.0001 | **+0.0020** | **+0.0068** |
| | `viterbi` | −0.0016 | −0.0014 | −0.0016 | −0.0016 | −0.0018 | −0.0025 |
| | `disposition` | −0.0012 | −0.0010 | −0.0012 | −0.0013 | −0.0013 | −0.0017 |

Stratum sizes at `off`: 1.90M, 1.30M, 1.09M, 0.74M, 0.47M keypoint-frames.
Unplaceable frames excluded: 235. No stratum refused.

**The per-stratum crossing is the result, and it does not depend on the pool
being representative.** Once the pool admits fast frames at all, `median_0.50`
goes positive in the top two strata while `viterbi` reaches its best there. The
two arms move in opposite directions across the speed range at every relaxed
threshold.

**The pooled row does depend on pool composition** and is reported second for
that reason. It is the frame-weighted sum over the strata — verified to six
decimal places, `DEVIATIONS.md` D2 — and the strata differ in size by 4×, so the
pooled figure is dominated by the slow strata where the median still wins.

## Threshold-invariance: true for two arms, false for the third

Range of each stratum's net across the four thresholds:

| arm | s1 | s2 | s3 | s4 | s5 | max |
|---|---:|---:|---:|---:|---:|---:|
| `viterbi` | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0007 | **0.0007** |
| `disposition` | 0.0001 | 0.0002 | 0.0000 | 0.0002 | 0.0002 | **0.0002** |
| `median_0.50` | 0.0009 | 0.0020 | 0.0023 | 0.0043 | 0.0087 | **0.0087** |

**`viterbi` and `disposition` are threshold-invariant.** Their profiles move by
at most 0.0007 and 0.0002 against nets of ~0.0016 — so for these arms the entire
pooled threshold effect **is** compositional: the arms behave identically, the
pool merely admits different frames. `viterbi`'s exact pooled invariance at
−0.0016 was a hypothesis this table tests, and it holds.

**`median_0.50` is not.** Its swing is 0.0087, larger than any of its own net
values. Its stratum-5 net runs −0.0018 → +0.0068 across the sweep.

## The post-stratified estimate is REFUSED, and the reason is structural

Task C asked for per-stratum nets reweighted to the corpus marginal of centre
speed. **It cannot be computed**, for three reasons, each naming what would have
to change:

**1. The stratifying variable does not exist outside the pool.** Strata are
quantiles of **segment median centre speed**, and a segment is *defined by the
pool criterion* — a contiguous run of frames that passed the bone check, the
continuity threshold, the presence check and the confidence floor. The corpus has
no segment-median-speed marginal to reweight to, because outside the pool there
are no segments. Fixing this means stratifying on a variable the corpus also has
— per-frame speed — which is a different stratum definition and needs its own
registration.

**2. The bin edges move with the threshold**, so the strata are not the same
population across the sweep. Median edges, body lengths per second:

| threshold | s1 lower | s5 upper |
|---|---:|---:|
| 0.02 | 0.0008 | **0.143** |
| 0.05 | 0.0009 | 0.413 |
| 0.10 | 0.0009 | 0.487 |
| off | 0.0009 | **0.487** |

The fastest stratum at `off` reaches **3.4× faster** than the fastest stratum at
0.02. Comparing "stratum 5" across thresholds compares different speed ranges.
This is also why `median_0.50` fails the invariance test above while the other
two pass: its net is strongly speed-dependent, so moving edges bite; theirs are
nearly flat, so they do not.

**3. Even at `off` the pool never reaches the corpus's fast tail.** The widest
pool's fastest segment-median edge is 0.487 bl/s; the corpus's per-frame centre
speed has p90 = 0.8415 and p99 = 2.2613. Any reweighting to the corpus marginal
would **extrapolate** the arms' behaviour past the fastest frames ever measured,
into precisely the regime the estimate exists to characterise.

Reporting a post-stratified number under these conditions would put the
least-supported figure in the most load-bearing position. **Refused.** The
per-stratum crossing stands on its own and needs no reweighting.

## The five predictions

| # | prediction | outcome | |
|---|---|---|---|
| 1 | fast share above 5% at `off` | **2.74%**, from 0.95% | **failed** |
| 2 | trunk contamination at `off` > 2× its 0.02 value | 0.000% → 0.026% | **unfalsifiable as written** |
| 3 | median's advantage falls with speed, Viterbi's rises | monotonic in both, every relaxed threshold | **held** |
| 4 | they cross in the fastest adequately powered stratum | they cross at s3–s4; the median goes **positive** | **held** |
| 5 | `median_0.50` best pooled at every threshold | beaten from 0.05 up | **failed** |

**Prediction 1 failed for the reason the registration named in advance:**
continuity was never the binding constraint — **confidence at 10.55% is**.
Turning continuity fully off reaches only 2.74% fast share against 10% unbiased,
and 0.10 and `off` are near-identical because continuity saturates by 0.10.

**Prediction 2 was badly specified by me.** The contamination baseline is
**0.000%**, so "exceeds 2×" is satisfied by any positive number and cannot fail.
What the measurement says: trunk violations inside the pool rise from 0.000% to
0.026%, against a 2.40% corpus-wide trunk rate. **The pool stays clean as it
widens**, which is the substantive point.

## Corrections made during the F3 audit

Three things in the first version of this document were wrong. All are in
`DEVIATIONS.md`; the two that were reported as findings are withdrawn here.

* **"`viterbi` is inert in the slowest stratum, net exactly 0.0000."**
  **Withdrawn.** That 0.0000 was `assign_bin`'s **refusal bucket** (bin −1, 235
  keypoint-frames), sorted first and printed as a stratum. Viterbi's real
  slowest-stratum net is **−0.0014**.
* **"`median_0.50` runs −0.0331 in the slowest stratum."** **Withdrawn.** Same
  bucket. The real value is **−0.0021**.
* **The pooled/per-stratum arithmetic "did not close."** It closes exactly. The
  pooled figure is the frame-weighted stratum sum; the comparison had been
  against an *unweighted* mean, which for these strata has the **opposite sign**
  (+0.0012 against −0.00004).

The crossing survives all three corrections, and is cleaner without the spurious
bucket.

## Also corrected

Phase F's per-recording corruption seed was salted per process
(`abs(hash(...))`), so `SEED = 0` controlled nothing. Fixed with blake2b, audited
as a class in `SEED_AUDIT.md`, pinned by `tests/test_seeds.py`. All four arms
re-run under the stable seed; Phase F's figures move ~1% relative and **no
verdict, sign or ordering changed**.

`recovery_read` had never issued a negative before flagging `median_0.50`. A
true-negative control on `viterbi` at `off` is now pinned, with a positive
control beside it so it cannot pass vacuously.
