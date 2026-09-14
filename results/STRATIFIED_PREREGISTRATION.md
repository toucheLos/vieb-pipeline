# Pre-registration — the injection benchmark in the fast regime

Written and committed **before the sweep ran**. Phase F's result is banked and is
not re-scored; this adds strata it could not reach and reports them beside it.

## Why

Phase F found `median_0.50` closest to known truth — net **−0.0026** [−0.0029,
−0.0024] against `viterbi`'s −0.0017 — and a pre-registered prediction that it
would make things worse **failed**. That result stands.

What it cannot speak to is the regime the dispute is actually about. The pool is
**4.45%** of keypoint-frames and it is slow:

| | corpus | pool | ratio |
|---|---:|---:|---:|
| median centre speed | 0.1795 | 0.1040 | 0.58× |
| p90 | 0.8415 | 0.4245 | 0.50× |
| p99 | 2.2613 | 1.0034 | 0.44× |

**Only 1.81% of pool frames exceed the corpus p90 speed, where 10% would be
unbiased — fast frames are under-represented 5.5×.** And the two arms trend in
opposite directions across the strata that do exist: `median_0.50`'s benefit
falls 5× from slowest to fastest while `viterbi`'s rises from exactly zero. They
narrow from unbounded to 1.4×. Whether they cross is unmeasured.

On a fear-conditioning corpus the fast rare events are the signal, so the
unmeasured regime is the one that matters.

## The cause, and the one thing being changed

The pool requires a continuity spike **≤ 0.02 body lengths**. The continuity
residual is computed after removing the animal's rigid motion and scale, so
locomotion alone does not inflate it — but a fast animal also moves its
landmarks relative to its own body, so the criterion excludes fast frames by
construction. It admits 37.3% of frames overall.

**One parameter is swept and nothing else changes**: `truth.CLEAN_SPIKE_BL` over
`{0.02 (banked), 0.05, 0.10, off}`. Same corruptions, same rates, same seed, same
arms, same scoring, same common-denominator rule.

Relaxing it admits genuinely-wrong frames into a pool that is supposed to be
ground truth. That trade is the point and it is measured, not assumed:
**bone-violation rate inside the pool** is reported at every threshold as a
contamination proxy, since the bone check is independent of the continuity
criterion being relaxed.

## Fixed in advance

| | value |
|---|---|
| swept | `CLEAN_SPIKE_BL` ∈ {0.02, 0.05, 0.10, off} |
| unchanged | corruption kinds, rates, seed 0, arms, `MIN_SEGMENT_FRAMES = 30`, confidence floor 0.60, bone check, common-denominator scoring |
| strata | 5 speed quantiles of segment median centre speed, as now |
| **minimum stratum n** | **20,000 scored keypoint-frames**; below it the stratum is **refused**, not reported thin |
| contamination proxy | bone-violation rate within the admitted pool, per threshold, per stratum |
| bootstrap | animals, 2000 replicates |
| dominance | non-overlapping animal-bootstrap intervals |

## Predictions

1. **Pool size rises and fast frames rise faster.** At `off`, the share of pool
   frames above the corpus p90 speed goes above **5%**, from 1.81%.
2. **Contamination rises with it.** Bone-violation rate inside the pool at `off`
   exceeds **2×** its value at 0.02. If it does not, the continuity criterion was
   not protecting purity and the whole pool design should be reconsidered.
3. **The trends continue.** `median_0.50`'s net advantage keeps falling with
   speed and `viterbi`'s keeps rising, at every threshold.
4. **They cross.** In the fastest stratum with n above the registered minimum,
   `viterbi`'s net is **below** `median_0.50`'s — with non-overlapping intervals.
   **This is the falsifier: if `median_0.50` still wins in the fastest adequately
   powered stratum, the movement-retention argument against smoothing is wrong on
   this corpus, and the smoother is the right default without qualification.**
5. **`median_0.50` remains the best arm pooled**, at every threshold. The pooled
   number is not where the disagreement lives and it is not expected to move.

## What this may not do

* It may not re-score or restate Phase F. `results/injection.json` stands as the
  pre-registered result at threshold 0.02.
* It may not change the corruption model. `PARK_FRAMES = (4, 30)` is known to be
  drawn from the post-filter run-length distribution rather than the pre-filter
  one (`RUNLEN.md`); that defect is recorded and **not fixed here**, because
  changing the corruption model needs its own registration.
* It may not report a stratum below 20,000 scored keypoint-frames.
* It may not re-score Q1.

## The limitation that survives the fix

Relaxing the criterion widens the pool; it does not make the pool representative.
Even at `off`, entry still requires the bone check, no missing keypoint, and DLC
confidence ≥ 0.60 — and **confidence is the binding constraint at 10.55%**, not
continuity. A pool gated on confidence is a pool of frames the detector found
easy, and the fastest frames are not those. So this measures further into the
fast regime than Phase F could, and still not all the way.
