# The trend filter refuses — grid-limited, unconverged, and the bar was mis-set

Registered in `results/TRENDFILTER_PREREGISTRATION.md`, committed before
`vieb/seg/trendfilter.py` or `scripts/trendfilter.py` existed. **`tune` split,
60 animals**, F3 `raw` arm, `shape` group. λ scale changed from §3's formula and
recorded as `DEVIATIONS.md` **D13**. Digest `198eb14ff258c7f6`.

## The verdict

> **`GRID_LIMITED`. No λ\* was selected, so the plant comparison never ran and
> predictions 2–5 are untested.** Prediction 1 — that λ\* exists on the grid —
> **failed**.

Refusal is a registered outcome and this is one. Two independent things went
wrong, and only one of them is about trend filtering.

## The sweep

| α | corpus rate | jitter-only rate | **jitter share** | **solver converged** |
|---:|---:|---:|---:|---:|
| 0.5 | 12.983/s | 10.024/s | 0.7751 | **0.95** |
| 1 | 10.617/s | 7.723/s | 0.7300 | 0.73 |
| 2 | 8.660/s | 5.969/s | 0.6924 | 0.21 |
| 5 | 6.567/s | 4.118/s | 0.6310 | 0.03 |
| 10 | 5.251/s | 2.908/s | 0.5588 | **0.00** |
| 20 | 4.088/s | 1.498/s | 0.3714 | 0.00 |
| 50 | 2.794/s | 0.370/s | 0.1360 | 0.00 |
| 100 | 2.058/s | 0.140/s | **0.0704** | 0.00 |

**Reason 1 — grid-limited.** The registered bar is a jitter share below **0.05**.
The best point on the grid is **0.0704**, at the sparse end. §3 forbids
extending the grid after seeing where the answer falls, and it is not extended.

**Reason 2 — the solver does not converge where the answer is.** ADMM converges
on 95% of recordings at α = 0.5 and on **0%** from α = 10 upward. Every α whose
jitter share is even close to the bar sits in a region the solver could not
reach, so those rows are not measurements regardless of the grid. This is a
statement about **this implementation**, not about trend filtering: a
group-sparse `D³` problem is badly conditioned, and banded ADMM with residual
balancing is not enough solver for its sparse end.

The registration forbids raising the iteration budget against an outcome
(§6), and raising it until a number appears is precisely that. The budget was
4,000 iterations with warm starts down the α path, fixed before the run.

## The bar was set without reference to the incumbent, and that is my error

The 5% was invented in §3 and named there. **What §3 never did was check what
the detector being replaced scores on it.**

| | jitter share |
|---|---:|
| **frozen detector** (`NOISEFLOOR.md`: 0.2599 / 0.4746) | **0.5475** |
| trend filter, best grid point (α = 100) | **0.0704** |
| the registered bar | 0.05 |

**The incumbent fails the bar by a factor of 11.** The trend filter reaches a
share **7.8× better than the frozen detector** and is still refused. A bar that
the thing you are replacing misses by an order of magnitude is not a bar, it is
an unexamined constant — and it was registered in advance, so it stands.

**This is why the number is not moved.** Moving it now, having seen that the
challenger lands at 0.0704, would be choosing a threshold to admit a result.
The bar stays, the stage refuses, and the mis-setting is reported as
`METHODS_FINDINGS.md` **M12** rather than quietly corrected.

## What is descriptively true, and carries no verdict

Both of these are read off unconverged solves above α = 5 and license nothing:

* **The jitter share falls monotonically**, 0.775 → 0.070 across the grid. The
  direction is what an error model is supposed to buy, and it is large.
* **Every α tested cuts far more often than the frozen detector.** At α = 100
  the corpus rate is 2.058/s against 0.443/s — still **4.6×**. The operating
  point that would match the incumbent's rate lies beyond the registered grid.

Taken together these say the estimator is plausibly doing what it was built to
do — suppressing jitter-driven knots much harder than a twice-differenced
threshold — at a scale the grid does not reach, on solves that have not
converged. **None of that is evidence.** It is the reason the next attempt is
worth making, not a result from this one.

## What did not happen

**The plant comparison did not run.** It needed λ\*, and there is no λ\*.
`PLANT.md`'s numbers for the frozen detector stand alone and have no challenger.
Nothing here says whether estimating the break beats differencing toward it —
**prediction 2, the headline claim, is untested.**

Nothing published moved. `SEGRECUR.md`, `VOCAB.md` and `CONTEXT.md` stand on the
detector they were computed with, as
`DETECTOR_PREREGISTRATION.md:97-100` requires. Segment recurrence, clustering
and coverage were not restarted, as §6 forbids.

## What is owed

1. **A solver that reaches the sparse end.** The binding constraint is
   numerical. Specialised trend-filter algorithms exist for the univariate
   ℓ1 case; the group-sparse variant needs either a better splitting, a
   preconditioner, or a path algorithm. Until one converges above α = 10 there
   is no measurement to be had here at any grid.
2. **A re-registration with a bar calibrated against the incumbent.** The next
   version should ask the challenger to beat **0.5475** — what the frozen
   detector actually achieves — by a stated margin, rather than to clear an
   absolute constant nothing in this programme has ever met.
3. **A grid that brackets the incumbent's rate.** Every α tested cuts 4.6× to
   29× more often than the frozen detector. A sweep that never reaches the
   comparison point cannot produce a comparison.

All three are one registration, and none of them may be written by editing this
stage's.
