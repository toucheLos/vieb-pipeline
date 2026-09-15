# Task 2 - what the alphabet throws away, and whether a symbol shares a future

Pose is continuous. Between any two poses there is another, so `N` is not
approximating a true vocabulary size the way a token count approximates a
language's -- it is **choosing a resolution**, and every symbol necessarily spans
a range of distinct positions. `LADDER.md` selected N = 256 on total description
length alone, and MDL reports a total and never splits it, so that selection
could not be read as either "cheap to encode" or "genuinely tighter prediction".

Two columns close that gap. **Distortion** is geometric: how far a frame sits
from the centroid standing in for it. **Homogeneity** is behavioural: whether the
near and far members of a symbol do the same thing next. The second is the one
that matters, because behaviour is defined by futures rather than by position --
a symbol can have high distortion and be perfectly coherent, and that is exactly
the case where coarsening is free.

| | |
|---|---|
| split | **report**, 20,783,507 non-abstained frames, 89 animals |
| pose arm | `raw` / `bodylen`, F3-carried. Every alphabet **retired** by the run-length condition |
| distortion | `D(N) = E||x - c||^2` in the standardised 17-dim space |
| homogeneity | excess total variation over a permutation null, jackknife interval |

## The table

`MDL` is the best of rungs 0-1 from `LADDER.md` (rung 2 is closed). Lower is
better for MDL and for `D`; the homogeneity excess is zero when a symbol's
members share a future.

| arm | N | MDL nats/s | D(N) | shape | twist | homogeneity excess | FDR failing |
|---|---:|---:|---:|---:|---:|---|---:|
| `plain` | 256 | 45.7 | **3.164** | 2.587 | 0.576 | +0.1574 [+0.0866, +0.2283] | 98.8% |
| `plain` | 512 | 60.2 | **2.766** | 2.265 | 0.501 | +0.1397 [+0.0899, +0.1894] | 98.4% |
| `plain` | 1024 | 81.4 | **2.425** | 1.982 | 0.443 | +0.1240 [+0.0878, +0.1603] | 96.1% |
| `plain` | 2048 | 95.7 | **2.163** | 1.762 | 0.401 | +0.1143 [+0.0925, +0.1361] | 87.7% |
| `speed` | 256 | 84.9 | **4.201** | 3.577 | 0.625 | +0.1510 [+0.0802, +0.2219] | 93.3% |
| `speed` | 512 | 99.8 | **3.615** | 3.068 | 0.547 | +0.1375 [+0.0914, +0.1836] | 91.8% |
| `speed` | 1024 | 138.8 | **3.105** | 2.662 | 0.443 | +0.1268 [+0.0945, +0.1590] | 91.3% |
| `speed` | 2048 | 168.5 | **2.690** | 2.305 | 0.385 | +0.1230 [+0.0971, +0.1489] | 88.5% |

## Reading the three columns together

**MDL falls toward coarse. Distortion rises toward coarse. Homogeneity gets
worse toward coarse.** All three move the same way and they disagree about which
end is right.

On `plain`, going from N = 2048 to N = 256 buys **95.7 to 57.0 nats/s** of
description length and pays **2.16 to 3.16** in distortion, a 46 percent rise.
That is not a gentle price: the fine alphabets were not merely resolving noise.

And the homogeneity excess rises monotonically as N falls -- **+0.114 at
N = 2048 to +0.157 at N = 256** on `plain`, with the same ordering on `speed`.
Coarser symbols are more under-resolved, which is what "under-resolved" ought to
mean and is a check that the statistic behaves.

**So the N = 256 selection is a compression preference, not a resolution
finding.** MDL is buying bits at a real cost in both geometry and behaviour, and
nothing in `LADDER.md` could have shown that.

## Every alphabet on this grid is under-resolved

All eight intervals exclude zero. Near and far runs of the same symbol differ in
what they do next by more than a permutation of their own labels produces --
**position within the cell predicts the future**, at every granularity tested.

The raw TV is **not** the effect and must never be quoted as one. It is biased
upward by roughly `sqrt(K/n)`, and the bias is visible in the table: the null
climbs from 0.358 at N = 256 to 0.479 at N = 2048 on `plain` purely because the
support widens. Bare, it would make every fine alphabet look heterogeneous and
every coarse one look coherent -- the exact opposite of the truth. The excess
over a null with the same `K`, the same `n` and the same marginal is what
cancels it.

## Where the distortion sits

Per channel at `plain`/N=256, worst five, in standardised units and in the
corpus's own (body lengths for shape, body lengths/s and rad/s for twist):

| channel | MSE (standardised) | RMS (native) |
|---|---:|---:|
| `v_y` | 0.2624 | 0.2598 |
| `s_nose_x` | 0.2603 | 0.0549 |
| `s_right_ear_x` | 0.2494 | 0.0366 |
| `s_left_ear_x` | 0.2477 | 0.0374 |
| `s_right_ear_y` | 0.2457 | 0.0353 |

The twist block carries **0.576 of 3.164** -- about a fifth of the total across
three channels against fourteen, so per channel it is the expensive part. The
worst decile of symbols carries **15.8 percent** of all the error.

## The interval took five attempts, and four of them were wrong

Recorded because each failure had a different mechanism and the same symptom:
**an interval that did not contain its own point estimate**.

1. **Bootstrap, null computed once on the full data.** `[+0.3198, +0.3350]`
   around `+0.3097`. A resample holds about 63 percent distinct runs, so plug-in
   TV rises and every replicate read high against a fixed null.
2. **Reflected into a basic interval.** When the bias exceeds half the width the
   reflected interval simply sits on the other side.
3. **Bootstrap, null redrawn per replicate.** The gap grew with `N`: `+0.1706`
   against `[+0.2553, +0.3016]` at N = 2048. A duplicated animal doubles its
   counts without adding independent runs, and a hypergeometric over those counts
   believes there are twice as many exchangeable items as there are.
4. **Subsample without replacement at m = n/2, plus a fixed comparison size.**
   The offset changed sign and shrank, and it still missed: `+0.1574` against
   `[+0.1594, +0.1827]`. **A duplicated or omitted animal's runs all land on the
   same side of the near/far split**, and that is clustering no item-level null
   reproduces, however the sizes are fixed.
5. **Leave-one-animal-out jackknife.** No animal is ever duplicated, so there is
   no clustering artifact; the sample changes by one part in 89, so the bias does
   not shift. All eight intervals bracket their estimates, and the jackknife mean
   agrees with the full-data value to five decimal places.

`tests/test_homogeneity.py::TestJackknife` pins the bracketing property under
both a true null and a real effect.

Note that the point estimate moved between attempts 3 and 4, from `+0.3097` to
`+0.1574`, because the comparison size was fixed at 100 runs per side. That is a
different and **better defined** statistic, not a correction to the old one: it
is computed at one sample size everywhere rather than at whatever each symbol
happened to have.

## The FDR fraction is reported and gates nothing

Between 88 and 99 percent of symbols reject homogeneity at FDR 0.05, carrying
essentially all the frame mass. It is in the table because it was specified, and
it **must not** be used to compare alphabets: significance scales with how much
data each symbol holds, and a symbol at N = 8 would hold roughly 200,000 runs
against 900 at N = 2048. The count would move with power rather than with
resolution, in the opposite direction to the TV plug-in bias, so the two do not
cancel. Task 3's floor is set on the effect interval.

## What this does not say

**Not that a finer alphabet would fix it.** The excess falls with `N` but is
still `+0.114` at the finest point on the grid, and `D(N)` never reaches zero for
any finite `N` on a continuous space. There may be no resolution at which these
symbols share a future.

**Not that the model learned behaviour.** The surrogate falsifier has not run.
Every number here inherits it.
