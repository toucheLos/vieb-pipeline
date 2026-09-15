# Step 1 — where the trajectory stops being smooth

**This is a statistic, not a finding.** A changepoint detector fires on noise,
and a boundary rate with no null beside it is not evidence of anything. Step 2
calibrates the threshold against the surrogate families and is the gate. Nothing
here is a claim that these boundaries are real.

| | |
|---|---|
| criterion | one-sided acceleration mismatch, ported from `exbias.py:69-150` |
| input | 17-dim egocentric, **`raw` / `bodylen`**, asserted by name at load |
| standardisation | `basis.json` `sd_used` — segment-balanced, corpus-pooled, tune-fitted |
| split | **report**, 89 animals |
| threshold | `k_mad = 3.0`, a **reference** with no stated false-positive rate |

The arm is asserted rather than assumed because a low-pass filter manufactures
exactly the smoothness whose breaks this detects, and Wiener's effect is five
times larger in the twist channels (0.0179) than in the pose block (0.0037).

## The sweep

18 cells: three channel groups × three derivative half-windows × two polynomial
degrees. The primary cell is starred.

| group | deriv (s) | degree | peaks | segments | boundaries/s | median dur (s) | median clean (s) | mean adj R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `both` | 0.067 | 2 | 127,486 | 280,891 | **0.573** | 0.300 | 0.433 | 0.3704 |
| `both` | 0.067 | 3 | 111,382 | 265,517 | **0.500** | 0.267 | 0.500 | 0.4375 |
| `both` | 0.133 | 2 | 100,204 | 256,068 | **0.450** | 0.233 | 0.467 | 0.3567 |
| `both` **| 0.133 | 3 | 89,317 | 245,613 | **0.401** | 0.233 | 0.533 | 0.4264 |
| `both` | 0.267 | 2 | 63,722 | 222,609 | **0.286** | 0.167 | 0.467 | 0.3425 |
| `both` | 0.267 | 3 | 57,948 | 217,027 | **0.260** | 0.133 | 0.433 | 0.4129 |
| `shape` | 0.067 | 2 | 139,553 | 293,451 | **0.627** | 0.300 | 0.433 | 0.3684 |
| `shape` | 0.067 | 3 | 121,293 | 275,879 | **0.545** | 0.300 | 0.533 | 0.4345 |
| `shape` | 0.133 | 2 | 111,018 | 266,525 | **0.499** | 0.233 | 0.467 | 0.3564 |
| `shape` | 0.133 | 3 | 98,660 | 254,734 | **0.443** | 0.217 | 0.533 | 0.4262 |
| `shape` | 0.267 | 2 | 71,356 | 229,836 | **0.321** | 0.167 | 0.500 | 0.3457 |
| `shape` | 0.267 | 3 | 64,834 | 223,533 | **0.291** | 0.167 | 0.467 | 0.4150 |
| `twist` | 0.067 | 2 | 143,226 | 297,998 | **0.643** | 0.367 | 0.433 | 0.3657 |
| `twist` | 0.067 | 3 | 124,664 | 280,133 | **0.560** | 0.333 | 0.533 | 0.4341 |
| `twist` | 0.133 | 2 | 109,046 | 265,792 | **0.490** | 0.300 | 0.500 | 0.3595 |
| `twist` | 0.133 | 3 | 97,186 | 254,398 | **0.437** | 0.267 | 0.567 | 0.4268 |
| `twist` | 0.267 | 2 | 80,361 | 239,563 | **0.361** | 0.233 | 0.567 | 0.3501 |
| `twist` | 0.267 | 3 | 73,010 | 232,401 | **0.328** | 0.200 | 0.567 | 0.4188 |

Boundary rate spans **0.260 to 0.643 per second** across the whole sweep — a
factor of 2.5, and every cell is within a factor of 2.5 of every other. Degree 3
gives a consistently higher adjusted R² than degree 2 (0.41–0.44 against
0.34–0.37), which is what more parameters buy and not evidence about the corpus.

## Two things the table hides, and both matter

**Abstain creates more boundaries than the detector does.** At the primary cell
on one animal: **1,081 detected peaks against 2,532 segments**. Abstain is 7.03%
of frames but arrives in short runs, so it fragments the stream far more than it
covers it. A median duration over all segments is therefore mostly a statement
about tracking failure — `0.233 s` against `0.533 s` for segments containing no
abstained frame. Both columns are in the table; the clean one is what the
detector actually delimits.

**The channel groups agree on rate and disagree on when.** Their boundary rates
sit within 1.5% of each other at the primary window. Their boundary *sets*, at
±2 frames, do not:

| pair | mean Jaccard |
|---|---:|
| `both` vs `shape` | 0.443 |
| `both` vs `twist` | 0.436 |
| **`shape` vs `twist`** | **0.226** |

**Shape breaks and twist breaks are largely different events** — under a quarter
of them coincide. Reported rather than resolved by fiat, as registered: a
posture change and a turn are not obviously the same kind of boundary, and
picking one channel group would have hidden that they are not.

## Against ExBias, as context and not as comparison

`exbias_002` reports 409,812 segments at a 700 ms median and mean adjusted R²
0.6404. This stage gives 245,613 at 0.533 s clean median and
0.4264. The numbers are **not comparable** and should not be read as one:
ExBias fitted 14 raw-DLC pose channels in alphabetical bodypart order, upstream
of the gap policy, the swap correction and every QC mask; this fits 17
standardised egocentric channels including three velocity channels on the
F3-carried arm. The velocity block is the noisiest part of the representation
and is exactly what a polynomial fits worst.

What *is* comparable is the order of magnitude: both put a boundary roughly
every one to two seconds, and both sit far below the ">90% a cleanly
piecewise-smooth process would give" that ExBias's own C4 calibration names as
the standard the axiom would meet if it held cleanly.

## What is deliberately absent

No minimum-duration rule, no stickiness prior, no smoothing of the boundary
series — that imports persistence, which is what retracted the dwell result. The
only length floor is `identifiability_floor`, a property of the estimator: a
degree-3 polynomial fitted to fewer than four samples per channel is
interpolation whose R² is 1 by construction.

Non-maximum suppression is the detector's **resolution limit**, not a duration
prior, and runs strongest-peak-first — a left-to-right rule lets an earlier
weaker peak suppress a later stronger one, putting the boundary on noise. `D` is
written out unsmoothed beside the boundaries so both choices are auditable.

## What happens next

`work/tok/breaks/<tag>_D.npz` holds the primary cell's statistic per animal.
Step 2 generates the four surrogate families through the identical pipeline,
runs the separability check on each, sweeps the threshold, and compares boundary
rates. **If the corpus rate sits inside the surrogate range across that sweep,
this route closes** as cleanly as the falsifier closed the tokenizer.
