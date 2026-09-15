# Task 1 - is the falling hazard duration dependence, or mixing?

`LADDER.md` reports a hazard that falls steeply with elapsed time and **refuses**
to read it as evidence that a bout becomes harder to leave the longer it lasts.
A falling pooled hazard is also exactly what unmodelled heterogeneity produces:
the runs still alive at large elapsed time are increasingly the long-lived kind,
so the population's hazard falls because its composition changes. That is
frailty, and it is the failure class that retracted the dwell result.

**Answer: the fall largely survives conditioning. It is not principally a speed
mixture.**

| | |
|---|---|
| cell | `plain` / N = 256, the MDL-selected alphabet (**retired** by the run-length condition) |
| split | **report** - 89 animals, 1,747,514 runs |
| pooled fall | **175.7x**, bins 0 to 12 |
| conditioned on run speed | **0.756 [0.742, 0.769]** of the log fall survives |
| verdict | **PASS** - the interval is entirely above the 0.50 floor |

The pooled figure here is 175.7x where `LADDER.md` reports 148x. That is not
a discrepancy: `LADDER.md`'s came from the **fit**-split model, and this is
recomputed on **report** so that the bootstrap sits on the 89 animals every other
reported interval sits on. Both are stated rather than reconciled away.

## The pooled hazard, with the counts that make it readable

A flat curve computed on twelve runs is not a flat hazard, so every bin carries
its realised exit count and a cell below 50 exits is marked unreportable
rather than plotted.

| bin | elapsed (frames) | hazard | exits |
|---:|---|---:|---:|
| 0 | 0-0 | 0.5514 | 935,236 |
| 1 | 1-1 | 0.3980 | 297,859 |
| 2 | 2-2 | 0.3258 | 144,897 |
| 3 | 3-4 | 0.2572 | 131,256 |
| 4 | 5-7 | 0.1771 | 70,530 |
| 5 | 8-12 | 0.1157 | 40,324 |
| 6 | 13-20 | 0.0669 | 20,084 |
| 7 | 21-33 | 0.0375 | 10,686 |
| 8 | 34-56 | 0.0227 | 6,773 |
| 9 | 57-93 | 0.0168 | 4,459 |
| 10 | 94-155 | 0.0105 | 2,489 |
| 11 | 156-258 | 0.0069 | 1,339 |
| 12 | 259+ | 0.0031 | 1,223 |

## The correction that reversed this result

The first version of this measurement reported **"THE FALL IS SUBSTANTIALLY
FRAILTY"**, `retained = 0.431`. That was wrong, and the mechanism is worth
stating because it is not obvious.

Each speed quintile was compared against the **full-range** pooled fall. But a
fast quintile's runs never last long enough to populate the late elapsed bins:
the fastest spans bins 0-5 where the pool spans 0-12. Comparing them compares a
6-bin fall against a 13-bin fall, and the slice looks flatter because its range
is shorter, not because its hazard is.

The tell was visible in the per-slice table and was missed on the first reading:
the **slowest** quintile spans the same 0-12 as the pool and falls 143.7x
against the pool's 175.7x - barely reduced at all.

Every fall is now taken over the elapsed-bin window reportable in **both** the
slice and the pool, and the pooled fall is recomputed over those same two bins.
`tests/test_frailty.py::TestTheCommonSpan` pins it: a slice whose hazard equals
the pool's over a short range must read as retaining *all* of the fall, and the
unrestricted comparison scored that case at 5/12.

## Speed quintiles, like for like

| quintile | speed | span | fall within | pooled, same span | exits |
|---:|---|---|---:|---:|---:|
| 0 | slowest | 0-12 | **143.66** | 175.71 | 370,727 |
| 1 | 2nd | 0-11 | **42.07** | 80.39 | 343,071 |
| 2 | 3rd | 0-8 | **9.38** | 24.33 | 333,068 |
| 3 | 4th | 0-7 | **5.93** | 14.71 | 321,578 |
| 4 | fastest | 0-5 | **1.96** | 4.77 | 298,711 |

Exit-weighted, the fall is **16.2x within quintiles against 32.7x
pooled over the same spans**. On a log scale that is 75.6% retained, and the
animal-level interval excludes the floor.

Run speed is a **run-level** covariate - the mean of `quantize.speed` over the
run's own frames. Per frame it would reintroduce the frame-mass bias run-length
encoding exists to remove: a long freeze contributes hundreds of slow frames and
a brief dart a handful of fast ones, so the quintile edges would describe how the
animal spends its time rather than what it is doing. Edges are cut on **tune**.

## State x animal: inconclusive, and the reason is thinness

0.538 [0.465, 0.610] retained, straddling the floor -
**INCONCLUSIVE**. Only **116 of 890** (state, animal) slices reach
50 exits in two bins, and those that do span very few bins: the fall is
1.2x within against 1.4x pooled *over the same spans*, both so close
to 1 that their log ratio is unstable. 51 of 89 animals were usable.

This is the outcome the brief anticipated for this slice. It is reported as
unresolvable rather than as a flat hazard, because **a flat curve on a handful of
runs is not a flat hazard**.

## What this licenses, and what it still does not

**It licenses removing one specific alternative.** The falling hazard is not
principally an artifact of mixing fast and slow runs. Roughly a quarter of the
log fall is that mixture; three quarters is not.

**It does not license the per-bout reading, and `LADDER.md`'s refusal stands.**
Frailty is about *unobserved* heterogeneity, and ruling out one observed
covariate does not rule out the unobserved ones. A mixture along some axis not
measured here would look exactly like this. The `PASS` above says the fall
survives *this* conditioning, and its own reason string says so.

**It does not make the alphabet good.** `plain`/N=256 is retired by the
run-length condition; everything here inherits that.
