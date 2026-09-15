# Step 0 — the dwell-matched surrogate. The corpus wins, and the falsifier's statistic was measuring run rate.

Registered in `DWELL_PREREGISTRATION.md` (commit `45a7160`) before any surrogate
was generated, amended with a construction-validity check (`eb8f100`) before any
ladder number existed.

## The result

Both dwell-matched arms **PASS** at both alphabet sizes: the corpus's
`rung1 − rung0` exceeds the surrogate's, animal-level at n = 89, interval
excluding zero.

| arm | N | runs/s vs corpus | advantage n/s | advantage n/run | gap n/s, animal-level |
|---|---:|---:|---:|---:|---|
| corpus | 256 | 1.000 | +11.32 | **+1.614** | — |
| `phase` | 256 | 2.078 | +22.48 | **+1.419** | -11.208 [-11.753, -10.666] |
| `var5` | 256 | 2.102 | +22.93 | **+1.427** | -11.653 [-12.167, -11.143] |
| `microstate` | 256 | 1.739 | +10.01 | **+0.845** | +1.317 [+0.899, +1.728] |
| `microstate0` | 256 | 1.459 | +1.26 | **+0.297** | +10.045 [+9.370, +10.710] |
| corpus | 512 | 1.000 | +8.95 | **+1.953** | — |
| `phase` | 512 | 2.032 | +22.65 | **+1.699** | -13.756 [-14.389, -13.070] |
| `var5` | 512 | 2.040 | +22.86 | **+1.703** | -13.956 [-14.568, -13.276] |
| `microstate` | 512 | 1.619 | +5.06 | **+0.971** | +3.888 [+3.300, +4.483] |
| `microstate0` | 512 | 1.382 | -4.10 | **+0.428** | +13.031 [+12.163, +13.876] |

Per the registered reading: **the corpus still wins, so dwell carries structure
beyond first-order dwell statistics, and segmentation has a target.**

## The thing this actually settled

`FALSIFIER.md` recorded a run-rate confound it could not resolve with two arms.
Four arms resolve it. Sorted by how much faster the surrogate's symbol stream
turns over, at N = 256:

| arm | runs/s vs corpus | gap n/s |
|---|---:|---:|
| `var5` | 2.102 | **-11.65** |
| `phase` | 2.078 | **-11.21** |
| `microstate` | 1.739 | **+1.32** |
| `microstate0` | 1.459 | **+10.04** |

**The gap is monotone in the run-rate ratio and changes sign inside it.** A
surrogate with roughly twice the corpus's transitions per second beats it; one
with 1.46× loses to it by ten nats per second. That is not four facts about four
surrogates — it is one fact about the statistic.

> **Under a per-second statistic, the verdict tracks how fast the surrogate's
> symbol stream turns over, not how much structure it has.**

The per-run column needs no such argument. **The corpus beats every one of the
four arms per run**, at both N, and the ordering is exactly what the
constructions predict: `microstate0` lowest (dwell kept, order destroyed), then
`microstate` (dwell and one-step kept), then `phase` and `var5` (spectrum kept,
dwell destroyed), then the corpus.

## What is and is not withdrawn

**`FALSIFIER.md`'s verdict stands as registered.** Per-second was the registered
statistic there, the phase and VAR arms lost under it, and
`DWELL_PREREGISTRATION.md` §2 registered per-run as primary **for future work
only, explicitly not retroactively**. That commitment is kept here. A reversal
available only under a statistic chosen after seeing the result is not a
reversal, and this document does not perform one.

**What is new is evidence about the statistic itself**, not about the past
numbers. The monotone table above was not available when `FALSIFIER.md` was
written; it is now, and it bears directly on whether per-second was the right
choice. `FALSIFIER.md` already said that decision needs its own registration and
belongs to the project owner. It still does.

**`LADDER.md`'s P1 remains withdrawn.** It was withdrawn on the registered
statistic against the registered nulls, and nothing here un-withdraws it.

## The two arms separate, which is the informative part

At N = 256, holding dwell fixed and destroying order costs the surrogate almost
everything — `microstate0`'s advantage is **+1.264 nats/s against the corpus's
+11.316**. Restoring one-step visit dynamics recovers most of it back:
`microstate` reaches **+10.008**.

So once dwell is held fixed, **sequence matters, and most of what it carries is
first order**. That is consistent with `LADDER.md`'s k\* = 0 and with rung 2
buying nothing, and it is the first positive evidence in this programme that the
symbol *order* carries anything at all.

> **`microstate` must be read weakly.** It preserves one-step visit dynamics and
> rung 1 **is** a one-step model, so the two overlap by construction and its
> +1.317 gap is a lower bound rather than a measurement. Its verdict string says
> so. `microstate0` carries no such overlap and is the clean arm.

## What is weak here, stated plainly

**The arms are not dwell-matched exactly.** Realised run-rate ratios are 1.74
(`microstate`) and 1.46 (`microstate0`) against phase's 2.08 — much closer, not
equal. The pilot in the registration's amendment measured 1.36 through the
corpus's own tokenizer as a common ruler; the ladder refits each arm's own
partition, which moves the number. Both are reported rather than the flattering
one.

**The residual runs in the surrogate's favour, which makes the result
conservative.** A higher run rate inflated phase and VAR's per-second advantage,
and `microstate0` has 1.46× the corpus's run rate too — it had the same thumb on
the scale and lost by ten nats per second anyway.

**A win does not mean the model learned behaviour.** Four alternatives are four
alternatives. Both dwell arms are built from the corpus's own frames, which is
what makes them tight on dwell and says nothing about anything else.

## Provenance

17-dim egocentric, `raw`/`bodylen`, F3-carried. Null partition `N = 500` fitted
on the corpus's tune split (1,205,769 visit transitions, prior share 0.1717,
hash `3c9a806dd84e3cd1`); the tokenizer's own partition is fitted separately on
each surrogate's tune split. `scripts/quantize.py` and `scripts/ladder.py` ran
**unmodified** on redirected paths. Corpus abstain mask carried over unchanged.
Every alphabet here is **retired** by the run-length condition and every number
inherits that.
