# The falsifier fired. The transition-table claim is withdrawn.

`FALSIFIER_PREREGISTRATION.md` was committed before any surrogate was generated.
Its stopping rule, quoted from §7:

> | corpus lower | `FAIL` | worse than the null; every ladder number is withdrawn |

**The corpus is lower.** Both surrogates beat it on the registered statistic, at
both alphabet sizes, with animal-level intervals excluding zero.

| surrogate | N | corpus | surrogate | gap, animal-level n=89 | verdict |
|---|---:|---:|---:|---|---|
| `phase` | 256 | +11.300 | +22.508 | **-11.208 [-11.753, -10.666]** | FAIL |
| `var5` | 256 | +11.300 | +22.952 | **-11.653 [-12.167, -11.143]** | FAIL |
| `phase` | 512 | +8.917 | +22.673 | **-13.756 [-14.389, -13.070]** | FAIL |
| `var5` | 512 | +8.917 | +22.873 | **-13.956 [-14.568, -13.276]** | FAIL |

The claim at risk was `LADDER.md`'s headline: **rung 1 beats rung 0 by +11.3
nats/s at `plain`/N=256, better on 89 of 89 animals** — the transition table
carries information the marginal does not. A phase-randomised signal with the
same power spectrum, and a VAR(5) fitted to each recording, both get **twice**
that advantage on the identical pipeline.

## The decomposition, which I did not anticipate and which matters

| source | N | runs/s | rung 0 n/s | rung 1 n/s | advantage n/s | rung 0 n/run | rung 1 n/run | advantage n/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corpus | 256 | 8.46 | 57.0 | 45.7 | **+11.32** | 6.742 | 5.127 | **+1.614** |
| `phase` | 256 | 17.57 | 109.4 | 86.9 | **+22.48** | 6.224 | 4.806 | **+1.419** |
| `var5` | 256 | 17.78 | 110.1 | 87.2 | **+22.93** | 6.194 | 4.766 | **+1.427** |
| corpus | 512 | 9.43 | 69.1 | 60.2 | **+8.95** | 7.329 | 5.376 | **+1.953** |
| `phase` | 512 | 19.16 | 130.3 | 107.7 | **+22.65** | 6.801 | 5.101 | **+1.699** |
| `var5` | 512 | 19.24 | 130.6 | 107.7 | **+22.86** | 6.784 | 5.081 | **+1.703** |

**The surrogates have 2.1x as many runs per second.** Phase randomisation
preserves the power spectrum but Gaussianises the signal, and a Gaussianised
mouse never sits still: the corpus's tune p99.9 run length is 259 frames and the
phase surrogate's is 34. So the corpus dwells, and the surrogate does not.

The registered statistic is **nats per second**, and a signal with twice as many
transitions per second has twice as many of everything per second. That is a
run-rate confound in the registered comparison, of exactly the shape that
reversed the frailty reading — two quantities measured over different
denominators.

**Three framings, and they do not agree:**

* **per second, as registered** — corpus 11.32 against 22.48 and 22.93. The
  corpus loses roughly two to one.
* **per run** — corpus **1.614** against 1.419 and 1.427. The corpus **wins**, by
  about 14 percent, and by 15 percent at N = 512.
* **as a fraction of rung 0** — corpus 19.84 percent against 20.55 and 20.82 at
  N = 256; 12.94 against 17.38 and 17.51 at N = 512. The corpus loses.

Two of the three go against the corpus, and the registered one is among them.

## What is withdrawn, and what is not

**Withdrawn: `LADDER.md`'s reading of rung 1 over rung 0 as evidence that the
symbol sequence carries behavioural information.** That was the claim the
falsifier was registered against and it failed on the registered statistic. It
may not be quoted as evidence of sequential structure.

**Not withdrawn, because none of it was the claim at risk and none depends on
it:** rung 2's failure to beat rung 1 (a comparison between two models of the
corpus, unaffected by how either compares to a surrogate); `k* = 0`; the hazard
shape; `FRAILTY.md`; `RESOLUTION.md`. Each stands on its own footing and each
already carries its own caveats.

## What the falsifier did NOT find, and this is the substantive part

The original falsifier was worded *"if surrogates achieve comparable MDL"*. They
do not, and not remotely: the corpus costs **45.7 nats/s** where its phase
surrogate costs **86.9**. The corpus is described in half the bits.

But per run the corpus is **more** expensive — 5.127 against 4.806. Every bit of
its compressibility advantage is that it has fewer transitions per second to
encode.

> **The corpus's advantage over a spectrum-matched surrogate is dwell duration,
> and nothing else.** Per transition, its symbol stream is no more predictable
> than a structureless signal's — slightly less. The thing that distinguishes a
> mouse from phase-randomised noise here is that a mouse holds still.

That is a real finding and it is not a null result. It also sits consistently
beside `LADDER.md`: rung 2, which conditions duration on state, bought nothing —
so the duration structure is real, large and **shared across symbols**, exactly
as `FRAILTY.md` found. The information is in how long, not in what next.

## The registered consequence, and the one decision I am not taking alone

`FALSIFIER_PREREGISTRATION.md` §7: a `FAIL` means **the coarse sweep does not
run**. It has not been started.

The confound above is not a reason to set the verdict aside, and I have not set
it aside. But it is also not something to act on silently in either direction:
switching to the per-run statistic *after* seeing that it reverses the result is
precisely what a pre-registration exists to prevent. Amending the comparison
needs its own registration, written before the amended number is read, and that
is a decision for the project owner rather than for me.

## What would settle it

A surrogate that **matches the corpus's run-rate** as well as its spectrum. The
phase and VAR arms match second-order structure in the continuous signal and
then diverge by a factor of two in the discrete stream, which leaves the
comparison measuring dwell rather than sequence. `recur.null.microstate` emits
real visits through a fitted Markov chain and would preserve dwell by
construction — it was excluded here as circular for a *first-order* comparison,
which is right, and it is the obvious arm for a run-rate-matched one.
