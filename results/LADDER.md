# Step 4 - the hazard ladder. Rung 2 does not beat rung 1, on any alphabet

**The registered stopping rule fired.** `TOK_PREREGISTRATION.md` SS5: *if rung 2
does not beat rung 1, stop and report. That is a complete finding - duration
carries no information beyond the transition table on this corpus.* It does not
beat rung 1 in **any** of the sixteen cell x abstain-arm combinations.

> **Read SS1 of the pre-registration before any number here.** All eight
> alphabets were retired by the run-length condition, and the ladder ran anyway
> under an override. Every shard carries `retired_by_runlength: true`.

| | |
|---|---|
| pose arm | `raw`, scale arm `bodylen`, F3-carried |
| duration grid | 12 log-spaced bins from **tune**: 0, 1, 2, 3, 5, 8, ... frames, last open |
| fitted on | `fit` - 149 animals, 3,099,564 runs at N = 256 |
| scored on | `report` - **89 animals**, 1,829,022 runs |
| interval | animal-level, 2,000 replicates, `how="mean"` |

## The ladder - total code length, nats per second (abstain conditioned away)

Lower is better. **Bold is the winner in each row.**

| arm | N | rung 0 | rung 1 | rung 2 (k=0) | rung 1 over 0 | rung 2 over 1 |
|---|---:|---:|---:|---:|---|---|
| `plain` | 256 | 57.0 | **45.7** | 71.5 | PASS +11.3 | FAIL -25.8 |
| `plain` | 512 | 69.1 | **60.2** | 163.1 | PASS +8.9 | FAIL -103.0 |
| `plain` | 1024 | **81.4** | 97.2 | 481.3 | FAIL -15.8 | FAIL -384.1 |
| `plain` | 2048 | **95.7** | 224.9 | 1647.6 | FAIL -129.3 | FAIL -1422.8 |
| `speed` | 256 | 120.2 | **84.9** | 104.4 | PASS +35.3 | FAIL -19.6 |
| `speed` | 512 | 135.8 | **99.8** | 176.1 | PASS +36.0 | FAIL -76.3 |
| `speed` | 1024 | 151.7 | **138.8** | 422.7 | PASS +12.9 | FAIL -284.0 |
| `speed` | 2048 | **168.5** | 271.6 | 1327.4 | FAIL -103.2 | FAIL -1055.8 |

**Rung 1 wins wherever it is affordable, and rung 2 wins nowhere.** At N = 256
and 512 rung 1 is the cheapest model on both arms; at N = 1024 and 2048 the
N-squared transition table costs more than it earns and **rung 0 is cheaper**.
The best model anywhere in the sweep is `plain` / N = 256 / **rung 1** at
**45.71 nats/s**, which is also the coarsest alphabet on the grid.

## Against the four registered predictions

**P1 - rung 1 beats rung 0. CONFIRMED where the codebook allows it.** At N = 256
and 512 it passes on both arms, by +8.9 to +36.0 nats/s, better on **89 of 89
animals**. The transition table carries real information. At N = 1024 and 2048 it
**fails**, because a first-order table costs N-squared parameters and 4.2M of
them at N = 2048 outweigh what they buy. That is the codebook term doing its job.

**P2 - rung 2 beats rung 1. REFUTED, and not only on the parameter charge.**
Rung 2 loses in all sixteen combinations. The important detail is that it loses
on the **data term alone**, before any codebook cost: 44.295 against rung 1's
43.365 nats/s at `plain`/N=256. Adding elapsed time conditioned on state makes
held-out prediction *worse*, not merely more expensive.

**P3 - k\* = 0 or refused. CONFIRMED, emphatically.** Parameter counts at
N = 256: rung 0 267, rung 1 65,036, rung 2 k=0 754,800, **k=1 135,027,600**, **k=2 867,135,405**.
Every step deeper is catastrophically worse - k=1 loses by roughly 4,900 nats/s
and k=2 by another 26,000. The registered expectation was that history beyond the
current state cannot be estimated at these alphabet sizes and that the codebook
charge would say so. It said so.

> **k\* = 0.** No completed `(u, d)` pair of history pays for itself, at any
> alphabet size, on either abstain arm. This is a `PASS` of the registered
> expectation rather than `GRID_LIMITED` - the grid did not end before the
> answer did.

**P4 - rung 2 wins and rung 3 does not beat it. The first clause is REFUTED and
the second remains UNTESTED.** Rung 3 was not run, is backlog, and nothing here
speaks to it. The registration said in advance that this phase could test only
half of P4, and it is reported as half-tested.

## The hazard shape, which is the positive finding

The honest replacement for the retracted dwell test - obtained with **no
persistence prior anywhere upstream**. Total hazard of leaving a state, against
how long the animal has already been in it, over the 8 busiest states at
`plain`/N=256:

| bin | elapsed (frames) | median hazard | spread across states |
|---:|---|---:|---:|
| 0 | 0-0 | 0.4890 | 1.09x |
| 1 | 1-1 | 0.3673 | 1.13x |
| 2 | 2-2 | 0.3176 | 1.10x |
| 3 | 3-4 | 0.2514 | 1.13x |
| 4 | 5-7 | 0.1561 | 1.15x |
| 5 | 8-12 | 0.1007 | 1.23x |
| 6 | 13-20 | 0.0558 | 1.26x |
| 7 | 21-33 | 0.0316 | 1.19x |
| 8 | 34-56 | 0.0205 | 1.21x |
| 9 | 57-93 | 0.0168 | 1.41x |
| 10 | 94-155 | 0.0108 | 1.44x |
| 11 | 156-258 | 0.0067 | 1.43x |
| 12 | 259+ | 0.0033 | 2.08x |

**Two things, and the second explains the whole result.**

**The hazard falls by 148x** across elapsed time, from 0.489 at the first frame
to 0.0033 beyond 259. Dwell here is emphatically **not** memoryless: a geometric
dwell has a flat hazard and this is nothing like flat.

**But it falls the same way in every state.** The spread between the busiest
states is **1.09x to 1.44x** in every bin carrying real mass, against that 148x
fall along the bins. The duration structure is large and it is **shared**. So the
pooled marginal `f(d)` that rungs 0 and 1 already use captures it, and rung 2's
754,800 parameters buy a refinement of at most 1.4x.

**That is why rung 2 fails, and it is a statement about the corpus rather than
about the model:** the symbols differ in *what* they are and not in *how long
they last*.

## What a falling hazard does not license

A falling hazard is also exactly what unmodelled heterogeneity produces. If runs
are a mixture of shorter- and longer-lived processes, the ones still alive at
large elapsed time are enriched for the long-lived kind, and the pooled hazard
falls whether or not any single process has memory. **Nothing here separates
those two**, and the 148x must not be read as evidence that individual bouts
become harder to leave the longer they last.

That is the same class of error that retracted the dwell result: a shape that
looks like structure and is produced by the instrument. It is stated here rather
than left for a reader to find.

## The gates

**Abstain reported twice.** Every number above is the `conditioned` arm; the
`symbol` arm is in `results/ladder.json` and agrees on every verdict - rung 1
beats rung 0 in the same 10 of 16, rung 2 loses in all 16. MDL does **not**
improve only when abstain is a symbol, so the model is not predicting tracking
dropout.

**Rare-transition recall.** At `plain`/N=256 the five busiest edges carry
8,638 of 1,667,155 scored transitions - **0.5%** - at 2.88 nats/run against
5.46 for everything else. The model is not winning by nailing a handful of
edges, because there is almost no mass on them to win.

**Identity probe: the gate is UNRUN.** Ten of the sixteen probes returned
`NOT_A_RESULT [DEGENERATE]` — the solver stopped on `max_iter = 400` over 89
classes, the same convergence guard that fired in `EGO.md`. The six that returned
a number are all on the `speed` arm, which MDL rejects at every N.

> That is an **absence of evidence, not evidence of absence**. No claim that the
> model avoided learning animal style is made or may be made from this, and
> nothing in this document leans on it.

The probe was also a **reduced** one — the pre-registration says to feed the
next-state distribution to `leak_read`, and at 2,048 symbols that is tens of
gigabytes per cell, so what went in was the per-window mean and SD of the model's
own code length. Both facts are in `DEVIATIONS.md` D5. Running the gate properly
needs either a larger iteration budget, which would break comparability with
`audit_moseq.json`'s 0.283 nats, or a probe whose feature count does not scale
with `N`. Neither is done.

## MDL and the run-length condition agree

The override existed because the run-length floor and MDL might disagree. They do
not. MDL prefers **`plain` over `speed` at every N**, and prefers **N = 256, the
coarsest point on the grid**, where it is still pushing downward - 256 is the
registered lower edge, so the selection is **`GRID_LIMITED` in the direction of
coarser alphabets**. That is the same thing the median-run condition was saying.

The grid was **not** extended below 256 to chase it. Doing so after seeing which
way MDL pushed would be selecting a hyperparameter against an outcome, which is
what the three-way split exists to prevent. **A coarser sweep is the obvious next
experiment and it needs its own registration.**

## What is not claimed

**Not that the model learned behaviour.** The falsifier named in
`TOK_PREREGISTRATION.md` SS7 - surrogates through the identical pipeline - was
not run. The most this shows is that one model codes this symbol stream more
cheaply than another.

**Not that dwell is non-geometric in any individual state.** See the
heterogeneity caveat above.

**Not that these alphabets are good.** All eight are retired. The ladder ran
under an override and every result inherits that.
