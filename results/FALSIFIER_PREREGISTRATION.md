# Pre-registration — the surrogate falsifier

Written and committed **before any surrogate is generated**. Configuration fixed
below; the prediction is falsifiable; the consequence of each outcome is decided
here rather than after seeing it.

## 1. Why this runs now, and why it is blocking

`TOK_PREREGISTRATION.md` §7 named the falsifier and left it as backlog:

> If surrogates achieve comparable MDL, the stack is fitting quantization
> structure rather than behaviour. **No result from this phase may be described
> as showing that the model has learned behaviour.**

Every number in `LADDER.md`, `FRAILTY.md` and `RESOLUTION.md` inherits that
caveat. It has stayed backlog through three phases, and the coarse sweep would
add a fourth set of numbers inheriting it. So it runs **before** the coarse
sweep, and its outcome decides whether the coarse sweep runs at all.

## 2. The claim at risk

**Rung 1 beats rung 0 by +11.3 nats/s at `plain`/N = 256**, better on 89 of 89
animals (`LADDER.md`). That is the whole positive result of the ladder: the
transition table carries information the marginal does not.

The falsifier asks whether a signal with **no behavioural structure but the same
smoothness** produces the same advantage. If it does, `rung1 − rung0` is
measuring what quantizing a smooth trajectory does, not what a mouse does.

## 3. The surrogates

Both are generated **per recording, never across a seam**, on the 17-dimensional
egocentric representation — the same array the corpus pipeline consumes, not on
raw pose.

**`phase` — the primary.** `recur.recurrence.phase.randomize_block`: one random
phase per frequency, **shared by every channel**. It preserves each channel's
power spectrum exactly, hence its autocorrelation, hence the run-length
structure that quantizing a smooth signal produces. It preserves the
cross-spectrum too, so the channels stay as correlated as they were. It destroys
everything above second order — every stereotyped sequence, every repeated
motif. It is the tightest available match on "smooth like the corpus, structured
like nothing".

Shared phase also preserves **linear relationships between channels exactly**,
which matters here: the egocentric representation has three rank-deficient
directions by construction, and a surrogate that broke them would carry variance
in a direction the corpus never moves in and would be trivially separable.

**`var5` — the second arm.** `recur.recurrence.surrogate.fit_var` /
`simulate_var` at order 5, fitted on each recording's own channels and
re-simulated. VAR(5) is the registered null behind Q1's +1.639%, so this keeps
the two results on one footing. It is fitted on the **live channels only**; the
identically-zero ones are copied, not simulated, for the reason above.

## 4. The pipeline is re-run end to end, not scored in the corpus's alphabet

The surrogate gets **its own** standardising SD and **its own** k-means
partition, both fitted on the surrogate's own `tune` split, then rungs 0–1 on its
`fit` and `report`. Reusing the corpus's centroids would answer a different
question — "does the corpus's alphabet describe the surrogate" — and would
guarantee the surrogate looked worse.

Guaranteed identical by construction rather than by inspection: the surrogate
runs through **`scripts/quantize.py` and `scripts/ladder.py` unmodified**, with
`VIEB_EGO_DIR` and `VIEB_TOK_DIR` pointed at the surrogate tree. No second
implementation of any stage exists.

## 5. The configuration, fixed in advance

| | value | why |
|---|---|---|
| arms | `phase`, `var5` | `phase` is primary; `var5` for continuity with Q1 |
| alphabet | **`plain` only**, N ∈ {256, 512} | the two cells where rung 1 passes on the corpus |
| rungs | **0 and 1 only** | rung 2 is closed (`LADDER.md`) |
| abstain | the corpus's mask, **carried over unchanged** | a phase-randomised signal has no tracking failures; 0% abstain against the corpus's 7.03% would confound the comparison |
| splits | the same tune/fit/report animals | never resampled |
| seeds | `vieb.seeds.stable_seed`, per (kind, animal) | never `hash()` on a string |
| interval | animal-level, n = 89, 2,000 replicates | `boot.animal_interval`, `how="mean"` |
| comparison | `rung1 − rung0` per animal, corpus against surrogate | the same statistic `LADDER.md` reports |

## 6. The prediction

**F1 — the surrogates' `rung1 − rung0` is positive but materially smaller than
the corpus's.** Quantizing any smooth trajectory yields a transition table that
beats a marginal, so a surrogate advantage of zero is *not* expected and would
itself be suspicious. What is expected is a gap.

**F2 — `phase` scores higher than `var5`.** Phase randomisation preserves the
full spectrum; a VAR(5) keeps only what five lags can express.

## 7. The stopping rule, decided now

Let `d_corpus` and `d_surr` be the per-animal `rung1 − rung0` and let the
comparison be the animal-level interval on `d_corpus − d_surr`, n = 89.

| outcome | verdict | consequence |
|---|---|---|
| interval excludes zero, corpus higher | `PASS` | the transition-table advantage is not explained by smoothness. **Task 3 runs.** |
| interval contains zero | `FAIL` | the stack is fitting quantization structure. **Task 3 does not run**, and that is the finding. |
| corpus lower | `FAIL` | worse than the null; every ladder number is withdrawn |

**No threshold is set on the size of the gap**, only on whether it excludes
zero. A threshold chosen now would be arbitrary and a threshold chosen later
would be selected against the outcome.

## 8. What a PASS would and would not license

It would license saying that `rung1 − rung0` exceeds what a smooth,
second-order-matched signal produces on the identical pipeline.

It would **not** license "the model learned behaviour". Two surrogates are two
alternatives, not all of them, and both are linear-Gaussian. A non-linear
structureless process could still match the corpus. The claim stays bounded to
what was actually tested, and `LADDER.md`'s "what is not claimed" section stands
either way.
