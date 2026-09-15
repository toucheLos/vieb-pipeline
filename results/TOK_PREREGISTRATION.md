# Pre-registration — the hazard ladder, and an overridden stop condition

Written and committed **before any ladder job is submitted**. Configuration fixed
below; predictions falsifiable; the thing that makes this document necessary
stated first rather than buried.

## 1. The override, and who made it

`results/ALPHABET.md` records that **all eight alphabets were retired** by a
pre-registered stop condition: median run 1.0 frame against a 3-frame floor. That
condition was written as **blocking**. The ladder is running anyway.

That decision was taken **after** seeing the condition fire, by the project owner,
on this argument: the run-length floor is a heuristic pre-filter, and the
principled selector for a vocabulary is held-out description length — which is
the whole design of this branch. MDL charges duration explicitly, so a flickering
vocabulary pays for its flicker in the objective rather than needing a guard.

**This is an override, not a reinterpretation.** It is written here because a
stop condition that can be stepped over silently is not a stop condition, and
because a reader six months from now must be able to see that the ladder ran on
alphabets the registered guard rejected.

Two things follow and are binding:

* **The retirement is not withdrawn.** Every cell stays marked `retired: true` in
  `results/alphabet.json`, and every ladder number inherits that label.
* **The stop condition is not amended.** It was not moved to a frame-weighted
  statistic, and the grid was not widened below N = 256. Either would be choosing
  a parameter against an outcome after seeing it.

## 2. What is being asked

One number: **memory depth k\***, the largest history length whose per-animal
held-out MDL improvement has an animal-level bootstrap interval excluding zero —
plus the hazard **shape** per state, obtained with no persistence prior anywhere
upstream.

## 3. The configuration, fixed in advance

| | value | why this and not something tuned |
|---|---|---|
| input | `results/alphabet.json`, all 8 cells | the retired set, carried forward under §1 |
| pose arm | `raw`, scale arm `bodylen` | F3-carried; Wiener is not benchmarkable and not carried |
| hyperparameters on | **tune** (60 animals) | the duration grid, and nothing else, is chosen here |
| fits on | **fit** (149 animals, 1,927 recordings) | |
| every reported number on | **report** (89 animals, 1,149 recordings) | `n_effective = 89` |
| duration grid | **12 log-spaced bins**, 1 frame to the **tune** p99.9 of run length, last bin open | a rule, not a list: determined by tune data before any report number exists |
| smoothing | Laplace `alpha = 1.0`, off-diagonal only | inherited from `recur.seq.motif.transition_matrix` |
| diagonal | **zero**, at every rung | RLE makes `AA` impossible; a model able to emit it inflates every real 2-gram |
| duration resolution | **one frame, `delta = 1/30` s**, identical in every arm and rung | arms with different duration resolutions are not comparable |
| codebook cost | `|params| * c`, `c = 0.5 * log(n_runs_fit)` nats | the standard BIC parameter charge, stated so it is not mistaken for a tuned knob |
| bootstrap | `recur.boot.animal_interval(..., how="mean")`, 2,000 replicates | animal-level, never frame-level |
| `boot.frame_interval` | printed beside it | to show how much narrower the wrong method looks. **It never licenses a claim** |

### The ladder

| rung | symbol model | duration model |
|---|---|---|
| 0 | marginal `p(u)` over runs | marginal `f(d)`, pooled |
| 1 | first-order `p(u_next \| u)`, zero diagonal | marginal `f(d)` — **unchanged from rung 0** |
| 2 | cause-specific hazard `lambda_j(tau \| u, h)` | the same hazard, jointly |

Rungs 0 and 1 share a duration model **on purpose**, so that rung 1 − rung 0
isolates the transition table and rung 2 − rung 1 isolates duration structure and
elapsed-time dependence. Every rung charges **both** symbol and duration; a rung
that charged only one would win by hiding information in the other.

**History depth** `k in {0, 1, 2}` completed `(u, d)` pairs, inside rung 2. `k = 0`
conditions on the current state and elapsed time alone.

**Elapsed time in the current state is its own input**, not part of the history
window. At prediction time the current segment is right-censored and has no
completed duration; it is probably the single most informative variable and no
`(u, d)` history carries it.

**A covariate slot is built now and left empty** — zero-width array, asserted by
test. `recur.labels.parse` already yields day, context and protocol. Adding it
later would reshape every fitted model and invalidate the scoring harness.

**Right-censoring at recording boundaries.** A terminal run contributes the
survival term only, never survival times hazard. Getting this wrong biases every
duration estimate downward.

## 4. The predictions

**P1 — rung 1 beats rung 0.** The transition table carries information. If it
does not, the alphabet is not describing anything sequential.

**P2 — rung 2 beats rung 1.** Duration and elapsed time carry information beyond
the transition table.

**P3 — k\* = 0 or is refused.** At 256 to 2,048 symbols the `k = 1` context space
is 65,536 to 4.2M cells and `k = 2` is 16.7M to 8.6 billion, against 5.8M to 14.4M
runs. **The registered expectation is that history beyond the current state
cannot be estimated at these alphabet sizes**, and that the MDL codebook charge
will say so by refusing to pay for the parameters.

**P4 — rung 2 wins and rung 3 does not beat it.** The headline claim: mouse
behaviour in this paradigm may be memoryless given the current state and its
elapsed duration. It contradicts the implicit assumption behind every
syllable-grammar result in the field.

> **P4 is registered and only half of it is testable here.** Rung 3 is backlog
> and will not be run in this phase. The second clause must be reported as
> **untested**, never as supported. Registering a prediction and then reporting
> it as confirmed by a phase that never ran it would be worse than not
> registering it at all.

## 5. Stopping rules, fixed in advance

* **If rung 2 does not beat rung 1**, stop and report. That is a complete
  finding: duration carries no information beyond the transition table here.
* **If a context is too thin to estimate**, refuse it rather than reporting a
  smoothed prior as a result. A `GRID_LIMITED` or `NOT_A_RESULT` verdict is a
  correct outcome.
* **k\* = 2 is reported as `GRID_LIMITED`**, never as "exactly 2". The grid ends
  at 2 and depth 3 and above is backlog, so the reading is "at least 2, and the
  grid ended".

## 6. Gates, each of which can invalidate the result

* **Abstain reported twice** — as a symbol, and conditioned away. If MDL improves
  only when abstain is a symbol, the model is predicting tracking dropout.
* **Identity probe on the predictions, not the labels.** The next-state
  distribution goes through `recur.audit.leak.leak_read`, held out by window
  within recording. If animal identity is decodable, the model learned style.
* **Artifact ablation with a delta.** Scored with and without flagged frames, and
  the difference reported.
* **Rare-transition recall reported separately.** Average NLL rewards a model
  that only nails the top five edges.

## 7. What would make this whole branch uninterpretable

If surrogates achieve comparable MDL, the stack is fitting quantization structure
rather than behaviour. That falsifier is named here and is backlog, not run in
this phase — so **no result from this phase may be described as showing that the
model has learned behaviour**. The most it can show is that one model codes the
symbol stream more cheaply than another.
