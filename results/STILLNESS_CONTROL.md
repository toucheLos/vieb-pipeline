# The island is not a kinematic bin — the registered prediction is refuted

Registered in `results/STILLNESS_CONTROL_PREREGISTRATION.md`, committed before
`scripts/stillness_control.py` existed and before any matched set was drawn.
`report` split, **89 animals, 439 (animal, day) cells**, `shape` group,
`k_mad = 3.0`, clump 0, F3 `raw` arm. Digest `198eb14ff258c7f6`.

## The result

> **Clump membership carries context information beyond being slow and long.**
> Residual **−0.00685 [−0.01371, −0.00147]**, pair-flip p = **0.0270**, after
> removing 0.246× a duration- and speed-matched control fitted through the
> origin.

**The registered prediction was the opposite** — that a matched arm would absorb
most or all of the effect and the residual would span zero, making the detector
a freeze scorer. It does not.

## The arm, and the precondition it had to clear first

Every one of the **358** clump-0 segments was matched to a partner from **the
same animal's own** non-clump-0 selected segments, nearest in joint standardised
(`log duration`, `log mean speed`), without replacement. All 358 matched.

| variable | island | matched control | SMD |
|---|---:|---:|---:|
| duration | **1.86 s** | 1.84 s | **+0.0098** |
| mean speed | **0.0451 bl/s** | 0.0460 bl/s | **−0.0240** |

Both are an order of magnitude inside the registered **0.10** bound, fixed
before the draw. **The control is the same animals' own segments, the same
length, moving at the same speed** — and it is not the island.

> **This precondition is balance, not overlap, and §3 fixed that in advance.**
> `controls.overlap_read` refuses an arm selecting almost none of the island's
> frames; applying it here would refuse a working arm, because a matched-segment
> control is disjoint from the island by construction. Getting the precondition
> right for the arm is the whole of what `METHODS_FINDINGS.md` M11 asks.

## The three arms side by side

| arm | Δ occupancy (B − A) | pair-flip p |
|---|---|---:|
| `island` (published) | **−0.00811 [−0.01520, −0.00243]** | 0.0050 |
| `matched` — same animal, same duration, same speed | **−0.00512 [−0.00961, −0.00113]** | 0.0085 |
| `windows` — length-matched random intervals (`CONTEXT_CONTROLS.md`) | −0.00270 [−0.00758, +0.00172] | 0.2110 |

**Being slow and long is itself worth something.** The matched arm moves with
context on its own, and that is a real finding: a set of segments selected only
for duration and speed tracks the manipulation. It is why this control was
necessary and why the frame-threshold arm was not a substitute.

**But it explains only a quarter of the island.** β = 0.246, and removing it
leaves **84%** of the published effect standing. Slow-and-long is a component of
the island's context dependence, not the whole of it.

## The ladder is now complete, and the island passes all three

| control | rules out | outcome |
|---|---|---|
| **length-matched random windows** | "boundary placement doesn't matter" | **passed** — random intervals carry no context information at all |
| **duration- and speed-matched segments** | "it's just slow and long" | **passed** — residual excludes zero |
| ~~frame-level stillness threshold~~ | — | **vacuous**, M11 |

Three independent ways for this result to have been an artifact. Two were
testable and it survived both.

## What this licenses

**Licenses:** occupancy of clump 0 differs by context within animal and day, on
89 animals, and that difference is **not** reproducible by intervals of the same
length placed elsewhere, nor by the same animals' own segments of the same
duration and mean speed. **The island is a behavioural unit and not a kinematic
bin.**

**Does not license calling it freezing.** That still needs the context-to-shock
mapping and an ethogram, and this repository holds neither. `CONTEXT.md`'s
statement of that is unchanged.

**Does not license** reading the island as a single behaviour.
`BEHAVIOUR.md`'s chaining verdict stands: clump 0 is a single-linkage chain
whose typical pair sits 2.1× the linking distance apart. A context-dependent
**basin** is what moved.

**Does not clear the tracking confound.** `CONTEXT_CONTROLS.md` measured held-pose
frames at 6.8× higher rate in Context A. That is orthogonal to this control —
matching on duration and speed does not match on tracking quality — and it
remains the largest untested threat to every occupancy in this design.

## A note on how the match was validated

The first draw **failed balance** at SMD +0.4508 on duration, and the refusal
was correct: `matched_partners` had defaulted its candidate pool to "everything
that is not a target", which admitted segments from outside the scored cells
whose mean speed is undefined. A NaN distance sorts *last* rather than being
refused, so a handful of them were matched at the end of the greedy pass. The
eligible pool is now named explicitly and `tests/test_controls.py` asserts it.

**No number was read before the precondition passed**, which is the only reason
that sequence is a bug report rather than a retraction. Re-drawing with a
different seed to obtain balance is forbidden by §6 and was not done; the pool
was a coding error, not an unlucky draw.
