# The island's context effect — one control it passes, one that was not a control

Registered in `results/CONTEXT_CONTROLS_PREREGISTRATION.md`, committed before
`scripts/context_controls.py` existed. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm; segment table
`work/tok/seg_vocab/shape__k3__corpus.npz`; `shape` group, `k_mad = 3.0`,
clump 0, **`report` split, 89 animals, 439 (animal, day) cells** — the same
cells as `CONTEXT.md`, asserted cell-for-cell across all three arms.
Digest `198eb14ff258c7f6`.

## The short version

| | |
|---|---|
| **length-matched random windows** | a real control, and **the island passes it** |
| **stillness** | **not a control**. It selects 2.27% of the island's own frames, so it cannot test the hypothesis it was registered for |
| **incidental, unregistered** | **tracking dropout is 6.8× higher in Context A than B.** Named as a threat, not an explanation |

**The registered prediction — that a stillness scalar would absorb the context
effect — is neither confirmed nor refuted.** The arm built to test it did not
test it. That is reported as a failed control, not as a passed test.

## The three arms

All three are occupancies over **one** denominator: that cell's total selectable
segment frames. The island arm reproduces `CONTEXT.md` exactly, which is the
check that the harness is measuring the published object.

| arm | mean A | mean B | Δ (B − A) | pair-flip p |
|---|---:|---:|---|---:|
| `island` | 1.275% | 0.470% | **−0.00811 [−0.01520, −0.00243]** | 0.0050 |
| `stillness` | — | — | **−0.01373 [−0.01734, −0.01037]** | 0.0005 |
| `windows` | — | — | **−0.00270 [−0.00758, +0.00172]** | 0.2110 |

Both MDE gates passed before any residual was read — 0.0087 and 0.0083 against
the registered plausible effect of 0.0200.

## The windows control, which worked

`embed.matched_windows` drew **298** intervals, count- and length-matched to
clump 0 per animal, avoiding abstain and never straddling a recording seam.
They ignore boundary placement entirely.

**They carry no context information.** Δ = −0.00270 [−0.00758, +0.00172], p =
0.2110 — the interval spans zero. A set of intervals with the island's length
distribution, placed without regard to where the detector cut, does not move
with context.

And the island survives removing them: residual **−0.00699 [−0.01319, −0.00148]**,
p = 0.0195, after removing 0.415× the windows delta fitted through the origin.

> **Boundary placement matters.** The context effect is not a property of
> "intervals of this length exist"; it is a property of *where* the detector
> put them. That is the first evidence in this programme that the boundaries
> themselves carry information, and it needed no human and no ceiling.

## The stillness control, which was not one

`θ_still` was fixed by the registered rule — the speed whose corpus-wide
below-share equals clump 0's occupancy. That rule produced:

| | |
|---|---:|
| target occupancy (clump 0, these cells) | 0.012105 |
| **θ_still** | **0.000199 body lengths/s** |
| realised below-share | 0.012105 — an exact match |
| **share of island frames below θ_still** | **0.0227** |
| island frame speed, median | 0.0279 — **140× θ_still** |

**The arm selects almost none of the thing it is controlling for.** Matching on
*occupancy* was the error: the island is a set of slow **segments**, and a slow
segment still contains fast frames. Matching frame-share therefore reaches into
the extreme tail of the *frame* speed distribution instead of selecting frames
like the island's.

So the residual — **−0.00667 [−0.01381, −0.00097]**, p = 0.0355, β = 0.105 —
excludes zero, and **that means almost nothing**. Two near-disjoint frame sets
do not explain each other, and a residual saying so is a statement about the
arms, not about the detector. The slope β = 0.105 says the same thing: across
cells the two deltas are nearly uncorrelated.

`controls.overlap_read` returns `NOT_A_RESULT` and names the arm vacuous in the
result itself, so this cannot be read off the number alone.
`METHODS_FINDINGS.md` M5 — *a control that returns nothing is usually testing
the control* — and this is its converse: a control that returns **everything**
is usually testing the control too.

**The freeze-scorer hypothesis is untested and still stands.** `BEHAVIOUR.md`'s
3.7× is a within-animal ratio of **segment mean** speeds, so the control that
tests it has to select **segments** matched on mean speed and duration, not
frames below a threshold. That is a different arm and needs its own
registration; §9 forbids reporting a θ_still other than the one §4 fixed, and
inventing a second threshold now would be choosing one after seeing a number.

## Tracking quality differs by context, and that reaches everything here

Unregistered, added after seeing θ_still, recorded in `DEVIATIONS.md` D10.

Roughly 39% of what the stillness arm selected is frames at **exactly zero** ego
speed — the whole pose held, which is a tracking dropout and not a slow animal.
Measured on its own:

| | |
|---|---:|
| held-frame share, **Context A** | **0.643%** |
| held-frame share, **Context B** | **0.094%** |
| difference, paired within (animal, day) | **−0.00551 [−0.00843, −0.00311]** |
| pair-flip p | **0.0005** |

**Dropout is 6.8× higher in Context A**, the same context where the island is
2.7× more occupied, in the same direction, at a comparable magnitude to the
island effect itself.

**It does not explain the island cell for cell.** The two deltas correlate
**−0.119** across the 439 cells, and the island residual on held frames is
**−0.00956 [−0.01645, −0.00381]** — larger than the raw delta, not smaller.

**But no occupancy in this design is free of it.** Every arm here, the published
island arm included, is a rate over segment frames, and how many frames a
session contributes depends on how well it tracked. Whether a freezing animal is
harder to track, or a harder-to-track session looks stiller, is not separable
from anything this repository holds. `CONTEXT.md` carries this caveat from now
on.

## What this licenses

**Licenses:** the island's context effect is **not** reproducible by a
length-matched set of intervals placed without regard to boundary position, on
89 animals, within animal and day, against a paired sign-flip null. Where the
detector cuts carries information that the existence of intervals does not.

**Does not license:** any claim that the detector is more than a freeze scorer.
That control was not built. It remains the most important open question about
this result, and the stillness arm as registered could not answer it.

**Does not license** reading the held-frame finding as a refutation of
`CONTEXT.md`. It is a confound that has been measured and does not account for
the effect, which is a different and weaker statement than "it is clean".

## What is owed

1. **A segment-level stillness control, with its own registration.** Draw
   non-clump-0 segments matched to clump 0 on joint (log duration, log mean
   speed) — the matching `ISLAND_LOOK.md` already specifies for its `matched`
   arm — and re-run the nested residual against that. This is the control the
   registration meant to build.
2. **A tracking-quality covariate in any future context contrast.** Held-frame
   share moves with context and is not going to stop.
