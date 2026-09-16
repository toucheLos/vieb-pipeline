# What the island actually is — a basin of near-immobility, not a token

`VOCAB.md` found one clump that survived every registered test: **361 segments
in 46 of 89 animals**, top animal supplying 16%. This document asks what it *is*,
before anything is named or published. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm, `shape` channel group,
`k_mad = 3.0`, θ = 0.190 inherited from Step 2. Digest `198eb14ff258c7f6`.

## The answer, in one line

**It is not one behaviour. It is a broad, chained region of the space in which
the animal is barely moving** — and being a *state* rather than a *unit* is
exactly why the token search had nothing to merge.

## The six checks

| check | result | what it rules in or out |
|---|---|---|
| **chaining** | `FAIL` — widest pair **4.8× θ**, typical pair **2.1× θ** | it is a **chain**, not a group. Must not be named as one behaviour |
| **speed**, paired within animal | ratio **0.272 [0.223, 0.337]**, 46 animals | **3.7× slower** than the same animals' other segments |
| **duration** | median 1.30 s vs 0.93 s; p90 9.8 s vs 4.7 s; max 65 s | longer and far heavier-tailed than the corpus |
| **arena position**, paired | edgeness **1.69 vs 3.61**, difference **−1.92 [−2.95, −0.83]** | **not** wall-driven — closer to the centre of the animal's own cloud |
| **tracking artifact** | zero flagged frames, by construction | cannot be a bone-violation artifact |
| **session** | 59 dates, 7 days, 3 contexts; context A 73.7% | not one session wearing a crowd's clothes |

## Why "chain" and "3.7× slower" are one finding, not two

A connected component is a statement about **paths**: A links to B and B to C
puts A and C in one clump however far apart A and C are. Clump 0's typical pair
sits **0.398** apart against a linking distance of **0.190** — farther apart than
the threshold that built it, and close to the **0.361** median of segments that
were never assigned at all.

So the members are not copies of one another. What they share is a *region*: all
of them are periods of unusual stillness. That is a **behavioural state with a
wide basin**, not a stereotyped motif with a tight core — and a basin is what a
single-linkage chain looks like when the thing underneath it is genuinely broad.

## What the eye said, and why the measurement overruled it

Ten segments were sampled across the whole duration range — 0.53 s to 65 s — and
rendered as first / middle / last frames with the skeleton drawn
(`results/behaviour/sheet_shape_c0.png`, index in `shape_c0.json`).

Every row shows a compact, hunched animal with the skeleton correctly placed on a
plainly visible mouse, and essentially **no change across the three frames** —
including the row that spans **65 seconds**. Nothing in the sheet looks like a
tracker parked on a static object: the animal is visible, and it is still.

**And my reading of the sheet was wrong about the arena.** Every row looked like
an animal pressed against the wall, and `CONCENTRATION.md` records tracking
failure rising 3.7× monotone from arena centre to wall, so a wall-enriched clump
would be finding the arena rather than behaviour. Measured, the clump sits at
edgeness **1.69** against **3.61** for the same animals' other segments — it is
**less than half as far out**, the opposite of the impression. The crops are
centred on the animal, so the wall fills the frame whenever the animal is
anywhere near it, and the eye cannot judge arena position from that.

This is the second time in this programme that the by-eye scorer has been
corrected by a measurement (`ADJUDICATION.md` records the first: two frames read
as "rearing against a wall" were teleports). The sheet is evidence about whether
the skeleton is on the animal. It is not evidence about where the animal is.

## What it is consistent with, stated at the right strength

Immobility of this kind — crouched, sustained, far slower than the animal's own
baseline, spread across 46 animals and 59 dates — is what **freezing** looks like
in a fear-conditioning assay, and freezing is this assay's canonical readout.

**This document does not claim the clump is freezing.** It has no behavioural
labels, no scored ethogram, and no comparison against shock timing or context.
It reports a kinematic state and names the obvious hypothesis so that someone can
test it. The test it needs is trial structure — does occupancy of this basin rise
after conditioning, and differ by context — and none of that is measured here.

## The registered secondary, finally computed

`SEGRECUR_PREREGISTRATION.md` registered open-end alignment as the secondary
distance and it was never run (`DEVIATIONS.md` D8). On clump 0's own nearest
pairs:

| | |
|---|---:|
| median time-normalised distance (the primary) | **0.173** |
| median open-end distance (the secondary) | **0.716** |
| Spearman between them | 0.616 |
| mean absolute log duration gap of matched pairs | **1.30** — a factor of **3.7×** |

**The matching is substantially tempo-invariance.** Segments called nearest
neighbours differ by 3.7× in duration on average, and without the warp they sit
4.1× farther apart. This is the number that explains the 122× duration range
inside one clump, and it was owed.

## What this does to the vocabulary question

`VOCAB.md`'s verdict stands and is sharpened. There is clump structure the
dwell-matched nulls do not reproduce; it covers 1.9% of segments; and the largest
piece of it is **a state, not an entry**. A vocabulary needs units that are
interchangeable with one another. These are units that are all *near* one thing
and not near *each other*.

That is also the cleanest available explanation for why Step 4 was refused. The
most recurrent structure in this corpus is a broad low-motion basin, and a basin
does not tokenize.
