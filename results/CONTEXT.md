# Step C — the island moves with context

Registered in `results/FREEZING_PREREGISTRATION.md`, committed at `cd6bec6`
before any occupancy was computed per context. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm. `shape` group, `k_mad = 3.0`,
clump 0. Digest `198eb14ff258c7f6`.

## The result

| | |
|---|---:|
| mean occupancy, **Context A** | **1.275%** |
| mean occupancy, **Context B** | **0.470%** |
| difference, B − A, paired within (animal, day) | **−0.00811 [−0.01520, −0.00243]** |
| pair-flip p, 2,000 within-pair sign flips | **0.0050** |
| animals | **89** |
| (animal, day) cells holding both contexts | **439** (2 dropped as incomplete) |

**The island is occupied 2.7× more in Context A than in Context B** — a
difference of 0.81 percentage points, which is **64% of the Context-A
occupancy**. The interval excludes zero and the sign-flip null agrees.

This is the first result in this programme in which a discovered unit is
validated against the **experimental design** rather than against its
appearance.

## The gate ran first, and it passed

`MDE = 0.0087` at 80% power over 439 pairs, against a plausible effect of
**0.0200** fixed in the registration. So the design could have seen an effect of
the registered size, and a null here would have been informative rather than
empty. Had it failed, no contrast would have been computed and the MDE would
have been the result.

`within_sd = 0.0651`.

## Why the day/context confound does not reach this

`CONCENTRATION.md` records context and day as **completely confounded** —
Context C occurs only on Day 2, and both are exactly 479,962 frames. That is a
fact about the whole-corpus marginal grouping.

It does not apply here, and the registration said so before the numbers existed.
Measured on the report split: **days 3–7 each carry both Context A and Context
B**, and 441 of 708 (animal, day) cells hold both. **Day 2 and Context C are
excluded.** Every comparison is within one animal on one day, so day cannot
carry the effect — it is held fixed inside every pair.

## The design of the test, and the three traps it avoids

**Occupancy is a rate**, clump-0 frames over that cell's total selectable segment
frames. Counting instead would have measured how much of each session survived
QC. `vocab.session_composition` reports counts with no denominator and is
deliberately not what this used.

**One Δ per cell, signed.** The null flips signs, so an unsigned or unpaired
quantity would have nothing to flip — the failure `journeys.py` records for the
raw W2 contrast, whose symmetric non-negative form "could not have failed".

Three implementation traps, all avoided and all recorded in the registration:

* **`simplex.journey_read` is not used.** It reads `flip.get("p")` while
  `boot.pair_flip_null` returns `p_two_sided`, so its p is always NaN and it
  always takes its FAIL branch. The p above is read directly.
* **`pair_flip_null` ignores its `pairs` argument** beyond a finite mask and
  flips per element. With one Δ per (animal, day) cell that is the correct
  design, and it is stated rather than assumed.
* **`journeys.py` is not split-restricted** and reports n = 298. This filters to
  `report` and asserts the cohort with `Read.assert_cardinality`, which would
  exit rather than quietly report a different n.

## One number that is smaller here than elsewhere, and why

The animal-level interval is **1.07×** the width of the frame-level one, where
this programme usually reports ~19×. That is not a weakening of the rule — it is
what the rule predicts when the unit of analysis is already close to the animal:
439 cells over 89 animals is about five per animal, so there is little clustering
left for the animal bootstrap to account for. The frame-level interval is printed
for comparison only and licenses nothing, as always.

## Two things measured afterwards, which this result now carries

**A length-matched window control, which it passes.** `CONTEXT_CONTROLS.md`:
298 intervals count- and length-matched to clump 0, placed without regard to
boundary position, carry **no** context information — Δ = −0.00270 [−0.00758,
+0.00172], p = 0.2110 — and the island's effect survives removing them,
residual −0.00699 [−0.01319, −0.00148]. **Where the detector cuts carries
information that the existence of intervals of this length does not.**

**A stillness control, which was not a control.** Matched on occupancy, the
threshold landed at 0.000199 body lengths/s and selected **2.27%** of the
island's own frames. It cannot test whether this result is a speed effect, and
that question remains open. `BEHAVIOUR.md`'s 3.7× is a ratio of segment **mean**
speeds, so the control that tests it must match segments, not threshold frames.

**Tracking quality differs by context.** Frames whose whole pose is held — a
dropout, not a slow animal — are **6.8× more common in Context A** (0.643%)
than in B (0.094%), Δ = −0.00551 [−0.00843, −0.00311], p = 0.0005: the same
direction as the effect above, at a comparable size. Across the 439 cells the
two correlate only **−0.119** and the island residual on held frames is
−0.00956 [−0.01645, −0.00381], so dropout does **not** explain this result cell
for cell. But occupancy is a rate over segment frames, and how many frames a
session contributes depends on how well it tracked. **No number on this page is
free of that**, and whether a freezing animal is harder to track or a
harder-to-track session looks stiller is not separable from anything this
repository holds.

## What this licenses, and what it does not

**Licenses:** occupancy of this kinematic state — sustained near-immobility,
3.7× slower than the same animals' other segments — differs by context, within
animal and day, on 89 animals, against a paired sign-flip null.

**Does not license calling it freezing.** That needs the context-to-shock
mapping and an ethogram, and **this repository holds neither**. Nothing here
establishes which context was paired with shock, which is also why the
registration predicted no direction: predicting a sign would have meant inventing
the mapping.

What it does is make freezing the leading hypothesis and hand the next person a
measured effect to test it against, which is more than appearance gave. The
honest form of the claim is: *the state this detector found is not uniformly
distributed across the experiment — it tracks a manipulation the experimenters
made.*

It also does not license the clump being **one** behaviour. `BEHAVIOUR.md`'s
chaining verdict stands: clump 0 is a single-linkage chain whose typical pair
sits 2.1× the linking distance apart. A context-dependent **basin** is what moved,
not a context-dependent motif.
