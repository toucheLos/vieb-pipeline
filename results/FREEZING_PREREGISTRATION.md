# Pre-registration — does the island move with context?

Written and committed **before any occupancy is computed per context**. One
clump, one contrast, fixed in advance.

## 1. The question

`BEHAVIOUR.md` found that the widest clump is **3.7× slower** than the same
animals' other segments (speed ratio 0.272 [0.223, 0.337], 46 animals, paired)
and named the obvious hypothesis: in a contextual fear-conditioning corpus,
sustained immobility is **freezing**. That document was explicit that it does not
test it — *"the test it needs is trial structure, and none of that is measured
here."*

This is that test. It converts the island from something that **looks** like a
behaviour into something validated against the **experimental design**, or it
does not.

## 2. Why this design supports it, and where the confound isn't

`CONCENTRATION.md` records context and day as **completely confounded** — Context
C occurs only on Day 2, and both are exactly 479,962 frames. That is true of the
whole-corpus marginal grouping and it is **not** true of the contrast below.

Measured on the report split: **days 3–7 each carry both Context A and Context
B**, and **441 of 708 (animal, day) cells hold both**. Every one of the 89 report
animals sees more than one context. **Context C and day 2 are excluded**, so the
confounded factor never enters.

| day | contexts present |
|---|---|
| 0, 1 | A only — excluded |
| 2 | C only — excluded |
| **3, 4, 5, 6, 7** | **A and B** — the contrast |

## 3. The statistic

For each **(animal, day)** cell holding both contexts, on days 3–7:

> `Δ = occupancy_B − occupancy_A`

where **occupancy** is that cell's clump-0 frames divided by its total selectable
segment frames. A rate, not a count: a session with more usable segments would
otherwise contribute more clump-0 frames for reasons that have nothing to do with
context. `vocab.session_composition` reports counts with no denominator and is
**not** what this uses.

**One Δ per cell.** The sign is what the null flips, so the quantity has to be
signed and paired — an unsigned or unpaired form has nothing to flip, which is
the failure `journeys.py` records for the raw W2 contrast.

## 4. Gate first: the MDE, before the contrast is read

`recur.journey.simplex.mde_read`, the pattern `journeys.py` and
`learning_curve.py` both use. The minimum detectable effect is computed from the
spread of Δ and compared against a plausible effect **named before it runs**.

**Plausible effect: 0.02** — two percentage points of occupancy. Registered
rationale: clump-0 occupancy is on the order of 1.9% of segments corpus-wide, so
a context effect smaller than about two points could not be distinguished from
the between-session variability this corpus already shows, and a null from a
design that could not have seen it is not evidence of absence.

**If the MDE gate does not PASS, no contrast is computed** — no permutation, no
bootstrap, and the MDE is the result. That is the registered behaviour and it is
reported whether it passes or not.

## 5. The null and the interval

* **`recur.boot.pair_flip_null`** — within-pair sign flip, 2,000 permutations.
  It flips **per element** of Δ and ignores its `pairs` argument beyond a finite
  mask; with one Δ per (animal, day) cell that is the correct design, and it is
  stated here rather than assumed.
* **`recur.boot.animal_interval`**, `how="mean"`, 2,000 replicates, on the
  **report split only**, so `n_animals = 89`. `journeys.py` is not
  split-restricted and reports n = 298; this one filters.
* **The verdict follows the interval, not the p-value.** `LEARNING_CURVE.md`
  fixed that precedent: a stabilisation arm at p = 0.0475 still read `FAIL`
  because its CI spanned zero.

**`simplex.journey_read` is not used.** It reads `flip.get("p")` while
`pair_flip_null` returns `p_two_sided`, so its p is always NaN and it always
takes the FAIL branch. The p is read directly, as `journeys.py:351` does.

## 6. One clump, one contrast

**Only clump 0**, the widest and the one `BEHAVIOUR.md` characterises. Nineteen
contrasts is a fishing expedition, and the multiplicity correction it would need
would destroy the power this design has.

## 7. The verdicts, fixed in advance

| outcome | verdict | reading |
|---|---|---|
| Δ interval excludes zero | `PASS` | the island moves with context — validated against the experimental design rather than against appearance |
| Δ interval spans zero, MDE passed | `FAIL` | the state is real and is not context-dependent. The freezing hypothesis comes off the page |
| MDE gate did not pass | `INCONCLUSIVE` | the design could not have seen the registered effect. The MDE is the result |

**Direction is not predicted.** Freezing would plausibly be higher in the shocked
context, but this corpus's context-to-protocol mapping is not established in any
artifact this repository carries, so predicting a sign would be inventing one.
A two-sided interval is what is registered.

## 8. What a PASS would and would not license

It would license: occupancy of this kinematic state differs by context within
animal and day, on 89 animals, against a paired sign-flip null.

It would **not** license calling it freezing. That needs the context-to-shock
mapping and an ethogram, neither of which this repository holds. It would make
freezing the leading hypothesis and give the next person a measured effect to
test it against — which is more than appearance gave.
