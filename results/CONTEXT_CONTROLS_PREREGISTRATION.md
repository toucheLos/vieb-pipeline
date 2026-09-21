# Pre-registration — is the island's context effect just a speed effect?

Written and committed **before `scripts/context_controls.py` exists** and before
any control occupancy is computed. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm; segment table
`work/tok/seg_vocab/shape__k3__corpus.npz`; `shape` group, `k_mad = 3.0`,
clump 0. Digest `198eb14ff258c7f6`.

## 1. The question, and why it is owed

`CONTEXT.md` is the one result in this programme validated against the
**experimental design** rather than against appearance: clump-0 occupancy
**1.275%** in Context A against **0.470%** in Context B, Δ(B−A) =
**−0.00811 [−0.01520, −0.00243]**, pair-flip p = **0.0050**, 89 animals, 439
(animal, day) cells, paired within animal and day.

`BEHAVIOUR.md` measures the same clump at a within-animal speed ratio of
**0.272 [0.223, 0.337]** over 46 animals — **3.7× slower** than the same
animals' other segments.

**Those two facts together mean the context result may be reproducible by a
plain stillness scalar, and nothing on disk rules that out.** No speed-adjusted
context test exists anywhere in `results/`. If a frame-level speed threshold
produces the same Δ, the detector is a freeze scorer: a useful instrument, but
not a tokenizer, and the one design-validated finding is a speed effect.

**A speed control was attempted once and was disqualified.**
`ISLAND_LOOK.md`'s `matched` arm drew controls nearest on joint (log duration,
log speed). The scorer failed the positive control — `unmatched` 37.8%
[0.280, 0.467] against chance 33.3% — so under §8 of its registration neither
arm is interpretable. The control was tried through human eyes and lost. This
stage does it statistically, where no scorer is in the loop.

## 2. Split: `report`, 89 animals, and why that is not contamination

`report`, matching `CONTEXT.md` exactly. A control on a `report`-split result has
to be computed where that result lives, or it controls nothing.

The one parameter this stage sets — the stillness threshold — is fixed by
**matching a marginal, not by fitting the contrast** (§4). No quantity derived
from the A-versus-B comparison enters its choice, and the contrast is not looked
at until the threshold is already written down.

## 3. The three arms, on one denominator

Every arm is an **occupancy**: a rate, not a count. `vocab.session_composition`
reports counts with no denominator and is deliberately not what this uses — a
session with more usable segments would otherwise contribute more frames for
reasons unrelated to context.

**The denominator is identical across all three arms**: that cell's total
selectable segment frames, exactly `cell_occupancy`'s `den`.

| arm | numerator | what it rules out |
|---|---|---|
| `island` | clump-0 segment frames — the published Δ | — |
| `stillness` | frames inside selectable segments whose **per-frame** speed is below θ_still | **"it's just speed"** |
| `windows` | frames inside **length-matched random windows** from `embed.matched_windows` | **"boundary placement doesn't matter"** |

**`stillness` never consults a boundary.** It is a per-frame threshold on
`quantize.speed(X)` — `hypot` of ego channels 14–15, body lengths per second,
ω excluded because a spin is not a displacement. That it is evaluated inside the
segment table is only to hold the denominator fixed; which frames it selects
depends on speed alone.

**`windows` never consults speed or a clump.** Count- and length-matched to
clump 0 per animal, drawn by the function the repo already uses and already
asserts never straddles a recording seam (`tests/test_seg_embed.py:84`).

## 4. The stillness threshold is matched, not tuned

> **θ_still is the speed at which the `stillness` arm's corpus-wide occupancy
> equals clump 0's corpus-wide occupancy.**

The two arms are then **matched on rate** and differ only in *how frames were
selected*, which is the entire question. A threshold chosen any other way —
by eye, by a quantile picked for roundness, or by whichever value maximises or
minimises a Δ — would be fitting, and the comparison would be against a straw
arm of this stage's own construction.

θ_still is computed, written into the result, and reported **whatever it turns
out to be**. If no threshold achieves the match, the stage refuses rather than
taking the nearest.

## 5. The statistic is nested, not three Δs side by side

Three separate Δs would invite reading "both are significant" as "both are
real", which is exactly the error this stage exists to prevent. The question is
not whether stillness also moves with context — it almost certainly does, since
the island is slow — but whether **clump membership adds anything to it**.

For each (animal, day) cell holding both contexts on days 3–7:

> `Δ_arm = occupancy_B(arm) − occupancy_A(arm)`, one signed value per cell
>
> **the residual `r = Δ_island − β·Δ_stillness`**, with β the least-squares slope
> of Δ_island on Δ_stillness across cells

`recur.boot.pair_flip_null` (2,000 within-pair sign flips) is applied to `r`, and
`recur.boot.animal_interval` (2,000 replicates, `how="mean"`, clustered on
animal) gives its interval. The same is reported for `Δ_windows` as a second
residual, and the three raw Δs are reported beside them.

**The verdict follows the interval, not the p-value** — the precedent
`LEARNING_CURVE.md` set, where a stabilisation arm at p = 0.0475 still read
`FAIL` because its CI spanned zero.

## 6. The gate runs first

`recur.journey.simplex.mde_read` on the spread of `r`, against a plausible
effect **fixed now at 0.0200** — the same two percentage points
`FREEZING_PREREGISTRATION.md` §4 registered, inherited rather than re-chosen so
the control and the result it controls are held to one standard.

**If the MDE gate does not pass, no residual is read** — no permutation, no
bootstrap — and the MDE is the result. A null from a design that could not have
seen the effect is not evidence of absence.

## 7. Reuse, and the three traps that come with it

Everything comes from the modules `CONTEXT.md` already used —
`context.cell_occupancy`, `context.paired_deltas`, `labels.paired_context_mask`,
`boot.pair_flip_null`, `boot.animal_interval` — including its three recorded
traps, restated rather than rediscovered:

* **`simplex.journey_read` is not used.** It reads `flip.get("p")` while
  `pair_flip_null` returns `p_two_sided`, so its p is always NaN and it always
  takes its FAIL branch. The p is read directly.
* **`pair_flip_null` ignores its `pairs` argument** beyond a finite mask and
  flips per element. With one Δ per (animal, day) cell that is the correct
  design, and it is stated rather than assumed.
* **The cohort is asserted, not reported.** Filtered to `report` and checked with
  `Read.assert_cardinality(..., 89, what="animals")`, which exits rather than
  quietly reporting a different n. `journeys.py` is not split-restricted and
  reports n = 298.

**Days 3–7 only. Day 2 and Context C are excluded** — they are the same 298
recordings and completely confounded (`CONCENTRATION.md`).

**No shock factor exists and none is invented.** Only `Context_A,_No_Shock` is
tagged, and those 298 recordings are the same recordings as CFC day 1 —
confounded three ways with day and protocol (`recur/recur/labels.py:38-44`).
Nothing here can say which context was shocked, and nothing here will try.

## 8. The prediction, and it is the unflattering one

**Registered: the `stillness` arm reproduces most or all of the context effect,
and the nested residual spans zero.** The island is 3.7× slower than its
animals' other segments, so a speed-selected frame set matched on occupancy is
expected to track the same manipulation.

Direction is predicted here where `FREEZING_PREREGISTRATION.md` refused to
predict one. That refusal was right: predicting a **sign** for the A-versus-B
effect would have meant inventing a context-to-shock mapping the repository does
not hold. This prediction is about a **mechanism already measured** — the clump
is slow, and slowness is what the control arm selects — and needs no such
mapping.

| outcome | reading, fixed now |
|---|---|
| residual **spans zero** | the detector is a **freeze scorer**. The one design-validated finding is a speed effect, and `CONTEXT.md`, `BEHAVIOUR.md` and any write-up say so plainly |
| residual **excludes zero** | boundary placement carries context information **beyond** speed. Behavioural reality, independent of any human |
| `windows` residual excludes zero | something is wrong with the control, not right with the detector — random intervals should carry no context information, and this is reported as an instrument failure |

**A freeze scorer is still a useful instrument.** It is not a tokenizer, and no
write-up may let the second read as the first.

## 9. What this stage may not do

* It may not **restate or re-derive** `CONTEXT.md`'s Δ. That number is quoted and
  the arms are added beside it; the published value does not move.
* It may not **choose θ_still against a Δ**, or report a θ_still other than the
  one §4 fixes.
* It may not **drop cells** beyond the incompleteness rule `paired_deltas`
  already applies, or change `DAYS`, `CONTEXTS` or the clump.
* It may not **re-score Q1**, restate **Phase F**, or use any array from the
  **Wiener** arm.
* It may not report a stratum below **20,000 scored keypoint-frames**.
* It may not **compute across a recording boundary**.

## 10. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only, 2,000
replicates, `how="mean"`; the frame-level interval printed beside it only to show
how much narrower the wrong method looks. Ranked on the effect-CI lower bound,
never a p-value. Seed 0, `seeds.stable_seed`. `mypy --strict` on new modules. The
**`raw`** arm asserted by name.
