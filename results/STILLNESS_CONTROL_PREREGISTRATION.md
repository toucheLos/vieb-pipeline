# Pre-registration — the segment-level control the last one should have been

Written and committed **before `scripts/stillness_control.py` exists** and
before any matched set is drawn. Source array `work/ego/raw__bodylen__*.npz`,
the F3 `raw` arm; segment table `work/tok/seg_vocab/shape__k3__corpus.npz`;
`shape` group, `k_mad = 3.0`, clump 0, **`report` split, 89 animals**.
Digest `198eb14ff258c7f6`.

## 1. Why this exists, and what went wrong last time

`CONTEXT_CONTROLS_PREREGISTRATION.md` §4 fixed a stillness arm by matching on
**occupancy**: the speed whose corpus-wide below-share equals clump 0's. The
rule was not tunable, which was the point, and it was wrong. It put
`θ_still` at **0.000199 body lengths/s** and selected **2.27%** of the island's
own frames — 140× below the island's median frame speed, and about 39% of what
it selected was frames at exactly zero speed, which is a held pose and not a
slow animal.

`METHODS_FINDINGS.md` **M11** records the rule that failure leaves behind:

> match a control on the statistic that **defines** the object, at the **level**
> the definition lives.

The island is defined by a **segment-level** quantity. `BEHAVIOUR.md`'s 3.7× is
a ratio of **segment mean** speeds, and its duration distribution is the other
half of what makes it distinctive (median 1.30 s against the corpus's 0.93 s,
p90 9.8 s against 4.7 s). So the control has to be **segments matched on
duration and mean speed**, and this registration fixes that before it is drawn.

## 2. The arm

For every clump-0 segment, a **partner drawn from the same animal's own
non-clump-0 selected segments**, nearest in joint standardised
(`log duration`, `log mean speed`), **without replacement**.

* **Joint, not either alone.** `ISLAND_LOOK_PREREGISTRATION.md:61-63` already
  specifies exactly this pairing for its own `matched` arm, and it is inherited
  rather than reinvented. Matching on duration alone leaves speed free, which is
  the variable in dispute; matching on speed alone leaves duration free, and
  duration is what `DISTANCE.md` shows the metric is most sensitive to.
* **Logs, because both quantities span orders of magnitude** — the island's own
  durations run 0.53 s to 65.0 s, a factor of 122.
* **Within animal.** The contrast is within (animal, day), and a partner from
  another animal would import that animal's tracking quality and body size.
* **Without replacement**, so one convenient partner cannot stand in for many
  island segments and shrink the control's effective size silently.
* **Mean speed is `quantize.speed(X)` averaged over the segment's frames** —
  `hypot` of ego channels 14–15, body lengths per second, ω excluded because a
  spin is not a displacement. The same definition `scripts/seg_vocab.py:281`
  used to produce the 3.7×.

## 3. The precondition, and why it is balance and not overlap

`controls.overlap_read` refuses an arm that selects almost none of the island's
own frames. **That test is wrong for this arm and must not be applied to it.**
A matched-segment control is *disjoint from the island by construction* — it is
made of different segments — and its overlap is zero by design rather than by
failure.

The right precondition for a matched arm is **balance on the matching
variables**:

> **Registered: the absolute standardised mean difference between the island
> and its matched partners must be below 0.10 on BOTH `log duration` and
> `log mean speed`.**

0.10 is the conventional balance bound and is fixed here before the draw. **If
either exceeds it the match has failed, the arm is refused, and no residual is
read.** Reporting an unbalanced matched arm would repeat M11 in a new costume.

## 4. The statistic

Identical to `CONTEXT_CONTROLS_PREREGISTRATION.md` §5 — same cells, same
denominator, same null, same bootstrap — with `matched` in place of
`stillness`:

> `Δ_arm = occupancy_B(arm) − occupancy_A(arm)`, one signed value per
> (animal, day) cell holding both contexts on days 3–7
>
> **residual `r = Δ_island − β·Δ_matched`**, β the least-squares slope **through
> the origin**

Through the origin, not ordinary least squares: with an intercept the residual
mean is zero by construction, the sign-flip null could never reject, and the
comparison could not have failed. `tests/test_controls.py` asserts this both
ways and the same function is reused.

`recur.boot.pair_flip_null`, 2,000 within-pair sign flips;
`recur.boot.animal_interval`, 2,000 replicates, `how="mean"`, clustered on
animal; cohort filtered to `report` and asserted at 89 with
`Read.assert_cardinality`. Days 3–7 only; Day 2 and Context C excluded.
**The MDE gate runs first**, against the same plausible effect of **0.0200**
inherited from `FREEZING_PREREGISTRATION.md` §4; if it does not pass, no
residual is read and the MDE is the result.

## 5. The prediction, unchanged in substance

**Registered: the matched arm absorbs most or all of the context effect and the
residual spans zero.** The island is slow and long; a set of the same animals'
own segments matched to be slow and long is expected to track the same
manipulation.

| outcome | reading, fixed now |
|---|---|
| residual **spans zero** | **the detector is a freeze scorer.** What moves with context is "slow, long segment", not "this clump". The one design-validated finding is a speed-and-duration effect and `CONTEXT.md`, `BEHAVIOUR.md` and any write-up say so |
| residual **excludes zero** | clump membership carries context information beyond being slow and long. The island is a behavioural unit and not a kinematic bin |
| balance fails | the arm is refused and the question stays open |

This is the third attempt at this question — `ISLAND_LOOK.md` asked it of human
scorers and disqualified the scorer; `CONTEXT_CONTROLS.md` asked it of a
frame-threshold arm and the arm was vacuous. **If this one is refused as well,
that is reported as three failures to control a single result, not as three
inconclusive footnotes.**

## 6. What this stage may not do

* It may not **restate or re-derive** `CONTEXT.md`'s published Δ.
* It may not change the matching variables, the balance bound, `DAYS`,
  `CONTEXTS` or the clump after seeing any number.
* It may not **re-run the draw** with a different seed to obtain balance. One
  draw, seed 0; if it fails balance, that is the result.
* It may not apply `overlap_read`'s frame-overlap criterion to this arm (§3).
* It may not **re-score Q1**, restate **Phase F**, or use any **Wiener** array.
* It may not report a stratum below **20,000 scored keypoint-frames**.
* It may not **compute across a recording boundary**.

## 7. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only; the
frame-level interval printed beside it only to show how much narrower the wrong
method looks. Ranked on the effect-CI lower bound, never a p-value. Seed 0,
`seeds.stable_seed`. `mypy --strict` on new modules. The **`raw`** arm asserted
by name.
