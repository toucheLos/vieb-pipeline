# Pre-registration — the island, transferred to 149 held-out animals

Written and committed **before `scripts/fit_replicate.py` exists** and before any
fit segment is assigned. Source `work/ego/raw__bodylen__*.npz`, the F3 `raw` arm;
`shape` group; **`fit` split, 149 animals, 1,927 recordings**.
Digest `198eb14ff258c7f6`.

## 1. Why, and why now

Every context claim in this programme rests on **439 (animal, day) cells from 89
`report` animals**. The `fit` split holds **149 animals** and roughly **745 more
complete A/B cells**, and has never been touched for a context contrast. A
replication there is worth more than any further re-analysis of `report`, and it
has to happen **before** the result is written up, not after.

## 2. The island is TRANSFERRED, not re-derived

`work/tok/seg_vocab/shape__k3__corpus.npz` contains **only the 89 report
animals** — there are no clump labels for `fit`. And re-running the clustering on
`fit` would produce **different clumps**: single linkage on a different
population gives different components, and nothing would guarantee that a new
"clump 0" is the same object. **Re-clustering is therefore forbidden here**, and
the reason is recorded rather than discovered later.

**The rule, fixed now:**

> Run the **frozen** detector on the 149 fit animals — `deriv_sec = 0.133`,
> `degree = 3`, `k_mad = 3.0`, `shape` channels — take `embed.selectable`
> segments, resample each to the **`SEG_GRID = 40`**-point stack on the
> standardised ego array, and compute its Euclidean distance to the **nearest of
> the 361 report clump-0 members** in that same space. Divide by the **report
> arm's ambient scale, 25.319997787475586**. **Transfer iff the normalised
> distance ≤ θ = 0.18998060778738327.**

**Both constants are inherited, neither is re-derived.** θ is the null's own
quantile from Step 2; the scale is the report arm's median random cross-animal
distance. Using *fit's* own ambient scale would make θ mean something different
on the two splits, which is the one thing a replication may not do.

**The space is reconstructed exactly, and this was verified before registering.**
`seg_vocab` builds its bank through a PCA fitted on **tune** with
`n_components = SEG_GRID × 14 = 560` — the full rank — so the projection is a
rotation and Euclidean distances are invariant to it. Checked on 7 stored
nearest-neighbour pairs, 4 of them island members: recomputed raw-space
distances matched the stored `nn_all` to **six decimal places, ratio 1.0000 on
all seven**. So no PCA is refitted, nothing is re-estimated on `fit`, and the
transfer inherits the report geometry exactly.

## 3. The transfer rule gets its own independence check

The island's members define θ's neighbourhood, so a fit segment landing inside
0.18998 of a report member **could reflect a shared low-speed regime rather than
the same state**. The rule is an instrument and it gets the control the island
itself already passed.

> **Two rates are reported, always together:**
> **(a)** the fraction of fit segments that transfer;
> **(b)** the fraction of **length-and-speed-matched** fit segments that
> transfer.
>
> **If (a) does not exceed (b) with non-overlapping animal-level intervals, the
> assignment is capturing slowness, not the island** — and the contrast is
> **not read**, whichever way it would have landed.

Partners are drawn by `controls.matched_partners` on joint (log duration, log
mean speed), within animal, without replacement, with the balance precondition
at **|SMD| < 0.10** on both variables — the same machinery and the same bound
`STILLNESS_CONTROL.md` used, applied one level up. **If balance fails, the
check fails and the stage refuses**; re-drawing for balance is forbidden.

## 4. The contrast, identical to the published one

For each (animal, day) cell holding both contexts on days 3–7:
`Δ = occupancy_B − occupancy_A`, occupancy being transferred-island frames over
that cell's total selectable segment frames. `context.cell_occupancy` and
`paired_deltas` unchanged; `boot.pair_flip_null` at 2,000 within-pair sign
flips; `boot.animal_interval` at 2,000 replicates, `how="mean"`; **cohort
asserted at 149 animals** with `Read.assert_cardinality`. Days 3–7 only; Day 2
and Context C excluded. **The MDE gate runs first** against the inherited
plausible effect of **0.0200**; if it fails, the MDE is the result.

## 5. The incumbent's values, stated in advance — M12

| quantity | report value |
|---|---|
| island Δ (B − A) | **−0.00811 [−0.01520, −0.00243]**, p = 0.0050 |
| residual after duration-and-speed matching | **−0.00685 [−0.01371, −0.00147]** |
| clump-0 segments | 361, in 46 of 89 animals |
| clump-0 occupancy, these cells | 0.012105 |

## 6. Predictions, fixed now

1. **At least 200 fit segments transfer.** Below that the cell occupancies are
   too sparse to carry a contrast and the stage refuses.
2. **The transfer rate exceeds the matched-partner rate**, non-overlapping at
   the animal level. *Falsifier: if they overlap, the rule selects slow long
   segments rather than the island, and §3 stops the stage.*
3. **The fit Δ has the same sign as report's** — negative, occupancy higher in
   Context A.
4. **The fit Δ interval excludes zero.**

## 7. What this stage may not do

* It may not **re-cluster** `fit`, or re-derive θ or the ambient scale (§2).
* It may not **re-draw** the matched partners for balance (§3).
* It may not **restate or re-score** `CONTEXT.md`, `STILLNESS_CONTROL.md` or any
  published number. A replication reports its own value beside the original.
* It may not **touch `report`**, or pool the two splits.
* It may not **re-score Q1**, restate **Phase F**, or use any **Wiener** array.
* It may not report a stratum below **20,000 scored keypoint-frames**.
* It may not **compute across a recording boundary**.
* It may not describe any context contrast as **shock versus no-shock**. It is
  Context A versus Context B; `recur/labels.py:41-43` refuses a `shocked` flag.

## 8. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only, 2,000
replicates, `how="mean"`; the frame-level interval printed beside it only to
show how much narrower the wrong method looks. Ranked on the effect-CI lower
bound, never a p-value. Seed 0, `seeds.stable_seed`. `mypy --strict` on new
modules. The **`raw`** arm asserted by name.
