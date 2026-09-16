# Pre-registration — do segments recur across animals?

Written and committed **before any segment bank, any search, and any
corpus-versus-null number exists**. It fixes the unit, the distance, the nulls,
the comparator, the statistic and the verdict in advance.

## 1. The question

Q1 measured cross-animal recurrence with **fixed-length windows** as the unit.
This asks the same question with **segments** — the intervals between detected
breaks — as the unit. The unit is the only thing that changes; the statistic,
the bootstrap and the exactness gate are Q1's.

## 2. What this is NOT read against, and why

**Not against Q1's +1.639%.** That is the `wiener` cell
(`results/Q1_BANKED.md`). The same file's `unfiltered` cell at the identical
window and dimension is **−0.1991% [−0.4636%, +0.0418%]**, an interval spanning
zero. This arm runs on `raw`, because a low-pass filter manufactures exactly the
smoothness whose breaks the detector looks for. Reading a raw-arm segment result
against a Wiener-arm window result would compare two things that differ in two
ways at once.

**Registered comparator: a fixed-window control in the identical space.** Same
17-dim ego `raw` arm, same abstain mask, same PCA procedure, same bank
construction, same nulls, same statistic — windows instead of segments. Its
lengths are **drawn from that arm's own clean-segment duration distribution**,
per recording, and its count is matched to the segment count, so bank size and
length distribution are both controlled. That is what "rises", "matches" and
"falls" are measured against.

## 3. The unit, fixed now

Segments from `vieb/seg/breaks.py` at `degree = 3`, `deriv_sec = 0.133`, on
`work/ego/raw__bodylen__<tag>.npz` with `pose_arm == "raw"` asserted by name.

**Excluded, not flagged:**

* segments containing any abstained frame (`abstain_frac > 0`);
* **segments adjacent to an abstain block**, which is new here. Abstain is 7.03%
  of frames arriving in short runs, and it creates more boundaries than the
  detector does — 1,081 peaks against 2,532 segments on one animal. A segment
  with no abstained frame inside it can still have had a boundary manufactured
  by the dropout edge beside it, and **a segment whose boundary was manufactured
  by a dropout is not a behavioural unit**. The count before and after this
  exclusion is reported.

**`k_mad` = 3.0 primary; sensitivity at {2.5, 3.0, 4.0}. `k ≤ 2.0` is excluded
from every sweep in this arm, and the reason is registered rather than
discovered:** at `k = 1` the four nulls of the closed boundary-rate gate agreed
to **0.40%** at 53% of the NMS resolution ceiling (30/14 = 2.143 b/s), so rates
there are a property of the refractory period rather than of the signal; and a
MAD-standardised threshold penalises the corpus's heavy-tailed `D` at the bottom
of a sweep by construction. Both defects are recorded in `SEGMENTATION.md` and
are not inherited.

**Three channel groups — `shape`, `twist`, `both` — run separately and never
pooled.** Their boundary sets at ±2 frames give Jaccard 0.443 (both|shape),
0.436 (both|twist) and **0.226 (shape|twist)**: largely different events. If
they disagree about which segments recur, **that disagreement is the finding**
and is not resolved by picking one.

## 4. The distance, registered as primary before either is seen

**PRIMARY: time-normalised.** Each segment resampled to a common 40-point grid
(`recur/render/meanskel.py:resample`, `np.interp`). Duration is treated as
nuisance and shape as the behaviour.

**Why this one is primary.** Fixed-length windows hold duration constant, so
Q1's excess is *already* a pure shape statistic; time-normalising is what makes
the windowed comparator like-for-like. Duration is not being thrown away from
the programme, only from the distance: `FRAILTY.md` characterises it separately
and found the 148× hazard fall surviving speed conditioning at 0.756 [0.742,
0.769].

**SECONDARY: open-end alignment**, no warp, compared over the overlapping
extent. Reported beside the primary. **The gate reads the primary.** This is
registered because four independent lines in this project say duration carries
signal, so a reader is entitled to both — but choosing after seeing them would
not be a choice.

**The honesty item ExBias records:** *"duration is discarded — time-normalization
separates shape from tempo, then tempo is never modeled."* The primary repeats
that choice **for the distance only**. And its warning that resampling before
fitting manufactures smoothness applies to resampling before comparing: the null
segments are resampled identically, so whatever it manufactures applies to both.

## 5. The nulls

**`microstate` and `microstate0`** — the dwell-matched pair, banked in
`results/falsifier.json`. Not the closed gate's four families: `phase` and
`var5` diverge **2.1× in runs per second** from the corpus, which is the
confound that made a per-second comparison against them uninterpretable.

Null streams go through the **identical detector at the identical `k_mad`**, so
their segments are produced the same way the corpus's are. `randpairs` is re-run
**per arm** because bank size and ambient scale are both confounds and
`paired_excess` divides each arm by its own scale.

`microstate` must be read **weakly**, per `DWELL.md`: it preserves one-step
visit dynamics and therefore much of what any sequence comparison measures.

## 6. No separability precondition. What replaces it.

The separability precondition is **contradictory for this class of question**
and is recorded as such in `DEVIATIONS.md` D7: it demands a null that lacks the
structure under test while being indistinguishable from data that has it. It is
not used here, and reinstating it later would need its own registration.

**In its place, the narrow assertion that the earlier comparison actually
needed:** the null's **segment-duration distribution matches the corpus's**.
Two-sample on log duration — KS statistic plus quantile differences at
{0.1, 0.25, 0.5, 0.75, 0.9} — **reported beside the comparison, not gating it**.
A null whose segments are systematically shorter would have a larger bank of
shorter units and the comparison would partly be about that; the reader is told
the size of that effect rather than being given a pass/fail on it.

## 7. The statistic

Q1's, unchanged. `paired_excess`: per-animal fraction of queries whose
**cross-animal** nearest-neighbour distance falls below θ, where **θ is set by
the NULL's** quantile at `NULL_RATE = 0.01`, and both arms are divided by their
own **random cross-animal pair** median first. The observed arm is never asked
to clear a threshold chosen from itself, and the normaliser has no
nearest-neighbour feedback.

* **Animal-level bootstrap, n = 89, 2000 replicates, `how="mean"`.**
  `boot.frame_interval` printed beside it **solely** to show how much narrower
  the wrong method looks (~19×), never to license a claim.
* **Exactness gate before any excess is quoted**: `search_reference` +
  `exactness_read` against the float64 CPU reference. `--device cuda`
  **asserted, never auto-detected**; there is no CPU fallback that finishes.
* **Dimension**: `d = min(192, 40 · n_channels)` per group, with `pca_read`'s
  retained variance and tail-preservation reported. At `twist` that is no
  reduction, and the read will say so.

## 8. The planted floor — every null result carries what would have been seen

Synthetic recurring segments planted at occupancies **{0.0025, 0.005, 0.01,
0.02, 0.05, 0.1}** — Q1's own ladder, including the 1% cell — into the ego
stream, segmented by the identical detector, scored against `microstate`.
Reported on the **realised** fraction, not the requested one, because `plant`
places whole instances and the two diverge at long templates. The smallest
recoverable occupancy is a headline number whether the gate passes or fails.

## 9. Two controls reported beside the headline, each able to undercut it

**Length-matched matching control.** `partition.py:20-37` records a rater
scoring **+0.183** — a third of the real effect — purely by calling the six
longest clips "same". A nearest-neighbour hit between two segments of similar
duration may be a length match rather than a shape match. Report the duration
similarity of NN partners against chance.

**`d_within` is an upper bound, and this is a limitation not a result.**
`search`'s `exclusion` is a start-offset test (`|q_start − s| <= exclusion`) and
applies only to same-animal pairs. For unequal-length segments two intervals can
overlap on most of their extent while their starts are far apart, so `d_within`
under-excludes. **The headline is unaffected**: `paired_excess` reads only
`d_cross`, which the exclusion never touches. `d_within` is reported as a
diagnostic with this bound stated.

## 10. The gate, fixed in advance

| outcome | verdict | consequence |
|---|---|---|
| excess **rises** above the windowed control, interval excluding zero | `PASS` | boundaries are behavioural and crop contamination was suppressing the measurement. **Step 3 runs.** |
| excess **matches** the windowed control | `PASS` (weak) | segments are a valid unit that adds nothing. Reported, Step 3 runs, and the vocabulary question stays open. |
| excess **falls** below the windowed control | `FAIL` | boundaries cut through behaviours rather than between them. **Step 1's criterion is wrong. Stop and report** — that closes this route with an answer. |
| excess does not exceed **`microstate0`** | `FAIL` | a unit that does not beat its own dwell-matched shuffle is not recurring. |

"Rises", "matches" and "falls" are decided by the **animal-level interval on the
per-animal difference** between the segment excess and the windowed-control
excess: excluding zero upward is a rise, excluding zero downward is a fall,
spanning zero is a match. No threshold on gap size — one fixed now would be
arbitrary and one fixed later would be selected against the outcome.

## 11. What a PASS would not license

It would not license "these are behaviours", and it would not license a
vocabulary. It would say segments recur across animals above a dwell-matched
surrogate and above a length-matched windowed control in the same space.
Whether the recurring things **clump** is Step 3's question, and Step 3 can
return "continuum" — which is also a result.
