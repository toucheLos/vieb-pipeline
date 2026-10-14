# STEP 1 — CalMS21 degrade-to-match: **O2 (input-limited) → branch 3B, as registered**

Registration: `step1/REGISTRATION.md` (8b92160) and Amendment 1 (c8e3e19,
D32), both committed before any CalMS21 sequence was degraded or scored.
Result: `step1/recovery.json` (66 Reads); figure: `step1/recovery.svg`; noise
and partitions: `step1/noise.json`, `step1/partitions.json`.

**Every interval is a bootstrap over evaluation sequences.** CalMS21 has no
animal identities, so a resident can recur across sequences undetectably, and
the intervals may be too narrow.

## The verdict

| rule (§7) | finding |
|---|---|
| negative controls (white, phase; every instrument) | **pass**: none recoverable; every control's normalised score lies within ±0.012 of 0 |
| native noise vs Luna | CalMS21 **r = 0.633** of Luna's level (skull estimator), so the sweep is valid |
| ≥ 1 behaviour recoverable at native | **yes**: investigation, by D (boundary F1), normalised 0.54 [0.33, 0.73] |
| groom recoverable at ≥ 1.0× | **no**: on no instrument, at no level |
| a deciding CI straddling 0.25 | **no**: every groom CI at 1.0× lies below 0.03 |
| **outcome** | **O2: input-limited → Step 3B** |

**Before choosing 3B, read the next section.** The registered table returns
O2, but 3B's premise does not hold.

## Branch 3B has no target

3B's target is groom's **breakdown noise**: the level at which grooming stops
being recoverable, converted into Luna jitter units. **Grooming is never
recoverable**, at any level, including CalMS21's own native noise, which is
**cleaner than Luna's** (0.63×). Its breakdown is therefore the bound
"< native", and there is no noise level for a pose ensemble to aim at:

| groom | D (boundary F1) | M (MoSeq, event F1) | C (clumps) |
|---|---|---|---|
| native (0.63× Luna) | −0.018 [−0.049, +0.006] | +0.013 [+0.002, +0.024] | 0 (no clumps) |
| 1.0× | −0.007 [−0.041, +0.022] | −0.000 | 0 |

Read plainly, **these instruments do not recover grooming even from cleaner
keypoints than Luna has.** That is the pattern 3A ("method fails") is for, for
this behaviour. O2 fired because an *easy* positive survives, not because
grooming is held back by noise.
- **The registered rule is not changed.**
- **Which branch to run, or neither, is the human's decision.** §7 of the
  prompt asks for exactly this kind of call to be raised, not resolved.

**Two caveats weaken the groom null, both registered in advance:**
- groom's map partition fell to **3 sequences** after the collision rule
  (`partitions.json` `dropped_for_collision`);
- groom's ceiling is **1**: no second annotator exists (D-S3), so its 0.25 bar
  is stricter than the human-ceiling bar the Task 1 labels face.

## Recovery curves (normalised score, 95% CI)

**D, the boundary detector** (boundary F1 at ±0.5 s; the ceiling is the
inter-annotator boundary F1 between annotators 3 and 4):

| level × Luna | 0 (native) | 0.25† | 0.5† | **1.0** | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|
| investigation | **0.54** [0.33, 0.73] | 0.54 | 0.54 | **0.50 [0.31, 0.68]** | 0.37 [0.19, 0.54] | 0.24 [0.09, 0.38] | 0.11 [−0.00, 0.22] |
| mount | 0.19 [0.03, 0.36] | 0.19 | 0.19 | 0.16 [0.02, 0.32] | 0.14 | 0.10 | 0.04 |
| groom | −0.02 | −0.02 | −0.02 | −0.01 | −0.01 | 0.00 | 0.01 |

† At 0.25× and 0.5×, the jitter target is below CalMS21's native level, so
those arms carry native jitter plus that level's dropout (§2.3).

- **Investigation boundaries are recoverable at Luna's noise level.**
  Breakdown is registered at **1.5×**, but the 1.5× CI straddles the bar, so
  the transition lies between 1.0× and about 2×.
- **Mount is never recoverable**: its lower bound is below 0.25 throughout,
  and its CIs straddle the bar at levels 0–1.5.
- **Attack is `GRID_LIMITED`**: 258 bouts in only 5 evaluation sequences. This
  was predicted in the registration.

**M, Keypoint-MoSeq** (κ 1e6, the labeller of record's untuned value):
- **never recoverable.** At native, investigation scores 0.04 [0.03, 0.07],
  mount 0.03 and groom 0.01.
- **The scores sit above chance's p95 at native.** Investigation scores 0.024
  against p95 0.003. Frame NMI with the labels is 0.03–0.07, against
  0.006–0.015 on white noise.
- So **MoSeq's syllables carry some label information, but the many-to-one
  map cannot turn them into matched bouts.** The syllables have a median dwell
  of 21 frames (about 0.7 s), and 20 are in use. Mapped bouts come out
  fragmented and fail the registered IoU ≥ 0.5 event match.
- This is a property of the registered metric meeting MoSeq's unit size. It is
  reported, **not** used to rescore. By 1.0×, even the above-chance signal is
  gone.

**C, segment clumps:**
- **formed zero clumps in every arm**, native and controls alike. On native,
  only 1,470 of the bank's nearest-neighbour edges fall under the θ inherited
  from Luna; the bank holds 19,971 segments from 133 videos.
- Every frame is "unassigned", so **C scores exactly chance everywhere**.
- On Luna, C left 98–99% of segments unassigned (`VOCAB.md`). On CalMS21 it is
  100%.

## The noise that was added

| quantity | value | source |
|---|---|---|
| Luna σ̄ (skull estimator) | 0.06180 BL | `plant.json` |
| CalMS21 native σ̄ (same estimator, MARS scores) | 0.03910 BL, **r = 0.633** | `noise.json` |
| jitter gain g (Amendment 1) | 1.092 | the estimator reads 0.0613 at 1.0×, and saturates to 0.148 at 3× |
| Luna dropout | 4.5% per keypoint; abstain 1.1%; run median 6 frames, p90 22; 94% of runs longer than the 3-frame hold | 100 Luna fit recordings |
| Luna skull bones | 0.282 / 0.268 / 0.260 BL; CalMS21 ≈ 0.35 | `noise.json` |

## Reads

66 in total:
- **4 PASS**: D on investigation at 0, 0.25, 0.5 and 1.0×;
- **52 FAIL**;
- **7 INCONCLUSIVE at the cell level**: D on mount at 0–1.5×, and D on
  investigation at 1.5× and 2×. None of them is a deciding CI under §7;
- **3 GRID_LIMITED**: attack, on each instrument.

## Disclosures

- **A 20-shift scoring smoke of D at native** was run and looked at before the
  registered scoring. It was deleted, and the registered run used the same
  code with 1,000 shifts. No setting changed between them.
- **The neck fills `center`'s slot** (primary arm). The 6-keypoint
  sensitivity arm, the jitter-only and dropout-only arms, and the §1.8
  ablation were registered after the verdict and have since run; see the
  Addendum.
- **"Easy positives survive" says nothing about behaviour in a fear session.**
  Investigation is a social behaviour, scored on a resident–intruder assay.
- **Nothing here enters Paper 1.**

## Stop

Per the prompt, **Step 3 does not begin.** The human decides between:
1. **3B as registered.** But its target noise level does not exist, because
   groom is never recoverable.
2. **Treat grooming as a method failure (3A).** That means localising where it
   is lost: a linear probe on the pipeline's features, then stage by stage.
3. **Stop here.**

---

# Addendum: secondary arms and the decision-importance ablation

Registered in `ABLATION_REGISTRATION.md` (3a5e371) after the verdict was seen
and before any of these arms existed. The data are in `step1/secondary.json`
(27 curves, 153 Reads) and `step1/ablation.json` (43 rows). **Everything here
is descriptive, and the O2 outcome is unchanged.** A re-combine of the primary
cells, which now also store per-sequence counts, reproduced `recovery.json`
exactly: the same decision, curves and controls.

## Secondary arms: which part of Luna's noise does the damage

D on **investigation** (boundary F1, normalised, with the 95% lower bound):

| arm | 1.0× | 1.5× | 2.0× | 3.0× | breakdown |
|---|---|---|---|---|---|
| joint (primary) | 0.50 (lo 0.31) | 0.37 (0.19) | 0.24 (0.09) | 0.11 (−0.00) | 1.5× |
| **jitter only** | 0.51 (0.31) | 0.39 (0.19) | 0.27 (0.09) | 0.14 (−0.01) | **1.5×** |
| **dropout only** | 0.54 (0.33) | 0.52 (0.32) | 0.51 (0.33) | 0.49 (0.32) | **> 3×** |
| 6 keypoints (neck dropped) | 0.43 (0.25) | 0.28 (0.12) | 0.18 (0.06) | 0.08 (−0.01) | 1.5× |

- **Jitter is what defeats the detector; Luna-pattern dropout barely matters,
  even at three times Luna's rate.** Luna's dropout runs are mostly long, so
  the `raw` arm holds them and abstain blocks boundaries there, rather than
  creating spurious breaks.
- **Discarding the neck**, with `center`'s slot filled by the midpoint between
  the ear and hip midpoints, costs about 0.07 at 1.0× and leaves the breakdown
  unchanged. The primary arm's neck-for-centre substitution is not what made
  investigation recoverable.
- **Groom is never recoverable in any arm**, on any instrument. M stays below
  0.1 everywhere. C forms no clumps in any arm.

## Ablation: which decision matters, for which behaviour, at which noise

Paired Δ = variant − primary, at the same seeds and the same noise draws. The
interval is a sequence bootstrap. Rows are shown where the CI excludes 0 and
|Δ| ≥ 0.02.

| decision | instrument | level | behaviour | Δ normalised [95% CI] |
|---|---|---|---|---|
| smoothing: median 0.25 s | D | 1.0× | investigation | **+0.214 [+0.061, +0.372]** (0.50 → 0.72) |
| smoothing: median 0.25 s | D | native | investigation | +0.162 [+0.067, +0.264] |
| window: DERIV_SEC 0.267 s | D | 1.0× | investigation | +0.157 [+0.063, +0.255] |
| representation: pose only | D | 1.0× | investigation | **−0.121 [−0.191, −0.056]** |
| frame rate: 15 fps | D | 1.0× | mount | +0.099 [+0.053, +0.145] |
| frame rate: 10 fps | D | 1.0× | mount | +0.098 [+0.053, +0.145] |
| window: DERIV_SEC 0.267 s | D | native | mount | +0.091 [+0.048, +0.144] |
| window: DERIV_SEC 0.067 s | D | 1.0× | investigation | +0.081 [+0.041, +0.117] |
| keypoints: 6 (no neck) | D | 1.0× | investigation | −0.073 [−0.114, −0.029] |
| MoSeq κ 1e5 | M | native | groom | +0.065 [+0.018, +0.119] (0.02 → 0.09) |
| window: DERIV_SEC 0.267 s | D | 1.0× | mount | +0.061 [+0.037, +0.089] |
| frame rate: 15 / 10 fps | M | native | investigation | −0.049 / −0.050 |

- **Pooled ranking**, on the lower bound of |Δ| averaged over investigation,
  mount and groom:
  1. D window 0.267 s at 1.0×;
  2. D median smoothing at native;
  3. D pose-only at 1.0×;
  4. D median smoothing at 1.0×;
  5. D six keypoints at 1.0×.

  All lower bounds are between 0.02 and 0.03.
- **Smoothing inside the nulls.** The median-smoothed white and phase controls
  fired on **no** instrument, so the smoothing Δs stand. **Wiener smoothing is
  `NOT_A_RESULT`**: there is no committed implementation for non-Luna data
  (GAPS G11).

**Reading.**
- **For the detector, the decisions that matter all trade noise for
  bandwidth**: smoothing, a longer derivative window, lower frame rates for
  mount. They raise investigation and mount scores. The velocity channels
  matter too: removing them costs 0.12.
- **For grooming, no single decision moves the score anywhere near the bar.**
  The largest change, MoSeq at κ 1e5, takes it from 0.02 to 0.09. Grooming's
  failure is not a tuning problem in any of the registered decisions.
- **MoSeq on investigation and mount** is near zero, with or without any
  variant.

This leaves the Step 3 question as it was. It strengthens the reading in the
main report that grooming's failure points to method or representation (3A)
rather than to input noise (3B).
