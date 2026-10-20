# Registration — STEP 3A for grooming: does the representation hold what the instruments lose?

Committed **alone**, before any 3A code exists.

**This is a post-result decision.** Step 1's registered outcome was **O2, which
selects 3B** (37df6d6). The user switched to **3A** after seeing two results:
- grooming was never recoverable on any instrument at any noise level, including
  CalMS21's native noise;
- no single decision in the §1.8 ablation moved grooming toward the bar
  (b361249).

3B's gate is `NOT_A_RESULT` by construction (`step3b/REGISTRATION.md`), and 3B is not
withdrawn. This branch asks the 3A question of the prompt, for **grooming**.
Investigation and mount are reported beside it for comparison.

## Data, partitions, scoring: inherited unchanged from Step 1

- **Partitions:** `step1/partitions.json`. Groom maps on map_g and is scored on
  eval_g (10 sequences); attack, investigation and mount map on map_t1 and are scored
  on eval_t1.
- **Inputs:** the Step 1 inputs, `work/calms/raw/<arm>` and
  `work/calms/moseq/labels/<arm>_s0`, for arms **m0_s0** (native) and **m1_s0**
  (Luna 1.0×). The negative control uses **white_s0**.
- **Scoring:** `vieb/calms/score.py`, unchanged. Event F1 at IoU ≥ 0.5; ceiling 1
  for groom and Task 2 annotators 3 and 4 for the others; normalised score;
  2,000-replicate sequence bootstrap. **Recoverable** means the lower 95% bound
  > 0.25 and the score > the circular-shift null's 95th percentile.
- **One deviation, decided now:** the null uses **200** circular shifts, not 1,000,
  because the probe is refitted on every shift (registered for cost).

## 3A.1: a linear probe on the pipeline's features

- **Feature sets:**
  - **R (representation):** the 17-D ego representation that D and C consume
    (`ego.transform` of the `raw` arm, divided by Luna's frozen `basis.json`
    `sd_used`), plus each channel's mean and SD over a centred **±0.5 s** window.
    That makes 51 features.
  - **L (MoSeq latent):** the 10-D `latent_state` that `apply_model` returned, with
    the same window statistics. That makes 30 features.
- **Probe:** L2 logistic regression (C = 1, standardised features, `lbfgs`,
  max_iter 1000). It is fitted on the **map** partition's frames and predicts the
  **eval** partition, binary per label (one-vs-rest within T1). The predicted label
  is probability > 0.5, and its bouts are scored by event F1. Frame AUROC is reported
  beside it.
- **Negative control:** the same probe on features computed from **white_s0**, with
  the true labels. **If it is recoverable for any label, every 3A.1 verdict is
  `NOT_A_RESULT`.**
- **Reading, per label at native, fixed now:**

  | probe | instruments (Step 1) | reading |
  |---|---|---|
  | recoverable | not recoverable | **the method loses information the representation holds** |
  | not recoverable | not recoverable | **the representation is the problem** |

  At 1.0× the same reading tells whether Luna-level noise removes what the
  representation held.

## 3A.2: localise the loss, stage by stage (groom; the others beside it)

Concordance after each stage. The probe is the 3A.1 probe in every row, with each
row's features:

| stage | features carried forward | what is lost if this row falls below the one above |
|---|---|---|
| representation | R (3A.1) | — |
| segmentation | each frame gets its **D segment's mean of R**, where segments are the intervals between D boundaries | within-segment detail |
| embedding | each frame gets its **C segment's time-normalised embedding** (SEG_GRID × the 14 shape channels), PCA to 20 components fitted on the map partition, with zeros and an indicator outside selectable segments | what segment embedding keeps |
| clustering | Step 1's normalised scores for C and M, as committed | the discretisation |

The table reports, per stage, the normalised score with its CI and the recoverable
verdict. **The first stage at which groom stops being recoverable** is the
localisation. If groom is not recoverable at the representation, there is nothing to
localise, and that is the answer.

## Not in this registration

**"Register the fix, then re-run Step 1 in full"** (3A.3) follows only after this
result, as its own registration.

## Reads

One `Read` per label × feature set × level × stage. `n_effective` is the number of
evaluation sequences, with Step 1's caveat (identities unknown, so intervals may be
too narrow).

---

# Amendment 1 (post-result): a bout-aware probe, a scorer check, and a speed baseline

**Committed alone.** Decided **after** 3A.1–3A.2 were seen (`REPORT.md`): groom's
event score was near zero while frame AUROC was 0.86, and mount scored exactly zero
events at AUROC 0.87. The user directed this amendment and its three additions. The
3A.1–3A.2 results stay on record.

**Scope.** This probe is **supervised and diagnostic**.
- **If it succeeds,** the unsupervised method is losing information the features
  contain, and 3A goes on to find the stage where it is lost (a later
  registration).
- **It does not change the Step 1 verdict.**

## A1.1: check the event scorer before trusting any event score

1. **Identity check.** For every label and every evaluation partition,
   `score.event_counts(y, y)` and `score.boundary_counts` on the label's own
   boundaries must give **F1 = 1.0 exactly**. **If this fails, the 3A.1–3A.2 event
   verdicts are void** and are re-scored after the scorer is fixed.
2. **Planted flicker.** Flip a random fraction p ∈ {0.01, 0.05, 0.10, 0.20} of the
   frames of the true label sequence (seed 0), feed it in as the prediction, and
   report event F1 against p.
   - Also report it after the readout fitted in A1.3 smooths it, which shows how much
     the readout restores.
   - **This is descriptive:** it says how fragile event F1 at IoU ≥ 0.5 is to flicker.

## A1.2: a trivial baseline for frame-level information

- **B (speed and stillness only):** log centroid speed and log pose-change rate
  (`vieb.hsmm.mouse.signal` on the `raw` arm, w = 0.2 s), plus their ±0.5 s window
  mean and SD. That makes 6 features.
- **Metrics:** **AUROC and PR-AUC** on evaluation frames from the probe's
  probability, for B, R and L, at native and 1.0×. The interval is a bootstrap over
  evaluation sequences (2,000).
- **The rule:** a representation **contains a label beyond speed** iff the lower 95%
  bound of **ΔPR-AUC (representation − B)** is > 0. The bootstrap is paired, with the
  same sequences on both sides.

## A1.3: the readout fitted on the map partition, not fixed at 0.5

- **Probe:** the A1.2 feature sets with the same logistic regression.
- **Readout grid:**
  - threshold τ from 0.05 to 0.90 in steps of 0.05;
  - median smoothing of the probability over w ∈ {0, 0.2, 0.5, 1.0} s;
  - minimum bout duration d ∈ {0, 0.2, 0.5, 1.0} s.
- **Choosing the readout:** by **out-of-fold event F1 on the map partition**, with
  folds by sequence (T1: 5 folds; groom: 3 folds, one per map sequence). Ties go to
  smaller w, then smaller d, then larger τ. The probe is then refitted on the whole
  map partition, and the chosen readout is applied to evaluation.
- **Scoring:** Step 1's unchanged. Event F1 normalised by the ceiling; recoverable
  means the lower 95% bound > 0.25 and the score > the null's 95th percentile.
- **Null:** **100** circular shifts, each with the probe **and** the readout choice
  refitted.
- **Arms:** **m0_s0** (native) and **m1_s0** (Luna 1.0×); white_s0 as the negative
  control. **If the control is recoverable for any label, every A1.3 verdict is
  `NOT_A_RESULT`.**

## Readings (fixed now)

| outcome | reading |
|---|---|
| scorer identity check fails | the earlier 3A event verdicts are void |
| R (or L) recoverable for groom at native | **the unsupervised method loses information the features contain**; 3A proceeds to localise the stage |
| R and L not recoverable for groom at native, but contain it beyond speed (ΔPR-AUC > 0) | the information is at frame level, but a linear probe cannot assemble it into bouts |
| neither | the representation, under a linear probe, lacks grooming beyond speed |

The same readings are reported at 1.0×.
