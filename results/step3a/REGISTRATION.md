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
