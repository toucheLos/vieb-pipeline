# Registration — STEP 1: CalMS21 degrade-to-match

Committed **alone**, before any Step 1 code exists and before any instrument
has seen CalMS21. Its inputs are `INVENTORY.md` (b554290),
`step1/DATA_FACTS.md`, and two decisions the user made after the data facts
were read (rearing; the groom split). Deviations from the prompt are listed in
§10.

**Question.** Do the programme's unsupervised instruments recover known
behaviours on clean mouse keypoints, and at what noise level does recovery
fail? Is Luna's noise level above or below that point?

**CalMS21 labels are used only to score**, never to fit, tune or choose.

---

## 1. Data

- **Animal:** the resident only (mouse 0).
- **Keypoints:** reordered into Luna's order and names, with **neck in
  `center`'s slot**: `left_ear, right_ear, nose, center(=neck), left_hip,
  right_hip, tail_base`. From here on every Luna module (`ego`, `bones.SKULL`,
  the kpms config) runs unchanged.
- **Frame rate:** 30 Hz, so there is no resampling.
- **Pixel scale.** Each sequence's coordinates are multiplied by
  **113.72 / ℓ_seq**, where 113.72 px is Luna's median ℓ_a
  (`results/ego_bodylen.json` `ell_a.summary.median`). That puts every
  sequence on Luna's pixel scale, so the pixel-denominated priors of every
  fitted model mean what they meant on Luna.
  - ℓ_seq is `ego.body_length` (nose → tail_base) at its median over the
    sequence's frames. It is **per sequence**, because CalMS21 has no animal
    identities.
  - The scaling is applied before any degradation, and identically in every
    arm and every surrogate.
- **Unit of independence: the sequence** (a unique video, by keypoint hash).
  Every Read carries "intervals may be too narrow: a resident can recur across
  sequences undetectably".

**Partitions**, fixed here by `random.Random(0)` on sorted sequence names:

| partition | content | used for |
|---|---|---|
| **fit** | unlabelled sequences that share no video with any labelled file, drawn until the total reaches **573,872 frames**, the size of Luna's MoSeq fit set | fitting MoSeq; the clump bank's PCA and background |
| **map-T1** | Task 1 train, 70 sequences | fitting the state → label map for attack, investigation, mount |
| **eval-T1** | Task 1 test, 19 sequences (different animals from train, per the readme) | scoring attack, investigation, mount |
| **map-G** | 5 of the 15 groom sequences | fitting the groom map |
| **eval-G** | the other 10 groom sequences | scoring groom |
| **ceiling** | Task 2 annotators 3 and 4 on their 10 shared videos | the inter-annotator ceiling |

**No video may sit in both a map and an eval partition.** That is checked by
hash, and any collision drops the video from the map side. An instrument is
applied to every unique video once, and its states are then read by every
partition the video belongs to.

## 2. Noise

### 2.1 Luna's target, per keypoint

- **Level.** σ̄_L = **0.061799 BL** (`results/plant.json` `sigma_bar_bl`). It
  comes from the skull-bone estimator `vieb.seg.jitter` (`residuals`,
  `calibrate`), as the count-weighted mean over confidence bins
  (`scripts/plant.py sigma_bar`).
- **Profile.** The skull estimator sees only nose and ears, so the shape
  across keypoints comes from shapeflow's `sigma_hat_px`
  (`results/inherited/sf_calibration.json`):

  σ_L,k = σ̄_L · s_k / S, with S = √((s_nose² + s_left_ear² + s_right_ear²)/3)

  - S is what the skull estimator measures: a bone residual's variance is the
    sum of its two endpoints' variances, averaged over the three skull bones.
  - `center`'s profile entry is used for the neck.

### 2.2 CalMS21's native noise

- Measured with the **same estimator** on the scaled CalMS21 resident pose,
  pooled over fit ∪ map ∪ eval sequences:
  - `jitter.residuals` per sequence, with `missing = interpolated = False`
    (CalMS21 has no gaps) and ℓ = ℓ_seq;
  - then `jitter.calibrate` on MARS scores;
  - giving σ̄_N, the same count-weighted mean.
- **r = σ̄_N / σ̄_L.** The native profile is taken as r · σ_L,k: the same
  shape at CalMS21's level.
- **If r ≥ 1, the sweep is `NOT_A_RESULT`** ("CalMS21 is already at or above
  Luna's noise"). Clean recovery (level 0) and the controls are still scored
  and reported.

### 2.3 The jitter operator

- At level m, the per-keypoint added amplitude is

  σ_add,k = σ_L,k · √max(0, m² − r²)  (BL; × ℓ = 113.72 px).

  So the quadrature total is m · σ_L,k wherever m ≥ r. **A level with m < r is
  unreachable.** It equals native, and it is labelled as such rather than
  reported as a separate level.
- **Colour.** `vieb.seg.noise.draw(rng, T, 7, fps=30)` gives Luna's measured
  PSD, ACF and common mode. Each keypoint's draw is multiplied by
  σ_add,k(px) / s₁, where s₁ is the per-axis SD of `noise.draw` at scale 1.
  - s₁ is computed **once**, from one 1,000,000-frame draw at seed 0, and
    stored.
  - Per-keypoint rescaling keeps the colour and the common-mode share.

### 2.4 The dropout operator

- **Measured from Luna**, the registered measurement (GAP G6):
  - the 100 tune-split recordings of Luna's MoSeq fit set (`work/moseq/sets.json` `fit`);
  - per keypoint, `missing = DLC missing ∪ vetoed`, and per frame,
    `abstain` (disposition);
  - built exactly as `scripts/moseq_input.py build` builds them.
  - Reported: per-keypoint rate, run-length distribution and abstain rate.
- **Generation per sequence: masks are transplanted**, not simulated.
  - Draw donor recordings uniformly from the 100, with a seeded generator, and
    concatenate their (missing, abstain) masks from a random offset until the
    sequence's length is covered.
  - This carries Luna's rates, run lengths and cross-keypoint co-occurrence
    exactly.
- **Level m.**
  - m ≤ 1: every donor missing *run* is kept with probability m.
  - m > 1: the union of ⌊m⌋ independent full masks and one mask thinned at
    m − ⌊m⌋.
  - m = 0: no dropout.
- **Luna's fill, per instrument, exactly as on Luna:**
  - **MoSeq:** `vieb.moseq.inputs.real(pose, missing, vetoed = 0, abstain,
    scores)`. Gaps of ≤ 3 frames are held, longer ones are NaN with confidence
    0, and the coordinates are filled.
  - **Detector and clumps:** the `raw` arm, `arms.held_array(pose, missing ∪
    abstain)`, with `abstain` blocking boundaries (as `tok/abstain` does on
    Luna).

### 2.5 The sweep

- **m ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0}**, with jitter and dropout jointly
  (the **primary arm**).
- **Seeds 0–4** per level. The seed sets the noise draw, the dropout draw and
  the MoSeq Gibbs seed.
- **Every draw is per sequence**, seeded by `stable_seed(seed, m, sequence)`.
  Nothing is computed across a sequence boundary.

## 3. Instruments

Each consumes the input stage it consumes on Luna, at its committed
hyperparameters:

| instrument | Luna source (commit) | on CalMS21 | output scored |
|---|---|---|---|
| **D: boundary detector** | `vieb/seg/breaks.py` (7b2bfc0), primary cell `scripts/breaks.py` `PRIMARY = ("both", 0.133, 3)`, K_MAD 3.0 | per sequence: `ego.transform` (ℓ = 113.72 after scaling) → ÷ Luna's frozen `basis.json` `sd_used` → `discontinuity` → `mad_threshold` → `boundaries` | boundary frames |
| **C: segment clumps** | `scripts/seg_vocab.py` (6670e1b), group **shape** (`PRIMARY_GROUP`), K_MAD 3.0, SEG_GRID 40, KNN 10, MIN_CLUMP 20; θ **inherited** = `seg_recur.json` `shape|k3` `segment|microstate` θ = **0.189981** | segments from D's machinery on the shape group (`segments_of`, `embed.selectable`), time-normalised. The bank holds **fit + every scored video**, and "animal" = video. PCA is fitted on the fit partition. Cross-video kNN, the bank's own ambient scale, then components at θ | per frame: the clump id of its segment, or **unassigned** (one state) |
| **M: Keypoint-MoSeq** | `scripts/moseq_kpms.py` (5d1e40f): kpms 0.6.8, latent_dim 10, nlags 3, κ 1e4 (AR-HMM, 50 iterations) → **κ 1e6** (full model, to 550), x64 | fitted on the **fit** partition at each (level, seed); `apply_model` (20 iterations) to every scored video at the same (level, seed) | per frame: syllable |

- **κ = 1e6 is the labeller of record's value. It was never fitted.**
- On Luna, MoSeq failed seed stability (AMI 0.473). Its recovery curves are
  read with that caveat.
- **Excluded**, with the reasons in INVENTORY.md:
  - the encoder (no code; disqualified);
  - the recurrence statistics (not labellers).

## 4. Scoring

- **Labels are read only by `vieb/calms/score.py`.** A test asserts that no
  other module under `vieb/` or `scripts/` opens the label store or imports
  `score`, except the scoring script.

- **Map.** Each instrument × level × seed gets its own map. Each state goes to
  the label with the most frames in that state on the map partition. The
  vocabularies are:
  - {attack, investigation, mount, other} for T1;
  - {groom, other} for G.

  The map is many-to-one, and a state unseen on the map side maps to `other`.
- **Event F1** (the deciding metric for C and M), per label:
  - true bouts are runs of the label;
  - predicted bouts are runs of the mapped label;
  - matching is greedy one-to-one at **IoU ≥ 0.5**;
  - F1 is pooled over the evaluation sequences.
- **Frame NMI**, reported and not deciding: NMI (arithmetic) between the
  binary label indicator and the **unmapped** state sequence.
- **Boundary F1** (the deciding metric for D), per label:
  - the true boundaries are the label's bout onsets and offsets;
  - the predicted ones are D's boundaries;
  - matching is one-to-one within **±0.5 s (15 frames)**.
- **Chance.**
  - **1,000 circular shifts** of the labels within each sequence (seed 0),
    applied to map and eval partitions alike.
  - **The map is refitted under every shift.**
  - chance = the mean; the null's 95th percentile is p95.
- **Ceiling**, computed with the same metric functions:
  - **attack, investigation, mount:** annotator 4 scored against annotator 3,
    and 3 against 4, averaged, on their 10 shared videos;
  - **groom: ceiling = 1**, with no second annotator (user's decision).
- **Normalised score = (s − chance) / (ceiling − chance)**, with the assertion
  ceiling > chance.
- **Support.** A label is scored only with ≥ 30 true bouts in its evaluation
  partition, across ≥ 10 sequences. Otherwise it is **`GRID_LIMITED`**.
  - From DATA_FACTS, **attack** has 258 bouts in **5** eval-T1 sequences, so it
    is expected to be GRID_LIMITED.
  - That is stated here, before any score exists.
- **Interval.** **2,000 bootstrap replicates over evaluation sequences**
  (seed 0). Each replicate recomputes the pooled score for each of the 5 seeds
  and averages them. Chance and ceiling are held at their point values. The
  95% percentile interval follows.

## 5. Recovery

- **Recoverable** (behaviour × instrument × level) iff **the lower 95% bound
  of the normalised score is > 0.25 and the seed-mean score is > the chance
  null's p95.**
- **Breakdown noise.** The lowest swept level at which a behaviour that was
  recoverable at the level below stops being recoverable. It is asserted to lie
  inside the swept range. Otherwise it is reported as a bound:
  - "> 3.0×" if it is recoverable throughout;
  - "< first reachable level" if it is never recoverable.

## 6. Negative controls

- **white:** `recur.recurrence.surrogate.white_surrogate`, an i.i.d. Gaussian
  at each coordinate's own mean and SD.
- **phase:** `phase_surrogate`, common-phase randomisation of the (T, 14)
  pose.

Both are built **per sequence** from the scaled native resident pose, with no
dropout (level 0), and go through **every instrument and the identical
scoring**. MoSeq is refitted on the surrogate fit partition, with seeds 0–4
for each control.

> **The controls fail iff the §5 rule declares any scored label recoverable
> for any instrument on either control.** If they fail, **every Step 1 verdict
> is `NOT_A_RESULT`.**

## 7. Decision rule (verbatim from the prompt, with grooming as the sole Luna-relevant positive)

- **Luna-relevant positive: groom.** Rearing is untestable (§10, D-S1).
- **Easy positives:** attack, investigation, mount.
- "Recoverable at a level" means by **any** of D, C or M.

| outcome | condition | Step 3 branch |
|---|---|---|
| **O1: method fails** | no behaviour recoverable at native (m = 0) | 3A |
| **O2: input-limited** | ≥ 1 behaviour recoverable at native, and **groom not recoverable at 1.0×** | 3B |
| **O3: input sufficient** | **groom recoverable at ≥ 1.0×** | 3C |
| `NOT_A_RESULT` → stop | the controls fail (§6), or r ≥ 1 (§2.2) | — |
| `INCONCLUSIVE` → stop | a deciding CI straddles 0.25 | — |

**Deciding CIs.** These are the intervals whose crossing could change the
outcome:
- **for O1:** every native-level interval;
- **for O2 vs O3:** groom's intervals at 1.0×, for each instrument.

If O3 already holds on one instrument outright, a straddling groom interval on
another instrument does not make the result INCONCLUSIVE.

**If groom is `GRID_LIMITED`**, the O2/O3 distinction is `NOT_A_RESULT`, and
O1 vs not-O1 is still reported.

If only easy positives survive at 1.0×, the outcome is O2. The report then
says that vigorous behaviours survive, and that **this says nothing about
behaviour in a fear session.**

## 8. Every claim is a `Read`

`recur.read.Read`: verdict, scored_object (instrument, label, level, arm,
metric), reason, and `n_effective` = **evaluation sequences**. Audits assert
their own check counts. Refusals are listed with the same prominence as
results.

## 9. Registered but deferred (run only after the human has seen the verdict)

- **Secondary arms:**
  - jitter only;
  - dropout only;
  - the **6-keypoint arm**: neck dropped, and `center`'s slot filled by the
    midpoint between the ear midpoint and the hip midpoint.
- **§1.8 decision-importance ablation:** registered separately, before it
  runs.

## 10. Deviations from the prompt, decided before this registration

- **D-S1. Rearing is untestable.** CalMS21's "whiterearing" is the intruder's
  behaviour ("unless specifically noted", white = intruder), and it occurs in 3
  sequences. **Groom alone** is the Luna-relevant positive. *Decided by the
  user after reading DATA_FACTS.*
- **D-S2. The groom split is 5 map / 10 eval**, so support can be met. *Decided
  by the user.*
- **D-S3. Groom's ceiling is 1.** No second annotator exists. *Decided by the
  user.*
- **D-S4. Recurrence is not scored.** It assigns no states. The clumps built on
  it (C) are scored.
- **D-S5. "Fitted κ" does not exist.** κ = 1e6 (G4).
- **D-S6. No new AR fit for jitter colour.** `vieb.seg.noise` already
  synthesises Luna's measured spectrum, and it rejects AR(1).
- **D-S7. Per-sequence ℓ and per-sequence pixel scaling**, because there are no
  identities.
- **D-S8. The per-keypoint profile comes from shapeflow's `sigma_hat_px`,** a
  second estimator, scaled to the skull estimator's level (G7).

## 11. Prohibitions

- No hyperparameter, threshold, seed, partition or arm changes after any
  CalMS21 state exists.
- No label is read outside `score.py`.
- No Step 3 work before the human has seen the verdict.
- No Step 1 number enters Paper 1.
