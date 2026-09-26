# Registration — STATES: behaviour as maximally predictive states and a transfer operator

Committed **alone, before `vieb/states/` exists**, while the whole-corpus mask
run is in progress and before any repaired feature exists. Inherited digest
**`198eb14ff258c7f6`**.

**Why this algorithm.** Every discovery attempt so far failed or was refused:
- frame-level k-means tokens (`ALPHABET.md`: median run 1 frame);
- the hazard/MDL ladder (`FALSIFIER.md`: surrogates beat the corpus);
- the acceleration-break detector (`DETECTOR.md`: 12.8% of plants recovered);
- segment clumps (`VOCAB.md`: 98% unassigned);
- Keypoint-MoSeq, audited upstream: tracker-driven syllables and a white-noise grammar.

Density methods fail because 96–99% of the density is one mode. **Metastability
is a property of dynamics, not density** (Costa & Stephens et al.). It appears
in the slow eigenvalues of a transfer operator built on states that include
their recent history. This is the external review's lead recommendation, and
it has never been tried here. A **CONTINUUM** verdict is an allowed, reportable
outcome.

---

## 1. Input

The pose arm REPAIR adopts (`repaired`, or its registered fallback), through
`ego.transform` (17 channels), plus the **mask stream** if it passes §2's leak
gates:
- mask area / bl²;
- elongation (moment eigen-ratio);
- mask-centroid speed (bl/s);
- mask turning rate.

Validity: ego-valid, not abstained, and (for the mask stream) a carried mask.
Features are standardised with means and SDs fitted on **tune**,
segment-balanced as in `quantize`.

## 2. Gates on the representation (before any state is fitted)

* **Identity leak:** the `EGO.md` leak probe on the pose stream. PASS as for the
  existing representation.
* **Context during immobility, per stream:** on frames with ego speed 0,
  logistic regression from the stream's features to context A/B, with
  animal-grouped 5-fold cross-validation on **tune**. Behaviour is by
  definition equivalent across contexts on those frames, so above-chance
  decoding is arena leakage. **A stream whose balanced accuracy has a lower
  bound > 0.60 is dropped.** The mask stream is the one at risk, given the
  arena-dependent head coverage (`POINT_PROMPTS_GUARDED.md`).

## 3. Maximally predictive states (fitted on tune only)

1. **Delay embedding:** x_K(t) = [x(t), …, x(t−K+1)] within a recording, valid
   only if all K frames are valid.
2. **PCA to 30 dimensions**, fitted on tune.
3. **Microstates:** mini-batch k-means with **N = 1,000** (seed 0) on tune.
4. **Choosing K ∈ {1, 2, 4, 8, 16, 32} frames:** the tune animals are split in
   two halves (seed 0). The microstate partition and the one-frame transition
   matrix are fitted on one half, and the **cross-entropy of the next
   microstate** is scored on the other; then the halves swap. **K\* is the
   smallest K whose mean cross-entropy is within 1% of the grid minimum.** This
   is the "maximally predictive" choice, and it is made on held-out data, so
   larger K cannot win by overfitting.
5. **Lag τ ∈ {1, 2, 4, 8, 16, 32, 64} frames:** on tune, the implied
   timescales tᵢ(τ) = −τ / ln|λᵢ(τ)| of the transition matrix P(τ).
   **τ\* is the smallest τ whose t₂ changes by less than 10% at the next
   grid τ.**
6. **Fit:** microstates from tune are assigned to fit and report. P(τ\*) is
   estimated on **fit**, and everything is **reported on report** (89 animals).

## 4. Metastable states and gates

**Timescales with intervals:** tᵢ, i = 2…11, on report, from an animal-level
bootstrap (2,000 replicates, seed 0) of the summed transition counts.

* **Gate 1, the spectral gap beats the surrogates.** Each surrogate is
  generated per recording on the standardised feature stream
  (`vieb/tok/surrogate.generate`: `phase`, `var5`, `microstate`,
  `microstate0`, `white`), then passed through the **whole** pipeline: its own
  PCA, k-means, and P(τ\*). Timescale tᵢ **counts** iff its report lower bound
  exceeds the **maximum over surrogates** of their tᵢ upper bound.
  **M = 1 + the number of consecutive counting timescales from i = 2.**
  **If M = 1, the verdict is CONTINUUM** and no state is named.
* **The states:** PCCA+ (inner-simplex algorithm) on the first M right
  eigenvectors of P(τ\*) gives soft memberships. **A frame belongs to a state
  if its membership is ≥ 0.8; otherwise it abstains.**
* **Gate 2, reproducibility:** frame-level AMI **≥ 0.7** between the
  assignments of the fit-estimated model and a report-estimated model on
  report frames, **and** between k-means seeds 0 and 1.
* **Gate 3, planted recovery:** on a copy of the report stream, a binary
  switching signal (gamma-distributed dwells, mean 2 s) adds **+1 SD** to two
  channels (body-extension proxy and speed) of 20% of recordings. The pipeline,
  refitted on the planted copy, must produce a state whose membership separates
  planted from unplanted frames with **balanced accuracy ≥ 0.8**.
* **Reported as a headline, not gated:** the abstention share, state
  occupancy, and dwell distributions.

**The verdict:**
- **PASS** iff gates 1–3 pass with M ≥ 2;
- **CONTINUUM** if M = 1 (gate 3 still reported, to show the pipeline could have
  found states);
- **FAIL** if gate 2 or 3 fails with M ≥ 2.

## 5. What a PASS licenses

**License:**
- named metastable states, with their clips, occupancy by context × day, and
  dwell hazards;
- any head-scale state's context comparison **conditioned on per-frame head
  coverage** (`POINT_PROMPTS_GUARDED.md` §3).

**Do not license:**
- reading a state as a named behaviour without looking at its clips;
- shock-versus-no-shock framing.

## 6. Prohibitions

No report-split tuning: K, N, τ, the 30 PCs, the 0.8 membership rule and
every bar are fixed here or chosen on tune by the rules above. Surrogates pass
through the identical pipeline. Nothing is read as grooming.

---

# Amendment 1: a connected operator, a non-trivial choice of K, and a gate on the gap

**Committed alone. Recorded as `DEVIATIONS.md` D28.** Made before any real
feature exists: the whole-corpus masks are still running, so REPAIR has not
run and no mouse data has entered `scripts/states.py`. The evidence is the
synthetic end-to-end test (`tests/test_states_pipeline.py`): real
tune/fit/report animal IDs with fake features carrying a hidden two-state
switch (gamma dwells, mean 3 s). As registered, the pipeline returned
**CONTINUUM on data containing a genuine switch**. Three defects, each fixed
here:

1. **Disconnected chains (§3.5, §4).** Closed islands of microstates give an
   eigenvalue of exactly 1 and an "infinite" timescale (10¹² frames in the
   synthetic run) that only records a disconnection. **The operator is
   restricted to its largest strongly connected set**, standard Markov-state
   practice (`operator.transition_matrix`).

2. **K chosen by next-microstate cross-entropy is rigged (§3.4).** Consecutive
   delay windows overlap by K − 1 frames, so the next microstate becomes
   trivially predictable as K grows. In the synthetic run K\* went to the grid
   maximum (32). **Replacement:** K\* is chosen by the **held-out share of
   variance of the features H = 15 frames (0.5 s) beyond the window's end**
   that the microstate explains (`micro.predictive_r2`), on the same tune
   halves. It is the smallest K within 1% of the grid maximum. The window
   cannot overlap the target.

3. **Gate 1 compared raw timescales, which linear surrogates reproduce (§4).**
   Phase and VAR surrogates keep the data's autocorrelation, so their slowest
   timescale is as long as a genuine switch's, and comparing tᵢ cannot detect
   metastability. **Metastability is a gap:** a few slow modes separated from
   the rest. **Replacement:** gap ratios rᵢ = tᵢ / tᵢ₊₁, i = 2…10, with
   animal-level bootstrap intervals. A gap after mode i **counts** iff its
   report lower bound exceeds every surrogate's upper bound at the same i.
   **M = the largest counting i.** If none counts, the verdict is
   **CONTINUUM**.

**On the synthetic switch, after the fix:**

| | corpus | phase | var5 | microstate | microstate0 | white |
|---|---|---|---|---|---|---|
| t₂ / t₃ | **5.14 [4.34, …]** | 2.24 (hi 2.38) | 2.18 (hi 2.38) | 1.00 | 1.00 | 1.00 |

The gap is found (M = 2). **The linear surrogates sit near the ladder a linear
process gives, about 2**, and the memoryless ones at 1: the reference the gate
is meant to beat.

**Unchanged:** the input, the §2 leak gates, N = 1,000, 30 PCs, the τ rule,
the fit/report roles, PCCA+ with the 0.8 membership rule, gates 2 and 3, the
surrogates and their identical pipeline, and every other bar.
