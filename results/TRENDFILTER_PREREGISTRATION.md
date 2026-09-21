# Pre-registration — a detector with an explicit error model, scored on the same plant

Written and committed **before `vieb/seg/trendfilter.py` or
`scripts/trendfilter.py` exist**, and before any λ is chosen or any recovery
measured. Source array `work/ego/raw__bodylen__*.npz`, the F3 `raw` arm;
`shape` group; **`tune` split, 60 animals**. Digest `198eb14ff258c7f6`.

## 1. Why a new detector, and why now

`PLANT.md` measured what the frozen detector can actually do: to recover an
**acceleration** discontinuity — the only break its axiom claims — at 50% it
needs a peak displacement of **16 σ̄ ≈ 0.99 body lengths**, against **4 σ̄** for
a position jump. At 1 σ̄ it is at chance. `NOISEFLOOR.md` showed its floor is
structural — a 4× change in tracking noise moves its rate 6.5%, because
`discontinuity` is scale-equivariant and `mad_threshold` scale-adaptive — so
**the floor cannot be retracked away**. Only a different criterion can move it.

**The mechanism is identifiable.** The frozen detector takes a numerical second
derivative of DLC position (`breaks.second_deriv_weights`) and thresholds its
mismatch. Differentiating a noisy signal twice amplifies exactly the jitter
`NOISEFLOOR.md` calibrated. An estimator that never differentiates the data
should do better, and this registration fixes what "better" means before it is
measured.

## 2. The estimator

**ℓ1 trend filtering of the third difference, group-sparse across channels,
with a weighted fidelity term:**

> minimise over `X`:  `½ · Σ_t w_t ‖Y_t − X_t‖²  +  λ · Σ_t ‖(D³X)_t‖₂`

* **`D³` on position** gives a piecewise-**quadratic** fit — piecewise-linear
  velocity, piecewise-constant acceleration — whose knots are **exactly
  acceleration jumps**. The axiom, written as an estimator. It never
  differentiates the data.
* **Group norm across the 14 `shape` channels**, so a knot is shared: a
  behavioural boundary is a boundary in the animal, not in one keypoint.
* **`w_t = 1 / σ²(c_t)`**, the per-frame observation variance from
  `NOISEFLOOR.md`'s skull-bone calibration. This is where the error model
  enters. `vieb/qc/disposition.py:316`'s bar on confidence as a likelihood
  weight is about MDL code lengths and does not reach a detector's observation
  variance; the departure is already recorded in `NOISEFLOOR_PREREGISTRATION.md`
  §3 and is inherited here.
* Solved by ADMM with a banded Cholesky step. Convex, one parameter,
  deterministic. **No `cvxpy` or `ruptures` in this environment**, and the venv
  is shared with `~/recur`, so it is reimplemented against `scipy` following
  the precedent `vieb/clean/arms.py:11-22` set for `movement`.

**A knot at row `r` of `D³` is reported as a boundary at frame `r + 2`** — the
first sample after the break — fixed now so it is not chosen to flatter a
recovery number.

## 3. λ, chosen against the measured floor and not by eye

λ is swept as a fraction of each recording's own `λ_max = max_t ‖(D³Y)_t‖₂`,
so the parameter is scale-adaptive per recording exactly as `k_mad` is. The
grid is **α ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}**, fixed here.

> **λ\* is the smallest α whose jitter-only boundary rate has an animal-
> bootstrap upper bound below 5% of the corpus boundary rate at the same α.**

The jitter-only arm is `NOISEFLOOR.md`'s: a constant pose carrying noise drawn
at the calibrated σ(c). **The 0.05 is invented here and named**, the way `PAD_S`
was: at most one boundary in twenty at the finest retained scale is
attributable to jitter. Fixed before the sweep and not moved to meet a result.

**If no α on the grid satisfies it, the result is `GRID_LIMITED`** and the
direction is reported. The grid is not extended afterwards — doing so would be
selecting a hyperparameter against an outcome, which is what `LADDER.md`
refused to do when MDL pushed below N = 256.

## 4. The comparison, on the identical instrument

The new detector is scored on **`scripts/plant.py`'s harness, unchanged**: same
orders (0, 1, 2), same amplitude ladder ({1, 2, 4, 8, 16, 32} × σ̄), same ±2
band, same per-recording chance level, same `white` negative-control arm, same
`tune` animals. **Only the peak source differs.**

That is the whole point of having built the plant first. Two detectors scored on
one instrument is a comparison; two detectors each scored on its own probe is
not.

## 5. Predictions, fixed now

1. **λ\* exists on the grid** — not `GRID_LIMITED` at either end.
2. **At λ\*, order-2 recall at 8 σ̄ exceeds the frozen detector's 0.322.** The
   headline claim. *Falsifier: it does not, and an estimator built directly on
   the axiom is no better at the axiom's own break than a twice-differenced
   threshold — which would be strong evidence the limit is the data and not the
   criterion.*
3. **At λ\*, the jitter-only boundary rate is below the frozen detector's
   0.2599/s.** An error model that buys nothing on pure noise has not earned
   its parameter.
4. **Recovery stays ordered 0 ≥ 1 ≥ 2.** An estimator reporting order 2 as
   *easier* than order 0 is mis-locating knots, not finding them.
5. **The `white` arm stays below the corpus arm above 4 σ̄**, as it does for the
   frozen detector. A new detector re-incurs the negative control
   `DETECTOR_PREREGISTRATION.md:84-90` requires; it is not inherited.

## 6. What this stage may not do

* It may not **retroactively change any published number**.
  `DETECTOR_PREREGISTRATION.md:97-100`: a better detector makes a **new**
  measurement, and `SEGRECUR.md` and `VOCAB.md` stand on the detector they were
  computed with.
* It may not **re-run segment recurrence, re-cluster, or re-derive coverage**.
  That is a separate stage with its own registration and it is explicitly not
  started here.
* It may not **tune ρ, the iteration count, or the knot offset** against a
  recovery number. ADMM settings are numerical and are fixed by a convergence
  check, not by an outcome.
* It may not **extend the α grid** after seeing where λ\* falls (§3).
* It may not **re-score Q1**, restate **Phase F**, or use any **Wiener** array.
* It may not **compute across a recording boundary**.

## 7. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only,
2,000 replicates, `how="mean"`. Ranked on the effect-CI lower bound, never a
p-value. Seed 0, `seeds.stable_seed`. `mypy --strict` on new modules. The
**`raw`** arm asserted by name. `assert_threshold_in_range` does not exist in
`vieb/checks.py` and should be written when λ\* is derived from a distribution.
