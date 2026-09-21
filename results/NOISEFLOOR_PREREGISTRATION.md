# Pre-registration — the tracking-noise floor under the boundary detector

Written and committed **before `vieb/seg/jitter.py`, `vieb/seg/floor.py` or
`scripts/noise_floor.py` exist**, and before any floor is computed. Source array
`work/clean/<rid>.npz` via `spine.clean`, the **`raw`** arm
(`held_array(pose_unfiltered, missing)`), through the standard ego transform.
Digest `198eb14ff258c7f6`.

## 1. The question, and why it is not the one that just failed

VIEB has been asking one question that is really two.

**Question 1 — does the detector find what it claims to find?** A question about
the *instrument*. The detector defines a boundary as a point where acceleration
stops being continuous; whether it finds those points is answerable with
synthetic tests alone. No human and no biology.

**Question 2 — are those points behaviourally meaningful?** A question about the
*mouse*, answerable only against something outside the pose stream.

`ANNOTATION.md` asked both at once, of human raters, at 67 ms resolution on an
overhead view, and the ceiling failed. **This stage asks only Question 1**, and
its whole point is that it needs no ceiling to stand.

**The hypothesis.** The detector of record cuts **0.443 boundaries per second**
(`BREAKS.md`, `shape`, `deriv_sec = 0.133`, `degree = 3`) on a corpus that is
mostly a still mouse seen from above through 7 keypoints. A large share of those
boundaries may fire in low-motion periods where DLC jitter is comparable to real
movement. `ROUGHNESS.md` establishes the signal is piecewise-smooth — corpus log
mean-squared increment **−7.67** against `phase`'s **−5.45**, a factor of 9.2 in
15 of 17 channels — but **that is not the same claim**. A piecewise-smooth signal
can still be sampled by an instrument whose own noise dominates in the still
regime, and nothing on disk separates those.

So: **measure the floor.** Not assume it, not argue it from the axiom.

## 2. Split, and why it is `tune`

**`tune`: 60 animals, 770 recordings.** Not `fit`, not `report`.

The reason is not caution, it is contamination. The floor this stage produces is
intended to **select a parameter** later — a future detector's smoothing
constant is to be chosen as the smallest value whose jitter-only rate falls below
a registered fraction of the corpus rate. Choosing a parameter against a number
is fitting, and a number computed on `report` would contaminate any coverage
figure scored there. `ANNOTATION_PREREGISTRATION.md` §2 fixed exactly this
reasoning for exactly this reason, and it is inherited rather than re-argued.

**Nothing in this stage is reported on `fit` or `report`, and no number here may
be quoted as a corpus-wide rate.** It is a `tune`-split measurement of an
instrument property and says so on every line.

## 3. The calibration, and why bone length and not stillness

Everything below needs a jitter amplitude. The obvious estimator — the
high-frequency residual during still stretches — is **circular**: "still" is
defined by the smoothness whose noise is the thing being measured.

**The skull triangle is not circular.** `bones.SKULL = ((0,1), (0,2), (1,2))` —
left_ear–right_ear, left_ear–nose, right_ear–nose — spans a rigid structure. Its
length is physically constant whatever the animal does, so **all** of its
frame-to-frame variance is tracking noise. `bones.bone_lengths(pose, bones)`
already computes it.

**Trunk bones are excluded and the exclusion is not optional.** `vieb/qc/bones.py`
records that trunk bones flex, so a length change there is real posture at least
as often as it is tracking. Including them would attribute posture to noise.

**The deliverable is σ²(c)**: displacement noise variance as a function of DLC
confidence, binned on `conf` across its full observed range. Not a parametric
fit — the empirical distribution, inverse-CDF interpolated, following the
precedent `DISPLACEMENT_Q` set in `INJECTION_PREREGISTRATION.md`.

### Two precedents this stage departs from, named rather than skipped

**Confidence as a weight.** `vieb/qc/disposition.py:316` fixes that confidence is
consumed "as a covariate, **never as a likelihood weight**", because a weighted
likelihood is not a code length — a model lowers its cost by downweighting what
it predicts badly. **That argument is about MDL and does not reach here.** An
observation variance inside a detector is not a code length and buys no model
its own reward. The departure is deliberate and is recorded in `DEVIATIONS.md`.

**Gaussian noise.** `vieb/qc/inject.py:3-7` refuses Gaussian perturbation
benchmarks: *"Gaussian noise is the thing a smoother is optimal against."* That
argument is about benchmarking **smoothers**, and this stage benchmarks a
**detector**, where the concern inverts. Even so the precedent is followed in
substance: jitter is drawn from the **measured** skull-residual distribution, not
from a Gaussian.

**`low_confidence` is not used.** It is identically all-False corpus-wide —
shapeflow's threshold estimator refused on this corpus for want of a bimodal mode
(`clean_thresholds.json`: all seven `None`). A mask with no variance is not a
covariate. Continuous `conf` is used throughout.

## 4. The floor: three injection arms, fixed now

Every arm goes through the **identical** detector at the identical settings —
`bk.discontinuity(sub, h=4)` → `bk.mad_threshold(d, k_mad=3.0)` →
`bk.boundaries(d, thr, min_gap=16, blocked=abstain)`, `DERIV_SEC = 0.133`,
`DEGREE = 3`, `guard = 2` at 30 fps — and through the same standardising SD and
the same ego transform. Jitter is added in **keypoint space**, before the ego
transform, because jitter is a property of the tracker in pixels and not of the
representation.

| arm | construction | what it measures |
|---|---|---|
| `static` | the animal's own median pose, held constant, **no noise** | the mechanical check. A signal with no variance must produce **no boundaries** |
| `jitter` | that same constant pose **+ noise drawn at σ²(c)** | **the noise floor.** Every boundary here is tracking noise, by construction |
| `jitter×0.5`, `jitter×2` | the same at half and double amplitude | dose-response. If the mechanism is jitter, rate rises with amplitude |

**The threshold is per-recording and adapts.** `mad_threshold` recomputes from
each stream's own `D`, so the jitter arm gets its own adaptive threshold exactly
as the corpus does. That is what makes the comparison fair, and it also means the
floor is **not expected to be zero**: a 3-MAD rule with a 16-frame refractory
period fires at some rate on any noise process. **That rate is the quantity of
interest.**

**Reported both ways.** Absolute boundaries/s, and as a fraction of the **NMS
resolution ceiling**, `fps / min_gap = 30 / 16 = 1.875` boundaries/s. The
detector of record sits at 0.443/s, which is **23.6%** of that ceiling.
`SEGMENTATION_PREREGISTRATION.md` already established that rates near the ceiling
are a property of the refractory period rather than of the signal, and the same
diagnostic applies to a floor.

## 5. The joint stratification, and the confound it exists to break

**One axis is not enough, and reporting one axis would mislead.** Low DLC
confidence is concentrated at the wall, where the mouse rears. Rearing is a real
behaviour that *also* breaks tracking. So "boundaries pile up where confidence is
low" is ambiguous between two opposite readings, and a marginal cannot resolve
it.

**Primary grid: speed quintile × confidence quintile × arena tercile = 75 cells.**
Speed is `quantize.speed(X)` — `hypot` of ego channels 14–15, body lengths per
second, excluding ω because a spin is not a displacement. Confidence is per-frame
`conf`, reduced across keypoints by **minimum** (the weakest keypoint is what
breaks a pose). Arena position is `concentration.edgeness(held[:, 3])` in IQR
units, cut at the corpus terciles.

**Marginals reported separately at full resolution**, including the 10-decile
arena profile against `results/concentration.json#/edge_profile`, so the
published decile structure stays comparable.

### The reading, fixed before the numbers exist

| observation | reading |
|---|---|
| corpus rate in **still, centre-arena, low-confidence** cells is **at or below** the floor | the noise diagnosis holds: those boundaries are jitter |
| boundary density concentrates in **fast** or **wall** cells at low confidence | rearing-through-bad-tracking. **This is not a noise result** and may not be reported as one |
| corpus rate is **separated from the floor in every cell** | the detector is above its own noise floor, the hypothesis in §1 is wrong, and the continuum reading strengthens |

**A self-check the stratification must pass.** `BEHAVIOUR.md` measured the island
at edgeness **1.69** against **3.61** for the same animals' other segments — it
sits *nearer the centre*, the opposite of the by-eye impression (`M7`). If the
stratification cannot recover that, it is measuring something else and the result
is refused.

## 6. Cross-arm concordance, and the sharpest prediction here

Re-run the frozen detector on **`raw`, `viterbi`, `disposition`** — the three
arms `F3_PREPROCESSING_FREEZE.md` carries.

**`wiener` and `butterworth` are excluded by name.** A low-pass filter
manufactures exactly the smoothness whose breaks this detects. They are also
`STORED_ARMS`, read off disk and not applicable to an array, so they could not be
run here regardless.

Boundary sets are compared with **`annot.match` / `annot.prf`** — greedy,
nearest-first, one-to-one — at the same ±2/±5/±10 bands, reporting F1 per band
and a per-boundary survival count across the three arms.
**`scripts/breaks.py:165 _agreement` is not used**: it counts unmatched hits,
which inflates agreement exactly where a detector is noisiest.

> **This is where the design has the most leverage.** `viterbi` reassigns
> **0.31%** of keypoint-frames (`CLEANING.md`). If changing three keypoint-frames
> in a thousand moves most of the boundaries, the boundaries are sitting on
> noise — and that inference needs no floor, no human and no surrogate.

## 7. Predictions, falsifiable, fixed now

1. **`static` produces exactly zero boundaries.** Mechanical. If it fails the
   harness is wrong and no number below means anything.
2. **The `jitter` floor is materially above zero** — the frozen detector fires on
   pure tracking noise. *Falsifier: a floor at or near zero means the detector is
   noise-immune at this amplitude, the §1 hypothesis is dead, and this stage
   reports that and stops.*
3. **Boundary rate rises monotonically with jitter amplitude** across ×0.5, ×1,
   ×2. *Falsifier: a flat or non-monotone dose-response means the rate is set by
   the threshold rule and the refractory period rather than by the noise, which
   would make the floor a property of the detector's plumbing and not of the
   tracker — reportable, but a different finding.*
4. **Cross-arm concordance at ±2 between `raw` and `viterbi` is below 0.5**,
   despite `viterbi` moving 0.31% of keypoint-frames. *Falsifier: concordance
   above 0.9 means the boundaries are robust to the cleaner and the noise reading
   is substantially weakened.*
5. **Corpus boundary rate in the still × centre × low-confidence cell is not
   separated from the floor.** This is the §1 hypothesis stated as a testable
   cell. *Falsifier: separation with a non-overlapping animal-level interval.*

**Direction is predicted here and was not in `FREEZING_PREREGISTRATION.md`,
deliberately.** There the sign would have required inventing a context-to-shock
mapping the repository does not hold. Here the direction follows from a mechanism
that is already measured — a numerical second derivative of a noisy signal — so
predicting it costs nothing and refusing to predict it would be false modesty.

## 8. What this stage may not do

* It may not **re-score Q1** or restate **Phase F**.
* It may not **score the frozen detector against the human marks**, at any
  tolerance. The ceiling failed and `ANNOTATION_PREREGISTRATION.md` §6 blocks it.
* It may not **tune**. No threshold, window, degree or `k_mad` is adjusted. The
  detector is frozen at the settings above and any change makes a different
  detector needing its own registration (`DETECTOR_PREREGISTRATION.md:97-100`).
* It may not use **any array from the Wiener arm**.
* It may not **report a pooled rate when strata disagree in sign or order of
  magnitude** — the rule `INJECTION_PREREGISTRATION.md` set after Phase F's pool
  turned out to be biased 5.5× toward slow frames.
* It may not report a stratum below **20,000 scored keypoint-frames**. Such cells
  are **refused**, and the count of refused cells is reported. A thin cell is not
  reported with a caveat.
* It may not **compute across a recording boundary**. Injection is per recording,
  as `planted.ego_plant` already is, and asserted by test.

## 9. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; **refusal is a correct outcome**. Animal-level bootstrap only —
`recur.boot.animal_interval`, 2,000 replicates, `how="mean"`, clustered on
animal; the frame-level interval printed beside it solely to show how much
narrower the wrong method looks. Ranked on the effect-CI lower bound, never on a
p-value. Windows in seconds, converted once via `recur.util.frames`. Body lengths
via `ego.ell_a`, per animal. Seed 0, `seeds.stable_seed`. `mypy --strict` on new
modules. The **`raw`** arm asserted by name.

## 10. The success rule for a vocabulary, decided before there is one

Registered here rather than later **because it is what makes the floor
load-bearing** rather than a diagnostic aside. If this stage produces a floor,
the following governs what any future tokenization may claim.

The expected outcome of a multi-scale detector is that coarse, persistent
boundaries form tokens — freeze, move, rear, groom-like — while the fine scale
behaves like a continuum. **That is also what every method in this field already
finds**, so recovering it is not on its own a result.

**A coarse vocabulary counts as success only with the floor under it.** Two
conditions, both fixed now:

1. **Above the floor.** Every retained token's boundaries sit above the §4
   measured noise floor **in the stratum they occupy**, by a non-overlapping
   animal-level interval — not above a floor asserted from theory.
2. **Arm-invariant.** The boundaries survive §6 cross-arm concordance across
   `raw`, `viterbi` and `disposition`. A token whose boundaries move with the
   cleaner is a cleaner artifact wearing a behaviour's name.

The contribution is then not that the tokens exist but that they are **the first
such tokens with a measured floor underneath them** — every prior report of this
vocabulary, this programme's own included, asserted the floor rather than
measuring it. **A coarse vocabulary failing either condition is reported as a
negative result about the detector, not as a vocabulary.**

**The fine scale is a second, independent prize**: a continuum-with-floor,
reported by an instrument that *could have* found tokens there and did not.
Neither prize exists without §4.

**A caveat that comes due before any rearing token is named.** Rearing seen from
an overhead camera partly tracks **foreshortening**: the projected geometry
changes because the camera sees a shortened body, not because the pose model
resolved a posture. `METHODS_FINDINGS.md` M7 records the general form, and
`ADJUDICATION.md` records the specific precedent — two frames read by eye as
"rearing against a wall" were **teleports**.

## 11. What this stage does not discharge

It does **not** discharge the registered negative control.
`DETECTOR_PREREGISTRATION.md:84-90` requires any detector cell clearing the 20%
isolation gate to be run on **`white`** and to fail there. No cell has ever
cleared (`DETECTOR.md:104-109`), so it has never run. Measuring a noise floor is
not the same test and does not substitute for it. It comes due the moment any
future detector passes.

It does **not** build the C² plant. `DETECTOR.md:98-102` states that a planted
instance whose insertion is a genuine acceleration discontinuity needs its own
registration, and that is still true. The floor is a precondition for designing
that probe — it fixes the amplitude the plant must stand above — not a
replacement for it.
