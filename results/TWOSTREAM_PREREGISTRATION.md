# Pre-registration — the two-stream detector, at the scope Stage 0 left

Written and committed **before `vieb/seg/twostream.py` or
`scripts/twostream.py` exist**. Source `work/ego/raw__bodylen__*.npz`, F3 `raw`
arm, `shape` group, **`tune` split for the floor, `report` for anything scored
against the design**. Digest `198eb14ff258c7f6`.

## 1. What Stage 0 left standing

`DYNAMICS.md` measured the descriptors against measured-colour tracking noise,
where 5% of corpus windows should exceed the null's 95th percentile by chance:

| descriptor | corpus above null p95 | usable? |
|---|---:|---|
| **`pole_radius`** | **0.887** | **yes** |
| `band_slow` 0.5–2 Hz | 0.587 | yes |
| `band_marginal` 4–6 | 0.383 | no |
| `band_fast` 6–15 | 0.273 | no |
| `band_mid` 2–4 | 0.245 | no |
| **`pole_frequency_hz`** | **0.021** | **no — below chance** |

**So the dynamics stream is a *persistence* stream in the slow band, and
nothing else.** Frequency is not used. Bands above 2 Hz are not used. This
registration does not revisit that.

## 2. The two streams

**Configuration.** The **frozen detector**, unchanged:
`breaks.discontinuity` → `mad_threshold(·, k_mad = 3.0)` → `boundaries`, at
`deriv_sec = 0.133`, `degree = 3`, on the 14 `shape` channels, plus
**`triage.body_extension`** (nose→tail_base over the recording's own median) as
one additional channel. Body extension is the stretch-attend descriptor and is
already built and tested; it is added because a crouch-to-elongate transition is
a configuration change the shape channels alone may express weakly.

> Adding a channel makes this a **different detector** from the one
> `SEGRECUR.md` and `VOCAB.md` stand on (`DETECTOR_PREREGISTRATION.md:49-52`
> fixes that a channel-group change is a different detector, not a filter). It
> therefore makes a **new** measurement and changes no published number, and
> the 15-channel arm is reported beside the unchanged 14-channel arm so the
> added channel's contribution is visible rather than assumed.

**Dynamics.** A two-window contrast on **`pole_radius` only**, half-window
`WIN_S = 1.0 s`, thresholded per recording at `median + 3·MAD` — the same
robust scale-adaptive rule the frozen detector uses, so the two streams'
parameters are comparable.

## 3. The scoring bands are different for the two streams, and derived

**Configuration: ±2 frames**, inherited. `PLANT.md` characterised the frozen
detector at that band and it is not re-chosen here.

**Dynamics: ±15 frames**, derived from Stage 0's measured localisation and
**registered before use**. At the one probe cell where the dynamics detector
demonstrably works (Δfrequency = 4 Hz, recall 0.325 against chance 0.042), the
offset distribution runs **p05 = −15.2 to p95 = +8.0**. At weaker cells the
spread is ±70–85 frames, which is the slice width and therefore chance.

> **±15 is the measured precision of this instrument, not a preference.**
> Deriving a band from an instrument's own probe is legitimate; deriving one
> from an agreement number after seeing it is what `ANNOTATION_PREREGISTRATION.md`
> §7 forbids, and this is the former. The asymmetry between the streams is a
> finding — they do not localise equally — and is reported, not averaged away.

## 4. Combination: neither union nor intersection

A boundary fires when **either** stream crosses **its own** floor, and every
boundary records **which stream drove it**:

* `config_only` — configuration fired, no dynamics boundary within ±15;
* `dyn_only` — dynamics fired, no configuration boundary within ±15;
* `both` — matched within ±15, by `annot.match`, greedy nearest-first and
  one-to-one.

`shape` and `twist` already share only **23%** of boundaries at ±2, so the
two-channel structure is expected to be the finding rather than a nuisance.
**The three-way split is the headline output**, not a single merged rate.

## 5. Location is a covariate, never part of state

Arena position concentrates tracking failure at the wall
(`CONCENTRATION.md`: 3.7× monotone centre→wall), so a detector that learned
"wall behaviour" would be learning the tracker. Boundary rates are reported
**separately by arena tercile**, and the combined rate is reported with and
without location conditioning. `ladder.covariate_block` /
`context_ids(covariates=)` already carry the slot; location is coarse
(terciles) because `n_context` multiplies by its levels.

## 6. Predictions, fixed now

1. **`dyn_only` is a non-trivial share** — above 10% of all boundaries. *If the
   dynamics stream fires only where the configuration stream already did, it
   adds nothing and the two-stream idea is refuted on this corpus.*
2. **The dynamics stream's rate exceeds its own coloured-noise floor**, with
   non-overlapping animal-level intervals. *The incumbent's value on the
   comparable quantity is the frozen detector's jitter share **0.5475**; the
   dynamics stream is asked to beat that, per M12.*
3. **Adding `body_extension` changes the configuration stream's boundary rate by
   less than 20%.** A channel that moved the rate more than that would be
   dominating rather than contributing.
4. **Boundary rate is higher at the wall than at the centre in the
   configuration stream** — it tracks tracking quality — **and the dynamics
   stream's wall/centre ratio is smaller**, because a persistence descriptor is
   scale-free where an acceleration threshold is not.

## 7. What this stage may not do

* It may not **re-score** `SEGRECUR.md`, `VOCAB.md` or `CONTEXT.md`, or restate
  any published number. A new detector makes a new measurement
  (`DETECTOR_PREREGISTRATION.md:97-100`).
* It may not use `pole_frequency_hz` or any band above 2 Hz (§1).
* It may not move the ±2 or ±15 bands after seeing a rate.
* It may not **re-score Q1**, restate **Phase F**, or use any **Wiener** array.
* It may not report a stratum below **20,000 scored keypoint-frames**.
* It may not **compute across a recording boundary**.
* It may not call any context contrast **shock versus no-shock**.

## 8. Conventions

Every claim a `Read` with non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only, 2,000
replicates, `how="mean"`. Ranked on the effect-CI lower bound, never a p-value.
Seed 0. `mypy --strict` on new modules. The **`raw`** arm asserted by name.
