# Pre-registration — frame-rate robustness, and whether the island splits

Written and committed **before `scripts/subsample.py` or
`scripts/island_split.py` exist**. Digest `198eb14ff258c7f6`.

## 1. What Stage 1 left, and what is therefore not attempted

`TWOSTREAM.md`: the dynamics stream cuts in different places (21.6% `dyn_only`)
and is far less contaminated by arena position (1.09× against 1.97×), but its
jitter share is **0.9665** against the frozen detector's **0.5475** — 97% of
its rate is noise.

> **Segment recurrence on the dynamics stream is not run.** Dwell-matched
> recurrence over a boundary set that is 97% noise would measure the noise's
> recurrence. This is a consequence of a registered prediction failing and it is
> recorded, not attempted.

Two things remain worth measuring and are registered here.

## 2. Frame-rate robustness

Re-run both streams at **15 fps** — every other frame — and ask which
boundaries survive. Everything temporal in this repo derives from
`recur.util.frames(seconds, fps)` at use time, so halving the rate re-derives
the windows in *seconds* and changes nothing else. **No frame-rate sweep has
ever been run in this programme**; this is the first.

A boundary detected at 15 fps sits at original frame `2i`. Survival is matched
with `annot.match`, greedy nearest-first and one-to-one, at each stream's own
registered band — **±2** for configuration, **±15** for dynamics.

**Predictions:**

1. **Configuration survival exceeds 0.5.** It is an acceleration threshold on a
   signal whose power is concentrated below 3 Hz, well inside the new 7.5 Hz
   Nyquist. *Falsifier: below 0.5 means the frozen detector's boundaries depend
   on the sampling rate, which would be a property of the instrument and not of
   the animal.*
2. **Dynamics survival is lower than configuration survival.** Its rate is 97%
   noise, and noise does not survive decimation.
3. **Neither stream's rate rises at 15 fps.** A rate that rises when half the
   data is thrown away is reading the sampling grid.

## 3. Does the island split, and on which channel?

`BEHAVIOUR.md`'s `chaining_clump0` **FAILs** — typical pair 0.398 against
θ = 0.190, **2.1×** — so clump 0 is a single-linkage chain and must not be
named as one behaviour. That is the motivation: a chain is the shape of
something that has sub-types.

**Per clump-0 segment**, on the `report` split, two candidate separators:

* **configuration** — mean `triage.body_extension` over the segment. Crouched
  freeze against stretch-attend is a posture: an elongated, flatter back.
* **dynamics** — mean `pole_radius` over the segment.

**The test is `vocab.gmm1d_bic_gain`**, which is already built and used:
`BIC(1 component) − BIC(2)` on a one-dimensional mixture. **It is scored
against a null, not a threshold** — the same discipline `vocab_read` uses. The
null is the identical statistic on **duration-matched non-clump-0 segments from
the same animals**, drawn once at seed 0, because any large set of segments has
some bimodality and the question is whether clump 0 has more.

**Pre-registered expectation: the island splits on CONFIGURATION, not
dynamics.** Stage 0 measured `pole_radius` separating corpus from noise on 88.7%
of windows, but Stage 1 measured the dynamics stream at 97% noise as a boundary
source; and the band budget says a low-amplitude in-place rhythm during
immobility sits near the floor. A split on **dynamics** would be the genuinely
novel outcome. **No split at all is reported as a unitary slow freezing state.**

**Predictions:**

4. **`body_extension` bimodality in clump 0 exceeds the duration-matched null.**
5. **`pole_radius` bimodality does not.**

## 4. What this stage may not do

* It may not **re-score** `SEGRECUR.md`, `VOCAB.md`, `CONTEXT.md` or
  `STILLNESS_CONTROL.md`, or restate any published number.
* It may not run segment recurrence on the dynamics stream (§1).
* It may not change the ±2 or ±15 bands, or the 15 fps factor.
* It may not describe any context contrast as **shock versus no-shock**. It is
  Context A versus Context B; `recur/labels.py:41-43` refuses a `shocked` flag.
* It may not report a stratum below **20,000 scored keypoint-frames**.
* It may not **compute across a recording boundary**.

## 5. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only, 2,000
replicates, `how="mean"`. Ranked on the effect-CI lower bound, never a p-value.
Seed 0. `mypy --strict` on new modules. The **`raw`** arm asserted by name.
Every gate states the incumbent's value on the same quantity — **M12**.
