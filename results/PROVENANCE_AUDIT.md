# Provenance audit — every hardcoded number in the document generators

Prompted by the run-length question, and widened because **this is the second
time a property of the filter was attributed to the data**. The first was the
Wiener filter's speed attenuation; the second was the 9-frame violating-run
median, which is a post-`median_0.50` residual used to justify the shape of an
operator that runs on the pre-filter array.

`bones_md.py` reads everything from JSON and has **zero** hardcoded numbers. It
is the standard the other three are measured against.

## The rule

A number in a published document either **traces to a result JSON** or is
**marked in the document itself** as unsourced. Nothing else is acceptable, and
"it was right when it was written" is not a third option — the reader cannot
check it and neither can we.

## The audit

| # | number | document | traces to | status |
|---|---|---|---|---|
| 1 | violation intervals ~1.43–1.79% vs ~1.49–1.87% | `cleaning_md.py:71` | `cleaning_report.json` | **traces** |
| 2 | `viterbi+median_0.50` 1.606% vs 1.592% | `cleaning_md.py:79` | `ranked` | **traces** |
| 3 | 1.544 px vs 1.573 px mean displacement | `cleaning_md.py:79` | `ranked` | **traces** |
| 4 | 27.1% vs 27.0% retention | `cleaning_md.py:80` | `ranked` | **traces** |
| 5 | Viterbi moves **0.31%** of keypoint-frames | `cleaning_md.py:147` | `cleaning_report.json` | **traces** |
| 6 | Wiener median move **1.42 px** | `cleaning_md.py:148` | `effect.json` | **traces** |
| 7 | Wiener moves **86%** of keypoint-frames | `cleaning_md.py:148` | `effect.json` | **traces** |
| 8 | Viterbi median move **46.65 px** | `cleaning_md.py:147` | — | **UNSOURCED** |
| 9–12 | violating runs: 207 / 2 frames / 32.3% / 70.4% (raw) | `cleaning_md.py:91` | — | **UNSOURCED → superseded** |
| 13–16 | violating runs: 66 / 9 frames / 53.1% / 96.1% (after median) | `cleaning_md.py:92` | — | **UNSOURCED → superseded** |
| 17–20 | SE(2) naive-twist R²: 0.9973 / 0.9853 / 0.9646 / 0.6492 | `ego_md.py:147–150` | — | **UNSOURCED, reconstruction failed** |
| 21 | 1.059% of coherent power above the crossover | `effect_md.py:146` | — | **UNSOURCED** |

Seven trace. Fourteen do not.

## The unsourced ones, individually

### The violating run-length table — superseded, not just unsourced

The whole block at `cleaning_md.py:89–92` is a hardcoded string. **No violating
run-length distribution is computed anywhere in this pipeline** — `runs_of` is
used to *build* Phase D's envelope and to count abstain runs, but nothing ever
recorded a length distribution. The only run statistic in any result JSON is
`abstain_runs_per_recording`, a count of abstain runs.

The numbers were also measured on the **eight worst recordings**, which is a
non-random subsample, and the 9-frame figure is **post-`median_0.50`** while the
corrector it was used to justify runs on the array *before* any smoother.

`vieb/qc/runlen.py` + `scripts/runlen.py` now measure it — corpus-wide, on
`held_array(pose_unfiltered, missing)`, by count **and** by mass, per arm — and
`results/runlen.json` supersedes the table. The document will cite the artifact.

### SE(2) naive-twist R² — I tried to reproduce it and could not

These four are a deterministic property of the transform, not a corpus statistic,
so they should be the easiest to verify. They are not.

Reconstructing the obvious procedure — synthetic mouse at each turn rate, exact
SE(2) twist for the labels, `ego.transform(..., naive=True)` scored by
`parity.exact_scores` — returns **1.0000 at every turn rate**, not 0.9973 /
0.9853 / 0.9646 / 0.6492. The synthetic fixture's `turn_frames=1` applies the
turn on a single frame, so R² over 900 frames is dominated by the non-turning
majority; whatever generated the published table did something else, and the code
does not record what.

**The claim the table supports is not in doubt** — `tests/test_se2.py` compares
`se2_log` against `scipy.linalg.expm` and fails the naive version at every turn
rate including zero, and that test passes. What is in doubt is the four specific
numbers and the "would pass this gate at every ordinary turn rate" framing built
on them.

Marked unsourced in `EGO.md`. Not re-derived: per the audit's own guard, a number
that needs a new run to source gets marked, not hunted.

### Viterbi median move 46.65 px

Computed during the Anipose work and printed, never written to an artifact. A
later 80-recording scan measured **43.79 px** median, which is in the same place
but is not the same number, and that one *is* recorded in `CLEANING.md`'s
per-keypoint table. The document should cite 43.79 with its sample, and drop
46.65.

### 1.059% of coherent power above the crossover

Attributed to shapeflow's calibration. `sf_calibration` contains the word
"coherent" but no key matching it and no value 1.059. Marked unsourced.

## What this does not change

Every number in this audit that **traces** is unaffected, and that includes the
whole bakeoff ranking — violation rates, displacements, retention — which is what
the cleaning verdict rests on. The injection benchmark, the continuity result and
the disposition result are all computed end-to-end from shards into JSON and
carry no hardcoded prose numbers.

The failure is concentrated in the *narrative* tables of three document
generators, which is both reassuring and exactly where it is hardest to notice.

## The structural fix, deliberately not taken

A build-time check that fails if a generator emits a numeric literal absent from
any result JSON would prevent recurrence. It was considered and not built: it is
a build rather than a measurement, and the guard on this phase says a build is
the signal to stop and re-scope. Recorded here so the option is not lost.
