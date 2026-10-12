# INVENTORY — committed instruments, before any step of the four-step programme

Written before Step 1 is registered. It lists the instruments the four-step
prompt names, each with its repo, path, commit, test, and evidence that it ran
end to end. **Steps 1 and 4 may use only what is listed here as runnable.**
Tests were run on 2026-10-12: vieb-pipeline 202 passed, across
`test_breaks`, `test_plant`, `test_seg_recur`, `test_seg_vocab`, `test_moseq`,
`test_floor`, `test_ego`, `test_pixel`, `test_repair` and `test_surrogate`;
recur `tests/test_surrogate.py` 20 passed.

## Instruments

| # | instrument | repo · path | commit | test | ran end to end | usable in Step 1 |
|---|---|---|---|---|---|---|
| 1 | **Boundary detector** (acceleration break) | vieb-pipeline · `vieb/seg/breaks.py` (`discontinuity`, `mad_threshold`, `boundaries`; DEGREE 3, DERIV_SEC 0.133, K_MAD 3.0), driver `scripts/breaks.py`, calibration `vieb/seg/plant.py` + `scripts/plant.py` | 7b2bfc0 · 6670e1b · 6c648d1 | `test_breaks`, `test_plant` | `results/breaks.json`, `results/plant.json` | **yes**: per recording, on ego `X` divided by the frozen `basis.json` `sd_used` |
| 2a | **Segment recurrence** (cross-animal excess) | vieb-pipeline · `vieb/seg/recur.py`, `combine_recur.py`, `scripts/seg_recur.py` | ce2d3e0 · 6670e1b | `test_seg_recur` | `results/seg_recur.json`, `SEGRECUR.md` | **no, as a labeller.** It is a *statistic* (does a segment have a close cross-animal neighbour), not an assignment of frames to states, so there is nothing to map onto labels |
| 2b | **Segment clumps** (the vocabulary built on 2a) | vieb-pipeline · `scripts/seg_vocab.py` (connected components of the θ-thresholded neighbour graph; θ inherited from `seg_recur.json`) | 6670e1b | `test_seg_vocab` | `results/seg_vocab.json`, `VOCAB.md` | **yes**: segments → clump id or *unassigned*. On Luna, 98–99% of segments were unassigned |
| 2c | `recur.recurrence` (window recurrence, `channels.py`) | recur · `recur/recurrence/*`, `scripts/channels.py` | e8ba893; Ledger 35 fix committed today as **38fe5c1** | recur `test_surrogate` | `results/channels_{wiener,unfiltered}.json` | **no, as a labeller**, for the same reason as 2a. Its white/phase generators are used for the negative controls |
| 3 | **Keypoint-MoSeq** | vieb-pipeline · `scripts/moseq_kpms.py`, `vieb/moseq/*`; protocol from `~/moseq/train_moseq.py` | 5d1e40f | `test_moseq` | `results/moseq.json` (11 fits) | **yes**. `~/moseq` is **not a git repo**: the protocol is pinned only through `moseq_kpms.py`, which restates it |
| 4 | **"Certified encoder"** | — | — | — | — | **no**: no code in vieb-pipeline, recur, vieb-atlas, vieb or `~/moseq`. It was **disqualified**, not certified, at a median seed AMI of 0.432 (cited in `MOSEQ_PREREGISTRATION.md` §4) |
| 5 | **Jitter estimator** (skull-bone residual) | vieb-pipeline · `vieb/seg/jitter.py` (`residuals`, `calibrate`), `scripts/noise_floor.py` | d972a01 · 6670e1b | `test_floor` | `results/noise_floor.json`; σ̄ = **0.061799 BL** at `results/plant.json` `sigma_bar_bl` (count-weighted over confidence bins, `scripts/plant.py sigma_bar`) | **yes**: needs nose, both ears and a confidence |
| 5b | Noise colour (spectral synthesis) | vieb-pipeline · `vieb/seg/noise.py` (`draw`, `validate`), from `results/inherited/sf_calibration.json` | 096a326 | `test_dynamics` | `results/dynamics_floor.json` | **yes**: Luna's measured PSD, ACF (lag-1 0.582, 267 ms) and common mode (0.226) |
| 6 | **Body length** | vieb-pipeline · `vieb/tok/ego.py` `body_length` (nose → tail_base distance), `ell_a` (per-animal median over usable frames) | 122c606 | `test_ego` | `results/ego_bodylen.json` | **yes, with one departure**: CalMS21 has no animal identities, so ℓ is the per-*sequence* median |
| 7 | **Pixel freeze score** | vieb-pipeline · `vieb/pixel/freeze.py` (`freeze_mask`, an ezTrack reimplementation, tested equal to ezTrack's source) | 64438ab | `test_pixel` | `results/pixel_pilot.json`, MOSEQ §7 | not needed in Step 1 (keypoints only) |
| 8 | Repair fill (hold ≤ 3 frames, NaN beyond; then `median_0.25`) | vieb-pipeline · `vieb/clean/repair.py` (`unsmoothed`, `repair`) | 5d1e40f | `test_repair` | `results/REPAIR.md` | **yes**: the interpolation the Luna pipeline applies |
| 9 | Ego representation (17-D: egocentric pose + SE(2) twist) | vieb-pipeline · `vieb/tok/ego.py` `transform` | 122c606 | `test_ego` | `work/ego/raw__bodylen__*.npz` | **yes**: the input to 1 and 2b; its origin is `center`, which CalMS21's `neck` fills |
| 10 | `Read` | recur · `recur/read.py` (verdicts PASS / FAIL / NOT_A_RESULT / GRID_LIMITED / INCONCLUSIVE; `n_effective` required) | recur HEAD | recur tests | used throughout | yes |
| 11 | Seconds → frames | recur · `recur/util.py` `frames(seconds, fps)` | recur HEAD | `test_util` | used throughout | yes |

## The prompt's premises, checked against the committed record

| prompt says | committed record | consequence |
|---|---|---|
| "the certified encoder's code assignment, if runnable" | not in any repo; disqualified at AMI 0.432 | excluded from Step 1 |
| "Keypoint-MoSeq at the **fitted** κ" | κ was **never fitted**: 1e4 (AR-HMM) → 1e6 (full), hardcoded in `train_moseq.py:55-56` and marked "TUNE THIS"; no scan exists | Step 1 uses κ = 1e6 and calls it *the labeller of record's value*. MoSeq also **failed** seed stability on Luna (median AMI 0.473, `MOSEQ.md`) |
| "segment clustering / recurrence" | recurrence (2a, 2c) is a statistic, not a labeller; the clustering built on it is the clumps (2b) | Step 1 scores the clumps; recurrence is not scored |
| σ̄ = 0.0618 BL; nose σ ≈ 3.10 px, centre ≈ 0.78 px | **two different estimators.** 0.0618 is the skull-bone estimator (5). The px figures are shapeflow's spectral `sigma_hat_px`, imported in `results/inherited/sf_calibration.json`; its code lives in `~/shapeflow`, outside these repos | native vs Luna is compared with **one** estimator (5). The per-keypoint *profile* comes from `sigma_hat_px` ratios (registered in Step 1) |
| "nothing above ~2 Hz clears the tracking-noise floor" | ~2 Hz is the **dynamics-descriptor** floor (`DYNAMICS.md`:14, :78). The tracking-noise crossover is f_c = **4.83 Hz** (`sf_calibration.json /f_c_hz`), unstable across segment lengths (4.83 → 1.71 Hz) | quoted with its own name |
| "SAM mask coverage 99% in A, 92% in B" | that is the mask **carried through REPAIR**: 98.9% A, 91.5% B (`REPAIR.md`:37). SAM keyframe coverage is 93.7% [92.3, 95.0] (`STABILISE4.md`:15) | quoted as the carried mask |
| "H.264 bits/frame 1.32× higher in A" | text only (`PIXEL_PREREGISTRATION.md`:206); **no committed script or JSON computes it** | GAPS.md |
| "grooming has been called unrecoverable" | **already withdrawn**: the grooming gate was NOT_A_RESULT at all 54 arms, and "unrecoverable" is not invoked (`PROGRESS.md`:548-555) | Step 1 tests the question; no standing claim to overturn |
| "a day-0 scalar was refused: 73% baseline noise" | recur HEAD `results/journeys_moseq.json` `baseline_noise_share` 0.7307, `JOURNEYS.md`:11, :13. The working copies of these files are **modified and uncommitted** | cite HEAD only |
| "bout hazard falls ~148×" | `LADDER.md`:119 (0.489 → 0.0033); `FRAILTY.md`:21 gives its own pooled 175.7× | quote both |
| "sessions in A ~17% longer (6,302 vs 5,392)" | `PIXEL_PREREGISTRATION.md`:225; data in `pixel_pilot.json` `n_frames` | as stated |
| "fit Luna's jitter AR model" | `vieb/seg/noise.py` already synthesises Luna's measured spectrum and **rejects AR(1)** (the ACF floors near 0.15 from the common mode) | Step 1 reuses `noise.draw`; no new AR fit |
| "Luna's dropout run-length distribution" | **does not exist.** Only a 3.4% rate (`vieb/qc/inject.py`) and REPAIR's ≤ 3-frame hold; `RUNLEN.md` is bone-violation runs, not missingness | measured in Step 1 as a registered quantity |

## Repository hygiene found on the way

- **recur** carries many uncommitted changes besides Ledger 35: `PROGRESS.md`, `LEDGER.md`, `INVENTORY.md`, `JOURNEYS.md`, `audit/leak.py`, `gates.py`, `journey/simplex.py`, `label/nonparametric.py`, `render/*`, `seq/motif.py`, deleted `results/instruments/b2/forced_choice/*`, and an untracked microstate null (`recur/null/`, `scripts/microstate_fit.py`, plus its wiring in `scripts/channels.py`). **None of it is used by Step 1.** Ledger 35 alone was committed (38fe5c1); the microstate wiring stays uncommitted.
- **`~/moseq` is not under version control.** The fit protocol survives only as `moseq_kpms.py`'s restatement.
