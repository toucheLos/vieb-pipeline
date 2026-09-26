# Registration — REPAIR: the pose, repaired with SAM as a second detector

Committed **alone, before `vieb/clean/repair.py` exists**, while the
whole-corpus mask run (plan Stage A1, `scripts/masks_corpus.py`) is in progress.
No repaired pose has been computed. Inherited digest **`198eb14ff258c7f6`**.

**Why.** Every tokeniser and segmenter in this programme ran on the **raw** DLC
pose (`held_array(pose_unfiltered, missing)`). The cleaning bakeoff's winner,
`median_0.25` (`CLEANING.md`: 32.0% [30.2, 33.7] of high-frequency power
retained against Wiener's 22.0%, lower distortion), was never applied
downstream. `AGREEMENT.md` and `POINT_PROMPTS_GUARDED.md` showed SAM's mask can
catch keypoints DLC puts off the animal. This stage builds the best pose the
two detectors support, and the behaviour stage (`STATES_PREREGISTRATION.md`)
consumes it.

---

## 1. The repaired pose, specified completely

Per animal, per recording, in this order:

1. **Start from `disposition`** (`work/disposition/<animal>.npz`). That is raw
   DLC, with bone-violating runs of ≤ 3 frames corrected by projection and
   longer runs abstained (`vieb/qc/disposition.py`, `DISPOSITION.md`). It is
   the best-validated arm the programme already carries.
2. **Mask veto.** For each frame with a mask (the nearest accepted keyframe of
   `work/masks/<rid>.npz`, every 6th frame, carried to the frame by the
   interpolated centroid of its two bracketing accepted keyframes, the
   `sam_videos` / STABILISE rule), a keypoint is set **missing** if it lies more
   than **0.3 body lengths** from the mask.
   * The veto is **deliberately far**. `AGREEMENT.md` §2 showed that near-edge
     head keypoints outside the mask are usually SAM clipping the head, not
     DLC erring. 0.3 bl only catches keypoints off the animal, like the wall
     points in context B.
   * Frames with no carried mask (not bracketed by two accepted keyframes) are
     not vetoed. The pose is kept as disposition left it.
3. **Holding.** Missing runs of **≤ 3 frames** (disposition's
   `MAX_CORRECT_FRAMES`) are linearly interpolated per keypoint. Longer runs
   stay missing, and the frame is flagged `abstain` for that keypoint.
4. **Smoothing: `median_0.25`** (`vieb/clean/arms.rolling_median`,
   window 0.25 s). As shapeflow does, missing values are held by interpolation
   for the filter only (`arms.held_array`), then re-set to missing after
   filtering. A held value is never stored as a measurement.

Stored at `work/repaired/<animal>.npz`: `pose`, `abstain` (frame),
`missing_kp` (frame × keypoint), `vetoed` (frame × keypoint), `has_mask`
(frame), `bounds`, `recording_ids`. Exposed as `pose_arm = "repaired"` in
`scripts/ego.py::pose_for`.

## 2. Gates: metrics that do not use the mask

The veto uses the mask, so no gate may score agreement with the mask. That
would be circular. **Report split, 89 animals, animal-level bootstrap (2,000
replicates, seed 0), paired within animal.**

| gate | statistic | PASS iff |
|---|---|---|
| 1. skull jitter | median over 2 s windows of the within-window SD of the three skull bone lengths, / bl (the D21 measure) | paired (repaired − raw) **and** (repaired − disposition) upper bounds both < 0 |
| 2. bone violations | skull-bone violation rate, reference refitted per arm (the bakeoff's `violation` axis) | (repaired − disposition) upper bound ≤ **+0.001** (not worse by 0.1 percentage point) |
| 3. high-frequency retention | `score.high_frequency_retention(before = raw held, after = repaired)` | lower bound ≥ **0.220**, Wiener's point value: no more smoothing than the pose shapeflow shipped |
| 4. distortion | mean displacement from raw held, px, over non-missing keypoint-frames | upper bound ≤ **1.555 px**, Wiener's mean: no more movement than shapeflow's pose |

**Reported, not gated:**
* the veto rate overall, per keypoint and **by context** (A/B). If the A:B
  ratio exceeds 2, it is flagged with the evidence for why, given the known
  arena difference in SAM's head coverage;
* the share of frames with a mask;
* the share abstained.

**Verdict: PASS iff gates 1–4 all pass.** PASS: the behaviour stage uses
`repaired`. **FAIL: it uses `disposition + median_0.25` without the veto**,
computed by the same code with the veto disabled, and the failing gate is
named. Either way, the choice is made by this rule and not after seeing the
behaviour results.

## 3. Prohibitions

No `report`-split tuning of the 0.3 bl veto, the 3-frame hold or the filter
window: all are fixed here. No gate reads the mask. Nothing is described as
shock versus no-shock.
