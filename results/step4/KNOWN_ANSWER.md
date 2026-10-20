# Registration — STEP 4 known-answer check: does the decoded "freeze" see the context difference?

Committed **alone**, before any code for it exists. This is a gate before any C5
learning fit, added by the user after run 2 (`HSMM.md`). **It is decided after
seeing run 2's descriptive result**: equal freezing in Context A (0.191) and
Context B (0.184), decoded mostly as sub-second bouts (median 0.17 s).

**Why that result is suspicious.** Two independent signals say the contexts differ:
- the pixel freeze score: about 12.6 points more freezing in Context A on the report
  split (MOSEQ §7);
- keypoint centroid speed: 0.758× in Context A.

The check does not need the shock protocol, and states no shock mapping.

## Data

- **Split:** **tune** only. The report split is not spent on a diagnostic.
- **Recordings:** CFD sessions on **Days 3–7** in Context A or Context B. For each
  animal, only the days on which **both** contexts were recorded are used.
- **Pairing:** by animal (`animal_tag`). Each animal contributes its mean over its
  paired days, per context. The unit is the animal, and intervals are an animal
  bootstrap (2,000 replicates, seed 0).
- **Decode:** the run-2 final model (plain HMM after the drop) and its saved smoothed
  posteriors (`work/hsmm/tune/<rid>_post.npz`). The decoded state is the argmax.
- **Pixel freeze:** `scripts/moseq_freeze.py`'s scan, unchanged, run on these tune
  recordings: full frame, mt_cutoff 10, FreezeThresh 800, MinDuration 15 frames.
  Frame 0 is never freezing. If the pose and video frame counts differ, both are
  truncated to the shorter, and the count of such recordings is reported.

## Measures

1. **Freeze time by bout length (HSMM).** Decoded freeze bouts are split into three
   bins: **< 1 s**, **1–2 s**, **≥ 2 s** (half-open: [0, 1), [1, 2), [2, ∞)). For
   each bin, the measure is the fraction of the recording's frames in freeze bouts of
   that length. Reported per context, with the paired **A − B** difference and its
   interval, per bin and in total.
2. **The same measure for the pixel score**, on the same recordings and bins. Pixel
   bouts are ≥ 0.5 s by construction, so its < 1 s bin covers only 0.5–1 s.
   - **The anchor:** the pixel A − B total on tune must have a lower 95% bound > 0.
     **If it does not, the known answer is absent on tune, and this check is
     `NOT_A_RESULT`.**
3. **Per-frame agreement within each context.** Cohen's κ between HSMM freeze and
   pixel freeze, per recording, averaged per animal, per context.
   - It is computed three ways: all decoded freeze frames, freeze frames in decoded
     bouts ≥ 1 s, and freeze frames in decoded bouts ≥ 2 s.
   - It is computed on all frames, with unflagged frames only reported beside it.
   - A within-context comparison is not affected by the image difference between
     contexts.
4. **Descriptive, beside it:** the paired A/B ratio of median centroid speed (v, BL/s)
   on these recordings, to check the 0.758× figure on tune.

## Readings (fixed now)

"Effect" means the paired A − B lower 95% bound > 0. "No effect" means the
interval includes 0. "Reverse" means the upper bound < 0.

| HSMM ≥ 2 s bin | HSMM < 1 s bin | reading |
|---|---|---|
| effect | no effect or reverse | **the freeze state mixes micro-pauses with real freezing.** The fix is a **registered change to the states**, not a duration prior |
| effect | effect | the decoded freeze sees the context difference at every bout length; the near-equal totals need explaining before use |
| no effect (and no effect in 1–2 s) | any | **either the keypoint signal misses freezing, or the pixel effect was partly the visual confound.** Either answer is a result |
| reverse | any | the decoded long freeze runs against both independent signals; the freeze state is not freezing |

**The within-context κ is a descriptive tiebreak for the third row,** not a verdict.
κ ≥ 0.4 in both contexts for the ≥ 1 s variant leans toward the visual confound: the
two signals agree on the same frames, yet the context effect differs. κ < 0.4 leans
toward the keypoint signal missing freezing.

## What it gates

The plain HMM is **cleared for C5** only if all of the following hold:
- Amendment 4's planted control passes;
- this check reads **row 1 after a registered state change has been re-run and passes
  this check**, or **row 2** with the totals explained.

Rows 3 and 4 keep it blocked. C5 also stays blocked on the lab's protocol whatever
this check reads.

## Reads

One `Read` per measure × context (or A − B) × bin. `n_effective` is the number of
paired animals.
