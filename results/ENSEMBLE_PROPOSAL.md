# H2 — a DLC ensemble: what it would cost, and what it would buy

A costing document. **Nothing here is built, trained or installed.**

## What it buys

Not in dispute, and worth restating because the cost below is large:

* **A real uncertainty estimate.** Across-network variance on a frame is a
  measurement. DLC's confidence is a heatmap peak value, and on this corpus it
  predicts a geometric violation at **AUC 0.60–0.67** — better than chance and
  not much better.
* **It is the only thing that unlocks EKS.** The Ensemble Kalman Smoother uses
  ensemble variance in place of learned observation noise. There is no
  single-network version.
* **The ensemble median beats any member, before any smoothing** — no
  post-processing needed for that part.
* **It is orthogonal to the F3 freeze.** `viterbi` and `disposition` still apply;
  an ensemble changes the DLC output, not the cleaning arm.

## The blocking problem

**There is no labeled set, no trained model, and no DLC environment on this
machine.**

| | |
|---|---|
| `~/dlc-training/trained_dlc` | 24 KB, **222 empty directories, zero files** |
| `CollectedData_*.h5` / `.csv`, filesystem-wide | **none** |
| any DLC `config.yaml` | **none** |
| `/home/tul26194/vieb/trained_dlc` — the path `dlc_run.slurm` points at | **does not exist** |
| `/home/tul26194/vieb/venv-dlc` | **does not exist** |

What survives: 3,846 `.mp4`, the inference outputs, `dlc_analyze.py` (inference
only — training is not in this repository), and **the 222 `labeled-data`
directory names**. Those names are the one real asset: they record exactly which
videos contributed labeled frames to `VIEBFeb11shuffle2`, so the original frame
*selection* is recoverable even though the frames are not.

So the first line item is a human labelling project, and the commonly quoted
"3–5× training (cheap, you have the labeled set)" does not apply here.

## The cost model

| stage | cost | basis |
|---|---|---|
| **Recreate the project and label frames** | **human time; the gating item** | 222 videos named. A single-animal DLC project typically labels 100–200 frames total. Someone has to click 7 keypoints on each. |
| Rebuild the DLC environment | hours | `venv-dlc` is gone; `dlc_analyze.py` expects a working `deeplabcut` |
| Train 5 members | hours of GPU each | `gpu` partition: 6 nodes × 3 A100 80GB |
| **Infer 5 × 3,846 videos** | **≈ 95 GPU-hours** | measured: job 138157 did the full corpus in **10 tasks × ~1h55m ≈ 19 GPU-hours**, 3.6 h wall. Five members ≈ 18 h wall at 10-way, ~6 h at 18 GPUs |
| **Re-run shapeflow** | full pipeline | new keypoints → new clean arrays, new calibration, new gauge, new splits |
| **Re-run everything downstream** | ~1 day of cluster | bones, ego, effect, cleaning, continuity, disposition, injection, runlen, concentration — every result in `results/` derives from shapeflow's arrays |
| **Re-run Q1** | its own registration | pre-registered against the banked numbers, reported both ways regardless of which is better |

**The inference is not the expensive part.** At 95 GPU-hours it is under a day of
wall time on this cluster. The expensive parts are the labelling at the front and
the full downstream re-run at the back, and the second is the one usually
omitted: every number this programme has produced would become pre-ensemble.

## What H1 says about the expected return

`CONCENTRATION.md`, measured, not assumed:

* Violation rates are **350× more dispersed** across recordings than independent
  frames would give. Half the corpus's failure sits in a tenth of its recordings.
* The rate rises **monotonically 3.7×** from the middle of the arena to the edge,
  across ten deciles of 2.2M frames each. This was read as the occlusion
  signature. **That reading is withdrawn** — see below.
* **Box has no effect** (2.14% / 2.14% / 2.20% across three apparatus units), so
  it is not the rig.
* One session type carries 7.34% against 1.13% elsewhere — though **context and
  day are completely confounded** there and it is one factor, not two.

The argument made here was: an ensemble averages away errors its members make
*independently*; an animal pressed against a wall is occluded for every member
equally; so errors with this shape are the ones an ensemble has least purchase
on.

**The premise was tested and is false.** `ADJUDICATION.md` scored 100 frames
blind — 25 from each cell of {worst decile, passing} × {flagged, not flagged},
`draw_skeleton` called with `flagged=False` on every frame, filenames a shuffled
index, key sealed until the scores were in.

| | occluded | visible but wrong | fine | unsure |
|---|---:|---:|---:|---:|
| **total** | **0** | **23** | 70 | 7 |

**Zero occluded frames in a hundred.** Not one showed an animal genuinely hidden.
Twenty-three showed a landmark plainly visible and the network putting it
somewhere else. Flagged frames scored visible-but-wrong 16/50 against unflagged
7/50, so the mask detects a real signal and is not measuring the wrong thing.

A named failure mode came out of it: six frames show **a keypoint locked onto a
bright object above the arena**, across three boxes, four days and two contexts.
Not occlusion, not one rig, and not an error a human labeller would make — the
kind of error members trained on different splits would plausibly disagree about,
which is the case *for* an ensemble.

The 3.7× edge gradient is real and still needs an explanation. The likeliest one,
visible in the contact sheets, is that the arena wall is a field of high-contrast
vertical bars and an animal in front of them sits against a strong distractor
texture. That is a contrast-and-distractor problem rather than a visibility one,
and unlike occlusion it is something better detections could fix.

**Two limits on this, stated plainly.** The blinding leaks with a known
direction — a bone violation *looks like* a stretched skeleton, so the scorer can
partly infer the flag despite the marker being off, which inflates the
32%-versus-14% contrast. That contrast should not be quoted as a measurement.
Zero-occluded and the bright-object mode do not depend on the leak. And the
scorer is a language model reading 320-pixel grayscale crops whose one previous
attempt at this task in this programme was wrong.

Hard footage and a network that fails on hard footage still look identical in the
aggregate, and only across-network variance separates them. What has changed is
that the aggregate is no longer the only evidence.

## Design requirements, adopted as given

1. **Vary train/val splits per member, not just the seed.** Seed-only variation
   gives correlated members and understates uncertainty.
2. **5 members, not 3.** Three is the minimum that yields a variance at all.
3. **Run the bone diagnostic on the ensemble median before committing**, and
   treat it as the decision gate rather than a check. If the violation rate drops
   materially the ensemble is doing the work. **If it does not, the errors are
   systematic — every network failing the same way on the same occlusions — and
   that is the more important finding, because it means the ceiling is the
   footage.** H1 already leans that way; the ensemble would settle it.

## Rejected, with the reason recorded

**Pseudo-labelling from the existing predictions.** Tempting, because it removes
the gating item: take high-confidence DLC output as labels and train members on
it. It must not be done. Members trained on one network's output are correlated
**by construction** — a worse version of the seed-only failure mode, because the
correlation sits in the labels rather than the initialisation. The variance
estimate would be an artifact of the teacher network and would look exactly like
a real uncertainty measurement.

## Where this leaves the decision

Step 1's standing answer is **2.134%** at ε = 0.10 (`results/bones.json`,
GRID_LIMITED, "inside the 2%–15% band"). Read strictly, that is 0.134 points
above the threshold at which the ensemble would be a waste — inside the band, at
its bottom edge.

The earlier conclusion — *"indicated but marginal, and H1 suggests its return
would be further limited by the shape of the errors"* — **is withdrawn**. The
shape argument rested on the errors being occlusions every network would share.
They are not occlusions.

The honest summary now: **the ensemble is indicated, marginal on the violation
rate alone, and no longer limited by the shape of the errors.** Every cost line
above stands unchanged — the labelling project at the front and the full
downstream re-run at the back are still the expensive parts, and the inference is
still under a day of wall time. What moved is the expected return, upward.

**The footage lever survives in modified form.** The edge gradient is still the
largest single structure in the failure, but it now points at arena contrast —
the high-contrast wall the animal is silhouetted against — rather than at
occlusion. That is fixable for data not yet collected, and cheaply, by changing
what the wall looks like. It remains unfixable retrospectively.

**What would settle it** is unchanged and is the third design requirement above:
run the bone diagnostic on the ensemble median. If the violation rate drops
materially the ensemble is doing the work; if it does not, the errors are
systematic across networks and the ceiling is elsewhere. Nothing short of an
ensemble answers that, which is the whole reason the question is still open.
