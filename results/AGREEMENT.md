# Do SAM's mask and the DLC keypoints agree? Mostly, and when not, SAM has usually clipped the head

`scripts/agreement.py`, thresholds fixed in its docstring and committed
(`agreement check` commit) before the first full run. Descriptive: no gate.
Inputs: STABILISE 7's saved SAM keyframe masks and the cleaned DLC pose; no
new SAM call. 300 recordings, 30 animals, animal-level bootstrap (2,000
replicates, seed 0). `results/agreement.json`.

**Headline.** On SAM's keyframes the two detectors **agree 81.0% [77.7, 84.1]**
of the time (85.0% [82.0, 87.6] when both produced something). **When they
disagree, the usual cause is SAM's mask missing the head.** The nose and ears
fall outside the mask on 13–16% of all keyframes and 54–64% of disagreeing
ones. The body centre falls outside on 0.5%.

## 1. Agreement, and the attribution as registered

| verdict per keyframe | share (animal mean) |
|---|---|
| agree (≥ 6 of 7 keypoints in the mask, centres ≤ 0.25 bl apart, axes ≤ 30°) | **80.96% [77.67, 84.08]** |
| "DLC suspect": keypoints fail their own plausibility check, mask passes | 7.12% [5.64, 8.80] |
| unresolved: both pass their own checks and still disagree | 5.87% [4.47, 7.48] |
| SAM refused (no accepted mask) | 4.78% [3.26, 6.58] |
| both off | 0.72% [0.41, 1.09] |
| "SAM suspect": mask fails its own check, keypoints pass | 0.54% [0.28, 0.85] |
| DLC missing | 0.00% |

**The attribution rows are not trustworthy, and example frames show why**
(`results/agreement_examples.jpg`). Each detector was judged only against
itself: keypoints on body length and skull bones, the mask on area alone. Those
checks are unequally sensitive. In the frames labelled "DLC suspect", the
keypoints sit on the mouse and **it is the mask that has cut off the head**. A
mask missing the head keeps a normal area, so it passes its check, while the
keypoint check trips on posture. The ratio of "DLC suspect" to "SAM suspect"
therefore measures the checks, not the detectors. It is reported as registered
and not interpreted.

## 2. The direct measure: which keypoints fall outside the mask

Added after the first run, once the example frames showed the problem. It is
descriptive and uses no threshold. The mask is taken with the same 0.05 bl edge
tolerance.

| keypoint | outside SAM's mask: all keyframes | disagreeing keyframes |
|---|---|---|
| right ear | **15.99%** | **63.97%** |
| nose | **13.82%** | **60.98%** |
| left ear | **12.96%** | **54.34%** |
| tail base | 5.80% | 19.85% |
| right hip | 3.25% | 17.57% |
| left hip | 2.39% | 12.95% |
| centre | 0.52% | 3.38% |

**SAM's mask misses the head on roughly one keyframe in seven, and almost never
misses the body.** Prompted with the padded keypoint box, SAM tends to return
the trunk and leave out the narrow, often darker head against the bars. The
keypoints put the head exactly where the video shows it.

## 3. What this means

* **Two detectors do give two checks.** 81% of keyframes pass both, and those
  frames can be trusted more than either detector alone.
* **The head is the weak point of the mask**, and the head is exactly the region
  a grooming measure depends on. Every still-frame gate STABILISE 7 passed was
  scored on the mask's pixels, so a clipped head is simply absent from them.
* **The obvious next step follows from §2:** prompt SAM with the keypoints as
  **points** (nose and ears as positive "this is mouse" points, plus the box)
  rather than the box alone, and re-measure the head's outside share. That is a
  new GPU run of SAM.

The 10 videos on the site (`scripts/sam_videos.py`) now carry this colouring:
the mask outline shows the nearest keyframe's verdict, and any keypoint outside
the mask is a red dot.
