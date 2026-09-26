# Keypoints as SAM point prompts: head clipping halves, but DLC's errors now reach SAM

`scripts/point_prompts.py`, metrics fixed in its docstring and committed
(`555aa05`) before the run. Descriptive. Every 30th frame of all 300 STABILISE
7 recordings: STABILISE 7's saved **box-only** masks against the same box
**plus nose, centre and tail base as positive points**, both accepted by
STABILISE 7's rules. 30 animals, paired frames, animal-level bootstrap.
`results/point_prompts.json`.

## 1. The head comes back, measured on keypoints SAM was never given

| measure | box only | box + points | paired change |
|---|---|---|---|
| **ears outside the mask** (not prompted) | 14.27% [10.96, 17.97] | **7.01% [4.90, 9.50]** | **−7.26 pts [−8.65, −5.95]** |
| hips outside (not prompted) | 2.73% | 1.69% | −1.04 pts [−1.47, −0.65] |
| frames refused | 4.83% | 1.47% | — |
| nose outside (*prompted: circular*) | 13.35% | 1.78% | — |

**Point prompts halve head clipping on the ears**, which is the honest measure
because SAM was never told where they are.

## 2. But a wrong keypoint now drags the mask with it

Mask area grows only 7% at the median (0.450 → 0.482 × bl²), but 20% at the
95th percentile (0.620 → 0.746). The frames behind that tail
(`results/point_prompts_big_masks.jpg`, six drawn from the 200 largest
growths, plus three typical frames) are all in the rounded arena, with the
mouse near its bright top wall. **In most of them a prompted DLC keypoint lies
on the wall, and SAM follows it onto the wall.**

| context | paired frames | point mask > 1.5× box mask | > 2× |
|---|---|---|---|
| A | 31,418 | 1.85% | 0.11% |
| B | 23,853 | **13.18%** | **5.82%** |

**The failure depends on the arena.** A context-B frame is about seven times as
likely as a context-A frame to get an inflated mask. Any A-versus-B analysis
built on plain point-prompted masks would inherit an arena artefact. And
because DLC's errors now feed into SAM, the two stop being independent checks
on each other.

## 3. What follows

**Plain point prompting is not adopted.** The combination that keeps both
detectors honest is **guarded**:
1. take SAM's box-only mask;
2. use a keypoint as a point prompt only if it lies within 0.2 body lengths of
   that mask, so a nose just past a clipped head qualifies and a keypoint on
   the wall does not;
3. keep the point-prompted mask only if its area is at most 1.3× the box mask's,
   and otherwise fall back to the box mask.

That guarded variant was **designed after seeing §2**, and it is run and
reported as such (`results/POINT_PROMPTS_GUARDED.md`), with the same metrics
plus the context split.
