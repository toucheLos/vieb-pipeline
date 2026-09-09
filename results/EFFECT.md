# Phase A — what the cleaning actually changes

**Corpus** `luna` | **recordings** 3,846 |
**frames** 22,355,989 | **animals** 298 |
**inherited digest** `198eb14ff258c7f6`

Nobody had measured this. Everything downstream — Q1 included — is computed on the
filtered array, and the size of what the filter does to it was not written down
anywhere in either upstream repo.

## The question has three answers and the bone check is not one of them

The bone check is a **flag, not an edit**. It feeds a gap policy, and the gap
policy is one of three layers that move the data:

| layer | what moves | mass it touches |
|---|---|---:|
| **L1 gap policy** | gaps ≤ 0.1 s linearly interpolated; longer runs abandoned | 1.340% interpolated, 3.385% abandoned |
| **L2 Wiener filter** | every frame, per-keypoint frequency-graded shrinkage | all of it |
| **L3 ε-violation flags** | nothing — they only flag | 4.104% |

The baseline is `raw_pose.npz`, **not** `pose_unfiltered`. The latter is the gap
policy's own output and already carries its interpolants, so measuring L1 against
it returns exactly zero — the one answer that cannot be right.

## The bone check's own cost is small, and perfectly targeted

`bone_flagged` runs 4.725% of frames. What that costs downstream:

| | value |
|---|---|
| median displacement, all keypoint-frames | **0.0000 px** |
| mean displacement on `interpolated` frames | 14.202 px |
| mean displacement **outside** the flags | 0.000 px |
| concentration | **∞** — touches only flagged frames |
| speed quantile ratio (p50, p90, p99) | 0.996, 0.966, 0.909 |

**The gap policy moves nothing outside the frames it flagged**, by construction
rather than by tuning, and it leaves the speed distribution within 10% at every
quantile. So the honest answer to "how much variance does the bone check create"
is: **very little**. It is the layer with the smallest effect of the three, and it
is the only one that is exactly targeted.

## The filter is where the data actually moves

| | Wiener (used downstream) | Butterworth (comparison arm) |
|---|---:|---:|
| median displacement | **0.8315 [0.7911, 0.8776] px** | 0.4564 [0.4327, 0.4824] px |
| p90 | 4.594 px | 3.000 px |
| in body lengths (median) | 0.00721 | 0.00397 |
| moved > 0.01 px | 99.4% | 96.3% |
| moved > 0.5 px | 56.3% | 41.5% |
| moved > 1 px | 38.9% | 24.1% |
| moved > 5 px | 7.9% | 4.4% |

It touches **99.4%** of keypoint-frames and moves
**56.3%** of them by more than half a pixel. That is a
transform applied to the whole corpus, which is what a shrinkage filter is — the
finding is not that it is wrong, but that the size of it had never been recorded
while every downstream number depends on it.

### It is targeted, more than a single recording suggested

| flag | inside | outside | concentration |
|---|---:|---:|---:|
| `bone_flagged` | 3.919 px | 0.785 px | **4.99×** |
| `eps_violation` | 3.175 px | 0.804 px | 3.95× |
| `interpolated` | 1.500 px | 0.834 px | 1.80× |

A spot measurement on one recording put this at 2.0× and the corpus puts it at
5.0×. The filter does
concentrate its work on the frames the geometry says are broken, and the
single-animal reading understated that. Butterworth concentrates harder still,
at 6.8×.

### Which landmark

| keypoint | median displacement (Wiener) |
|---|---:|
| left_ear | 0.597 px |
| right_ear | 0.676 px |
| nose | 2.578 px |
| center | 0.153 px |
| left_hip | 0.599 px |
| right_hip | 0.671 px |
| tail_base | 0.708 px |

A **17×** spread across
landmarks. The nose moves furthest and the centre barely moves, which tracks the
per-keypoint noise shapeflow measured (σ = 3.10 px at the nose against 0.79 at
the centre) and the Wiener gain that follows from it. The filter is not one
operation applied uniformly; it is seven different operations.

## What it does to behaviour, not just position

Quantile ratios, after ÷ before. This is the axis a displacement statistic cannot
see: a filter can preserve position while deleting the fast tail of the movement
distribution, and a behaviour model would never know.

| | p50 | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| **Wiener, speed** | **0.590** | **0.668** | **0.644** | **0.634** |
| Butterworth, speed | 0.602 | 0.709 | 0.682 | 0.685 |
| **Wiener, turn** | **0.760** | **0.802** | **0.664** | **0.606** |
| Butterworth, turn | 0.922 | 0.911 | 0.773 | 0.712 |
| gap policy, speed | 0.996 | 0.966 | 0.909 | 0.945 |

**The Wiener filter removes 41% of the median instantaneous
speed and 24% of the median turning.** Much of frame-to-frame
displacement at 30 fps genuinely is tracking noise — σ runs 0.79–3.10 px per
keypoint — so removing it may well be correct. The point is that the magnitude
was never recorded, and "1.059% of coherent power sits above the crossover" does
not prepare a reader for a 41% cut to median speed.

The comparison that carries information is between the two filters.
**Butterworth preserves turning far better than Wiener does**
(0.922 against
0.760 at the median) for a similar
cut to speed. Turning is exactly what the egocentric representation's `omega`
channel carries, and the two filters treat it very differently.

## What this licenses

**The bone check is cleared.** Its downstream cost is confined to
1.340% of keypoint-frames, it moves nothing else, and it leaves
the kinematics within 10%. It is not a source of variance worth worrying about.

**The filter is not cleared, and was never on trial.** It moves
56.3% of keypoint-frames by more than half a pixel and cuts
the median speed by 41%.
That is what shrinkage does, but it is a large intervention sitting under every
result this project has published, and it had never been quantified. Phase B puts
it in a bakeoff against the alternatives.

All intervals are animal bootstraps over 298 animals, never pooled
frames.
