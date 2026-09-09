# Phase B — the cleaning bakeoff

**Corpus** `luna` | **split** `report` | **animals**
89 | **ε** 0.1 |
**inherited digest** `198eb14ff258c7f6`

shapeflow's Wiener shrinkage was chosen on a good argument and never benchmarked.
This is the benchmark.

## Read — `PASS`

> median_0.25 beats the incumbent wiener on all three axes at once: violations 1.632% against 1.673%, displacement 1.112 px mean against 1.555, and it keeps 32.0% of the power above f_c against the incumbent's 22.0%. 1 arm(s) dominate; a change of cleaning method is licensed by this comparison

## The three axes, and why no single number

| arm | violations | vs raw | distortion median px | mean px | body lengths | HF retained | %/px |
|---|---:|---:|---:|---:|---:|---:|---:|
| `median_0.50` | 1.592% | +11.0% | 0.512 | 1.573 | 0.00445 | 0.270 | 7.0 |
| `median_0.25` | 1.632% | +8.7% | 0.345 | 1.112 | 0.00300 | 0.320 | 7.8 |
| `savgol_0.50` | 1.663% | +7.0% | 0.708 | 1.612 | 0.00615 | 0.136 | 4.3 |
| `wiener` | 1.673% | +6.4% | 0.768 | 1.555 | 0.00665 | 0.220 | 4.1 |
| `savgol_0.33` | 1.701% | +4.8% | 0.551 | 1.308 | 0.00479 | 0.145 | 3.7 |
| `median_0.10` | 1.721% | +3.7% | 0.055 | 0.434 | 0.00048 | 0.547 | 8.6 |
| `viterbi` | 1.731% | +3.2% | 0.000 | 0.189 | 0.00000 | 0.586 | 16.9 |
| `butterworth` | 1.742% | +2.5% | 0.406 | 0.993 | 0.00353 | 0.144 | 2.6 |
| `outlier_iqr3` | 1.765% | +1.3% | 0.000 | 0.456 | 0.00000 | 0.412 | 2.8 |
| `outlier_mad3` | 1.772% | +0.9% | 0.000 | 0.661 | 0.00000 | 0.418 | 1.3 |
| `outlier_mad5` | 1.775% | +0.7% | 0.000 | 0.537 | 0.00000 | 0.419 | 1.3 |
| `outlier_p99` | 1.779% | +0.5% | 0.000 | 0.208 | 0.00000 | 0.522 | 2.3 |
| `raw` | 1.788% | +0.0% | 0.000 | 0.000 | 0.00000 | 1.000 | — |
| `savgol_0.17` | 1.790% | -0.2% | 0.267 | 0.710 | 0.00233 | 0.435 | -0.2 |

Sorted by violation reduction, never reduced to a score. Combining the axes into
one number would hide the case this exists to find: an arm that wins on
violations by moving everything.

**Distortion is judged on the mean, not the median.** Every targeted arm — the
outlier gates and Viterbi — moves under half its frames, so its median
displacement is exactly 0 and `0 ≤ 0` would let all of them "dominate" on an axis
that never separated them.

**The last column is the one to read for the Viterbi arm.** Violation reduction
bought per pixel of mean displacement is a ratio of two axes rather than an axis,
but it is what separates a de-glitcher from a smoother, and the two columns side
by side do not make it obvious.

**Retention is not "higher is better" on its own.** `raw` retains 100% by
definition and is the worst arm on violations. The three axes are read together
or not at all.

Every arm receives the **identical** input — shapeflow's hold-for-filtering
array, which is what its Wiener filter consumed. `wiener` and `butterworth` are
read off disk rather than recomputed, so a second implementation cannot silently
disagree with the arrays every existing result was built on. The reference length
is refitted **per arm**, so an arm that shrinks every distance uniformly cannot
post a lower violation rate for having made the animal smaller.

## The Viterbi arm, and a correction

An earlier version of this work recorded Anipose's Viterbi filter as
**unavailable**, on the grounds that all 3,080 `_full.pickle` files carry exactly
one detection per bodypart-frame. The detection count is right and the conclusion
was wrong — it came from the paper's phrase "a set of top detections per frame"
rather than from the implementation.

`viterbi_path` builds its candidate set from the previous `n_back` frames as well
as the current one, weighted by `2^-j`, plus an explicit missing state. At one
detection per frame the state space is still `n_back + 1` wide, and the decision
it makes is whether to accept this frame's jump or carry an older position
forward. Anipose's own `wrap_points()` carries the comment `# n_possible = 1` for
exactly this input.

It is a **de-glitcher, not a low-pass**, and the measured signatures are opposite:

| | fraction of keypoint-frames moved | median move when it moves |
|---|---:|---:|
| Viterbi | **0.31%** | 46.65 px |
| Wiener | **86%** (>0.01 px) | 1.42 px |

Anipose 1.1.24 is a pinned dependency. Only its import chain is bypassed:
`filter_pose` imports `.common` → `aniposelib` → `numba`, which is calibration
code the filter never touches, and installing it in full would drag
`opencv-contrib-python` on top of recur's pinned `opencv-python-headless` in a
venv the two repos share. The two functions the filter needs are loaded from the
installed package's own source by AST — Anipose's implementation, verbatim and
version-recorded.

## What could not run, and why that is a result

**`anipose_viterbi`** — CORRECTION: this was recorded as unavailable on the grounds that all 3,080 _full.pickle files carry one detection per bodypart-frame. The count is right, the conclusion was wrong -- viterbi_path builds its candidate set from the previous n_back frames as well as the current one, so it runs on single-detection input. It is now a scored arm, not an excluded one

**`confidence_filter`** — inert -- shapeflow's threshold estimator refuses for all 7 keypoints; its gate masks 0.000% of keypoint-frames

**`movement_package`** — not installed -- resolves to ~70 transitive packages into a venv shared with recur. The savgol and median semantics are matched to it in vieb/clean/arms.py instead


## The axis this does not settle

Held-out MDL. It is the decision-relevant one — a cleaning method should be
judged by whether the behaviour model built on it predicts better — and it costs
a full tokenizer run per arm. It is deferred to after Step 4, and the arms that
carry through are the cheap branch: no filter, wiener, viterbi, and whichever
smoother places here.

Ensemble-DLC / EKS is **blocked**, and the blocker is concrete rather than a
matter of effort: `~/dlc-training/trained_dlc/` contains zero files and all 3,846
h5 outputs carry the same single scorer, so no ensemble exists and none can be
assembled without retraining and re-inferring the corpus. Monsees et al.'s
anatomically constrained model needs multiple calibrated views and limb joints;
this is one uncalibrated overhead camera with seven surface landmarks. Both sit
on the re-tracking branch that Step 1's 2–15% verdict said to cost out.

All intervals are animal bootstraps, never pooled frames.
