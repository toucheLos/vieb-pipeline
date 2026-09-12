# Phase B — the cleaning bakeoff

**Corpus** `luna` | **split** `report` | **animals**
89 | **ε** 0.1 |
**inherited digest** `198eb14ff258c7f6`

shapeflow's Wiener shrinkage was chosen on a good argument and never benchmarked.
This is the benchmark.

## Read — `PASS`

> 11 arm(s) beat the incumbent wiener with non-overlapping animal-bootstrap intervals and none is significantly worse on any axis. median_0.25 separates on distortion_mean_px, hf_retained: it retains 32.0% [30.2%, 33.7%] of the power above f_c against the incumbent's 22.0% [20.8%, 23.2%]. No arm separates on violation_rate, so the ordering on that axis is a ranking of point estimates and not a finding.

## The three axes, and why no single number

| arm | violations | vs raw | distortion median px | mean px | body lengths | HF retained | %/px |
|---|---:|---:|---:|---:|---:|---:|---:|
| `median_0.50` | 1.592% | +11.0% | 0.512 | 1.573 | 0.00445 | 0.270 | 7.0 |
| `viterbi+median_0.50` | 1.606% | +10.1% | 0.504 | 1.544 | 0.00438 | 0.271 | 6.6 |
| `median_0.25` | 1.632% | +8.7% | 0.345 | 1.112 | 0.00300 | 0.320 | 7.8 |
| `viterbi+median_0.25` | 1.640% | +8.3% | 0.340 | 1.085 | 0.00296 | 0.317 | 7.6 |
| `viterbi+savgol_0.33` | 1.661% | +7.1% | 0.542 | 1.264 | 0.00471 | 0.137 | 5.6 |
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

**Read the ordering as an ordering, not as a result.** Dominance is decided by
**non-overlapping animal-bootstrap intervals**, and on this corpus
`violation_rate` separates for no arm at all -- the intervals run ~1.43-1.79%
against the incumbent's ~1.49-1.87% and overlap almost entirely. An earlier
version of this read compared point estimates and announced that an arm "beats
the incumbent on all three axes"; it does not. What separates is retention, and
displacement for the targeted arms.

## Composing a de-glitcher with a smoother buys nothing

`viterbi+median_0.50` against `median_0.50` alone: 1.606% against 1.592%
violations, 1.544 px against 1.573 mean displacement, 27.1% against 27.0%
retention. **All three intervals overlap.** The hypothesis was that Viterbi would
remove the teleports so the median would not have to, giving lower violations at
lower distortion. It does not, and the residual measurement below says why.

## What the best arm still gets wrong, and why nothing temporal will fix it

Violating runs on the eight worst recordings, before and after `median_0.50`:

| | runs | median run | share of violating frames in runs > 0.5 s | > 3 frames |
|---|---:|---:|---:|---:|
| raw | 207 | 2 frames | 32.3% | 70.4% |
| after `median_0.50` | 66 | **9 frames** | **53.1%** | **96.1%** |

The median removes the short violations -- which is what a median is for -- and
the median *length* of what survives triples. **What a temporal filter leaves
behind is temporally smooth.** A keypoint parked off the body and held there
satisfies a median (most of the window is wrong) and satisfies Viterbi's motion
prior too (a stationary point is not a glitch). That is why composing them is
redundant rather than complementary, and it is visible in the residual clips: one
landmark off the animal, identical in all three panes.

The information needed to fix it is not on the time axis. It is either anatomical
-- the bone length used as a **corrector** rather than a flag, which is the cheap
2-D shadow of what an anatomically constrained model does properly in 3-D -- or it
is in better detections, which is the expensive branch.

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

### Which landmarks it reassigns

80 seeded report recordings, 464,153 frames, Anipose 1.1.24. These numbers exist
in `viterbi.reassignment` and had never been reported. The Wiener column is
`SNR/(1+SNR)` at Nyquist from shapeflow's calibration spectrum — **low means the
filter removes that landmark's fast content entirely**.

| keypoint | Viterbi reassigns | Wiener gain at Nyquist | continuity residual |
|---|---:|---:|---:|
| nose | **0.5315%** | 0.2903 | **0.0067** |
| left_hip | 0.4212% | 0.0161 | 0.0047 |
| right_hip | 0.3872% | 0.0000 | 0.0047 |
| right_ear | 0.2555% | 0.5773 | 0.0044 |
| tail_base | 0.1881% | 0.0000 | 0.0052 |
| left_ear | 0.1385% | 0.5007 | 0.0042 |
| center | **0.0224%** | **0.5832** | **0.0034** |
| **all** | **0.2778%** | — | 0.0046 |

Median move 43.79 px, p90 85.05 px, max 340.01 px.

**The nose is reassigned 24× more often than the centre**, and the centre is the
landmark every instrument trusts: Viterbi touches it least, Wiener smooths it
least, and it has the lowest continuity residual of the seven.

The three orderings agree, but only one pair does so significantly at n = 7
keypoints: reassignment against continuity residual **ρ = +0.775, p = 0.041**;
Wiener gain against continuity **ρ = −0.718, p = 0.069**; Wiener gain against
reassignment **ρ = −0.468, p = 0.289**. Seven points is not enough to separate
these, and the last is reported as not significant rather than as agreement.

Where they part company is the hips and tail. **Wiener's gain there is 0.016,
0.000 and 0.000** — it deletes the entire above-Nyquist band on three of seven
landmarks — while Viterbi reassigns 0.42%, 0.39% and 0.19% of their frames and
leaves the rest. A smoother cannot tell a hip that jumped from a hip that moved,
so it removes both; a path-selection filter decides that a specific jump was
wrong and keeps everything else. That is the mechanism behind the efficiency
difference, and it is why this arm is a better fit for a corpus where fast rare
movement is the signal.

`results/CONTINUITY.md` measures the third column and Phase D names the nose from
a fourth direction, as the suspect on 61.6% of corrections.

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
