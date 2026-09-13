# What each cleaning method actually does

Background for every arm that has been scored in this programme. Written because
"median 0.50 s looks solid" is a judgement about a filter whose mechanism had
never been written down, and the mechanism turns out to explain every number
measured about it.

The input to all of them is DeepLabCut's output: 7 keypoints × (x, y) per frame,
30 fps, one uncalibrated overhead camera, 3,846 recordings, 22,355,989 frames.

---

## The substrate: what arrives before any arm runs

**shapeflow's gap policy.** Keypoint-frames DLC marks missing, or that the bone
check flags, are removed. Gaps of **≤ 3 frames are linearly interpolated**
(1.340% of keypoint-frames); longer runs are **abandoned** (3.385%). This is not
an arm and it is not optional — it is what `pose_unfiltered` means, and every arm
below receives its output.

**What is wrong with the substrate.** Two independent instruments measure it:

| instrument | flags | what it can see |
|---|---:|---|
| bone check, ε = 0.10 | 1.12% of frames | a skull bone longer than the animal's own reference |
| continuity residual | 3.80% of frames | a keypoint disagreeing with both neighbouring frames |

They overlap on **8.2%** of what either finds. A length test is blind to a
keypoint sliding *along* a bone; a continuity test is blind to an error that is
stable over time. Roughly **4% of frames carry a detectable tracking error**.

---

## `median_0.50` — a 15-frame centred rolling median

**The mechanism, exactly.** For each keypoint and each coordinate *independently*,
the output at frame *t* is the median of frames *t−7 … t+7*. Fifteen samples,
centred, reflect-padded at the recording edges — reflect rather than edge-hold,
because holding the first value flat makes the start of every recording look like
freezing. `scipy.ndimage.median_filter`, matching
`movement.filtering.rolling_filter(statistic="median")`.

x and y are filtered separately, so the output position is generally **not any
position the animal was ever observed in** — it can take its x from one frame and
its y from another.

**The one fact that explains everything about it.** A median of 15 samples is
unchanged unless more than 7 of them are corrupted. So:

| corruption run | fraction of its error removed |
|---:|---:|
| 1 frame | 99.9% |
| 4 frames | 99.8% |
| **7 frames** | **99.6%** |
| **8 frames** | **0.2%** |
| 15 frames | 0.1% |

That is a **cliff, not a slope**. Measured on synthetic injection; the boundary is
arithmetic, not empirical tuning.

Everything else measured about this arm follows from the cliff:

* **It repairs 95.4% of teleport error.** Teleports are 1–3 frames. All below the
  cliff.
* **It repairs 10.5% of park error.** Parks are 4–30 frames. Only those ≤ 7 are
  reachable.
* **Violating runs go from a median of 2 frames to 9 after filtering.** This was
  previously described as "what survives is temporally smooth". The real
  mechanism is sharper: the filter deletes every run of 7 frames or fewer and
  leaves the rest **untouched**, so the surviving distribution is simply the
  original one truncated below 8. The median rises because the short runs are
  gone, not because anything got longer.
* **It deletes 86–90% of the animal's median movement.** A 15-frame median of a
  mostly-slow signal returns long constant stretches. Median centre speed falls
  from 0.1071 to 0.0129 body lengths per second.

That last point is why it must not be adopted casually. **An 88% cut in median
speed is a persistence prior**, and this project retracted a result once for
importing one through a labeller. It has still never been shown to *help* the
thing the programme is for, which is predicting behaviour.

---

## `wiener` — shapeflow's incumbent, and the array every published number uses

**The mechanism.** A frequency-domain shrinkage. shapeflow estimates, per
keypoint, a movement spectrum and a noise spectrum from a calibration, and
multiplies the Fourier transform of each keypoint's trajectory by

    G(f) = SNR(f) / (1 + SNR(f))

There is **no cutoff frequency**. Every frequency is attenuated by how much of it
the calibration thinks is signal. Where SNR is high, G ≈ 1 and the content passes;
where SNR is low, G ≈ 0 and it is removed.

**What that does per landmark**, gain at Nyquist:

| keypoint | gain | meaning |
|---|---:|---|
| center | 0.5832 | over half the fast content kept |
| right_ear | 0.5773 | |
| left_ear | 0.5007 | |
| nose | 0.2903 | |
| left_hip | 0.0161 | |
| right_hip | 0.0000 | **the entire fast band removed** |
| tail_base | 0.0000 | **the entire fast band removed** |

It touches **86% of keypoint-frames** by more than 0.01 px, median 1.42 px. It is
not a repair operator; it is a reweighting of the whole corpus.

**It cannot be benchmarked against known truth here.** This repo reads the Wiener
array off shapeflow's disk and has never reimplemented the filter — deliberately,
because a second implementation could silently disagree with the array every
existing result was computed on. An injection benchmark has to apply the arm to
an array it has just corrupted, and there is no callable to apply. So the
incumbent is the one arm whose recovery performance is unknown.

---

## `butterworth` — order-4 zero-phase low-pass at 4.833 Hz

Also shapeflow's, also read off disk. A conventional low-pass with a **hard
cutoff**: everything above 4.833 Hz is removed regardless of whether it is signal.
Zero-phase (forward-backward), so it introduces no lag. The cutoff was chosen
from the calibration's crossover. Same benchmarking limitation as `wiener`.

---

## `savgol` — Savitzky–Golay, polynomial order 2

A sliding window that fits a **quadratic** to each keypoint's trajectory and takes
the fitted value at the centre. `savgol_0.33` uses an 11-frame window.

The difference from a median: a polynomial fit **preserves peak height** where a
moving average flattens it, which is why it is the usual choice for kinematics.
The difference from a median in the other direction: it is a *least-squares* fit,
so a single outlier pulls the whole window, where a median ignores it entirely.
It smooths and does not de-glitch.

---

## `position_outlier` — refineDLC's rule: gate, drop, interpolate

Frame-to-frame displacement per keypoint is thresholded — by MAD, IQR or a
percentile — flagged frames are **dropped**, and the resulting gaps are linearly
interpolated.

Its distinguishing property: **ordinary frames are left exactly as measured**. It
is the only arm here that can lower the violation rate without moving anything
else. That is also its weakness — it has no opinion about where the dropped
keypoint should have been beyond a straight line.

---

## `viterbi` — Anipose's path selection, and why it is not a filter

**The mechanism.** For each keypoint independently, build a small set of candidate
positions per frame, then choose the sequence through them that maximises a
motion-prior likelihood by dynamic programming.

The candidates at frame *t* are: this frame's detection, the detections from the
previous `n_back = 3` frames carried forward and decayed by 2⁻ʲ, and an explicit
**missing** state. That is 4 particles per frame. Transitions further than
`thres_dist = 30 px` are penalised. Anipose 1.1.24, loaded from its installed
source.

**This is a de-glitcher, not a low-pass.** The decision it makes at each frame is
binary in character — *accept this frame's jump, or carry an older position
forward* — and if it accepts, the value passes through **unchanged**. It never
averages. So:

| | keypoint-frames moved | median move when it moves |
|---|---:|---:|
| Viterbi | **0.31%** | 46.65 px |
| Wiener | 86% | 1.42 px |

**It preserves 98–100% of median movement on every keypoint**, because 99.7% of
the corpus passes through it untouched. It is the only arm measured here that
does not alter the movement distribution.

Which landmarks it reassigns, from 80 report recordings:

| keypoint | reassigned |
|---|---:|
| nose | 0.5315% |
| left_hip | 0.4212% |
| right_hip | 0.3872% |
| right_ear | 0.2555% |
| tail_base | 0.1881% |
| left_ear | 0.1385% |
| center | **0.0224%** |

The nose is reassigned **24× more often than the centre**, which agrees with both
other instruments about which landmark is least reliable.

---

## `disposition` — this project's own, Phase D

Not a filter. A per-frame decision with three outcomes.

1. **Find the suspect.** On a frame where a skull bone is too long, the suspect is
   the keypoint common to every violating bone — well defined on 96–99% of
   violating frames. Ties broken by DLC confidence.
2. **Check the run length.** Runs of **≤ 3 frames** are eligible for correction.
   Longer runs are **abstained** — a 300 ms run is the tracker sitting on the
   wrong body part, and moving it to the nearest legal position would produce
   something anatomically valid and still wrong, which is worse than leaving it
   flagged.
3. **Correct, by prediction not by projection.** The suspect is replaced with its
   position in the nearest clean frame, carried through the similarity transform
   fitted on the other six keypoints. Then clipped to feasible if needed.

It moves **0.50%** of frames, by a median of 0.19 body lengths, and abstains on
1.32%. Its damage to correct frames is 0.00004 body lengths.

**It is the only arm with a gate that could have failed.** Constraining the skull
and evaluating on the trunk — bones the corrector never sees — trunk violations
fall 2.4042% → 2.3219%. It also fails its own atom check at 0.680 balanced
accuracy against a pre-registered 0.60, which is recorded rather than explained
away.

---

## How they score against known truth

Corruption with measured shapes injected into clean frames, each arm asked to put
it back. `net` is the change in total error against truth; negative is better.

| arm | teleport repaired | park repaired | damage | **net** |
|---|---:|---:|---:|---:|
| `raw` | 0.0% | 0.0% | 0.0000 | +0.0000 |
| `median_0.50` | **95.4%** | 10.5% | 0.0031 | **−0.0026** |
| `viterbi` | 35.3% | −0.0% | **0.0000** | −0.0017 |
| `disposition` | 30.9% | 0.0% | 0.0000 | −0.0013 |

**No arm repairs a parked landmark.** That is the error a detector produces when
it locks onto the wrong body part, and it is the one class nothing temporal or
geometric reaches. 10.5% is the ceiling, from the arm that gets there by
accident — the parks it fixes are the ones under its 7-frame cliff.

**The test pool under-represents fast movement 5.5×**, so these numbers settle the
slow regime and are silent about the fast one. `median_0.50`'s advantage falls 5×
from the slowest speed stratum to the fastest; `viterbi`'s rises from zero. Where
they end up in the regime that matters is not known and is decided by held-out
MDL, at Step 4, with all four arms.

## The honest summary

Nothing here fixes the data. Every arm is post-hoc repair of a detector that is
wrong on about 4% of frames, and the error class that matters most — a landmark
parked on the wrong body part — survives all of them. **Retraining the detector is
the only thing that would actually raise the ceiling.** It requires re-inferring
3,846 videos and has been costed out twice; that cost is the reason this page
exists instead.
