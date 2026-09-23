# Progress — VIEB pipeline

Plan: `~/.claude-1/plans/coding-prompt-streamed-milner.md`.

**Video Interpreter Excluding Bias.** Find the behaviours, find the patterns,
predict them to a reliable degree. This repo owns the tokenizer + hazard-model
programme and the cleaning work under it.

| # | Stage | State | Verdict | Artifacts |
|---|---|---|---|---|
| 0 | Repo split from `~/recur`, provenance frozen | **done** | `PASS` (digest `198eb14ff258c7f6`) | `results/inherited.json`, `vieb/io/spine.py` |
| 1 | Bone-length violations + ExBias R² join | **done** | `GRID_LIMITED` (2.13%, correct branch) · join `PASS` | `results/bones.json`, `BONES.md` |
| 2 | Egocentric SE(2) transform + A4 parity | **done** | A4 `PASS` (exact 1.000000) · leak `PASS` | `results/ego_{bodylen,raw}.json`, `EGO.md` |
| 2R | Step 1R: re-run on the F3-carried `raw` pose arm | **done** | A4 `PASS` (1.000000) · reversal `PASS` 10/10 · leak `PASS` | `results/ego_raw_bodylen.json`, `EGO.md` §Step 1R |
| A | What the cleaning actually changes | **done** | gap policy `PASS` (touches only flagged) · wiener `PASS` (4.99x) | `results/effect.json`, `EFFECT.md` |
| B | Cleaning bakeoff, 14 arms | **done** | `PASS` — `median_0.25` dominates the incumbent | `results/cleaning_report.json`, `CLEANING.md` |
| C | Before/after video | **rendered** | 42 clips, 4.6 MB, all h264 | `results/compare/` |
| C2 | Publish to the VIEB Atlas | next | — | — |
| 3 | Coarse alphabet + RLE, 8 cells | **done — STOPPED** | all 8 retired: median run **1.0 frame** · occupancy `PASS` in all 8 | `results/alphabet.json`, `ALPHABET.md`, `work/tok/` |
| 4 | Rungs 0–2, hazard + MDL | **done — STOPPED** | rung 1 `PASS` at N≤512 (**withdrawn, see 7**) · **rung 2 `FAIL` in all 16** · k\* = 0 | `results/ladder.json`, `LADDER.md`, `TOK_PREREGISTRATION.md` |
| 5 | Frailty: is the 148× fall mixing? | **done** | `PASS` — 0.756 [0.742, 0.769] of the log fall survives speed conditioning | `results/FRAILTY.md` |
| 6 | Distortion + symbol homogeneity | **done** | all 8 alphabets **under-resolved**; N=256 is a compression preference | `results/RESOLUTION.md`, `resolution.json` |
| 7 | **Surrogate falsifier** | **done — FAILED** | both surrogates BEAT the corpus on `rung1−rung0`; the transition-table claim is withdrawn | `results/FALSIFIER.md`, `FALSIFIER_PREREGISTRATION.md` |
| 8 | Coarse sweep N ∈ {8…128} | **not run** | blocked by 7, as registered | — |
| 9 | **Dwell-matched surrogate** | **done** | corpus **beats** both arms at both N; the per-second statistic tracks run rate | `results/DWELL.md`, `DWELL_PREREGISTRATION.md` |
| 10 | Segmentation Step 1: the boundary statistic | **done** | a statistic, not a finding — 0.26–0.64 boundaries/s across 18 cells | `results/BREAKS.md`, `breaks.json` |
| 11 | Segmentation Step 2: **the gate** | **done — NO VERDICT** | gate `FAIL` *and* separability `FAIL` on all four nulls, so the gap is not evidence in either direction | `results/SEGMENTATION.md`, `seg_gate.json`, `DEVIATIONS.md` D7 |
| 12 | **Bank the three passes** | **done** | Q1 is a WIENER result; its unfiltered cell is −0.199% | `results/Q1_BANKED.md`, `ROUGHNESS.md` |
| 13 | **Segment recurrence — the gate** | **done — PASS ×3** | shape +5.93% vs a length-matched windowed +0.74%; twist segments do **not** recur | `results/SEGRECUR.md`, `seg_recur.json` |
| 14 | **Vocabulary or continuum** | **done — a continuum with islands** | 19 clumps the nulls don't reproduce, covering **1.9%**; one shared by 46/89 animals | `results/VOCAB.md`, `seg_vocab.json` |
| 15 | Tokens, merge, MDL | **refused** | 852 mergeable adjacent pairs — three orders of magnitude under the thin-stratum floor | `results/VOCAB.md` |
| 16 | **What the island is** | **done** | a **chain**, 3.7× slower than baseline — a basin of near-immobility, not a token | `results/BEHAVIOUR.md` |
| 17 | Continuing the token search | **planned** | four levers, ranked by the measured blocker | `results/TOKENS_NEXT.md` |
| 18 | **Step A — does the ruler make the clumps?** | **done — PASS** | `open` survives on shape: +3.391% [+2.533, +4.214], 13 clumps vs the nulls' 4 | `results/DISTANCE.md` |
| 19 | **Step C — does the island move with context?** | **done — PASS** | occupancy 1.275% in A against 0.470% in B, paired, p = 0.0050 | `results/CONTEXT.md` |
| 20 | **Step B — can the detector see what is there?** | **done — FAIL** | 12.8% isolation against a 20% gate; `no_edge` 66%, `merged` 0% | `results/DETECTOR.md` |
| 21 | Step D — coverage, then tokens | **not run** | gated behind 20 | — |
| 22 | **Watch the island — blind odd-one-out** | **done — INSTRUMENT BLIND** | positive control 37.8% [0.280, 0.467] against chance 33.3%; neither arm interpretable, 180 trials published for a human scorer | `results/ISLAND_LOOK.md`, `island_look.json` |
| 23 | **Step 0 — is the plant findable at all?** | **done — INCONCLUSIVE** | planted edges sit at the 60th percentile of the criterion's own scalar against real firings' 96th; only 19.87% clear their own threshold | `results/PROBE_AUDIT.md`, `probe_audit.json` |
| 24 | **Step 1 — human boundary annotation** | **done — FAIL** | ceiling does not stand: F1 0.2615 / 0.3046 / 0.3247 at ±2/±5/±10 against chance 0.2510 / 0.2652 / 0.2810, every interval containing its own chance level. Two raters, 58 of 58 clips each. Step 2 blocked for this round; no detector is scored | `results/ANNOTATION.md`, `annot_ceiling.json` |
| 25 | **Step 6 — the island published** | **done** | 361 segments at their own extent against a duration-matched blind control; every published number now a record | `results/island/`, `scripts/island_clips.py` |
| 26 | **Controls on the context result** | **done — mixed** | window control PASSES: length-matched intervals carry no context information (p = 0.2110) and the island survives removing them, residual −0.00699 [−0.01319, −0.00148]. Stillness control was VACUOUS — selected 2.27% of the island's frames — so the freeze-scorer question is still open. Incidental: tracking dropout 6.8× higher in context A | `results/CONTEXT_CONTROLS.md`, `context_controls.json` |
| 27 | **The tracking-noise floor** | **done — mixed** | floor measured at 0.2599 [0.2554, 0.2644] boundaries/s on a constant pose carrying only DLC jitter, 54.8% of the corpus's 0.4746. Above the floor overall; NOT above it in the slow regime — 34 of 75 joint cells overlap, including the registered still × centre × low-confidence cell. The axis is speed, not confidence or wall | `results/NOISEFLOOR.md`, `noise_floor.json` |
| 28 | **Cross-arm concordance** | **done — INCONCLUSIVE** | raw-vs-viterbi boundary F1 0.7559 [0.7239, 0.7878] at ±2, refuting the registered <0.5 prediction; raw-vs-disposition 0.9542. 0.7212 [0.6871, 0.7553] of raw boundaries survive both arms. Stability tracks how far a cleaner moves a keypoint, not how often | `results/NOISEFLOOR.md` §6, `arm_concordance.json` |
| 29 | **Where the island's boundaries sit** | **done — diagnostic, no verdict** | island boundary frames run 2.4× faster than their own interiors (0.0654 vs 0.0275 bl/s); share in the at-or-below-floor strata halves, 72.4% → 36.7%. Still ~1.9× enriched there against boundaries generally (19.4%). Tension halved, not dissolved | `results/NOISEFLOOR.md`, `island_boundaries.json`, D11 |
| 30 | **Segment-level stillness control** | **done — PASS** | registered prediction REFUTED: clump membership adds beyond being slow and long. Residual −0.00685 [−0.01371, −0.00147], p = 0.0270, β = 0.246, after a duration- and speed-matched arm balanced at |SMD| ≤ 0.024. 84% of the published effect survives | `results/STILLNESS_CONTROL.md`, `stillness_control.json` |
| 31 | **Order-controlled plant — the first accuracy number** | **done — mixed** | order-2 (acceleration) recall reaches 50% at 16 σ̄ ≈ 0.99 body lengths; order 0 at 4 σ̄ ≈ 0.25. At 1 σ̄ order 2 is at chance. Predictions 1,2,3,5 held; 4 failed at ladder resolution. **Negative control discharged** | `results/PLANT.md`, `plant.json` |
| 32 | **Trend-filter detector** | **done — GRID_LIMITED, refused** | no α on the registered grid brings the jitter share below 0.05 (best 0.0704), and the solver converges on 0% of recordings above α = 10. Plant comparison never ran; prediction 2 untested. Bar was set without reference to the incumbent's 0.5475 — M12 | `results/TRENDFILTER.md`, `trendfilter_sweep.json` |
| 33 | **Stage 0 — dynamics noise floor and probe** | **done — narrow PASS** | noise model reproduces the measured colour (lag-1 0.5905 vs 0.5817); published floor survives colour, +14.6% (D14). Only `pole_radius` clears the descriptor floor (88.7% vs 5% chance); frequency does not (2.1%); nothing above 2 Hz clears. Probe never reaches 50% recall — best 0.325 at Δf = 4 Hz, but 99.1% within ±30 frames | `results/DYNAMICS.md`, `dynamics_floor.json`, `colour_check.json` |
| 34 | **Stage 1 — two-stream detector** | **done — mixed** | dynamics stream cuts elsewhere (21.6% [20.7, 22.5] `dyn_only`) and is far cleaner at the wall (1.09× vs config's 1.97×), but its jitter share is **0.9665** against the incumbent's 0.5475 — 97% of its rate is noise. Not usable as a boundary source. body_extension moves the config rate 0.2% | `results/TWOSTREAM.md`, `twostream.json` |
| 35 | **Stages 2–3 — frame-rate robustness, island split** | **done — one confounded, one unitary** | 15 fps survival: config 0.2604, dyn 0.3194 — but the refractory floor is 12 SAMPLES so it is 1.75× longer at 15 fps (M13), so the test is NOT settled. Neither rate rose. Island does not split: BIC-gain excess −9.50 (body_extension) and −8.62 (pole_radius) against duration-matched nulls — a unitary slow freezing state | `results/VALIDATION.md`, `subsample.json`, `island_split.json` |

**Tests:** 783 passing, CPU only, `pytest tests/`. `mypy --strict` clean over
`vieb/tok`, `vieb/qc`, `vieb/audit`, `vieb/clean`, `vieb/seg` — 47 files.

## The dwell-matched arms reverse the sign, and show why

`microstate` and `microstate0` stitch the corpus's **own real visits**, so they
preserve its dwell distribution. The corpus **beats both**, at both N, on the
same registered per-second statistic — `+1.317 [+0.899, +1.728]` against
`microstate` and `+10.045 [+9.370, +10.710]` against `microstate0` at N = 256.

With four arms the run-rate confound is no longer a caveat but a measurement:

| arm | runs/s vs corpus | gap n/s |
|---|---:|---:|
| `var5` | 2.102 | −11.65 |
| `phase` | 2.078 | −11.21 |
| `microstate` | 1.739 | +1.32 |
| `microstate0` | 1.459 | +10.05 |

**Monotone, and it changes sign inside the range.** Under a per-second statistic
the verdict tracks how fast the surrogate's symbol stream turns over. Per run —
which needs no correction — **the corpus beats all four**.

`FALSIFIER.md`'s verdict and `LADDER.md`'s P1 withdrawal both **stand as
registered**: per-run was registered as non-retroactive and that is kept. What is
new is evidence about the statistic, and that decision has its own registration
waiting.

The two arms also separate usefully: destroying visit order costs the surrogate
almost everything (+1.26 against the corpus's +11.32), and restoring one-step
dynamics recovers most of it (+10.01). So once dwell is held fixed, **sequence
matters and most of what it carries is first order** — consistent with k\* = 0
and with rung 2 buying nothing.

## The falsifier fired, and what it leaves standing

Both ego-space surrogates — phase randomisation and VAR(5), per recording, run
through `scripts/quantize.py` and `scripts/ladder.py` unmodified on redirected
paths — get roughly **twice** the corpus's `rung1 − rung0`. So that advantage is
not evidence of sequential structure, and `LADDER.md`'s P1 reading is withdrawn.

**The substantive replacement finding.** The corpus costs **45.7 nats/s** where
its phase surrogate costs **86.9** — described in half the bits. But **per run**
the corpus is more expensive, 5.127 against 4.806. Its entire compressibility
advantage is that it has 2.1× fewer transitions per second.

> The corpus's advantage over a spectrum-matched surrogate is **dwell duration,
> and nothing else.** Per transition its symbol stream is no more predictable
> than a structureless signal's. That sits consistently with rung 2 buying
> nothing and with `FRAILTY.md`: the information is in how long, not what next.

A run-rate confound in the registered statistic is recorded in `FALSIFIER.md` —
it reverses the comparison per run — and amending the statistic after seeing that
needs its own registration. The coarse sweep has **not** been started, as
registered.

## Where Step 3 stopped, and why Step 4 is not built

Two arms (`plain`, speed-stratified) by four alphabet sizes (256–2048), fitted on
**tune**, assigned over 22,355,989 frames. **Every cell returned a median run of
1.0 frame** against a pre-registered floor of 3, and all eight are retired.

Occupancy was never the problem: the worst symbol in any alphabet holds 1.23% of
frames, **no symbol at any N is unused**, and `self_transitions` is 0 everywhere.
The failure is on the time axis.

Three things came out of the diagnosis (`work/tok/flicker.json`, tune only):

* **It is not the velocity channels.** Dropping the three twist columns leaves
  the median at 1 frame.
* **Wiener doubles the median run** (1 → 2 frames, mean 3.84 → 5.18). Any dwell
  measured on the standard pipeline is partly the filter's own autocorrelation —
  the third appearance of that failure mode here, now quantified.
* **The speed-stratified arm is worse than plain**, 46–52% of frames in one-frame
  runs against 16–24%. A hard stratum cut on a noisy scalar flips a frame's whole
  symbol block on a speed wobble.

The grid was **not** widened below N = 256 and the stop condition was **not**
moved to a frame-weighted statistic, though `ALPHABET.md` records that the
frame-weighted picture differs for the `plain` arm (two thirds of frame mass sits
in runs longer than 3 frames). Both would be choosing a parameter against an
outcome after seeing it. `results/TOK_PREREGISTRATION.md` is deliberately not
written: registering a prediction about a blocked experiment is ceremonial.

**The ladder then ran anyway**, by the project owner's decision, on the argument
that the run-length floor is a heuristic pre-filter while MDL is the principled
selector and charges duration explicitly. That override is recorded in
`TOK_PREREGISTRATION.md` §1, committed before any ladder job was submitted, and
every ladder shard carries `retired_by_runlength: true`.

## What the ladder found

**Rung 2 does not beat rung 1 in any of the sixteen cell × abstain-arm
combinations**, so the registered stopping rule fired. It loses on the **data
term alone** (44.295 against 43.365 nats/s at `plain`/N=256), before any codebook
charge — conditioning on elapsed time makes held-out prediction worse, not merely
more expensive.

* **Rung 1 beats rung 0** at N = 256 and 512 on both arms, by +8.9 to +36.0
  nats/s, better on **89 of 89 animals**. The transition table carries real
  information. At N ≥ 1024 the N² table costs more than it earns.
* **k\* = 0.** No completed `(u, d)` pair pays for itself. At N = 256, k = 1
  needs 136M parameters and k = 2 needs 929M, against 3.1M fit runs.
* **The hazard falls 148×** with elapsed time (0.489 → 0.0033) — dwell is not
  memoryless — **but it falls the same way in every state**, spread 1.09–1.44×.
  The duration structure is large and *shared*, which is exactly why rung 2's
  761,088 parameters buy nothing. The symbols differ in what they are, not in
  how long they last. A falling hazard is also what unmodelled heterogeneity
  produces, and `LADDER.md` refuses to read it as per-bout memory.
* **MDL and the run-length condition agree.** MDL prefers `plain` over `speed` at
  every N and prefers N = 256, the coarsest point, where it is still pushing
  downward. The selection is `GRID_LIMITED` toward coarser alphabets. The grid
  was not extended to chase it; that needs its own registration.

---

## How this repo relates to `~/recur` and `~/shapeflow`

**Hash the artifacts, import the utilities.**

recur reads shapeflow's *artifacts* and never its *code* — `qc/bones.py` even
re-implements `pair_indices` rather than importing it. That rule is about **data
provenance**: shapeflow is a live repo whose stages could re-run underneath,
changing a number without changing anything a reader could see. It is not a rule
against reusing a verdict type.

So this repo splits the two:

| | treatment |
|---|---|
| recur / shapeflow / exbias **artifacts** (`.npz`, `.json`) | content-hash contract in `vieb/io/spine.py`, frozen once, re-checked in every job preamble |
| recur's **utility layer** (`Read`, `boot`, `splits`, `anchors`, `util`, `labels`, `kendall`, `geom/*`, `qc/swap`) | imported via `PYTHONPATH` |

**Nothing is copied.** Forking `Read` would give the project two verdict
vocabularies; forking `splits` would give it two copies of a split whose whole
point is being fixed once and never resampled.

One divergence from recur's spine, deliberate: the digest excludes `git_sha`.
Upstream folds it in, so an unrelated commit changes the label every shard
stamps, and two shards computed on byte-identical data end up looking different.
The digest answers *was this the same data*; the git sha answers *which commit
produced it*. Both are recorded; only the first is the label.

The venv is a **symlink to recur's** — the pins are identical by design and two
copies would be ~5 GB of duplicated torch. `pip install` from either repo lands
in both, so anything installed here must be added to both `requirements.txt`
files. `env.sh` says so.

---

## Step 1 — bone-length violations

Skull-triangle violations run **2.134%** of frames at ε = 0.10 (unfiltered pose,
raw metric) — inside the 2–15% band, so *correct rather than exclude*, and cost
out an ensemble-DLC path. Shuffled-keypoint ceiling 79.6%, a 37× margin.

Four findings the sweep produced that were not in the brief:

1. **The excess is not perspective.** Removing the per-frame common scale does
   not lower the rate (2.134% → 2.319%); it raises it slightly, at every ε. The
   two-metric design existed to test exactly this and it settles against the
   rearing hypothesis.
2. **There is no elbow.** Relative drops run 34%, 37%, 44%, 54% — monotonically
   increasing, one population thinning out. No ε on this grid is a principled
   cut, and the branch itself moves with the threshold: at ε = 0.20 the rate is
   1.203% and lands in the *exclude* branch instead.
3. **The R² join reverses under its own confound.** ExBias's R² is largely a
   readout of segment length (mean duration 4.00 s in the worst decile against
   0.53 s in the best; ρ(duration, R²) = −0.538), and the violation rate's
   denominator *is* segment length. Marginally ρ = −0.054 and the join reads
   `GRID_LIMITED`; with duration held fixed ρ = **−0.104 [−0.115, −0.094]** and
   it reads `PASS`. The confound was suppressing the association, not creating it.
4. **The Wiener filter absorbs 23% of them**, so **1.648%** of frames in the
   feature space every downstream stage consumes — Q1's included — carry
   impossible skull geometry. Shrinkage makes an excursion smaller without making
   the frame correct. This is why the artifact ablation at Step 4 is mandatory.

`results/BONES.md`.

---

## Step 2 — the egocentric transform

**A4 passes exactly.** Closed-form recovery of centroid speed and body-axis
angular velocity from (s, ξ, ℓ_a) reads **R² = 1.000000 on both**, over 6,403,616
frames in 1,149 `report` recordings, against thresholds of 0.98 and 0.94. The map
is a bijection given the equivariance it removed. 17 dims in, 17 out.

1. **A4 cannot catch the error the brief warns about.** Substituting separate
   differencing for the SE(2) logarithm costs almost nothing until the animal
   turns hard — speed R² 0.9973 at 0.9 rad/s, 0.9853 at 3 rad/s, breaching 0.98
   only above ~4.5 rad/s. The guard is `tests/test_se2.py`, which checks
   `se2_log` against `scipy.linalg.expm` and fails the naive version at every
   turn rate including zero.
2. **The brief's ξ is dimensionally wrong for the rotation.** f/ℓ_a on ω gives
   rad·s⁻¹·px⁻¹ and makes a large mouse's turning read systematically slower.
   Translation takes f/ℓ_a, rotation takes f alone.
3. **Body-length normalisation works, for a reason the brief got wrong.** It cuts
   animal identification from **15.0% to 6.9%** against 0.336% chance, and
   session identification from 17.9% to 1.4%. Quoted in **accuracy, not nats**:
   both `raw` probes stopped on `max_iter`, so their nats figures are refused.
   Body length spans 89–172 px over 298 animals, a 1.92× range. The brief asked
   for this to fix "5.88 nats", which was the previous instrument's number.
4. **The identity-leak metric can hide a leak entirely, and did.** The `raw`
   session probe reads 0.0000 nats while picking the right session out of 3,846
   **17.9%** of the time, 687× chance. `identity_leak` returns
   `max(log N − CE, 0)` and CE exceeds log N when the probe is confidently wrong
   — which it is, because the solver stops on `max_iter=400`. Fixed in
   **recur**, where the defect lives: `leak_read` now refuses an unconverged
   probe and a clamped zero above 3× chance, and `identity_leak` records
   `converged`/`n_iter`/`max_iter`.
5. **Single-frame pose-only recovery is negative** (−0.185 / −0.065) held out by
   animal — worse than predicting the test mean.

Rank is **11 of 14 on every one of the 298 animals**, exactly SE(2)'s three
degrees of freedom. The reversal audit runs **10 checks** — recur's seven,
composed with three for the twist, which is odd in every component and needs a
one-frame shift because it lives on the interval between frames.

`results/EGO.md`.

---

## Phases A–C

**The bone check creates almost no variance.** The gap policy moves nothing
outside the frames it flagged, touches 1.34% of keypoint-frames, and leaves the
speed distribution within 10% at every quantile. That was the question, and the
answer is that it is the smallest of the three layers.

**The filter is where the data moves, and it had never been measured.** Wiener
shifts 99.4% of keypoint-frames, 56.3% by more than half a pixel, median 0.83 px,
concentrated 4.99x on flagged frames and varying 17x across landmarks (nose 2.578
px, centre 0.153). It removes **41% of median instantaneous speed and 24% of
median turning**; Butterworth preserves turning far better (0.922 against 0.760)
for a similar cut to speed, and turning is what the egocentric `omega` channel
carries. `results/EFFECT.md`.

**The bakeoff licenses a change, but not on the axis it first appeared to.**
Dominance is decided by **non-overlapping animal-bootstrap intervals**, and
`violation_rate` separates for **no arm at all** — the intervals run ~1.43–1.79%
against the incumbent's ~1.49–1.87%. An earlier version of this read compared
point estimates and announced that an arm beat the incumbent on all three axes;
it does not. Eleven arms do separate on **retention** (and the targeted ones on
displacement): `median_0.25` keeps 32.0% [30.2, 33.7] of the power above f_c
against Wiener's 22.0% [20.8, 23.2]. The case for a change rests on preserving
fast movement at the same violation rate and the same distortion.

**Composing a de-glitcher with a smoother buys nothing.** `viterbi+median_0.50`
against `median_0.50` alone: all three intervals overlap.

**And nothing temporal will close the gap.** On the eight worst recordings the
median filter cuts violating runs from 207 to 66 while the median run LENGTH
triples, 2 frames to 9; 96% of what survives lasts longer than 3 frames. What a
temporal filter leaves behind is temporally smooth — a landmark parked off the
body and held there satisfies a median and satisfies Viterbi's motion prior. The
remaining gain needs an anatomical prior or better detections, not another
filter.

**Anipose's Viterbi arm is the efficiency outlier.** It buys **16.9%** of
violation reduction per pixel of displacement against the incumbent's 4.1%, moves
0.31% of keypoint-frames, and retains 58.6% of the power above f_c against 22.0%.
It reduces violations least in absolute terms and disturbs the data least by a
wide margin — the trade-off a corpus where fast rare movement is the signal
should care about. Recorded as *unavailable* in an earlier version of this repo;
that was wrong and is corrected in `vieb/clean/viterbi.py`.

**None of this is settled until MDL.** The three axes disagree about which arm
wins, and the decision-relevant axis — does the behaviour model built on it
predict better — costs a full tokenizer run per arm and waits for Step 4.

## Phase D — suspect-keypoint disposition

`results/DISPOSITION.md`, `results/disposition.json`. Pre-registered in its own
commit before any code touched data, and **amended** after the first run.
**Q1 was not re-scored.**

**The gate passes.** Constraining the skull lowers violations on the **trunk —
bones the corrector never sees — from 2.4042% to 2.3219%**, a 3.42% relative fall
on 89 report animals. That is the only check in this phase that could have failed
on its own terms: distortion, retention and MDL all measure magnitude, not
direction, and a corrector that is confidently wrong scores well on every one.

**The atom check fails, twice.** A probe held out by three-second block separates
corrected frames from clean ones at **0.680** balanced accuracy against a
pre-registered limit of 0.60. The same probe on the array the corrector was
*handed* scores **0.810**, so the correction moves those frames towards the clean
distribution without making them ordinary. Whether the residue is the selection,
the 13.2% still landing on the constraint surface, or a reason to prefer
down-weighting is not settled here — it is settled at Step 3 on the MDL axis,
where the disposition is one cleaning arm among several.

**The first run did not implement what was pre-registered, and the pre-registered
prediction is what caught it.** 100.0% of corrections landed exactly on a
codimension-1 surface where the table fixed the target as the interior; the
constraint set was all 21 pairs rather than the SKULL, so a nose correction
constrained trunk bone (2,3) directly on 61.6% of corrections; and 0.5% exited
still violating and were reported as corrected. Fixed, re-run, and both runs
reported. The gate's verdict survived the leak — the direct path covered 0.0059%
of frames against a 2.40% trunk rate, under a tenth of the observed fall — and
with the trunk genuinely unconstrained the fall gets *larger*, 3.10% → 3.42%.

**The mass is small and the magnitude is not.** 0.5043% of frames corrected,
1.3209% abstained — but the median correction moves the suspect **0.194 body
lengths** and the p90 moves it **0.807**. Inside a ≤3-frame envelope the predictor
is not nudging a noisy keypoint; it is saying the keypoint was most of a body away.

`reliability` is emitted per keypoint-frame (mean 0.6984) and enters Step 4 as a
**covariate, never a likelihood weight**.

## Phase E1 — continuity, and the Anipose arm made visible

`results/CONTINUITY.md`, `results/continuity.json`. Pre-registered in its own
commit before the corpus run. The bakeoff verdict is **not** restated and Q1 is
not re-scored.

**The bone check is the wrong net for most of what a viewer sees.** A continuity
residual — fit the similarity carrying the six other keypoints from a
neighbouring frame onto this one, take what is left, in body lengths — flags
**3.798%** of report frames against the bone check's **1.122%**, and **only 8.2%
of spiking frames are bone-flagged**. Jaccard 0.068. A length test is
structurally blind to a keypoint sliding *along* a bone, and 232,792 frames carry
a discontinuity no ε could have found.

**Anipose Viterbi is the efficiency outlier by a wider margin than the bakeoff
showed.** It removes 67.0% of spiking frames at 0.189 px mean displacement —
**354% per pixel against the next best arm's 83%** — while retaining 58.6% of the
power above f_c against the two smoothers' 22% and 27%.

**Four instruments name the nose.** Worst continuity residual of the seven
(0.0067 against the centre's 0.0034); most-reassigned landmark for Viterbi
(0.5315% against the centre's 0.0224%); heavily smoothed by Wiener (gain 0.2903
against the centre's 0.5832); and Phase D's suspect on 61.6% of corrections.
Reassignment and continuity rank-correlate at ρ = +0.775 (p = 0.041); the other
two pairs do not reach significance at n = 7 keypoints and are reported as such.

**Two predictions did not hold.** Prediction 2 said no arm would lower `step`;
two did, and the honest reading is that **the falsifier was badly constructed** —
a 15-frame median smooths *across* a 9-frame park, shrinking the step while the
landmark stays off the animal. The `CLEANING.md` claim it was meant to test is
not withdrawn, and the run-length measurement that does test it is unchanged.
Prediction 3 said the disposition would lower spike on the held-out side; it
falls 4.4% and **the intervals overlap raw's**. Against 10.9% on the donor side —
so better than half of Phase D's apparent continuity gain is the corrector
agreeing with its own predictor, which nothing inside Phase D could have shown.

## Phase F — the injection benchmark

`results/INJECTION.md`, `results/injection.json`. Pre-registered before the run.
Corruption with known truth, injected into frames every instrument here calls
clean, with every parameter taken from an earlier measurement rather than chosen.

**A pre-registered falsifier fired and reversed a decision.** Prediction 3 said
`median_0.50` would leave the data further from the truth than it found it. Its
net is **−0.0026** [−0.0029, −0.0024], the best of any arm, negative in every
speed stratum. The registration named the consequence in advance, so
**`median_0.50` goes back on the Step 4 MDL branch**: the arms carried through
are `raw`, `viterbi`, `median_0.50`, `disposition`.

| arm | repair | damage | net |
|---|---:|---:|---:|
| `raw` | 0.3635 | 0.0000 | +0.0000 |
| `median_0.50` | **0.1122** | 0.0031 | **−0.0026** |
| `viterbi` | 0.2894 | 0.0000 | −0.0017 |
| `disposition` | 0.3035 | 0.0000 | −0.0013 |

The movement-retention evidence is unchanged and still true — `median_0.50`
deletes 86–90% of median movement — but on this pool the movement it deletes is
more wrong than right: 95.4% of teleport error removed against Viterbi's 35.3%.

**No arm repairs a sustained park.** 10.5% / −0.0% / 0.0%. This is Phase E's
failed prediction 2 rebuilt on an instrument that can falsify it — a park here
has a known true position, so a median smoothing *across* it earns nothing —
and it gives `CLEANING.md`'s standing claim direct support for the first time.

**The pool is 4.45% of frames, not the 23.4% the feasibility check suggested**,
bound by DLC confidence (10.55% of frames have every keypoint above 0.60). It is
also slow: mean speed 0.50× the corpus, and only 1.81% of pool frames exceed the
corpus p90 where 10% would be unbiased. `median_0.50`'s benefit falls 5× from the
slowest stratum to the fastest while `viterbi`'s rises from zero — opposite
trends — so this phase is decisive about slow frames and silent about fast ones.

**`wiener` could not be benchmarked.** It is read off disk from shapeflow and was
never reimplemented here, so it cannot be applied to a freshly corrupted array.
The pre-registration listed it; it was dropped and that is recorded.

## Held-out replication — the island's context effect does not survive

`results/FIT_REPLICATION.md`, registered alone as `28f9d0d` before the code
existed. The frozen detector ran on the **149 `fit` animals**, and each of their
**108,678** segments was assigned to the `report` island by distance to the
island's own members at the frozen **θ = 0.18998**, normalised by the `report`
arm's scale. **Nothing was re-clustered and no PCA was refitted** — the bank's
PCA is full-rank, so it is a rotation and raw-space distances reproduce stored
ones exactly (ratio 1.0000 on 7/7 stored neighbour pairs, checked first).

**The transfer rule passed its own gate.** Fit segments land within θ of an
island member at **0.541% [0.389%, 0.705%]** against **0.192% [0.123%, 0.276%]**
for a duration-and-speed-matched bank of 361 non-island `report` segments —
2.82×, non-overlapping, balance |SMD| 0.0240 checked before scoring. And the
rule *is* the single-linkage membership criterion rather than a stand-in for it:
run back over `report` it selects exactly the 361 labelled members, 0 admitted
and 0 missed. **That agreement is a tautology** — θ is the merge height that
defined the clump — and is reported as one.

**The island recurs; its context asymmetry does not.** Prevalence reproduces
almost exactly (0.561% of report segments, 0.541% of fit). But occupancy B − A
is **−0.00216 [−0.00536, +0.00123]**, p = **0.2020**, over 739 cells and 149
animals, against the published **−0.00811 [−0.01520, −0.00243]**, p = 0.0050.
A / B falls from **2.71× to 1.34×**, and the gap closes from both sides — A
occupancy down a third, B occupancy up a third.

**Not an underpowered null.** MDE 0.0047 over 739 pairs against `report`'s own
0.0087 over 439 — **the held-out split is the better-powered test** — and power
against the published point estimate is **99.8%**. What it does *not* exclude is
an effect a third that size: against −0.00243 power is only 30.5%, and the two
intervals still overlap over 22.9% of the report interval's width.

So `CONTEXT.md`'s headline is now a **`report`-split finding that did not
replicate**, qualified rather than withdrawn, and it must be written that way
wherever it appears. **D15** records that the registration's phrase
"length-and-speed-matched" was ambiguous and that the executed check matches the
*target* bank, the only reading that is not circular.

## The pixel channel — a standard freeze measure finds what the island lost

`results/PIXEL_NOISEFLOOR.md` and `results/PIXEL_PILOT.md`, under a registration
committed alone and then amended alone (D16). ezTrack was **read from source and
not installed**; the reimplementation is checked against a literal transcription
of `Measure_Freezing` on 300 random inputs rather than against expectation.

**The arena calibrates itself, and it has to.** With no animal-free video, the
animal is masked out per frame and the remaining pixels supply ezTrack's own
rule, `mt_cutoff = 2 x percentile(|delta|, 99.99)`. Each recording contributes
**1.0-1.5 billion** arena pixel-pairs against ezTrack's own ~3 million, and 0 of
1,440 recordings fell below the registered minimum. The derived cutoff spans
**6.50 to 128.50 grey levels** — a factor of 20 — while ezTrack's published
default is a single global **10**. A global threshold is not defensible here, and
that is now measured rather than argued.

**The arenas differ, and the pilot could not see it.** Paired within (animal,
day), the arena's own noise floor is **−1.296 [−2.020, −0.505]** grey levels
B − A over 144 animals: Context A is noisier, agreeing in sign with the 32%
bits-per-frame excess. The registered 300-recording pilot gave **−0.430 [−1.927,
+1.366]** and would have licensed the opposite conclusion. **The pilot was
underpowered for the one question it existed to interpret** — the same lesson as
D16, one level up.

**Q2 answers, and it is §0's first branch.** On the same 30 `fit` animals where
the island's context effect is null, the freeze score gives **B − A = −0.1264
[−0.1706, −0.0851]**, p = **0.0005** — 12.6 points more freezing in Context A —
and it **survives** subtracting each frame's own arena noise (**−0.1611 [−0.1994,
−0.1217]**, further from zero). §0 fixed the reading in advance: *a standard
pixel measure finds a context effect where the island does not; the island is
the worse instrument.* The **magnitude** is not established — the only arm
defined everywhere is ezTrack's untransplanted absolute threshold, a pixel count
tuned on 320x240 for a 640x480 corpus — and it is not new biology either, since
keypoint speed already said animals move 24% less in A. What is new is that the
crude channel carries the contrast on held-out animals and the island does not.

**Q1 refused by one animal, and the bar did not move.** 19 animals yield a
defined rank correlation against a registered minimum of 20; the value is **not
reported**. The diagnosis is the useful part: **90.7% of cells hold exactly zero
island frames**, so at 0.54% prevalence the island is too sparse to
rank-correlate at cell resolution. The convergent-validity question survives at
segment resolution and belongs in its own registration.

**The held-pose question is settled, against the incumbent description.** A
census of 1,440 recordings and **43,206** zero-ego-speed frames: **14.2%
duplicate video frames, 9.6% tracking dropouts, 76.1% genuine immobility**, and
the dropout share falls to 0.9% at the most generous margin. `CONTEXT.md` and
`CONTEXT_CONTROLS.md` carry the correction, as registered, and their intervals
are unchanged — what changed is what conditioning on held poses *means*.

**Two defects only running could find.** **D17**: §3's noise-derived cutoff and
§4's percentile threshold are each defensible alone and degenerate together, so
27 of 28 arms could not be read and the registered headline was degenerate on
62% of cells. **D18**: §7's refusal compared grey levels with a fraction and
therefore fired on every possible result; replaced by a correction that could
have failed and did not.

## Open

Steps 3–4, with **four** arms and the MDL entropy normalisation: a flattened
corpus has fewer, longer runs and a shorter code, so an absolute code-length
comparison across cleaning arms measures how much signal each deleted. Report in
bits relative to each arm's own rung-0 marginal entropy, as the plan already
requires for the labeller bakeoff.
