# Step 2-3 — the alphabet sweep, and the stop condition that fired

**Every one of the eight cells is retired.** The registered stop condition is a
median run of three frames or fewer; all eight read a median of **1.0 frame**.
The ladder does not run on any of them.

| | |
|---|---|
| pose arm | `raw` = `held_array(pose_unfiltered, missing)`, F3-carried |
| scale arm | `bodylen`, `ell_a` per animal |
| fitted on | **tune** — 60 animals, 770 recordings, 4,475,368 frames |
| inherited digest | `198eb14ff258c7f6` |
| corpus | 3,846 recordings, 22,355,989 frames, anchor-checked |

## The sweep

Occupancy is measured on labelled frames; run length excludes abstain runs.

| arm | N | top symbol | Gini | median | p75 | p99 | mean | runs =1 | frames in =1 | frames in <=3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `plain` | 256 | 1.23% | 0.384 | **1.0** | 3.0 | 34.0 | 3.59 | 56.8% | 15.8% | 32.8% |
| `plain` | 512 | 0.66% | 0.361 | **1.0** | 2.0 | 29.0 | 3.23 | 59.9% | 18.5% | 36.7% |
| `plain` | 1024 | 0.37% | 0.380 | **1.0** | 2.0 | 25.0 | 2.97 | 63.0% | 21.2% | 40.0% |
| `plain` | 2048 | 0.19% | 0.357 | **1.0** | 2.0 | 23.0 | 2.74 | 65.9% | 24.0% | 43.1% |
| `speed` | 256 | 1.06% | 0.328 | **1.0** | 2.0 | 6.0 | 1.54 | 71.1% | 46.1% | 81.1% |
| `speed` | 512 | 0.52% | 0.347 | **1.0** | 2.0 | 6.0 | 1.50 | 72.6% | 48.3% | 82.7% |
| `speed` | 1024 | 0.28% | 0.364 | **1.0** | 2.0 | 6.0 | 1.47 | 74.1% | 50.4% | 83.8% |
| `speed` | 2048 | 0.18% | 0.362 | **1.0** | 1.0 | 6.0 | 1.44 | 75.5% | 52.4% | 84.7% |

`self_transitions` is **0** in every cell — no run spans a recording seam and no
abstain gap was spliced out. Abstain is **7.03%** of frames corpus-wide
(6.75% fit / 7.18% report / 7.53% tune).

## Three readings, and they do not all point the same way

**1. Occupancy is healthy everywhere and was not the problem.** No symbol comes
near the 40% dominance threshold — the worst is 1.23% — and **not one symbol of
any alphabet is unused**, at any N up to 2048. The worry that k-means would spend
its budget on immobility and leave the moving frames sharing a handful of symbols
did not happen. The alphabet is well spread; it is the *time axis* that fails.

**2. The stop condition is run-weighted, and the frame-weighted picture differs.**
On `plain` at N = 256, 56.8% of runs are one frame long — but those runs hold only
**15.8% of frames**, and two thirds of the frame mass sits in runs longer than
three frames. A median over runs counts a one-frame flicker and a nine-second
freeze once each, which is precisely the frame-mass-versus-run-mass distinction
`rle.py` exists to handle, now appearing in the stop condition itself.

**This is recorded, not acted on.** The condition was registered on the median
and the median fired. Moving it to a frame-weighted statistic after seeing which
way that would go is tuning, and tuning is what this branch exists to avoid.
Amending it is a decision that needs its own registration.

**3. The mandatory second arm made it worse, and the mechanism is visible.**
`speed` was included because k-means finds density modes and the density is
single-mode. It backfires: **46-52% of frames** land in one-frame runs against
`plain`'s 16-24%, and 81-85% in runs of three frames or fewer. Stratum
membership is a hard cut on a noisy scalar, so a frame sitting near a quintile
edge changes stratum — and therefore its whole symbol block — on a speed wobble
far below any behavioural change. The stratification bought alphabet coverage in
the fast tail and paid for it in temporal stability, at a rate that makes the arm
unusable. Reported as measured; it was the right arm to run and it lost.

## Why the median is one frame — the diagnosis

Run only because the condition fired on every cell. Tune split, plain alphabet at
N = 256, three fits differing in exactly one thing each. Nothing here is carried
forward.

| row | channels | median | p75 | mean | runs |
|---|---:|---:|---:|---:|---:|
| `full/raw` | 17 | 1.0 | 3.0 | 3.84 | 1,076,629 |
| `pose/raw` | 14 | 1.0 | 3.0 | 4.17 | 992,556 |
| `full/wiener` | 17 | 2.0 | 3.0 | 5.18 | 798,861 |

**Dropping the three twist channels changes almost nothing.** The obvious
suspect was the velocity block — `v_x`, `v_y`, `omega` are frame-to-frame
differences and the noisiest thing in the representation. Removing them leaves
the median at 1 frame and moves the mean from 3.84 to 4.17. The flicker is in the
egocentric pose coordinates too, and a velocity-free tokenizer would not fix it.

**Wiener doubles the median run and lifts the mean 35%** — 1 to 2 frames,
3.84 to 5.18, with 26% fewer runs over the same frames. That is the number this
whole re-run existed to produce.

> A longer run on Wiener is **not** evidence that Wiener is better. A low-pass
> filter makes adjacent frames more similar; that is what it is for. It means
> any dwell measured on the standard pipeline is **partly the filter's own
> autocorrelation**, and a memory-depth result computed there would have
> inherited it.

This is the third appearance of one failure mode on this project — a property
attributed to the corpus that belongs to the filter — after the Wiener
recomputation and the 9-frame run length. Here it is quantified rather than
suspected. And note that **Wiener does not rescue the alphabet either**: at a
median of 2 frames it still trips the same floor.

## What this does and does not say

**It is not a failure of the egocentric transform.** Step 1R's A4 gate reads
R^2 = 1.000000 in closed form on this arm, the reversal audit passes 10 checks,
and the pose block's rank is 11 of 14 on every animal. The representation is
exact. The tokenization of it, at every granularity on this grid, is not stable
in time.

**It is not a claim that behaviour has no dwell structure.** Two thirds of the
frame mass on the best cell sits in runs longer than three frames, and p99 reaches
34 frames — 1.1 s. There is a dwell distribution; it sits underneath a large
one-frame mode that a nearest-centroid assignment creates at every cluster
boundary the trajectory brushes against.

**The grid was not widened.** N = 256 is the coarsest point registered, and
extending downward after seeing that every registered point failed would be
choosing a hyperparameter against an outcome. The pre-registered rule — every N
tripping the condition is a finding about the representation, not a reason to
widen the grid — is the rule that was followed.

## What is not built

`vieb/tok/{hazard,mdl,ladder}.py`, `scripts/ladder.py`, `jobs/ladder.slurm` and
`results/TOK_PREREGISTRATION.md` are **not written**. The ladder would be fitted
on a symbol stream that is retired, and the preregistration commits to a claim
about rung 2 beating rung 1 on exactly that stream. Registering a prediction
about an experiment that is blocked would make the registration ceremonial.

**k\* is not reported, and no value of it is implied.**

## Artifacts

`results/alphabet.json` (eight cells, every read), `work/tok/basis.json` (the
frozen SD and speed cuts), `work/tok/flicker.json` (the diagnosis),
`work/tok/abstain/` (298 shards), `work/tok/<arm>_N<N>/` (eight label sets).
Every one carries `inherited_digest` and `preprocessing_freeze: "F3"`.
