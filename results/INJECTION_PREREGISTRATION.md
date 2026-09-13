# Pre-registration — the injection benchmark

Written and committed **before the benchmark ran on any corpus data**.
Configuration fixed below; predictions falsifiable; the reason it exists first.

## Why this phase exists

Watching the Atlas comparison clips produced the reading *"median does look
solid"*. That reading is exactly what the measurements predict, and it is the
trap this phase is built to escape.

Median speed retained per keypoint, 12 tune recordings:

| keypoint | wiener | `median_0.50` | viterbi |
|---|---:|---:|---:|
| nose | 51.8% | **14.2%** | 98.4% |
| centre | 75.5% | **12.0%** | 100.0% |
| left_hip | 45.4% | **10.1%** | 98.2% |
| tail_base | 31.2% | **11.3%** | 99.1% |

`median_0.50` deletes **86–90% of the animal's median movement**. It looks solid
because a flattened mouse looks well-tracked. On a fear-conditioning corpus the
fast rare events are the signal, and an 88% cut in median speed is the
persistence prior that retracted the dwell result, arriving through a filter
instead of a labeller.

**Every axis used so far is blind to this.** Violation rate, distortion,
retention and held-out MDL are all computed without knowing where the keypoint
actually was, so none can distinguish a repair from a confident error, and none
can charge an arm for moving a *correct* point. MDL is worse than blind: a
flattened corpus has fewer, longer runs and therefore a shorter code, so Step 4
as currently specified would reward `median_0.50` **for having deleted the
behaviour**.

So: inject corruption with known truth, and measure both what the arm fixes and
what it breaks, in the same units.

## The configuration, fixed in advance

| | value | why this and not something tuned |
|---|---|---|
| pool | no bone violation at ε = 0.10, continuity spike ≤ 0.02 bl, no missing keypoint, min DLC confidence ≥ 0.60 | every instrument this repo has, at once. Measured: 23.4% of frames, 14.4% in runs ≥ 1 s |
| segment | ≥ 30 frames | shorter and a temporal arm has no context, so the comparison measures edge effects |
| teleport magnitude | Phase D's measured suspect-displacement quantiles (median 22.15 px, p90 92.23, max 355.55), inverse-CDF interpolated | not a parametric fit, and not Gaussian |
| corrupted keypoint | weighted by `CLEANING.md`'s per-keypoint Viterbi reassignment (nose 0.5315%, centre 0.0224%) | uniform corruption would test a corpus that does not exist |
| teleport | 1–3 frames, tracks the animal at constant offset | Phase D's envelope |
| park | 4–30 frames, **holds one position** | `CLEANING.md`: runs surviving `median_0.50` have median 9 frames, 96.1% over 3 |
| swap | 1–8 frames, bilateral pairs only | `recur/qc/swap.py`, `MAX_RUN_S = 0.25`, median 3 frames |
| dropout | NaN, not a sentinel | a `(0,0)` sentinel reads downstream as a teleport to the origin |
| rates | teleport 1.2%, park 0.4%, swap 0.3%, dropout 3.4% of pool keypoint-frames | matched to corpus-wide measured incidence |
| arms | `raw`, `wiener`, `median_0.50`, `viterbi`, `disposition` | the bakeoff's cheap branch plus both smoothers, so nothing is ruled out by assertion |
| units | body lengths, `ego.ell_a`, per animal | never per frame |
| bootstrap | animals, 2000 replicates | frame-level resampling gave intervals ~19× too narrow |
| dominance | **non-overlapping** animal-bootstrap intervals | a point comparison crowned an arm on 0.029 px once already |
| seed | 0 | realised rates recorded in the result |

### The three numbers

* **repair** — mean error against truth on **corrupted** keypoint-frames.
* **damage** — mean error against truth on **uncorrupted** keypoint-frames.
  Zero is achievable: a perfect arm leaves correct data alone.
* **net** — total error against truth over *every* pool keypoint-frame, minus the
  same for `raw`. **Negative means the arm left the data closer to the truth than
  it found it.** This is the figure an arm cannot win by flattening.

## Predictions

1. **`raw` scores damage exactly 0 and repair exactly the injected magnitude.**
   Mechanical, and registered so the harness is checkable rather than trusted. If
   it fails, nothing else in the table means anything.
2. **`viterbi` has the lowest damage of any arm that moves anything** — under
   **0.01 body lengths**.
3. **`median_0.50`'s net is positive**: its damage exceeds its repair and it
   leaves the data further from the truth than it found it. **Falsifier: if
   `median_0.50`'s net is negative, dropping the smoothers from the Step 4 MDL
   branch is wrong and that decision is reversed.**
4. **`viterbi` repairs teleports and not parks.** It selects a path, and a park is
   a consistent path.
5. **No arm repairs sustained parks** — fraction of park error removed under 25%
   for every arm. This is Phase E's failed prediction 2 rebuilt on an instrument
   that can actually falsify it: a park here has a known true position, so a
   filter smoothing *across* it scores no repair, where the smoothness statistic
   could not tell those apart.
6. **`disposition` has damage under 0.002 body lengths** — it touches 0.5% of
   frames — and repairs teleports inside its ≤3-frame envelope only.

## What this phase may not do

* It may not re-score Q1.
* It may not restate the bakeoff verdict in `results/cleaning.json`.
* It may not modify Phase D or widen its suspect channel.
* It may not report a pooled `net` when the speed strata disagree in sign. The
  pool is biased towards slow behaviour and an arm that helps still frames while
  harming moving ones must be reported as such, not averaged into a PASS.

## The limitation that most threatens the conclusion

**The pool is selected for cleanliness, so it is biased towards slow and still
behaviour** — a continuity spike is what a fast keypoint produces, so fast frames
are preferentially excluded. Damage measured here therefore **understates** what
a smoother does to fast movement, which is the exact quantity in dispute.

This runs *against* the conclusion the phase is likely to reach, so it is a
conservative bias rather than a flattering one. It is handled by stratifying
every result on segment speed and by refusing a pooled verdict when the strata
disagree, not by claiming it does not matter.

It is a *pseudo*-ground truth. No human labelled any of it, and nothing written
from it may say otherwise.
