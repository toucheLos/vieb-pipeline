# STABILISE 3: SAM's mask drifts to the arena wall, and gate 0 catches it

Registered in `STABILISE3_PREREGISTRATION.md` (`c027a6b`), with bars from the
incumbent-only calibration (`6857460`). `vieb/pixel/sam.py`,
`scripts/stabilise3.py`, `jobs/stabilise3.slurm` (12 A100 shards, all
`COMPLETED`, all 300 recordings). The SAM checkpoint's SHA-256 was checked
against the registered value in every shard. §7 holds: K reproduces the
incumbent with **max |diff| 0.0**.

**Headline.** SB and SP are **`NOT_A_RESULT`**: **144 of 300 recordings
(48%)** are refused, against the 20% limit. On top of that, **gate 0 fails for
every arm**. The head disc lies inside SAM's mask on only **66.9% [58.5,
75.0]** of accepted keyframes, against a registered lower bound of 90%. **No
arm is eligible.**

## 1. The failure, seen directly

`results/stabilise3/drift_20251117_Box_3_Day_3_A_975.png`: SAM's mask (red),
its prompt box (yellow), DLC keypoints (green) and the head-disc centre, on six
keyframes of the recording with the worst on-animal share. At *t* = 0 the
keypoint seed yields the mouse. **By *t* = 30, SAM has moved to the bright top
wall of the arena, with a predicted IoU of 0.94–0.96**, and it stays there for
the rest of the recording.

The mechanism is §1's propagation. Each prompt is the last accepted mask's box
padded by 0.25 body lengths. Once that box touches a high-contrast arena
structure, SAM cuts the more salient object, confidently. From then on the
design contains nothing that returns it to the animal. **Neither refusal rule
can see this.** Predicted IoU is high, and the area rule is relative to the
recording's own median, which is now the wall's. Gate 0 exists for exactly this
lower tail (D24), and it is where the failure shows.

**It is common, not an edge case.** Across recordings the on-animal share has a
median of 0.839, but a 10th percentile of **0.024**. Four recordings score
exactly 0 with almost no frames refused. That is SAM stably tracking something
else.

## 2. The gates

| gate | K | SB | SP |
|---|---|---|---|
| 0. disc on the animal (≥ 90%) | **FAIL** 63.9% | **FAIL** 66.9% | **FAIL** 66.9% |
| 1. duplicate frames exactly zero | **FAIL** 100% non-zero | **FAIL** 97.4% non-zero | **PASS** 0 of 15,444 |
| 2. jitter decoupling (< 0.10, below K) | reference +0.340 [+0.285, +0.392] | FAIL +0.095 [+0.039, +0.148] | FAIL +0.255 [+0.208, +0.299] |
| 3. identity ratio in [0.80, 1.25] | FAIL 3.54 [2.36, 5.02] | `NOT_A_RESULT` (14 animals) | `NOT_A_RESULT` (14 animals) |
| 4. oracle ratio in [0.80, 1.25] | FAIL 3.93 [3.14, 4.87] | FAIL 17.4 [15.0, 19.9] | FAIL 16.4 [13.6, 19.5] |
| **arm** | **FAIL** | **`NOT_A_RESULT`** (48% refused) | **`NOT_A_RESULT`** (48% refused) |

**What the gates add, read with §1 in mind:**

* **Gates 3 and 4 behaved as calibrated on the incumbent.** K scored 3.54×
  and 3.93×, in line with the calibration's 4.43× and 3.09×. Both now measure
  what they were built to measure.
* **SB and SP's gate 4 (16–17×) is the drift.** The plant's SAM track locks
  onto non-animal structure, so the registration follows the static floor while
  the pasted animal moves. It says nothing about ECC or the moment warp on a
  correct mask.
* **SB fails gate 1 by construction of §1.** Between keyframes its pose is
  *interpolated*, so on a byte-identical pair t−1 → t the interpolated warp
  still moves. A registration that interpolates pose is not image-driven per
  frame. SP, which registers by ECC on each pair, passes gate 1 exactly.
* **SP's gate 2 (+0.255) cannot be read as residual pose leakage.** SP's
  registration reads no keypoint, and on drifted recordings its disc is off
  the animal. There is also a problem with the gate's own premise, recorded as
  `DEVIATIONS.md` D25: skull bone-length variance is a noise proxy only if it
  does not rise when the head *really* moves. Motion blur and pose change make
  that unlikely. If DLC jitters more on moving heads, gate 2's < 0.10 bar
  penalises honest registrations too. It also means part of D21's +0.276 and
  STABILISE's +0.340 could be genuine co-movement. This stage cannot separate
  the two.
* **Gate 5:** every arm, K included, detects the planted local motion at the
  null rate (about 5%) at every amplitude up to 4 px. On the moving plants, no
  registration here sees a 0.15 bl patch oscillating inside a 0.6 bl disc.

## 3. What this licenses

**License:** SAM ViT-B, prompted by its own last mask's padded box, **drifts
off the mouse onto arena structure** on a substantial share of this corpus, with
high predicted confidence. Neither a predicted-IoU floor nor a self-relative
area rule detects it.

**License:** calibrated on the incumbent, the rebuilt gates 3 and 4 reproduce
the incumbent's calibration on the full run.

**Do not license** any claim about SAM masks on a correctly located animal, or
about ECC or moment registration on such masks. Every candidate arm was refused
or carried the drift.

## 4. What is owed

* **Prompt every keyframe from the located animal, not from the last mask.**
  Using the keypoint box, padded, as the prompt at *every* keyframe is §0's
  *locate*: a few pixels of jitter in a box does not move a boundary SAM draws
  from image edges. Gate 1 and gate 2 would measure whether that holds.
* **Drop interpolated-pose registration (SB).** Register per pair by ECC (SP),
  or segment every frame.
* **Replace gate 2's premise** with a jitter proxy that is independent of real
  head motion, or with a planted-jitter test: add known jitter to K's poses and
  measure the arm's energy response.
* Each of these is a new registration, not an amendment of this one.
