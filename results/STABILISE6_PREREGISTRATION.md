# Registration — STABILISE 6: masked ECC from the better of two starts, and a chained keypoint gate

Committed **alone, before the new arm has aligned any real frame**. Inherited
digest **`198eb14ff258c7f6`**.

`STABILISE5.md`: SM (ECC on SAM's eroded masks, both frames) follows moving
animals. Moving plant 1.31× perfect; keypoint-agreement bracket 0.83–1.27. But
it fails the still-frame gates SP passes: 0.62% of duplicate pairs non-zero,
and 2.29× the identity on immobile frames. The inferred cause is SM's starting
warp, SAM's interpolated centroid shift, which wobbles even on a still animal.
This stage changes that start, and replaces gate 6's lag-1 statistic, which
D27 showed is inflated by arm noise. **Everything else is
`STABILISE5_PREREGISTRATION.md` with its Amendment 1, unchanged:** the SAM
track and every refusal, the eroded masks on both frames through
`findTransformECCWithMask`, gates 0–5 with their bars, the planted-jitter gate
2 with K as control, the sample and the prohibitions.

---

## 1. The new arm: SM6, masked ECC from the better of two starts

For each pair, masked ECC runs **twice**, from the **identity** and from
**SAM's interpolated centroid shift**. The fit with the **higher ECC
correlation**, the image fit's own objective, is kept. Ties go to the
identity. Both fits use STABILISE 5's masks, criteria and pre-blur, and no
keypoint enters either the fits or the choice.

**Why both starts, from evidence gathered before this registration**
(synthetic, plus descriptive statistics of STABILISE 5's saved keypoint data,
which is not an SM6 result):

* On the bar-floor synthetic scene, the identity start recovers the full
  motion up to 2.5 px/frame. It fails to converge on some pairs at 5 px/frame
  and on all at 8 px and above. The centroid start converges at every speed
  tested (0–12 px/frame). The better-of-two rule recovers 1.00 at every speed
  and exactly 0 when still.
* On real video, STABILISE 5's fastest quarter of frame pairs moves a median
  1.3 px/frame (90th percentile 4.7, 99th 14.5). **8.8% exceed 5 px.** An
  identity start alone would lose the fastest pairs, and a centroid start alone
  is what failed the still gates.

**SM** (STABILISE 5's centroid-start arm) and **SP** are carried beside SM6,
so the change is measured against both. K is the incumbent, with §7's exact
check.

## 2. Gate 6, chained over ten frames

Lag-1 gate 6 is inflated by the arm's noise (D27). **Composing the arm's warps
over k consecutive pairs** and comparing with the keypoint displacement over
the same k frames makes the motion signal grow about k-fold while both noises
grow only about √k. So the attenuated slope β and the inflated ratio f close
toward the true scale from either side.

* **Chains:** non-overlapping runs of **k = 10** consecutive pairs, all with a
  valid (non-refused) warp. `CENTER` must be measured, not interpolated, at
  both ends, and the chain's mean keypoint speed must be above the recording's
  75th percentile.
* **Displacements:** `d_arm = W_chain·c − c`, with `c = CENTER(t−10)` and
  `W_chain` the composed warp, and `d_kp = CENTER(t) − CENTER(t−10)`.
* **Statistics, per animal, pooled over its chains:**
  `β = Σ d_kp·d_arm / Σ|d_kp|²` and `f = Σ|d_arm|² / Σ d_kp·d_arm`.
* **PASS iff both** animal-level intervals (2,000 replicates, seed 0) lie
  within **[0.90, 1.10]**, over ≥ 20 animals with ≥ 20 chains each.

**Disclosed calibration of the statistic** (on STABILISE 5's saved warps; the
arms there are already published, and SM6 has not run). Animal-mean brackets:

| arm (STABILISE 5) | k = 1 | k = 5 | k = 10 |
|---|---|---|---|
| SM | 0.786 … 1.290 | 0.924 … 1.116 | **0.958 … 1.084** |
| SP | 0.131 … 0.798 | 0.164 … 0.588 | 0.165 … 0.560 |

At k = 10 the bracket closes around 1 for an arm that follows the body, and
stays far from it for one that does not. That is the property a gate needs.
Choosing k from this table is recorded as a calibration of the statistic
against an already-published arm. **It is not a result about SM6.**

K is not scored on gate 6 (circular). SP and SM are reported.

## 3. Verdicts

**SM6 is PASS iff gates 0–4 and 6 all pass**; FAIL if any fails;
`NOT_A_RESULT` under the refusals. A PASS makes SM6 eligible for a separately
registered grooming stage, which, with no grooming ground truth (`REVIEW.md`),
could only ever be scored for precision. **Nothing here is read as grooming**,
and nothing as shock versus no-shock.

## 4. Incumbents (M12)

| quantity | incumbent | source |
|---|---|---|
| SM, gates 1 / 3 | FAIL 0.62% / FAIL 2.286 [1.854, 2.779] | `STABILISE5.md` |
| SM, gate 4 | 1.305 [1.266, 1.344] | `STABILISE5.md` |
| SP, gates 1 / 3 / 4 | PASS / PASS 1.038 / FAIL 17.05 | `STABILISE5.md` |
| lag-1 gate 6 brackets | SM 0.83–1.27, SP 0.14–0.82 | `STABILISE5.md` |
