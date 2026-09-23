# The arena's own noise floor, measured from arena pixels

Registered in `PIXEL_PREREGISTRATION.md` §3 (+ Amendment 1). Produced by
`vieb/pixel/motion.py` and `scripts/pixel_pilot.py`, inherited digest
**`198eb14ff258c7f6`**.

**Named distinctly on purpose.** `NOISEFLOOR.md` is the **DLC keypoint jitter**
floor, in body lengths, synthesised from skull-bone residuals. This is **sensor
read noise, H.264 quantisation and lighting flicker**, in grey levels. Different
physical quantities. They are never compared and never share a document.

---

## 1. The measurement

ezTrack derives its motion cutoff from an **animal-free** calibration video
(`Calibrate`, line 1077): sample pixels, take `percentile(|Δ|, 99.99)`, and set
`mt_cutoff = 2 ×` that. **We have no animal-free video**, so the arena supplies
the calibration from the video we do have — the animal is masked out by its
keypoint bounding box dilated by one body length, a pixel counts as arena only
if it is outside the animal in **both** frames of the pair, and a frame with
fewer than 3 locatable keypoints contributes **no** arena pixels.

**Sample size is not the limitation.** ezTrack's own calibration uses 10,000
pixels × 300 frames ≈ 3 × 10⁶ samples. Each recording here contributes
**1.0–1.5 × 10⁹** arena pixel-pairs, roughly 400× more, and **0 of 1,440
recordings** fell below the registered 200,000 minimum.

## 2. A single global cutoff is not defensible on this corpus

| | registered pilot (300) | census (1,440) |
|---|---|---|
| `mt_cutoff` span | **6.50 – 116.50** | **6.50 – 128.50** |
| Context A, median | 13.50 | **12.75** |
| Context B, median | 9.50 | **9.50** |
| Context A, mean (sd) | 16.09 (7.48) | 16.67 (12.54) |
| Context B, mean (sd) | 15.23 (16.83) | 14.08 (16.31) |
| **ezTrack's published default** | **10.0** | **10.0** |

**The derived cutoff spans a factor of 20**, and ezTrack's single global default
of 10 sits near the *bottom* of that range. For the median Context A recording
the default is ~25% too permissive; for the noisiest it is **13× too
permissive**, and at that cutoff sensor noise alone would be counted as motion.
**This is the registered reason for deriving the cutoff per recording, and it is
now a measurement rather than an argument.**

**Report the median, not the mean.** The distribution is heavy-tailed — 9 of the
first 324 recordings sit above 20 while the bulk sits near 5 — and the outliers
are Context B while the bulk difference runs the other way. On the pilot the
mean gap (0.86) and the median gap (4.00) point to different conclusions. Every
gap quoted here is a median.

## 3. The floor differs by context, and the pilot could not see it

§7 predicted this: Context A and Context B are **visually different arenas**
(A frames cost 32% more bits while keypoint speed is 24% lower — opposite
directions, so motion cannot explain it). Paired within (animal, day):

| | B − A, grey levels | pairs | animals |
|---|---|---|---|
| registered pilot (300 recordings) | **−0.430 [−1.927, +1.366]** | 150 | 30 |
| **census (1,440 recordings)** | **−1.296 [−2.020, −0.505]** | **720** | **144** |

**Context A's arena is noisier**, by about 1.3 grey levels on the floor and
therefore ~2.6 on the derived cutoff. The sign agrees with the bits-per-frame
result: more spatial detail → more high-frequency content → more compression
noise in static regions.

> **The registered 300-recording pilot was underpowered for the one question it
> existed to interpret.** Its interval spans zero and would have licensed "the
> A/B image difference does not reach the sensor-noise channel". The census
> says it does. Both are printed, in that order, in `pixel_pilot.json`. The
> pilot size was chosen for balance and for §7's *interpretive* caution, and
> neither of those is a power calculation — that is the lesson, and it is the
> same one **D16** records one level up.

**This is §7's confound, made numerical, on exactly the axis Q2 uses.** It does
not invalidate a cross-context pixel result by itself — **D18** replaces §7's
unit-broken refusal with a correction that can actually be applied — but it must
be quoted beside every one.

## 4. Provenance

| number | source |
|---|---|
| the algorithm, `SIGMA = 1`, `mt_cutoff = 2 × p99.99` | `github.com/DeniseCaiLab/ezTrack` `master`, `FreezeAnalysis/FreezeAnalysis_Functions.py:196,1077`, read from source. **ezTrack is not installed** |
| default `mt_cutoff = 10` | the three shipped notebooks, verified |
| arena mask | `spine.clean(rid)["pose"]`, keypoint bbox dilated by 1.0 × `ego.body_length` median |
| 1,440 recordings | every `fit` animal holding all ten (context, day) cells on days 3–7 — a census, not a draw |
| 300-recording pilot | `PIXEL_PREREGISTRATION.md` Amendment 1, seed 0 |
