# STABILISE 5: masked ECC follows moving animals, and breaks on still ones

Registered in `STABILISE5_PREREGISTRATION.md` (`79b055a`), amended alone
before any real frame (Amendment 1, D26, `13912b3`). `scripts/stabilise5.py`,
`jobs/stabilise5.slurm` (12 A100 shards, all `COMPLETED`, all 300
recordings). The checkpoint hash was verified in every shard. §7 holds:
**max |diff| 0.0**. This is the first result JSON whose `env.git_sha` is this
repository's own commit (`vieb.provenance`).

**Headline. SM: `FAIL`. SP: `FAIL`.** They fail on **opposite** gates.
Restricting ECC to the animal's eroded masks makes it follow a moving body:
gate 4 falls from SP's 17× to **1.31×**, and gate 6's bracket contains 1. But
it breaks the still-frame gates SP passed. **Neither arm is eligible.**

| gate | SP (unmasked ECC) | SM (masked ECC, Amendment 1) |
|---|---|---|
| 0. disc on the animal, ≥ 90% | PASS 93.7% [92.3, 95.0] | PASS 93.7% [92.3, 95.0] |
| 1. duplicate frames exactly zero | PASS 0 of 22,950 | **FAIL** 143 of 22,932 (0.62%) |
| 2. planted jitter, in [0.90, 1.10] | PASS 1.000 [0.999, 1.000] | PASS 1.001 [1.000, 1.002] |
| 3. still frames vs identity, in [0.80, 1.25] | PASS 1.038 [1.016, 1.063] | **FAIL** 2.286 [1.854, 2.779] |
| 4. moving plant vs oracle, in [0.80, 1.25] | FAIL 17.05 [15.34, 18.74] | FAIL **1.305 [1.266, 1.344]** |
| 6. keypoint agreement f, in [0.90, 1.10] | FAIL 0.816 [0.773, 0.859] | FAIL 1.271 [1.245, 1.298] |
| gate 6 bracket, **not gated** (D27): β … f | **0.136** … 0.816 | **0.825** … 1.271 |
| refused recordings | 56 / 300 | 56 / 300 |

K, the incumbent: gate 1 FAIL (100%), gate 2 positive control 11.79×
[8.93, 15.14], gate 3 5.22×, gate 4 3.93×. Gate 6 is not scored for K,
because its warp is built from the keypoints.

## 1. Gate 6, the real-motion test, read with its bracket

Gate 6's registered f is **inflated by the arm's own noise** (D27, found on the
smoke run), so both arms' f values sit above their true recovery. The ordinary
slope β is attenuated by keypoint noise instead. Under noise in both, the true
fraction of body motion recovered lies **between β and f**. The bracket was
added after the smoke run and is labelled as ungated everywhere.

* **SP: between 0.14 and 0.82.** Its lower end is far below 1. On real fast
  frames SP recovers only a small part of the body's motion, consistent with
  gate 4's 17× and with STABILISE 4's image evidence.
* **SM: between 0.83 and 1.27.** The bracket contains 1. On real video, masked
  ECC moves with the body. It does so noisily, which is why f overshoots and
  the registered gate fails on its upper edge.

**This is the investigator's proposal working as intended.** The keypoints are
too noisy to see fine head motion, but they are a sound reference for where the
body went. Gate 6 tells SP and SM apart on real video, which the planted gate 4
could only do on synthetic pastes.

## 2. Why SM fails on still frames

Gates 1 and 3 test the still animal. SP passes both exactly, and SM fails both.
On a byte-identical pair SM should return the identity. On 0.62% of pairs it
does not, and on genuinely immobile frames it leaves **2.3×** the identity's
difference. **The likely mechanism is SM's initial warp**, SAM's interpolated
centroid shift (§1). SAM's mask centroid wobbles slightly between keyframes
even when the animal is still. ECC over an eroded interior, with few pixels
and little texture, does not reliably pull that offset back to zero. SAM's own
jitter then enters SM the way keypoint jitter entered K. This mechanism is
inferred from which gates fail, and was not separately measured.

## 3. What this licenses

**License:** ECC restricted to SAM's eroded masks on both frames follows a
moving animal. On the planted trajectories it comes within **1.31×** of a
perfect registration (SP: 17×; the jittered incumbent: 3.9×). On real fast
frames its recovered fraction of keypoint-measured body motion is bracketed
between 0.83 and 1.27.

**License:** the two arms fail on opposite gates. **SP is exact on still
frames and does not follow motion. SM follows motion and is not exact on still
frames.** Neither keypoint-free alignment here passes both.

**Do not license** SM for still-body analyses, which is where a grooming
question would live. It is 2.3× worse than no alignment on immobile frames.

## 4. What is owed

The design that follows from §2 is an SM whose initial warp is the **identity**,
not SAM's centroid shift, so a still animal starts at the right answer. It
could also select per pair between SP's and SM's fits by ECC's own correlation.
Either is a new registration: gates 1–4 and 6 as here, with gate 6's bracket
promoted to the registered statistic (D27). No grooming question follows from
any of it until such an arm passes, and even then only for precision: there is
no grooming ground truth (`REVIEW.md`).
