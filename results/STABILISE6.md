# STABILISE 6: the identity start fixes duplicate frames; the still-frame gate measures the floor

Registered in `STABILISE6_PREREGISTRATION.md` (`90460e7`), committed alone
before the new arm aligned any real frame. `scripts/stabilise6.py`,
`jobs/stabilise6.slurm` (12 A100 shards, all `COMPLETED`, all 300
recordings). §7 holds: **max |diff| 0.0**.

**Headline. SM6: `FAIL`**, on gates 3 and 4. It passes gates 0, 1, 2 and 6:
it is exact on byte-identical frames, immune to planted jitter, and follows
the body on real video. **Gate 3's failure looks like a property of the gate's
region more than of the alignment** (§2), but that is a diagnostic reading.
No arm is eligible.

| gate | SP | SM (STABILISE 5) | **SM6** |
|---|---|---|---|
| 0. disc on the animal, ≥ 90% | PASS 93.7% | PASS 93.7% | **PASS** 93.7% [92.3, 95.0] |
| 1. duplicate frames exactly zero | PASS 0 / 22,950 | FAIL 143 / 22,932 | **PASS 0 / 22,932** |
| 2. planted jitter, in [0.90, 1.10] | PASS 1.000 | PASS 1.001 | **PASS** 1.001 [1.000, 1.002] |
| 3. still frames vs identity, in [0.80, 1.25] | PASS 1.038 | FAIL 2.286 | **FAIL** 2.279 [1.844, 2.773] |
| 4. moving plant vs oracle, in [0.80, 1.25] | FAIL 17.05 | FAIL 1.305 | **FAIL** 1.305 [1.266, 1.344] |
| 6. chained keypoint agreement, β and f in [0.90, 1.10] | FAIL β 0.166, f 0.562 | PASS β 0.969, f 1.084 | **PASS β 0.952 [0.924, 0.973], f 1.067 [1.056, 1.078]** |

## 1. What the identity start fixed, and what it did not

**Duplicate frames: fixed.** Starting from the identity, SM6 returns exactly
zero on every one of 22,932 byte-identical pairs, where SM left 143 non-zero.

**The chained keypoint gate passes for both masked arms.** Over 10-frame chains
on the fastest quarter of real frames, SM6's body motion against the
keypoints' is bracketed between 0.95 and 1.07. SP's sits between 0.17 and 0.56.
The masked alignment follows the body; the unmasked one does not. The
investigator's proposal, using the keypoints as the referee for bulk motion, is
now a working registered gate.

**The still-frame gate did not move** (2.279, against SM's 2.286). **So the
starting warp was not the cause.** `STABILISE5.md` §2 inferred that it was, and
carries a correction.

## 2. Why gate 3 fails: the disc reaches the floor (diagnostic, not gated)

Gate 3 compares in-disc |Δ| on immobile frames against the identity's, in a
**0.6 body-length disc**. That disc reaches well past the body onto the
high-contrast bar floor. Two measurements from this run's saved data:

| | disc 0.4 bl | disc 0.6 bl | disc 0.8 bl |
|---|---|---|---|
| SM6, gate-3 ratio | 1.748 | 2.279 | 2.835 |
| SP, gate-3 ratio | 1.019 | 1.038 | 1.065 |

| on immobile frames | median | 90th percentile |
|---|---|---|
| SM6, implied body shift | 0.072 px | 0.289 px |
| SP, implied body shift | 0.013 px | 0.059 px |

**SM6's ratio grows with the disc, and SP's stays flat.** SM6 fits the
animal's small real or noise motion on immobile frames, a few hundredths of a
pixel, and applies it to the whole frame. Against bars of 20 and 230 grey
levels that moves the floor visibly, and a larger disc holds more floor. SP
follows the floor, so the floor stays registered. **This reading is not
complete.** Even the 0.4 bl disc reaches the floor, so this data cannot exclude
a residual problem on the animal itself.

**Gate 4** is scored at a 0.2 bl disc inside the body, and SM6 is 30% above a
perfect registration there. That is a real, modest shortfall on moving
animals: the interpolation and the eroded-mask fit leave some residual.

## 3. What this licenses

**License:** SM6 is exact on byte-identical frames, keypoint jitter does not
reach it (1.001×), and over 10-frame chains on real video it follows the body
(0.95–1.07 against the keypoints).

**License:** gate 3's failure grows with the share of floor in its disc, and
SM6 applies sub-pixel animal motion to the whole frame on immobile frames.

**Do not license** SM6 as passing. Gates 3 and 4 fail as registered.

## 4. What is owed

The grooming question concerns **pixels on the animal**. A disc that includes
floor measures the alignment's effect on the floor too. The next registration
should score gates 3–5 over **SAM's eroded mask intersected with the disc**,
the animal's own pixels. It would then either confirm the §2 reading (SM6's
ratio near 1 on animal pixels) or refute it. It must be calibrated first, as
STABILISE 3 was, on the incumbent alone.
