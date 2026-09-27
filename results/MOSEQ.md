# MOSEQ — Keypoint-MoSeq run as a test, with the answer fixed in advance

Registration: `MOSEQ_PREREGISTRATION.md` with Amendments 1–2 (D30, D31), all
committed before any registered fit finished. Result: `results/moseq.json`.
Input: the repaired pose before `median_0.25`. Fitted on the same 100 tune
recordings for every arm and scored on the report split (89 animals).

## Verdict: FAIL, on seed stability

| gate | statistic | result | bar | |
|---|---|---|---|---|
| 1. held-out gain | gain_real − max(gain_white, gain_phase), seed 0 | **2.63 [2.37, 2.91]** nats/frame | lower > 0.05 | PASS |
| 2. seed stability | median AMI, 10 pairs of real seeds | **0.473** (range 0.432–0.498) | ≥ 0.70 | **FAIL** |

- 93.0% of report frames were selected for scoring in every arm; none had a
  non-finite latent or log-likelihood (refusal rule as amended).
- **MoSeq's switching captures something real, but it does not reproduce which
  syllables that something is carved into.**

### Gate 1: the mice gain more than noise does

Gain over a single-regime AR(3), per frame, on MoSeq's own inferred latents
(Amendment 2):

| arm | seed 0 | seeds 0–4 / 0–2 |
|---|---|---|
| mice | 5.32 [4.67, 6.24] | 5.17–5.38 |
| phase-randomised | 2.69 [2.12, 3.51] | 2.69–2.74 |
| white noise | 1.63 [1.40, 1.87] | 1.63–1.64 |

**The switching prior buys 1.6 nats/frame on white noise and 2.7 on a
linear-Gaussian process.** Both are data with no regimes, so these gains are
the inference favouring its own model, as registered in advance. The mice gain
about twice as much. The pose therefore has structure a single linear AR
cannot express and a switching one can. This gate does not say what that
structure is: a coarse still/moving change in noise level would pass it.

### Gate 2: the syllables do not reproduce across seeds

| arm | median pairwise AMI | pairs |
|---|---|---|
| mice | **0.473** | 10 |
| white noise | 0.641 | 3 |
| phase-randomised | 0.556 | 3 |

- The mice's syllable partitions agree **less** between seeds than white
  noise's do.
- The 0.473 median is the level at which the VIEB bottleneck codes were
  disqualified (0.432).
- Same data and same fit set, differing only in the Gibbs seed: the frames are
  carved differently each time.

## White noise beside the mice (§5, not gated)

| arm (seeds) | syllables in use (> 0.5%) | used at all | median dwell (frames) | dwell var/mean² | top-10 share |
|---|---|---|---|---|---|
| mice (5) | 13–16 | 39–44 | 18–19 | 7.8–8.5 | 90–93% |
| white noise (3) | **24–30** | 37–45 | 21–25 | 7.4–8.0 | 66–72% |
| phase (3) | 27–31 | 52–57 | 39–43 | 5.6–5.9 | 68–70% |

**White noise yields more syllables in use than the mice, with dwells of the
same order.** Its dwell distribution is just as far from geometric
(var/mean² ≈ 8, against 1 for a geometric distribution). A syllable count or
a dwell histogram cannot tell mice from noise here, as the audit found.

## Context beyond freezing (§7): CONFOUNDED, and descriptive only

Pixel freezing: FreezeThresh 800, MinDuration 15, on 882 report recordings
(89 animals, days 3–7, A/B).

| features | balanced accuracy, context A vs B |
|---|---|
| F: freeze fraction | 0.669 [0.634, 0.703] |
| S: syllable occupancy, non-freezing frames | 0.749 [0.717, 0.779] |
| F + S | 0.765 [0.734, 0.797] |
| gain of F + S over F | **+0.097 [+0.063, +0.132]** |
| **leak guard**: S on freezing frames only | **0.710 [0.677, 0.744]** > 0.60 |

- Syllables add to freezing, but they decode the arena just as well on
  freezing frames, where behaviour is assumed equivalent. **The registered
  reading is CONFOUNDED.** Because MOSEQ did not PASS, even that is
  descriptive.
- **Caveat on the guard itself:** at the registered threshold, **63.8%** of
  frames score as freezing. That is far more than an immobility measure
  should. So the "freezing" frames still hold some movement, and the guard's
  premise (identical behaviour across contexts) is weaker than intended. Part
  of the 0.71 may be behaviour, not arena. **This does not change the
  registered reading.** It says the freeze threshold, transplanted by area
  scaling, is itself unvalidated here (`PIXEL_PILOT.md` already left its
  magnitude unquantified).

## What this licenses

- **No syllable is licensed as a behaviour token.** They do not reproduce
  across seeds.
- **The pose has non-linear, regime-like structure beyond both surrogates**
  (gate 1). This is the first positive, controlled result for discrete-ish
  dynamics in this programme. It agrees with STATES' post-hoc finding that the
  corpus beats white noise at short lags. The structure exists; no method
  tried so far carves it the same way twice.
