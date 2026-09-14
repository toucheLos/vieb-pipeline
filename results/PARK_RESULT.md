# Task D — the corrected park distribution

Registered in `PARK_REREGISTRATION.md`, committed before this ran. Four
thresholds, four arms, everything else unchanged. **Both distributions stand;
neither replaces the other.**

## The correction

`PARK_FRAMES = (4, 30)` uniform was drawn from runs *surviving* `median_0.50` —
by construction the runs the median cannot fix, since it deletes every run of 7
frames or fewer. Measured pre-filter, real park-like runs (continuity `step` >
`spike`, 532 runs over 40 seeded report recordings) have median **2** frames and
**87.6%** are ≤ 7, against **14.8%** under the legacy draw.

## Park repair, as predicted

| threshold | legacy | measured |
|---|---:|---:|
| 0.02 | 8.4% | **58.1%** |
| 0.05 | 7.8% | 57.9% |
| 0.10 | 8.8% | 57.9% |
| off | 8.8% | **57.2%** |

A 7× improvement. **Prediction 1 said "toward ~85%" and overshot** — 58% is where
it lands, because a park is *held* at one position while the animal moves, so
even a run entirely inside the window is not fully recoverable by a median.

## Net, old against new

| threshold | arm | net old | net new | Δ |
|---|---|---:|---:|---:|
| 0.02 | `median_0.50` | −0.0026 | **−0.0033** | −0.0008 |
| | `viterbi` | −0.0016 | −0.0019 | −0.0002 |
| | `disposition` | −0.0013 | −0.0014 | −0.0001 |
| off | `median_0.50` | −0.0000 | **−0.0007** | −0.0007 |
| | `viterbi` | −0.0016 | **−0.0019** | −0.0003 |
| | `disposition` | −0.0012 | −0.0015 | −0.0003 |

**Prediction 3 was a quantitative bet and I lost it in the safe direction.** I
predicted the improvement would be ≈ 0.0013; it is **0.0007–0.0008**, about half,
because park repair reached 58% rather than 85%.

**Prediction 4 held.** `viterbi` and `disposition` repair no parks at all, and
move only −0.0002 to −0.0003 through the shared denominator.

## The falsifier did not fire

Per-stratum at `off`, measured parks, slowest → fastest:

| arm | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| `median_0.50` | −0.0026 | −0.0013 | −0.0003 | **+0.0005** | **+0.0053** |
| `viterbi` | −0.0016 | −0.0018 | −0.0018 | −0.0023 | **−0.0030** |
| `disposition` | −0.0012 | −0.0013 | −0.0015 | −0.0017 | −0.0022 |

| | |
|---|---:|
| improvement required in s5 to reverse the crossing | **≥ 0.0093** |
| improvement measured in s5 | **0.0015** |

**The crossing survives**, at 1/6 of the magnitude needed to reverse it.
`median_0.50` is still positive — actively harmful — in the top two strata while
`viterbi` reaches its best there and `recovery_read` classifies the median as
**mixed by speed** at `off`: helps still frames, harms moving ones.

The registered partial-reversal case (s4 goes negative while s5 stays positive)
also did not occur: s4 is **+0.0005**, still positive. The crossing point moved
from between s2/s3 to between s3/s4 — **less than one stratum**.

## What this settles

The most serious known bias in the benchmark ran in `median_0.50`'s favour when
corrected, exactly as registered, and the conclusion did not move. That removes
the first objection a reviewer would raise.

It also sharpens the picture: with parks drawn correctly, **the median is the
best arm at the banked threshold** (−0.0033 pooled, better than `viterbi`'s
−0.0019) and still **harmful on fast frames**. Those are not in tension. The
median is a good teleport-and-short-park remover and a bad thing to run on an
animal that is moving, and which of those dominates is entirely a question of
what fraction of the corpus is moving.
