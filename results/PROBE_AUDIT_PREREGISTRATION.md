# Pre-registration — is the planted instance findable at all?

Registered before `vieb/seg/probe_audit.py` or `scripts/probe_audit.py` exist.
Source array `work/ego/raw__bodylen__*.npz`, the F3 **`raw`** arm. Digest
`198eb14ff258c7f6`.

## 1. The question, and why it is not the question Step B asked

Step B asked *can the detector isolate a planted instance* and answered **12.8%**
against a registered 20% gate, with `merged` 0.0% and `split` 0.0% in every cell
— so neither the refractory period nor the threshold is at fault, and
`DETECTOR.md:12, 44` records `no_edge` at **66.2%** in the best cell and
`DETECTOR.md:24` at **73.8%** in the diagnose cell. The detector produces no
boundary at either edge of the thing it is supposed to find.

`DETECTOR.md:83-86` already names the suspicion: the template is the mean of 40
real windows — very smooth — crossfaded into the signal over 3 frames. **It may
have no acceleration discontinuity to find.**

This stage asks the prior question. **Not** "does the detector find the plant",
but **"is there anything at the plant's edges for an acceleration criterion to
respond to?"** That is a property of the probe, and it is measurable without a
detector, without a threshold, and without a human.

It matters because the alternative to measuring it is tuning against the probe,
and this programme has a ledger entry for exactly that failure: the separability
probe that certified white noise as inseparable from a mouse and would have
licensed an entire gate. A criterion tuned until it finds crossfades is a
crossfade detector.

## 2. The measurement

The criterion's own scalar, unmodified: `breaks.discontinuity(x, h)`
(`vieb/seg/breaks.py:131`),

```
D(t) = ‖ẍ⁺(t) − ẍ⁻(t)‖
```

the gap between one-sided least-squares **quadratic** second derivatives. Note
`DEGREE = 3` does not enter `D` — degree drives only `fit_quality` and
`min_segment_frames` — so this stage sweeps nothing over degree.

Computed at the **primary** setting only: `deriv_sec = 0.133`, `h = frames(0.133,
fps)`, group `shape`, planted duration **0.5 s**, occupancy **0.1**, planted into
`microstate`, `EDGE_TOL = 2` — every one of them inherited from
`detector_sweep.py`, none of them chosen here. Twelve animals, as Step B used.

`D` is computed per recording span and **never across a recording boundary**.

## 3. The three populations, and the one number compared

Within each recording, `D` has its own scale, so a raw magnitude is not
comparable between recordings. Each frame is therefore scored by its
**percentile within that recording's own `D` distribution** — unit-free, bounded,
and directly interpretable.

| population | frames |
|---|---|
| **planted** | the `2 · EDGE_TOL + 1` frames around each planted instance's start and stop |
| **fired** | the frames the frozen detector actually selected as peaks, same recording, same setting |
| **random** | frames drawn uniformly from the same recording's selectable span, **count-matched to `planted`** |

`random` is the anchor and is expected to land at **0.50** by construction. A
value away from 0.50 means the scoring is wrong, and the stage refuses rather
than reports.

**The primary statistic is `pct_planted − pct_random`, paired within recording,
aggregated with `recur.boot.animal_interval(how="mean")`, 2000 replicates,
clustered on animal.** `n_effective` is animals. `pct_fired − pct_random` is
reported beside it as the scale of the effect the criterion does respond to.

Also reported, because it is the quantity the detector actually thresholds:
the share of planted edges whose `D` exceeds that recording's own
`mad_threshold(D, K_MAD)` — the fraction that *could* fire at all.

## 4. The reading, fixed before the code is written

| outcome | reading |
|---|---|
| `pct_planted − pct_random` interval **includes 0** | the plant has no edge to find. **Step B measured the probe, not the detector.** Its FAIL is **withdrawn as uninterpretable** — not reversed, because this says nothing about what the detector does on real behaviour |
| interval **excludes 0** but `pct_planted` sits well below `pct_fired` | the plant has a weak edge. 12.8% is part probe and part criterion, and neither can be separated from the other by this design |
| `pct_planted` **comparable to `pct_fired`** | the plant is findable and **the criterion really is limited. Step B stands** |

"Well below" is fixed now as **`pct_planted` below the lower bound of
`pct_fired`'s interval**. No threshold is chosen after seeing a number.

## 5. What this cannot do

It **cannot reverse Step B**, and a withdrawal is not a pass. Whether the frozen
detector finds real behavioural boundaries is a question about real boundaries,
and only human annotation answers it. This stage decides one thing: whether the
12.8% is evidence about the detector or evidence about the crossfade.

It also **does not discharge the registered negative control**.
`DETECTOR_PREREGISTRATION.md:84-90` requires any cell that clears the gate to be
run on `white` and fail there; no cell cleared, so it never ran
(`DETECTOR.md:104-109`). It comes due the moment any future detector passes, and
nothing here substitutes for it.

## 6. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only — the
frame-level interval is printed beside it solely to show how much narrower the
wrong method looks, never to license a claim. Ranked on the effect-CI lower
bound, never on a p-value. The **`raw`** arm asserted by name; no Wiener array
enters this stage, and a low-pass filter would manufacture exactly the smoothness
whose absence is under test. `mypy --strict`. No parameter is tuned: every
setting is inherited from `detector_sweep.py` by import, not by retyping.
