# Pre-registration — an order-controlled discontinuity plant, and the first accuracy number

Written and committed **before `vieb/seg/plant.py` or `scripts/plant.py`
exist**, and before any recovery is measured. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm; `shape` group; **`tune`
split, 60 animals**. Detector frozen at `deriv_sec = 0.133`, `degree = 3`,
`k_mad = 3.0`. Digest `198eb14ff258c7f6`.

## 1. Why this exists, and the authorisation for it

**This programme has never measured the detector's accuracy.** Every result on
disk is a comparison against a null — recurrence excess, a context contrast, a
rate against a floor. Those establish *not nothing*. None is a recall.

Both previous attempts at ground truth failed for diagnosable reasons.
`DETECTOR.md` scored **12.8%** isolation against a 20% gate, and `PROBE_AUDIT.md`
then showed why the target was unfair: the planted instance was the mean of 40
real windows crossfaded over 3 frames, sitting at the **60th percentile** of the
criterion's own scalar against real firings' **96th**, with only **19.87%** of
its edge frames clearing their own threshold. `DETECTOR.md:98-102` names the fix
and requires this registration first:

> The criterion looks for a discontinuity in acceleration. … **It has no
> acceleration discontinuity to find.** … **What would distinguish them is a
> planted instance whose insertion is a genuine C² discontinuity** — and that
> needs its own registration, because "make the control easier until the
> detector passes" is the failure mode this gate exists to prevent.

**Step B is not re-scored and its FAIL is not withdrawn.** This is a new probe
answering a different question: not "can the detector isolate a smooth template"
but "at what order and what magnitude does it recover a discontinuity at all".

## 2. What is planted, and why order is the axis

The detector's axiom is that behaviour is piecewise smooth and a boundary is a
break in acceleration. A probe that does not control the **order** of the break
cannot separate "the criterion is sound but mis-scaled" from "the criterion is
broken". So the plant is a one-parameter family in order.

Added to the `shape` channels of one recording, at frame `t0`, along a random
unit direction fixed per instance:

| order | added signal on `[t0, t0+W)` | what is continuous at `t0` | what jumps |
|---|---|---|---|
| **0** | `A` | nothing | position |
| **1** | `A·(t−t0)/W` | position | velocity |
| **2** | `A·((t−t0)/W)²` | position, velocity | **acceleration — the axiom** |

and mirrored back to zero over `[t0+W, t0+2W)` so the signal returns to baseline
at the same order. `W = frames(0.25, fps)`, a quarter second — longer than the
detector's own 0.133 s half-window, so the onset is resolvable in principle
rather than by luck.

**Only the onset at `t0` is scored.** The taper creates a second break of the
same order at `t0+W`; that frame and its ±2 neighbourhood are excluded from both
the hit test and the false-positive count, and the exclusion is registered here
rather than discovered later.

**The registered prediction is that recovery is ordered `0 > 1 > 2`.** A
numerical second derivative of a noisy signal responds most to the sharpest
break. What matters is not the ordering but where order 2 — the only one the
axiom claims — crosses from undetectable to detectable.

## 3. Amplitude, in units of the measured noise

`A` is swept in multiples of the **measured** per-keypoint jitter amplitude from
`NOISEFLOOR.md`'s calibration — σ(c) from rigid skull-bone residuals, in body
lengths — not in units chosen for convenience.

> **Ladder: A ∈ {1, 2, 4, 8, 16, 32} × σ̄**, where σ̄ is the calibration's
> confidence-weighted mean amplitude.

This is the link the floor was needed for. A plant below the noise the tracker
already injects is not a fair target for any detector, and one far above it is
not a behavioural event. **The number this stage exists to produce is the
amplitude at which order-2 recovery crosses 50%**, expressed in units of
measured tracking noise.

## 4. Scoring, and the chance level it is read against

A planted onset is **recovered** if the detector, run on the planted signal,
places a boundary within **±2 frames** of `t0` — the same band every other
stage in this programme uses, inherited and not chosen here.

**Recall alone is not readable.** A detector firing at the corpus rate hits a
±2 window by luck at `rate × (2·tol+1) / fps`; at the measured 0.443/s that is
**0.074**. So every recall is reported beside a **per-recording chance level**
computed from that recording's own boundary rate on its own planted stream, and
the effect is `recall − chance`. This is the same discipline `annot.chance_f1`
applies to the human ceiling.

Plants are placed **per recording**, never across a seam, never touching an
abstained frame, and never overlapping one another, following
`planted.ego_plant`'s existing contract and asserted by test.

## 5. The negative control, discharged here

`DETECTOR_PREREGISTRATION.md:84-90` has required since Step B that any cell
clearing the gate be run on **`white`** — i.i.d. noise at each recording's own
mean and SD — and fail there. No cell ever cleared, so it has never run
(`DETECTOR.md:104-109`).

**It runs here, for every cell of the sweep.** The identical plant goes into
`work/surrogate/white/ego/` and is scored identically.

> **A cell recovers its plant on `white` at the same rate as on the corpus ⇒
> that cell is recovering its own threshold, not the plant, and may not be
> reported as a recovery.**

This discharges the standing debt for the cells this stage covers. It does not
discharge it for any future detector.

## 6. Predictions, fixed now

1. **Order 0 is recovered at the top of the ladder**, above 0.8 at 32σ̄. Near
   mechanical; if it fails the harness is wrong and nothing else here means
   anything.
2. **Recovery is monotone in amplitude** within each order.
3. **Recovery is ordered 0 ≥ 1 ≥ 2** at every amplitude.
4. **Order 2 requires a larger amplitude than order 1 to reach 50%**, by at
   least one rung of the ladder. *Falsifier: if order 2 reaches 50% at the same
   rung as order 1, the criterion is not preferentially blind to the smoother
   break and Step B's failure is about the template's smoothness alone.*
5. **`white` recovery is below corpus recovery at every cell above 4σ̄.**
   *Falsifier: equality means the probe measures the threshold.*

## 7. What this stage may not do

* It may not **re-score Step B**, restate `DETECTOR.md`'s 12.8%, or withdraw it.
* It may not **tune**. The detector is frozen; a changed window, degree or
  threshold is a different detector needing its own registration.
* It may not **adjust the amplitude ladder, `W`, the ±2 band or the orders**
  after seeing a recovery number.
* It may not report a cell whose planted count is below **200 instances**.
  Refused, not caveated.
* It may not use any **Wiener** array, or compute across a recording boundary.
* It may not read a recovery number as a statement about the corpus's own
  boundaries. **A planted discontinuity is not a behaviour**, and recall against
  a plant bounds the instrument, not the biology.

## 8. Conventions

Every claim a `Read` with a non-empty `scored_object`, a reason and a required
`n_effective`; refusal is a correct outcome. Animal-level bootstrap only,
2,000 replicates, `how="mean"`. Ranked on the effect-CI lower bound, never a
p-value. Seed 0, `seeds.stable_seed`. `mypy --strict` on new modules. The
**`raw`** arm asserted by name.
