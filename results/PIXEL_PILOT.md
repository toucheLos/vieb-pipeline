# The pixel pilot: a standard freeze measure finds the context effect the island lost

Registered in `PIXEL_PREREGISTRATION.md` (+ Amendment 1), committed before the
code. `scripts/pixel_pilot.py`, `jobs/pixel.slurm`. Digest
**`198eb14ff258c7f6`**. The arena floor this all rests on is
`PIXEL_NOISEFLOOR.md`.

**Headline.** On the same 30 held-out `fit` animals where the island's context
effect is indistinguishable from zero, a frame-differencing freeze score
separates the contexts decisively: **B − A = −0.1264 [−0.1706, −0.0851]**,
pair-flip p = **0.0005**, over 150 paired (animal, day) cells — animals freeze
**12.6 percentage points more in Context A**. It **survives** subtracting each
frame's own arena noise (**−0.1611 [−0.1994, −0.1217]**, further from zero).

This is the **first branch of §0's registered table**, and §0 fixed its reading
before the number existed:

> *a standard pixel measure finds a context effect where the island does not.*
> **The island is the worse instrument**, and that is a finding about the
> island, not about the animals.

---

## 1. What may and may not be concluded from that number

**The arm that produced it is the one §9.5 forbids adopting**, and that is not a
technicality — it is the only arm the data left standing.

* **Established:** that a pixel freeze measure separates A from B on these
  animals, in this direction, at this significance, robust to arena-noise
  correction. The one arm defined on every cell is enough for existence and
  sign.
* **Not established:** the **magnitude**. `FreezeThresh = 200` is a pixel
  **count** tuned on 320 × 240 video; this corpus is 640 × 480, **4× the
  pixels** (§2). 12.6 points is what an untransplanted ezTrack reports, not what
  a correctly scaled one would.

So Q2's answer is *yes, and the size is unquantified pending a threshold that is
registered for this resolution.*

**And the two numbers are not the same quantity.** Island occupancy is a share of
*segment* frames; freeze fraction is a share of *video* frames. The comparable
statement is not "0.126 against 0.00216" but: **on the same animals, the same
days and the same cells, one measure detects a context difference and the other
does not.**

**It is also not new biology.** Keypoint speed already said animals move 24%
less in Context A. More freezing in A is the same fact on an independent
channel. The island was supposed to be a *finer* state than "slow" — that is
what would have made it worth having — and on held-out animals it is the crude
measure that carries the contrast.

## 2. Q1 refused, by one animal, and the bar did not move

**Q1 (does island occupancy track the freeze score within animal?) is
`NOT_A_RESULT`:** 19 animals yield a defined rank correlation against the
registered minimum of **20**.

**The value it would have given is not reported.** Reading past a refusal is
what the refusal exists to prevent, and a bar moved by one unit after seeing
that it binds is not a bar.

**The diagnosis is the useful part.** 11 of the 30 examined animals have island
occupancy that **never varies** across their ten cells, because **90.7% of
(animal, day, context) cells hold exactly zero island frames**. At 0.54%
prevalence the island is simply too sparse at cell resolution to rank-correlate
against anything. This is a property of the object, not of the pilot's size —
tripling the animals would triple the constant ones too.

> **The convergent-validity question is still open and still worth asking**, at
> a resolution where the island is not sparse: does an individual island
> **segment** coincide with pixel-defined freezing? 673 island segments against
> 108,678 others is not a sparse comparison. That is a different statistic from
> the one registered here, so it is **not computed**; it belongs in its own
> registration.

## 3. Twenty-seven of twenty-eight arms could not be read — D17

§3 derives `mt_cutoff` from the arena's noise; §4 sets `FreezeThresh` to a
percentile of `Motion`. **They do not compose.** At a cutoff that strict most
frames have *no* pixel changing at all, so `Motion` carries a large atom at zero
and a low percentile of it **is** zero — and `Measure_Freezing` tests `<`
strictly, so such an arm reports 0% freezing as though it had measured it.

| arm | degenerate cells |
|---|---|
| `eztrack_default` | **0%** — and §9.5 forbids adopting it |
| `m1_p40_*` | 5.7% — the best **derived** arm |
| `m1_p25_*` | 11.3% |
| `m1_p10_*` | 31.7% |
| **`m2_p25_*` (registered headline)** | **62.3%** |
| `m2_p10_*` | 85.0% |
| `m4_p25_*` | 90.7% |
| `m4_p10_*` | 98.0% |

**Degenerate arms are refused, not computed on survivors.** The cells such an arm
drops are the **quietest** recordings — exactly the ones most likely to contain
freezing — so a fraction computed on what is left is selection on the outcome.
An arm is readable only at **zero** degenerate cells; that bar does not decide
this stage, since the best derived arm sits at 5.7% and any bar from 0% to 5%
refuses the same set.

**So the registered headline carries no verdict**, and the fix — at a
noise-derived cutoff the natural criterion is `Motion == 0`, not a percentile —
is deliberately **not** applied here. Choosing an estimator after seeing which
one the data supports is what §9.4 prohibits.

## 4. The held-pose question is settled, and it was not a dropout — §6

A **census** of all 1,440 eligible `fit` recordings, **43,206** zero-ego-speed
frames — above the standing 20,000-frame floor, which the 300-recording pilot's
6,228 was not.

| | duplicate video frame | tracking dropout | genuine immobility |
|---|---|---|---|
| margin 0 (the literal §6 reading) | **14.2%** | **9.6%** | **76.1%** |
| margin 0.1% of animal area | 14.2% | 2.7% | 83.0% |
| margin 1% of animal area | 14.2% | 0.9% | 84.9% |

§6 left the *margin* free, and at a cutoff derived from the floor the arena
expectation is near zero, so the literal reading makes a **single changed pixel**
a dropout. A free choice inside an instrument is a variable (**M10**), so it is
swept. **The conclusion is the same at every margin**, and the most generous
setting for the incumbent description is the one reported as the headline.

> **`CONTEXT_CONTROLS.md` calls these frames "a tracking dropout". That is
> correct for at most 9.6% of them and for as little as 0.9%.** Three quarters
> or more are the animal genuinely holding still, and **14.2% are duplicate
> video frames** — an encoder artefact, in which no time passes at all. The
> duplicate share is margin-independent by construction: it is exact equality of
> the decoded greys.

Corrections land in `CONTEXT.md` and `CONTEXT_CONTROLS.md`, as registered,
whichever way this fell.

## 5. Two defects the registration could not have caught by reading

Both are recorded in full, and both were found by running the registered thing
rather than by inspecting it.

* **D17** — §3's cutoff and §4's threshold are each defensible alone and
  degenerate together.
* **D18** — §7's refusal compared **grey levels** with a **fraction**. As
  written it fires on every possible result, and on the first run it did.
  Replaced by subtracting each frame's own arena rate and requiring the effect to
  survive with its sign; that test could have failed and did not.

`PIXEL_NOISEFLOOR.md` §3 records a third, of a different kind: the registered
300-recording pilot was **underpowered for §7's own question**, and only the
census could see the arena difference the pilot was meant to interpret.

## 6. Provenance

| number | source |
|---|---|
| freeze algorithm and defaults | ezTrack `master`, read from source; **not installed**. Reimplementation checked against a literal transcription of `Measure_Freezing` on 300 random inputs, `tests/test_pixel.py` |
| `mt_cutoff` per recording | `PIXEL_NOISEFLOOR.md` |
| island occupancy | rebuilt from Step 1's own shards; reproduces `FIT_REPLICATION.md` exactly (B−A −0.00216 [−0.00536, +0.00123], 739 pairs, mean_A 0.008390, mean_B 0.006246) |
| incumbents quoted, never recomputed | island on `fit` −0.00216 [−0.00536, +0.00123]; on `report` −0.00811 [−0.01520, −0.00243]; residual on `report` −0.00685 [−0.01371, −0.00147] |
| 43,206 zero-ego-speed frames | census of 1,440 recordings, `qz.speed == 0` |

**Nothing here is framed as shock versus no-shock.**
