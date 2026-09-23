# The island does not replicate on 149 held-out animals

Registered in `FIT_REPLICATION_PREREGISTRATION.md`, committed alone as `28f9d0d`
before this code existed. Produced by `scripts/fit_replicate.py`
(`jobs/fit_replicate.slurm`, 12 shards). Inherited digest **`198eb14ff258c7f6`**,
verified in every job preamble.

**Headline.** On the 149 `fit` animals — never touched by any analysis in this
programme — island occupancy in Context B minus Context A is **−0.00216
[−0.00536, +0.00123]**, pair-flip p = **0.2020**, over **739** (animal, day)
cells on days 3–7. The published `report` value was **−0.00811 [−0.01520,
−0.00243]**, p = 0.0050. **The sign matches, the magnitude is 27% of it, and the
interval includes zero.** The fit split had **99.8% power** against the published
point estimate. Verdict `FAIL`.

---

## 1. The transfer rule passed its own gate first, and it is not an approximation

§3 stops the stage if the rule cannot be shown to select the island rather than
slowness. Against a second bank of **361 non-island `report` segments matched to
the island** on joint (log duration, log mean speed) — worst |SMD| **0.0240**,
against the registered 0.10 bound, computed before any fit segment was scored:

| target bank | fit segments landing within θ = 0.18998 | |
|---|---|---|
| the 361 island members | **0.541% [0.389%, 0.705%]** | 673 segments |
| 361 duration-and-speed-matched non-members | **0.192% [0.123%, 0.276%]** | 224 segments |

**2.82×, non-overlapping animal-level intervals.** θ is not measuring duration
and speed, so the contrast is read. (What "matched" attaches to is disambiguated
in `DEVIATIONS.md` **D15**: the *target* bank is matched, because the source
reading is circular and unimplementable.)

**The prevalence reproduces almost exactly.** The island is **0.561%** of the
64,341 `report` segments; the rule selects **0.541%** of the 108,678 `fit`
segments. The island is just as common in held-out animals. What changed is
where it sits.

**And the rule is the membership criterion, not a stand-in for it.** Run back
over `report` leave-one-animal-out (`fit_replicate_selfcheck.json`), it selects
**exactly** the 361 labelled members: 0 non-members admitted, 0 members missed.
**That agreement is a tautology and is reported as one** — θ *is* the
single-linkage merge height that defined clump 0, so a non-member within θ of a
member would have been a member, and no unfaithful rule was detectable there. Its
value is different: it establishes that a fit segment transfers exactly when
adding it to the report bank would have placed it in clump 0. The one gap is
that single linkage on a combined bank could also chain through new fit points,
and this rule does not — so it is the **conservative** version and can only
under-count. **It cannot manufacture a null.**

## 2. The contrast

Identical machinery to `CONTEXT.md`: `cell_occupancy` → `aligned_deltas` →
`animal_interval` (2,000 replicates, animal-level, `how="mean"`) and
`pair_flip_null`. Cohort asserted at **149** and it held.

| | `report` (published) | `fit` (held out) |
|---|---|---|
| animals | 89 | **149** |
| cells, days 3–7 | 439 | **739** |
| occupancy, Context A | 1.275% | **0.839%** |
| occupancy, Context B | 0.470% | **0.625%** |
| A / B | **2.71×** | **1.34×** |
| B − A | **−0.00811 [−0.01520, −0.00243]** | **−0.00216 [−0.00536, +0.00123]** |
| pair-flip p | 0.0050 | **0.2020** |

**The gap closes from both sides.** Context A occupancy falls by a third and
Context B occupancy rises by a third. This is not the island becoming rarer — §1
shows it is equally common — it is the island becoming **evenly distributed
across contexts**.

## 3. This is a null with teeth, and also a null with a limit

Both statements are true and neither may be dropped.

**It is not an underpowered null.** MDE at 80% power is **0.0047** over 739
cells (within-pair SD 0.04557, SE 0.001676), below the 0.0200 registered as a
biologically meaningful effect — the registered precondition for reading a null
at all. Against the **published point estimate of 0.00811** the fit split had
**99.8% power**. An effect of the size `CONTEXT.md` reported would have been
found here essentially every time. It was not found.

`report` itself ran at **MDE 0.0087 over 439 pairs**. **The held-out split is
the better-powered of the two** — 739 cells against 439, MDE 0.0047 against
0.0087 — so this is not a weaker test failing to see what a stronger one saw.

**It does not exclude the whole published interval.** The fit interval
[−0.00536, +0.00123] overlaps the report interval on [−0.00536, −0.00243] —
**22.9% of the report interval's width**. Against the *weakest* effect the
report's interval allows (−0.00243), power here is only **30.5%**. So:

> **The published effect size is ruled out. A real effect a third its size is
> not, and this design could not have ruled it out.**

**What is not claimed.** That the island is not a real kinematic state — §1 says
it recurs at the same prevalence in 149 unseen animals, which is itself the
first held-out evidence that the *state* generalises. What fails to replicate is
its **context asymmetry**, and only that.

## 4. What this does to the programme

`CONTEXT.md`'s headline — "the island is occupied 2.7× more in Context A" — is
now a **`report`-split finding that did not replicate on held-out animals**, and
it must be written that way wherever it appears. It is not withdrawn; it is
qualified by a preregistered test that it did not pass.

The arena confound compounds this rather than resolving it. `A` and `B` are
**visually different arenas** (A frames cost 32% more bits while keypoint speed
is 24% lower — opposite directions, so motion cannot explain it), and A sessions
run ~17% longer. Occupancy is a rate, so session length cancels; the image
difference does not, and it is a live candidate for why a keypoint-derived
occupancy would differ by context at all. **That question is now more important,
not less** — it is the first of the two asks in the Luna email.

**Nothing here is framed as shock versus no-shock.**

## 5. Provenance

| number | source |
|---|---|
| θ = 0.18998060778738327, scale = 25.319997787475586 | `work/tok/seg_vocab/shape__k3.json`, inherited unchanged |
| 361 island members, 64,341 report segments | `work/tok/seg_vocab/shape__k3__corpus.npz` |
| 108,678 fit segments | `scripts/seg_recur.py:segments_of` + `embed.selectable`, frozen parameters, 149 animals |
| balance |SMD| 0.0240 | `controls.matched_partners` + `balance_read`, before scoring |
| incumbent −0.00811 [−0.01520, −0.00243], p 0.0050 | `CONTEXT.md`, quoted per **M12**, not recomputed |
| 1.275% / 0.470% | `CONTEXT.md` |
| MDE 0.0047, power 0.998 / 0.305 | `sx.mde_read`; power from the same SE, normal approximation |
