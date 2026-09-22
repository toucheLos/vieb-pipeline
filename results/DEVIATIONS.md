# Declared deviations from registered configurations

Every place a run differed from what its registration fixed, with the cause.
Recorded rather than corrected silently.

## D1 — six columns where five strata were registered

**Registration:** `STRATIFIED_PREREGISTRATION.md` fixes "5 speed quantiles of
segment median centre speed".

**What ran:** six columns.

**Cause.** `truth.assign_bin` returns **−1** when a segment's speed cannot be
placed — a NaN speed, or fewer than two frames. That is a **refusal code, not a
stratum**. `scripts/injection.py` built its per-stratum table with
`sorted(strata.items())`, and −1 sorts before 0, so the refusal bucket was
printed first and read as *the slowest stratum*.

**What it corrupted.** The bucket carries **235 keypoint-frames** at `off`
(445 at 0.02) against 4.5M in the real strata — far below the registered
`MIN_STRATUM_FRAMES = 20,000`, so it should have been refused outright. Two
published claims came from it and **both are withdrawn**:

| claim | came from | actual |
|---|---|---|
| "`viterbi` reassigns nothing in the slowest stratum, net exactly 0.0000" | bin −1, no data | **−0.0014** |
| "`median_0.50`'s net runs −0.0331 in the slowest stratum" | bin −1, 235 frames | **−0.0021** |

Both appear in `STRATIFIED.md` as published; corrected there.

**Fix.** `combine` now excludes bins below 0, reports their frame count as
`n_unplaceable_frames`, and refuses any stratum under `MIN_STRATUM_FRAMES`
(emitting NaN, never a thin number). Pinned by `tests/test_strata.py`.

**Does the finding survive?** Yes, and more cleanly. The corrected five-stratum
profile at `off` is −0.0021, −0.0008, +0.0001, +0.0020, +0.0068 for
`median_0.50` against −0.0014, −0.0016, −0.0016, −0.0018, −0.0025 for `viterbi`.
The crossing is intact and monotonic in both directions.

## D2 — pooled and per-stratum figures were compared on different footings

**Not a deviation from a registration**, but a reporting error worth the same
treatment, since it is what made the arithmetic look broken.

`net` is a **frame-weighted** mean across animals of each animal's difference in
mean error per keypoint-frame. `net_by_speed_bin` was an **unweighted** mean
across animals of the within-bin difference. Reading a pooled −0.0000 against
per-stratum values whose unweighted mean is **+0.0012** suggested the figures
could not both be right.

They can. **The pooled figure is exactly the frame-weighted sum over strata**,
verified to 6 decimal places on the measured data:

```
frame-weighted sum over strata : -0.000033
pooled net as reported         : -0.000033
```

The strata differ in size by 4× (1.90M down to 0.47M keypoint-frames), and the
positive strata are the small ones, so weighting flips the sign relative to an
unweighted average. `combine` now computes the per-stratum figure
frame-weighted, matching the pooled path, and `tests/test_strata.py` pins the
identity.

## D3 — `wiener` registered as an arm, never benchmarked

**Registration:** `INJECTION_PREREGISTRATION.md` lists `wiener` among the arms.

**What ran:** four arms, without it. `clean_arms.STORED_ARMS` reads the Wiener
and Butterworth arrays off shapeflow's disk; this repo never reimplemented the
filter, so there is no callable to apply to a freshly corrupted array.

Recorded in `INJECTION.md` at the time rather than edited out. The incumbent
remains the one arm whose recovery performance is unknown.

## D4 — Phase F's seed did not control the run

**Registration:** `INJECTION_PREREGISTRATION.md` fixes "seed | 0" and says
"Seeded; the seed and the realised rates go in the result."

**What ran:** `abs(hash((SEED, tag, rid))) % 2**32`, and Python salts `hash()`
on `str` per interpreter. Every run drew a different corruption layout.

Found in F2, fixed with blake2b, audited as a class in `SEED_AUDIT.md`, pinned by
`tests/test_seeds.py`. Phase F's figures move ~1% relative under the stable seed;
**no verdict, sign or ordering changed**.

## D5 — the ladder's identity-leak gate is UNRUN, not merely reduced

**Registration:** `TOK_PREREGISTRATION.md` §6 — *"the next-state distribution
goes through `recur.audit.leak.leak_read`, held out by window within recording."*

**What ran:** the per-window mean and SD of the model's own per-run code length,
through the same `leak_read` at the same holdout. The next-state distribution is
an `N`-vector per run; at N = 2,048 and ~4.3M scored runs that is tens of
gigabytes per cell, times sixteen cells, for a gate rather than a headline.

**What came back: ten of the sixteen probes returned `NOT_A_RESULT
[DEGENERATE]`** on the inherited `max_iter = 400` convergence guard. The six that
passed are all on the `speed` arm, which MDL rejects at every N.

> **So the style-leak gate is UNRUN.** It is not "reduced but passing", and it is
> not weak evidence of no leakage — it is an absence of evidence. Nothing
> downstream leans on it, and nothing downstream may.

Stated this way because the earlier wording of this entry ("a reduced probe…
strictly weaker") invited exactly the misreading it was trying to prevent: a
future reader finding a D5 that says "reduced" and a `LADDER.md` paragraph that
says "six pass" could take the gate as cleared. It was not cleared. It did not
run.

**What running it would take:** either a larger `max_iter` — which would make the
figure incomparable to `audit_moseq.json`'s 0.283 nats, the reason the budget is
inherited in the first place — or a probe whose feature count does not scale with
`N`. Both are open, and neither is in the current plan.

## D6 — the duration bin width was off by one, and the rule it produced

**Registration:** `TOK_PREREGISTRATION.md` §3 — duration charged at one-frame
resolution, `delta = 1/30` s, identically in every arm.

**The mechanism, in full.** Elapsed-time bin `b` covers `[e[b], e[b+1])`, so it
holds exactly `e[b+1] - e[b]` whole durations. `ladder.duration_pmf` divided each
bin's mass by `e[b+1] - e[b] + 1` instead — the `+1` already being present in the
lower endpoint, since `d = elapsed + 1`. So `f(d) = p_bin / width` summed to
**1.115** rather than 1 and **every duration charge was too small by roughly 0.11
nats**.

**Why nothing looked wrong.** A constant offset applied to every arm cancels in
every comparison this project reports, and survives only in the absolute
nats-per-second figures — which nothing else is calibrated against. Rung 1 would
still have beaten rung 0 by the same margin. The bug was invisible in exactly the
numbers a reader would check.

**How it was found:** by `tests/test_mdl.py::test_the_duration_pmf_normalises_
over_frames`, written to pin the registered charge, **not by review**. It was
caught before any ladder job was submitted, so no published number was computed
under it.

### The rule

> **A pmf that is never asserted to sum to 1 is not a pmf.**

`vieb/checks.py` now holds `assert_pmf`, `assert_log_pmf`, `assert_unit` and
`assert_share`, and they are called at the point each distribution is built —
not in a test below it, where a new branch can route around them. Applied at
every site in the `vieb` package that constructs a probability vector or a
share:

| module | what is now asserted |
|---|---|
| `tok/ladder.py` | `f(d)` **at frame resolution** (the bin masses always summed to 1 — checking those would not have caught this); rung 0's marginal; rung 1's rows, per row, over the `a-1` allowed causes |
| `tok/hazard.py` | every `(context, elapsed bin)` cell is a simplex over stay + exits + prior, **including the unobserved cells** `_log_exit` backs off into |
| `tok/quantize.py` | occupancy frame share |
| `tok/rle.py` | frame mass and run mass per symbol; bucket shares bounded |
| `qc/runlen.py` | mass at-or-below the widest bucket plus mass above it is exactly 1 |
| `qc/bones.py` | per-keypoint blame share, with the no-violations case now flagged `share_defined: false` instead of returning zeros that sum to nothing |
| `qc/concentration.py` | per-recording rates bounded; Lorenz cumulative shares bounded **and monotone** |

`assert_unit` exists separately from `assert_pmf` for memory: a table of per-cell
simplexes would need a stacked extra axis to check with `assert_pmf(axis=-1)`,
which at the `k = 2` history depth is 7.5M contexts × 13 bins × 3 outcomes, about
2.3 GB of temporary, for an identity that holds cellwise.

---

## D7 — Step 2's separability precondition is unsatisfiable by any null the question admits

**Registered** (`SEGMENTATION_PREREGISTRATION.md` §3): the probe runs first, on
every null, on window features; **high separability is failure**; limit
`balanced accuracy ≤ 0.60`; a null above it *"differs from the corpus in ways
unrelated to boundaries, so a boundary-rate gap against it is uninterpretable."*

**What happened.** The first run passed all four nulls. It was wrong: `white` —
i.i.d. noise — came back inseparable at **AUC 0.496**, which is chance, and white
noise is not inseparable from a mouse trajectory. `separability` fits a logistic
regression, and smooth-versus-rough lives in the **second moment of the
increments**, which is quadratic in the raw window and outside a linear model's
reach. Per-channel `log` mean-squared-first-difference features were added,
`white` failed as it must, and **so did the other three**: `ou` 0.999, `white`
0.989, `phase` 0.800, `var5` 0.801.

**Why it is a design defect and not a result.** The post-hoc diagnostic
(`work/tok/seg_validate/_separability_diagnose.json`) shows what separates them.
It is not a roughness *level* — that would have been benign, since the threshold
is MAD-standardised per recording and divides a level out. `phase` preserves the
corpus's power spectrum exactly, hence its mean squared increment exactly, yet
the corpus's mean **log** window energy sits 2.22 nats lower (9.2×) in 15 of 17
channels with 2.15× the window-to-window spread. The corpus's roughness is
*concentrated*: smooth stretches, rare rough moments.

That is the hypothesis under test, visible in the probe's own features. A null
constructed to have **no** boundaries differs from a piecewise-smooth corpus in
its roughness distribution **because** it has no boundaries. So once the probe
could see roughness at all, no boundary-free null could have passed, and the
precondition rules out every null the question admits.

**What was done about it: nothing, deliberately.** The registration is not
amended and the limit is not moved. Both registered checks are reported as
failing, and the gate's `FAIL` is declared **non-licensing in both directions** —
it cannot say the route closes, exactly as a `PASS` could not have said the
boundaries are real. Step 2 returns no verdict; Steps 3 and 4 do not run.
Reopening requires a **new** registration whose precondition can separate "this
null is off-manifold in an irrelevant way" from "this null lacks the structure
under test". The current one cannot, and neither can a threshold chosen after
seeing this.

**Both probe runs are on disk**, the blind one preserved rather than deleted:
`_separability_rawonly.json` and `_separability.json`. The refactor that gave the
diagnostic access to the window builder was verified to reproduce the standing
probe **bit-for-bit** before anything was concluded from it.

**The rule this leaves:**

> **A probe with no negative control is not a probe.** The blind instrument
> passed four nulls and would have licensed the whole gate. Only the null whose
> answer was already known could say the probe was measuring nothing.

---

## D8 — the registered secondary distance was never computed

**Registered** (`SEGRECUR_PREREGISTRATION.md` §4): time-normalised as the
primary, and *"**SECONDARY: open-end alignment**, no warp, compared over the
overlapping extent. Reported beside the primary."*

**What happened.** `vieb/seg/embed.open_end_distance` was written, unit-tested,
and **never called by any driver**. `SEGRECUR.md` reports only the primary. It
does not claim the secondary ran — it simply omits it, which is worse in one
specific way: a reader checking the registration against the result would find a
promised comparison missing with nothing saying so.

**Why it mattered here specifically.** The primary warps every segment to 40
points, so duration is invisible to it, and clump 0 pools 0.53 s with 65 s. The
secondary is precisely the instrument that says whether those belong together.
Omitting it removed the one check aimed at the most suspicious property of the
headline result.

**Fixed, then half-withdrawn.** Computed in `results/BEHAVIOUR.md`: on clump 0's
own nearest pairs the Spearman between the two distances is 0.616 and matched
partners differ by a mean factor of **3.7×** in duration.

**The distance half of that was wrong and is withdrawn.** It read "median
open-end **0.716** against the primary's **0.173**" — a per-frame RMS in
standardised channel units against a 560-dimensional Euclidean norm already
divided by its ambient scale. Different units; the ratio meant nothing. The
open-end ambient scale, measured since on eight report animals, is **3.885**, so
the same pairs sit at **0.184** against **0.173** — near-identical, the opposite
of what was reported. It never reached the website.

So the clumping being "substantially tempo-invariance" rests on the **duration
gap alone**, which needs no distance scale. The distance comparison requires a
per-metric ambient scale and is what `DISTANCE_PREREGISTRATION.md` registers.

**The rule this leaves, which is the same failure in a new place:**

> **Two numbers are not comparable until something has divided them by the same
> thing.** A normalised distance and an unnormalised one look alike in a table
> and their ratio is meaningless. `paired_excess` exists precisely because raw
> distances are not comparable across arms, and this put an unnormalised
> quantity in a table beside a normalised one anyway.

**The rule:**

> **A registered secondary is a deliverable, not a courtesy.** If a registration
> promises a second measurement, the result document either carries it or says in
> its own text that it does not and why. Silence reads as "reported".

## D9 — the island's poster is a video frame, not a mean skeleton

**The brief:** "Mean-skeleton render as the poster frame, so the page is legible
before video loads and still legible if a clip 404s."

**What shipped:** a frame of the first clip in each panel, via
`sync_assets._poster_from_clip`. The legibility requirement is met; the mean
skeleton is not drawn.

**Cause, and it is the page's own argument.** The island is a **single-linkage
chain**: a typical pair of its members sits `0.3978` apart against a linking
distance of `0.1900` — 2.1×, further apart than the threshold that built it.
`BEHAVIOUR.md` states the consequence and `island.html` repeats it: **its
members are not copies of one another and it must not be named as one
behaviour.**

A mean skeleton is a picture of a prototype. Drawing one for a chain would put,
at the top of the page, an image asserting exactly what the page spends its
words denying — and a reader who looked at the poster and stopped would take
away the opposite of the result. `recur/scripts/meanskel.py` already refuses the
analogous move for v2 motifs: rather than borrow one component's skeleton and
call it the motif, "which would be a picture of something else", it lays the
components out side by side.

**What it costs.** The poster carries less information than a mean skeleton
would for a genuine motif. For this object it carries the right amount: one real
frame of one real segment, captioned as such.

**What would change it.** If a future clump passes `chaining_read` — a group
rather than a chain — its poster should be a mean skeleton, and this deviation
does not extend to it.

## D10 — two diagnostics added to the context controls after seeing the threshold

**What the registration said.** `CONTEXT_CONTROLS_PREREGISTRATION.md` fixed three
arms — `island`, `stillness`, `windows` — one nested residual per control arm,
and an MDE gate before any residual is read. It named no diagnostic beyond that.

**What was added.** Two reads, both `NOT_A_RESULT`, both written after
`θ_still = 0.000199` had been computed and its consequences seen:

* **`overlap|stillness`** — the share of the island's own frames the control arm
  also selects. It came out at **0.0227**, which is what shows the arm is not a
  control for this clump at all.
* **`held_frames`** — the share of frames at exactly zero ego speed, a tracking
  dropout rather than a slow animal, contrasted by context. It came out at
  **0.643% in A against 0.094% in B**, p = 0.0005.

**Why this is not a verdict moved after the fact.** Neither read carries a
verdict, and neither changes one. The registered residuals are reported exactly
as §5 specified, with the values they had before these were written. What the
diagnostics change is what a reader is told the residuals *mean* — and the
alternative was to publish a `PASS` that a reader would reasonably take as
"the detector is not a freeze scorer" when the arm producing it selects 2.27% of
the island.

**Why it had to be added rather than deferred.** A number that is vacuous is not
made less vacuous by being reported on its own and corrected later. `M5` is the
standing rule — *a control that returns nothing is usually testing the control* —
and the honest response to discovering it mid-run is to instrument it, not to
publish the number and file the doubt separately.

**What it costs.** The held-frame contrast is unregistered and no verdict rests
on it. It is a measured confound, reported at the strength a post-hoc
measurement earns: it moves with context, it does not explain the island cell
for cell, and it means no occupancy in this design is free of tracking quality.
A registered version would need its own contrast and its own gate.

**What would change it.** The segment-level stillness control
`CONTEXT_CONTROLS.md` names as owed — segments matched on joint (log duration,
log mean speed) — supersedes the vacuous arm. When it is registered and run, the
`overlap` read becomes a routine precondition rather than a finding.


## D11 — an unregistered diagnostic on where the island's boundaries sit

**What prompted it.** `NOISEFLOOR.md` found the detector at its own noise floor
in the slowest speed quintiles. `BEHAVIOUR.md` found the island **3.7× slower**
than its animals' other segments. Read together those say the one
design-validated unit in the programme sits where boundaries cannot be told from
jitter, and the first write-up of the floor said exactly that.

**Why it was wrong to leave there.** A segment's content and its edges are not
in the same regime. A long still segment is delimited by the animal entering and
leaving stillness, and those are motion events. The 3.7× describes interiors. No
measurement on disk had asked about the boundary frames.

**What was run.** Per-frame speed at island boundary frames, island interior
frames, every other selected segment's boundaries, and every scored frame,
placed in the **tune-derived** speed quintiles from `noise_floor.json`. No
verdict: the read is `NOT_A_RESULT` and changes no published number.

**The result.** Island boundary frames run 2.4× faster than their own interiors
(0.0654 against 0.0275 body lengths/s) and their share in the floor strata
halves, 72.4% [65.8, 78.4] to 36.7% [28.0, 44.8]. They remain ~1.9× enriched in
those strata against boundaries in general, 19.4% [16.5, 22.4].

**What it costs.** It is post-hoc and it improves the standing of a published
result, which is the direction in which a post-hoc analysis should be trusted
least. It is therefore reported without a verdict, with both the pooled and
animal-weighted shares (they disagree, because the island is animal-
concentrated), and with the split transfer named. A registered version would fix
the strata and the comparison before running, and would be needed before any
claim rests on it.

**What would change it.** The segment-level stillness control
`CONTEXT_CONTROLS.md` names as owed is the registered test of the same worry.
This diagnostic informs how to design it; it does not substitute for it.

## D12 — the plant's return to baseline is an exponential decay, not a mirror

**What the registration said.** `PLANT_PREREGISTRATION.md` §2 describes the
added signal as `A·((t−t0)/W)^k` on `[t0, t0+W)`, "mirrored back to zero over
`[t0+W, t0+2W)` so the signal returns to baseline at the same order", with the
midpoint break excluded from scoring.

**Why that construction could not be used.** It plants a second break that is
*sharper than the one under test*, and at a lower order.

* **Mirrored:** for order 2 the rising ramp meets its own reflection with
  slopes `+2A` and `−2A`. That is an order-**1** discontinuity at the midpoint,
  against the order-**2** discontinuity the probe exists to measure.
* **Clamped-then-tapered**, the first thing built instead: holding the
  polynomial at 1 after `u = 1` drops the slope from `2/w` to `0` there, which
  is the same order-1 break. Measured at `w = 16`, that junction's second
  difference was **15.5×** the planted order-2 break.

Non-maximum suppression keeps the strongest peak. Either construction would
have handed the detector a sharper, lower-order target a few frames away from
the one being scored, and the recovery numbers would have been about that
instead. Excluding the midpoint from *scoring*, as §2 provides, does not help:
the competing peak still wins NMS inside the same refractory window and
suppresses the onset.

**What is used instead.** `g(u) = u^k · exp(−u/τ)`, normalised to unit peak,
with `τ` placing the peak near `u = 1` and a smootherstep taper applied only in
the far tail at `u ≥ 8`, where the profile is already ~7e-6 of its peak. Below
`order` every derivative at `u = 0` is zero and the `order`-th is not, so **the
only discontinuity of any order in the instance is the planted one**, and the
profile returns to exactly zero. Measured: the planted break is the largest
second difference in the instance for all three orders, by at least 3.3×.

**Why this is a refinement and not a loosening.** §2's stated intent is a
plant whose only break is the registered one — "returns to baseline at the same
order". The mirror does not achieve that intent for `k = 2`; the exponential
does. Nothing about the order, the amplitude ladder, the ±2 band or the scoring
rule moved, and the change makes the target *harder* to find, not easier, by
removing a sharp competing edge.

**What it costs.** Instances are longer — `12w + 1` frames against `2w` — so
fewer fit per recording and the placement guard is wider. `tests/test_plant.py`
asserts the margin rather than the constant, so a future change to the taper
that reintroduced a competing break would fail rather than pass quietly.

**Caught before any number was read.** The first implementation's recovery
figures were computed on a smoke test and discarded unread when the test that
checks where the break lives failed.

## D13 — the trend filter's λ scale is a robust median, not §3's `λ_max`

**What the registration said.** `TRENDFILTER_PREREGISTRATION.md` §3 fixes λ as
a fraction of "each recording's own `λ_max = max_t ‖(D³Y)_t‖₂`", swept over
α ∈ {1e-4 … 3e-1}.

**Why that could not be used.** The name `λ_max` means "the smallest penalty at
which the fit has no knots". The formula §3 writes down is not that quantity and
has no fixed relationship to it. For the group-lasso trend filter the true
no-knot threshold is the **dual** norm `max_t ‖((D³D³ᵀ)^{-1} D³Y)_t‖₂`.
Measured on realistic data at `T = 3000`:

| | |
|---|---:|
| §3's formula | 5.55e−2 |
| true no-knot threshold (dual norm) | 1.29e+9 |
| **ratio** | **2.3e10** |

So **the entire registered α grid sat in the fully dense regime**. Every grid
point would have returned thousands of knots per recording, `lambda_read` would
have returned `GRID_LIMITED` for every animal, and §3's own instruction not to
extend the grid would have locked the stage into a refusal produced by an
arithmetic slip rather than by evidence.

The dual norm is not usable either: `D³` is severely ill-conditioned, so
`(D³D³ᵀ)^{-1}` is dominated by the smoothest modes and `λ_max` lands ~1e9,
nine orders above where knots actually become sparse. A grid that is a fraction
of it is just as useless in the other direction — measured, every α on the
registered grid gives **zero** knots.

**What is used instead.** `α · median_t ‖(D³Y)_t‖₂`, with
α ∈ {0.5, 1, 2, 5, 10, 20, 50, 100}, implemented as `trendfilter.scale_of`.

This is the deliberate analogue of the frozen detector's own rule:
`mad_threshold` cuts at `median(D) + k·MAD(D)` of its scalar, and this penalises
at α times a robust scale of the same quantity the estimator penalises. Both are
per recording, so both mean *"this far above this recording's own typical
roughness"*, which is what makes `k_mad` and α comparable parameters at all.
Measured, the grid spans boundary rates from ~12/s down to ~0.05/s and therefore
brackets the frozen detector's 0.443/s, which is exactly the span §3 needed.

**What did not change.** The selection *rule* — smallest setting whose
jitter-only rate has a bootstrap upper bound below 5% of the corpus rate — is
untouched, as is the 5% itself, the plant comparison, the ±2 band and every
prediction. Only the quantity α multiplies moved, and it moved because the
registered one could not span the path.

**What it costs.** α is no longer bounded in `[0, 1]`, so "α = 1 means no knots"
is not available as a sanity check. The convergence rate is reported per α
instead, which is the check that actually matters here.

**Caught before any number was read**, by sweeping the grid on synthetic data
during implementation and finding every cell either fully dense or fully empty.

## D14 — the published noise floor's margin is restated, not its verdict

**What changed.** `NOISEFLOOR.md` measured the frozen detector's floor by
injecting **white** noise at σ̄ = 0.0618 bl and reported **0.2599/s**, "54.8% of
the corpus's 0.4746/s". The frozen shapeflow calibration says the residual is
not white — lag-1 **+0.5817** over 267 ms — so that floor was measured under the
wrong noise colour.

**What was done.** `scripts/colour_check.py` re-ran the identical arm with the
identical frozen detector and a noise model carrying the **measured** colour, on
the same 60 `tune` animals. Registered prediction 2 set the tolerance at 25%
before the run.

| | floor |
|---|---|
| white (published) | 0.2599 [0.2553, 0.2645] /s |
| measured colour | **0.2977 [0.2918, 0.3039]** /s |
| move | **14.6%** |

**The verdict does not move.** The corpus at 0.4746 [0.4573, 0.4909] /s is still
non-overlapping with the floor, so `NOISEFLOOR.md`'s `PASS` stands and so does
the reading that the detector is above its own noise floor overall.

**One published figure does move and is restated here.** The floor is **62.7%**
of the corpus rate, not 54.8%. `NOISEFLOOR.md`'s headline sentence is therefore
conservative in the wrong direction by 8 points, and anywhere that number is
quoted it should be quoted as 62.7% with this deviation attached.

**Why the floor barely moved.** The threshold is `median + 3·MAD` of the
detector's own scalar, recomputed per recording, so it adapts to whatever noise
it is given. `PLANT.md` measured the same insensitivity to *amplitude* — 4×
moved the rate 6.5%. The floor is a property of the rule, not of the noise, and
this is now measured in both colour and amplitude rather than assumed in either.

**What it does not excuse.** `PLANT.md`'s recovery curve — order-2 at 16 σ̄ ≈
0.99 bl — was also measured under white noise and has **not** been re-run in
colour. Its amplitudes are large relative to the noise in every cell that
matters, so the effect is expected to be small, but that is an expectation and
it is recorded as untested rather than claimed.
