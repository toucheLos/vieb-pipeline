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
