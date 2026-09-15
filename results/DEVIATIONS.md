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

## D5 — the ladder's identity probe is a reduced one

**Registration:** `TOK_PREREGISTRATION.md` §6 — *"the next-state distribution
goes through `recur.audit.leak.leak_read`, held out by window within recording."*

**What ran:** the per-window **mean and SD of the model's own per-run code
length**, through the same `leak_read` at the same holdout.

**Why:** the next-state distribution is an `N`-vector per run. At N = 2,048 and
~4.3M scored runs that is tens of gigabytes per cell, times sixteen cells, for a
gate rather than a headline.

**What it costs.** The reduced probe asks whether *how expensive and how variable
the model finds a window* identifies the animal. That is a property of the
predictions rather than of the labels, so it is the right kind of question — but
it carries far less information than the full distribution, so it is **strictly
weaker**. A `PASS` from it does not establish the absence of style leakage, and
`LADDER.md` says so where it reports the probe rather than only here.

In the event ten of the sixteen probes returned `NOT_A_RESULT [DEGENERATE]` on
the inherited `max_iter = 400` convergence guard, so the gate is largely
**refused** rather than passed, and nothing downstream leans on it.

## D6 — the duration bin width was off by one, and it was a real bug

**Registration:** `TOK_PREREGISTRATION.md` §3 — duration charged at one-frame
resolution, `delta = 1/30` s, identically in every arm.

**What happened:** `ladder.duration_pmf` spread each bin's mass uniformly over
`e[b+1] - e[b] + 1` whole durations where the correct count is
`e[b+1] - e[b]`. The resulting `f(d)` summed to **1.115** rather than 1, so every
duration charge was wrong by a constant of roughly 0.11 nats.

Caught by `tests/test_mdl.py::test_the_duration_pmf_normalises_over_frames`
**before any ladder job was submitted**, and fixed. The pmf now sums to 1 to
within 1e-9. No published number was computed under the broken version.
