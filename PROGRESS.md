# Progress — VIEB pipeline

Plan: `~/.claude-1/plans/coding-prompt-streamed-milner.md`.

**Video Interpreter Excluding Bias.** Find the behaviours, find the patterns,
predict them to a reliable degree. This repo owns the tokenizer + hazard-model
programme and the cleaning work under it.

| # | Stage | State | Verdict | Artifacts |
|---|---|---|---|---|
| 0 | Repo split from `~/recur`, provenance frozen | **done** | `PASS` (digest `198eb14ff258c7f6`) | `results/inherited.json`, `vieb/io/spine.py` |
| 1 | Bone-length violations + ExBias R² join | **done** | `GRID_LIMITED` (2.13%, correct branch) · join `PASS` | `results/bones.json`, `BONES.md` |
| 2 | Egocentric SE(2) transform + A4 parity | **done** | A4 `PASS` (exact 1.000000) · leak `PASS` | `results/ego_{bodylen,raw}.json`, `EGO.md` |
| A | What the cleaning actually changes | **next** | — | — |
| B | Cleaning bakeoff | planned | — | — |
| C | Before/after video → VIEB Atlas | planned | — | — |
| 3 | Coarse alphabet + RLE | planned | — | — |
| 4 | Rungs 0–2, hazard + MDL | planned | — | — |

**Tests:** 122 passing, CPU only, `pytest tests/`. `mypy --strict` clean over
`vieb/tok` and `vieb/qc`.

---

## How this repo relates to `~/recur` and `~/shapeflow`

**Hash the artifacts, import the utilities.**

recur reads shapeflow's *artifacts* and never its *code* — `qc/bones.py` even
re-implements `pair_indices` rather than importing it. That rule is about **data
provenance**: shapeflow is a live repo whose stages could re-run underneath,
changing a number without changing anything a reader could see. It is not a rule
against reusing a verdict type.

So this repo splits the two:

| | treatment |
|---|---|
| recur / shapeflow / exbias **artifacts** (`.npz`, `.json`) | content-hash contract in `vieb/io/spine.py`, frozen once, re-checked in every job preamble |
| recur's **utility layer** (`Read`, `boot`, `splits`, `anchors`, `util`, `labels`, `kendall`, `geom/*`, `qc/swap`) | imported via `PYTHONPATH` |

**Nothing is copied.** Forking `Read` would give the project two verdict
vocabularies; forking `splits` would give it two copies of a split whose whole
point is being fixed once and never resampled.

One divergence from recur's spine, deliberate: the digest excludes `git_sha`.
Upstream folds it in, so an unrelated commit changes the label every shard
stamps, and two shards computed on byte-identical data end up looking different.
The digest answers *was this the same data*; the git sha answers *which commit
produced it*. Both are recorded; only the first is the label.

The venv is a **symlink to recur's** — the pins are identical by design and two
copies would be ~5 GB of duplicated torch. `pip install` from either repo lands
in both, so anything installed here must be added to both `requirements.txt`
files. `env.sh` says so.

---

## Step 1 — bone-length violations

Skull-triangle violations run **2.134%** of frames at ε = 0.10 (unfiltered pose,
raw metric) — inside the 2–15% band, so *correct rather than exclude*, and cost
out an ensemble-DLC path. Shuffled-keypoint ceiling 79.6%, a 37× margin.

Four findings the sweep produced that were not in the brief:

1. **The excess is not perspective.** Removing the per-frame common scale does
   not lower the rate (2.134% → 2.319%); it raises it slightly, at every ε. The
   two-metric design existed to test exactly this and it settles against the
   rearing hypothesis.
2. **There is no elbow.** Relative drops run 34%, 37%, 44%, 54% — monotonically
   increasing, one population thinning out. No ε on this grid is a principled
   cut, and the branch itself moves with the threshold: at ε = 0.20 the rate is
   1.203% and lands in the *exclude* branch instead.
3. **The R² join reverses under its own confound.** ExBias's R² is largely a
   readout of segment length (mean duration 4.00 s in the worst decile against
   0.53 s in the best; ρ(duration, R²) = −0.538), and the violation rate's
   denominator *is* segment length. Marginally ρ = −0.054 and the join reads
   `GRID_LIMITED`; with duration held fixed ρ = **−0.104 [−0.115, −0.094]** and
   it reads `PASS`. The confound was suppressing the association, not creating it.
4. **The Wiener filter absorbs 23% of them**, so **1.648%** of frames in the
   feature space every downstream stage consumes — Q1's included — carry
   impossible skull geometry. Shrinkage makes an excursion smaller without making
   the frame correct. This is why the artifact ablation at Step 4 is mandatory.

`results/BONES.md`.

---

## Step 2 — the egocentric transform

**A4 passes exactly.** Closed-form recovery of centroid speed and body-axis
angular velocity from (s, ξ, ℓ_a) reads **R² = 1.000000 on both**, over 6,403,616
frames in 1,149 `report` recordings, against thresholds of 0.98 and 0.94. The map
is a bijection given the equivariance it removed. 17 dims in, 17 out.

1. **A4 cannot catch the error the brief warns about.** Substituting separate
   differencing for the SE(2) logarithm costs almost nothing until the animal
   turns hard — speed R² 0.9973 at 0.9 rad/s, 0.9853 at 3 rad/s, breaching 0.98
   only above ~4.5 rad/s. The guard is `tests/test_se2.py`, which checks
   `se2_log` against `scipy.linalg.expm` and fails the naive version at every
   turn rate including zero.
2. **The brief's ξ is dimensionally wrong for the rotation.** f/ℓ_a on ω gives
   rad·s⁻¹·px⁻¹ and makes a large mouse's turning read systematically slower.
   Translation takes f/ℓ_a, rotation takes f alone.
3. **Body-length normalisation works, for a reason the brief got wrong.** It cuts
   animal identification from **15.0% to 6.9%** against 0.336% chance, and
   session identification from 17.9% to 1.4%. Quoted in **accuracy, not nats**:
   both `raw` probes stopped on `max_iter`, so their nats figures are refused.
   Body length spans 89–172 px over 298 animals, a 1.92× range. The brief asked
   for this to fix "5.88 nats", which was the previous instrument's number.
4. **The identity-leak metric can hide a leak entirely, and did.** The `raw`
   session probe reads 0.0000 nats while picking the right session out of 3,846
   **17.9%** of the time, 687× chance. `identity_leak` returns
   `max(log N − CE, 0)` and CE exceeds log N when the probe is confidently wrong
   — which it is, because the solver stops on `max_iter=400`. Fixed in
   **recur**, where the defect lives: `leak_read` now refuses an unconverged
   probe and a clamped zero above 3× chance, and `identity_leak` records
   `converged`/`n_iter`/`max_iter`.
5. **Single-frame pose-only recovery is negative** (−0.185 / −0.065) held out by
   animal — worse than predicting the test mean.

Rank is **11 of 14 on every one of the 298 animals**, exactly SE(2)'s three
degrees of freedom. The reversal audit runs **10 checks** — recur's seven,
composed with three for the twist, which is odd in every component and needs a
one-frame shift because it lives on the interval between frames.

`results/EGO.md`.

---

## Open

Phase A is the next thing and it is the one nobody has done: **what does the
cleaning actually change?** The bone check only flags; the gap policy and the
Wiener filter are what move the data, and a spot measurement puts the filter at a
median 1.77 px / p90 6.25 px / max 278.8 px with 86.2% of keypoint-frames
displaced by more than 0.5 px. Neither upstream repo records that anywhere.
