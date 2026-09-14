# F3 — the preprocessing default for the MDL branch

**Decided 2026-09-14.** This record closes the question. The Step 4 preprocessing
decision has reversed twice; the stopping rule below names the only two things
that may reopen it.

## 1. The decision

**Carried onto the MDL branch:**

| arm | what it is |
|---|---|
| `raw` | the gap-policy output, `held_array(pose_unfiltered, missing)`. The floor. |
| `viterbi` | Anipose 1.1.24 path selection, `n_back = 3`, `thres_dist = 30 px` |
| `disposition` | Phase D: correct runs ≤ 3 frames, abstain on longer |

**Reported and not carried:** `median_0.50`.

**Not benchmarkable, therefore not carried:** `wiener`, `butterworth`. Both are
read off shapeflow's disk and were never reimplemented here, so neither can be
applied to a corrupted array. The incumbent's recovery performance is unknown and
this record does not claim otherwise.

## 2. The evidence, with provenance per claim

Every figure below is computed on `held_array(pose_unfiltered, missing)` — the
pre-smoother array — unless the row names an arm, in which case that arm's output.

**The per-stratum crossing** — `STRATIFIED.md`, `results/injection_1e6.json`,
`results/injection_mp_1e6.json`. Net error against injected truth, slowest →
fastest, at the widest pool, with the corrected park distribution:

| arm | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| `median_0.50` | −0.0026 | −0.0013 | −0.0003 | **+0.0005** | **+0.0053** |
| `viterbi` | −0.0016 | −0.0018 | −0.0018 | −0.0023 | **−0.0030** |
| `disposition` | −0.0012 | −0.0013 | −0.0015 | −0.0017 | −0.0022 |

All five strata exceed the registered 20,000-keypoint-frame minimum; the smallest
is 472k. Positive means the arm left the data **further from the truth than it
found it**. This claim does not depend on the pool being representative — it is a
within-stratum comparison.

**`viterbi`'s threshold-invariance** — `STRATIFIED.md`. Its per-stratum profile
moves by at most **0.0007** across the four pool thresholds, against nets of
~0.0019; `disposition`'s by 0.0002. `median_0.50`'s moves by **0.0087**, larger
than any of its own net values. So for the two carried arms the pooled threshold
effect is purely compositional — the arms behave identically and only the pool
changes. For the median it is not.

**The PARK_FRAMES outcome** — `PARK_RESULT.md`, registered in
`PARK_REREGISTRATION.md` before running. The benchmark's most serious known bias
ran against `median_0.50`: injected parks were drawn from a post-filter
distribution, so 14.8% fell under the median's 7-frame cliff where **87.6%** of
real parks do. Corrected, the median's park repair rises 8.4% → **58.1%** and its
net improves by 0.0007–0.0008 at every threshold. **The crossing survives**: s5
improved by 0.0015 against the **0.0093** required to reverse it, and s4 remains
positive.

**The post-stratified estimate is refused, not missing** — `STRATIFIED.md`. The
stratifying variable is segment median speed, and a segment is *defined by* the
pool criterion, so the corpus has no marginal to reweight to. Two further
obstacles are recorded there.

**What is not claimed.** `median_0.50` is the best arm **pooled** at the banked
threshold with corrected parks (−0.0033 against `viterbi`'s −0.0019). It is
excluded on the per-stratum result, not on the pooled one, and that is the whole
basis of the decision: it is a good teleport-and-short-park remover that is
harmful on a moving animal. On a fear-conditioning corpus the moving frames are
the signal.

## 3. What was corrected to get here

| | finding | where |
|---|---|---|
| **Seed class audit** | `hash()` on a `str` is salted per process, so Phase F's `SEED = 0` controlled nothing. Swept every site in two repositories, classified **by demonstration** in three subprocesses. **The shard-hash architecture is real** — `spine.py` is sha256 throughout, the animal bootstrap and Q1 surrogates take integer seeds. One load-bearing site fixed here, one in `recur` recorded and deliberately left. | `SEED_AUDIT.md` |
| **Pooled/stratum reconciliation** | The arithmetic closes exactly: pooled = frame-weighted stratum sum, −0.000033 both ways. The apparent gap was a comparison against an *unweighted* mean, which has the **opposite sign**. | `DEVIATIONS.md` D2 |
| **The sixth stratum** | `assign_bin`'s **refusal code** (−1) was sorted first and printed as the slowest stratum on 235 keypoint-frames. **Two published findings came from it and are withdrawn**: "`viterbi` is inert in the slowest stratum at 0.0000" (really −0.0014) and "`median_0.50` runs −0.0331" (really −0.0021). | `DEVIATIONS.md` D1 |
| **`recovery_read` control** | The guard had fired for the first time with no recorded true negative. `viterbi` at `off` now pinned as one, with a positive control beside it. | `tests/test_strata.py` |
| **Arm coverage** | All four arms now carry full per-threshold and per-stratum numbers. `disposition` is no longer recommended on absent numbers. | `STRATIFIED.md` |

## 4. The stopping rule

**This decision is closed on the evidence above.** Exactly two things reopen it:

1. **A seed or provenance finding that invalidates the inputs.** `SEED_AUDIT.md`
   gave shard identity and the Q1 surrogates a clean bill; a later finding that
   contradicts it reopens this.
2. **A PARK_FRAMES outcome that reverses the per-stratum crossing**, under its
   own registration. Task D ran and did not reverse it. A further correction to
   the corruption model that does would reopen this.

**Nothing else reopens it.** Not a new candidate method, not a new filter, not a
better idea, not a result on the pooled axis, not an argument from first
principles about what a filter ought to do. The decision has reversed twice
already; a third reversal on anything other than the two items above would mean
the evidence base is not what this record says it is.

Methods deliberately excluded, with their reasons recorded elsewhere and **not**
grounds for reopening: keypoint-MoSeq as a cleaner (`METHODS_CLEANING.md` — it is
a contestant in the model comparison), EKS and ensemble-DLC (no ensemble exists;
`~/dlc-training/trained_dlc` holds zero files), the anatomical projection
corrector and the pose-subspace predictor (unbuilt by choice).

## 5. The line

**Every downstream result is tagged with which side of this freeze its
preprocessing falls on.**

The mechanism already exists and is not reinvented here: `vieb/io/spine.py`
hashes the inherited artifacts and `anchors.header` stamps every result
document. Results computed after 2026-09-14 carry
`preprocessing_freeze: "F3"` and the arm name in their header; results computed
before it carry neither and are pre-freeze by absence.

Inherited digest at the time of freezing: **`198eb14ff258c7f6`**.

Pre-freeze artifacts that remain valid and are **not** re-run: `results/q1*`,
`results/bones.json`, `results/ego_*.json`, `results/cleaning_report.json`,
`results/effect.json`, `results/continuity.json`, `results/disposition.json`.
Each was computed on the Wiener array or on the gap-policy output, which
`METHODS_CLEANING.md` records per document.

**Q1 was not re-scored at any point in F3.**
