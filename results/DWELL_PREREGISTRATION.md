# Pre-registration — the dwell-matched surrogate

Written and committed **before any surrogate is generated**. The configuration
is fixed below, the reading is fixed below, and the statistic is settled below
rather than after the number is seen.

## 1. The open loop this closes

`FALSIFIER.md` recorded that both surrogates beat the corpus on `rung1 − rung0`
and withdrew `LADDER.md`'s reading of that gap as sequential structure. It also
recorded a confound in the registered statistic that I did not anticipate:

> The surrogates have **2.1× as many runs per second**. Phase randomisation
> preserves the power spectrum but Gaussianises the signal, and a Gaussianised
> mouse never sits still — the corpus's tune p99.9 run length is 259 frames and
> the phase surrogate's is 34.

A per-second statistic counts twice as much of everything for a signal with twice
as many transitions per second. So the falsifier's phase and VAR arms match the
continuous spectrum and then diverge by a factor of two in the discrete stream,
which leaves the comparison measuring **dwell** rather than sequence.

`recur.null.microstate` emits **real visits** and therefore preserves dwell by
construction. This is the run-rate-matched arm that comparison needed.

## 2. The statistic, settled now and not later

**Per-second remains the verdict.** It is what `TOK_PREREGISTRATION.md` and
`FALSIFIER_PREREGISTRATION.md` registered, it is what `LADDER.md` reports, and it
is what the falsifier was read against.

The run-rate decomposition — runs/second, nats/run, and the excess as a fraction
of rung 0 — is reported beside it as the **explanation**, exactly as
`FALSIFIER.md` does.

> **A reversal available only under a statistic chosen after seeing the result is
> not a reversal.** Per-run is hereby registered as a primary statistic **for
> future work only**. It does not apply retroactively to `LADDER.md`, to
> `FALSIFIER.md`, or to this document's verdict, and no past number may be
> re-read under it.

## 3. The two arms

Both are **new `KINDS` in `vieb/tok/surrogate.py`**, generated in the 17-dim
egocentric space on the F3-carried `raw` arm, per recording, never across a seam.

**`microstate` — dwell *and* one-step matched.** A first-order chain fitted over
microstate visits, simulated, with real visits emitted. This is the arm the brief
specifies.

**`microstate0` — dwell matched, sequence destroyed.** The same real visits
emitted in **random order**, drawn from the recording's own visit marginal.

The second arm exists because the first is close to circular for this test:
`microstate` preserves one-step dynamics and rung 1 **is** a one-step model, so a
`FAIL` there would largely restate the null's construction rather than measure
the corpus. `microstate0` is the clean contrast — it holds dwell fixed and
removes sequence, which is exactly the axis in question.

**One trap, registered because it would be invisible:** the order-0 draw is
conditioned on **not repeating the previous state**. Run-length encoding
guarantees `AA` cannot occur in the data, `pooled_chain` zeroes its diagonal for
that reason, and an unconditioned marginal draw would emit self-repeats the
corpus cannot contain — handing the surrogate a transition the tokenizer would
then have to encode.

## 4. The configuration, fixed in advance

| | value | why |
|---|---|---|
| space | 17-dim ego, `raw` / `bodylen` | the F3-carried arm; Wiener is not carried |
| emitter | a new ego-space visit stitcher | `microstate.emit` is coupled to `kendall.decompose` on `(T,K,2)`. Its position/heading integration exists only because recur's channel space carries **absolutes**; the ego space carries shape plus SE(2) *increments*, so joins need no integration and stitching is concatenation |
| reused unchanged | `standardise`, `fit_partition`, `assign_states`, `visits`, `pooled_chain`, `simulate_visits` | all six are already generic over `(T, C)` |
| partition | fitted on the **surrogate's own tune split** | reusing the corpus's centroids answers a different question |
| `sd` | floored where `ego.dead_columns()` is zero | 2 of 17 directions are identically zero by SE(2) construction; dividing by them is a divide-by-zero |
| alphabet | `plain`, N ∈ {256, 512} | the two cells where rung 1 passes on the corpus |
| rungs | 0 and 1 only | rung 2 is closed |
| abstain | the corpus's mask, carried over unchanged | a surrogate has no tracking failures; 0% against 7.03% would confound |
| pipeline | `scripts/quantize.py` and `scripts/ladder.py` **unmodified**, on `VIEB_EGO_DIR` / `VIEB_TOK_DIR` | identical by construction, not by inspection |
| seeds | `vieb.seeds.stable_seed(SEED, kind, tag)` | never `hash()` on a string |
| interval | animal-level, n = 89, 2,000 replicates | `boot.animal_interval`, `how="mean"` |

## 5. The reading, fixed in advance

The comparison is the animal-level interval on `d_corpus − d_surrogate`, where
`d` is per-animal `rung1 − rung0` in nats per second.

| outcome | meaning |
|---|---|
| **corpus still wins** | dwell carries structure beyond first-order dwell statistics. Segmentation has a target. |
| **corpus loses** | the dwell advantage is not behaviour either. A harder finding, and it **bounds what segmentation could recover** — stated here, before Step 1 runs, so it cannot be discovered afterwards |

**The two arms are read separately and both are reported.** `microstate0` losing
while `microstate` wins would say sequence adds nothing beyond dwell.
`microstate` losing is the weaker evidence of the two, for the circularity reason
in §3, and the result document must say so where it reports it.

**No threshold on the size of the gap** — only on whether the interval excludes
zero. A threshold chosen now would be arbitrary; one chosen later would be
selected against the outcome.

## 6. What either outcome does not license

**A win does not mean "the model learned behaviour".** These are two more
alternatives, not all of them, and both are built from the corpus's own frames —
which is what makes them tight on dwell and weak on everything else.

**A loss does not withdraw anything further.** `LADDER.md`'s P1 reading is
already withdrawn; rung 2's failure, `k* = 0`, the hazard shape, `FRAILTY.md` and
`RESOLUTION.md` were not the claim at risk here and are not at risk from this.

---

## 7. Amendment — construction validity, checked before the ladder ran

Added **after the partition was fitted and before any ladder number existed.**

A null that claims to be dwell-matched has to actually be dwell-matched, and the
fitted partition raised a doubt worth settling: at `N_MICROSTATES = 500` the
median microstate visit is **1.0 frame**. Stitching one-frame visits is frame
shuffling, not dwell preservation.

So the property was measured directly, on **three tune animals only**, by
generating a `microstate0` surrogate at several partition sizes and scoring the
run rate of both corpus and surrogate through the corpus's own `plain`/N=256
tokenizer as a common yardstick:

| null granularity `N` | median visit (frames) | surrogate runs/s | **ratio to corpus** |
|---:|---:|---:|---:|
| 20 | 2.0 | 4.797 | 1.253 |
| 50 | 2.0 | 4.715 | **1.232** |
| 100 | 2.0 | 5.005 | 1.307 |
| 200 | 1.0 | 5.195 | 1.357 |
| **500** (registered) | 1.0 | 5.191 | **1.356** |

Corpus: 3.829 runs/s, mean run 7.52 frames.

**`N = 500` is kept.** It was chosen before this measurement, matching recur's
`microstate_N500k1` arm, and the sweep shows the choice cannot turn a verdict:
the run-rate ratio moves only between **1.23 and 1.36** across a 25-fold range of
granularity, against the **2.1** that made the phase and VAR comparison a dwell
confound. Picking `N = 50` for a 0.13 improvement would buy little and would look
like — and partly be — selecting a null parameter after measuring it.

**What this does license, and what it does not.** The arm is entitled to be
called far better dwell-matched than phase or VAR, and it is **not** entitled to
be called dwell-matched exactly: the residual is ~1.36 and is reported with every
number this arm produces. The yardstick above is the corpus's own tokenizer,
used only as a common ruler for this check; the ladder itself fits the
surrogate's own partition, as §4 requires.

**No ladder number was computed, inspected, or available when this was written.**
