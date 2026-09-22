# Stages 2 and 3 — the subsample test is confounded, and the island is unitary

Registered in `results/VALIDATION_PREREGISTRATION.md`, committed before
`scripts/subsample.py` or `scripts/island_split.py` existed. Stage 2 on `tune`,
60 animals; Stage 3 on `report`, 89 animals, 361 clump-0 segments.
Digest `198eb14ff258c7f6`.

---

# Stage 2 — frame-rate robustness

**The first frame-rate sweep this programme has run.** Everything temporal
derives from `recur.util.frames(seconds, fps)`, so passing 15 fps re-derives
every window in *seconds*.

| stream | band | survival at 15 fps | rate 30 → 15 fps |
|---|---:|---|---|
| `config` | ±2 | **0.2604 [0.2548, 0.2662]** | 0.4746 → 0.2975 /s (0.627×) |
| `dyn` | ±15 | 0.3194 [0.3071, 0.3317] | 0.2356 → 0.1711 /s (0.726×) |

| # | prediction | outcome |
|---|---|---|
| 1 | configuration survival above 0.5 | **FAIL** — 0.2604 |
| 2 | dynamics survives less than configuration | **FAIL** — and ill-posed, see below |
| 3 | neither rate rises | **PASS** |

## The test is confounded, and the confound is in the detector

> **`breaks.identifiability_floor` is `k·(degree+1) = 12` SAMPLES, not
> seconds.** So the detector's refractory period is **0.533 s at 30 fps** and
> **0.933 s at 15 fps** — **1.75× longer**. The 15 fps detector is mechanically
> forbidden from placing boundaries the 30 fps detector places.

That is most of the effect. The observed rate ratio is **0.627×**, and
1/1.75 = 0.571 — the refractory period alone predicts nearly the whole drop.

**So 0.2604 is a lower bound on survival, not a measurement of it**, and
prediction 1's failure may say more about the floor's units than about whether
the boundaries are real. The result is reported at the strength that leaves it:
**this test did not settle frame-rate robustness.** A corrected version, holding
the refractory period fixed in *seconds*, is owed and is not run here — changing
it now, after seeing the number, is the move every registration in this
programme exists to prevent.

**Prediction 2 was ill-posed and is withdrawn rather than read.** The two
streams are matched at ±2 and ±15 — a 7.5× difference in matching window — so
"dynamics survives more" is what a wider band buys, not a fact about the
streams. The bands were registered in advance and correctly reflect the two
instruments' measured precision; using them for a *between-stream* comparison
was the error, and it is mine.

**Prediction 3 held and is the one clean result here.** Neither rate rose when
half the data was discarded — 0.627× and 0.726×. A rate that rose would have
meant the detector was reading the sampling grid.

---

# Stage 3 — does the island split?

`BEHAVIOUR.md`'s `chaining_clump0` FAILs — typical pair 0.398 against
θ = 0.190, **2.1×** — so clump 0 is a single-linkage chain, and a chain is the
shape of something with sub-types. All **361** clump-0 segments were
duration-matched to non-clump-0 partners from the same animals.

| channel | island BIC gain | null BIC gain | **excess** | island median | null median |
|---|---:|---:|---:|---:|---:|
| `body_extension` | +23.44 | +32.94 | **−9.50** | 1.0083 | 0.9989 |
| `pole_radius` | −5.49 | +3.12 | **−8.62** | 0.6749 | 0.6742 |

| # | prediction | outcome |
|---|---|---|
| 4 | `body_extension` bimodality exceeds the null | **FAIL** — excess −9.50 |
| 5 | `pole_radius` bimodality does **not** exceed the null | **HELD** — excess −8.62 |

> **The island does not split on either channel, and it is *less* bimodal than
> its duration-matched partners on both.** On the registered reading it is a
> **unitary slow freezing state**.

The registered expectation — that it would split on configuration, crouched
freeze against stretch-attend — is refuted. The medians are also nearly
identical to the null's (1.0083 against 0.9989; 0.6749 against 0.6742), so the
island is not displaced on either axis, merely more homogeneous along it.

**This is stronger than "no split found."** A null result from a search for
structure usually means the search was underpowered. Here the comparison set —
the same animals' own segments, matched on duration — is *more* bimodal than
the island on both channels, so the island is not a mixture of two postural or
two dynamical types.

**It does not contradict the chaining FAIL, and the two are not about the same
space.** Chaining is measured in the 40-point time-normalised ego space where
the clump was built; this is measured in two one-dimensional projections chosen
in advance. Clump 0 can be a chain in the first and unimodal in both of the
second. What this rules out is the specific hypothesis that the chain is two
named postural sub-types.

---

## What these two stages license

**License:** the island is homogeneous in body extension and in pole
persistence relative to a matched control, and neither detector's boundary rate
rises when the frame rate is halved.

**Do not license** any claim that the frozen detector's boundaries are or are
not frame-rate robust — the test that was supposed to settle it is confounded
by its own refractory floor.

**Do not license** any between-stream survival comparison.

**Do not license** calling the island freezing. `CONTEXT.md`'s position is
unchanged: that needs the context-to-shock mapping and an ethogram, and this
repository holds neither. Nothing here was scored as shock versus no-shock.

## What is owed

1. **A frame-rate test with the refractory period held in seconds**, and its
   own registration. Until then frame-rate robustness is untested, not failed.
2. **A `white` negative control for the dynamics stream.** Registered as
   prediction 5 of `DYNAMICS_PREREGISTRATION.md` and still not run; a new
   detector re-incurs `DETECTOR_PREREGISTRATION.md:84-90` and does not inherit
   it.
3. **A like-for-like survival comparison**, both streams at one band, if a
   between-stream statement is ever wanted.
