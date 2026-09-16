# Step 3 — a continuum with a few real islands. **Step 4 does not run.**

Registered in `results/VOCAB_PREREGISTRATION.md`, committed at `e91d6fb` before
any neighbour graph existed. Source array: **`work/ego/raw__bodylen__*.npz`**,
the F3 `raw` arm. 89 report animals, `k_mad = 3.0`, segments time-normalised to
40 points, embedded at full dimension. θ **inherited** from
`results/seg_recur.json` — the null's own quantile at `NULL_RATE = 0.01` — so no
new free parameter entered at this stage.

## The headline: the coverage read refuses, on every group

| group | clumps | coverage | participation |
|---|---|---|---|
| **shape** | `PASS` — 19 vs the nulls' 2 and 4 | **`NOT_A_RESULT`** — 98.1% unassigned | `PASS` — 1 clump in ≥ 50% of animals |
| **both** | `PASS` — 15 vs 3 and 2 | **`NOT_A_RESULT`** — 98.9% unassigned | `FAIL` — best clump spans 25.8% |
| twist | `FAIL` — 1 vs 1 and 1 | **`NOT_A_RESULT`** — 99.0% unassigned | `FAIL` — best spans 36.0% |

**There is clump structure the dwell-matched nulls do not reproduce, and it
covers under 2% of segments.** Both halves of that sentence are the result.

The registered verdict for a coverage above 90% is `NOT_A_RESULT`, and its
reason was written before the number existed: *"a vocabulary built on them would
describe a fraction of behaviour while appearing to describe all of it."* That
is exactly the situation.

## The decomposition that makes it concrete

| group | nearest cross-animal neighbour within θ | in a clump of ≥ 20 | segments |
|---|---:|---:|---:|
| shape | 3.31% | **1.86%** | 64,341 |
| both | 2.40% | **1.13%** | 57,406 |
| twist | 1.04% | 1.01% | 65,993 |

Step 2's excess is real and it lives here: a few per cent of segments have a
genuinely close partner in another animal. But those partners mostly form
**pairs and small chains**, not groups — only about half of the linked segments
end up in a component of twenty or more.

## The structure is not what a null produces

| group | arm | clumps | unassigned | two-component BIC gain | BC | excess kurtosis |
|---|---|---:|---:|---:|---:|---:|
| shape | **corpus** | **19** | 98.14% | **+21,924** | 0.420 | +14.49 |
| | `microstate` | 2 | 99.84% | +10,369 | 0.262 | |
| | `microstate0` | 4 | 99.84% | +20,098 | 0.340 | |
| both | **corpus** | **15** | 98.87% | **+21,621** | 0.414 | +19.62 |
| | `microstate` | 3 | 99.79% | +10,966 | 0.319 | |
| | `microstate0` | 2 | 99.88% | +18,317 | 0.375 | |
| twist | corpus | 1 | 98.99% | +36,514 | 0.438 | +42.12 |
| | `microstate` | 1 | 99.08% | +28,593 | 0.393 | |
| | `microstate0` | 1 | 99.09% | **+42,357** | 0.439 | |

On shape the corpus produces 19 clumps where the nulls produce 2 and 4, through
the identical graph at the identical θ. **On twist the null beats the corpus on
the BIC gain** (+42,357 against +36,514), which is why twist reads `FAIL` — and
it is the registered rule doing its job: clumpiness a null reproduces is not
clumpiness.

**The bimodality coefficients are not read alone and should not be.** Every arm
carries a large **positive** excess kurtosis — +14.5 on shape, +42.1 on twist —
which is the signature of a heavy tail rather than of two peaks. A genuinely
two-peaked distribution has *negative* excess kurtosis. The registration
anticipated this: BC rises on heavy tails as well as on two peaks, an
exponential scores 0.58, and this corpus's distances are heavy-tailed for
reasons `ROUGHNESS.md` documents. **So the BIC gain and the clump counts carry
the verdict, and the BC values are reported only because they were registered.**

## The one entry that survives every test

On `shape`, a single clump of **361 segments present in 46 of 89 animals**, with
no animal supplying more than **16%** of its members. It is the only clump in the
whole study that reaches the registered 50% participation floor, and its low
top-animal share means it is not one animal's habit wearing a crowd's clothes.

The next largest spans 23 animals (173 segments, top share 28%), then 16
(89 segments, 44%). On `both`, the best spans 23 of 89 — below the floor, which
is why `both`'s participation read `FAIL`s while `shape`'s passes.

Mean animal fraction over shape's 19 clumps: **0.098 [0.056, 0.155]**,
animal-level interval. Most clumps are local to a handful of animals.

## What this is: a continuum with a few islands

Not a vocabulary. A vocabulary would assign most segments to an entry; this
assigns under 2%. Not a pure continuum either — a pure continuum would not
produce 19 components where its dwell-matched surrogate produces 2, and would
not contain a group of 361 segments shared by half the animals.

**Local neighbourhoods, plus a small number of genuinely shared, tight
islands.** That is enough for averaging power without a global partition, which
is what the registration said a continuum verdict would license, and it is the
first such claim here made by an instrument that could have returned the
opposite — it did return the opposite on `twist`, where the null won.

## Assignment is 38× clustered in time, which is a finding in its own right

Before deciding anything about Step 4, the adjacency was measured rather than
assumed. Consecutive segments within an animal:

| group | adjacent same-animal pairs | **both assigned** | expected if independent | enrichment |
|---|---:|---:|---:|---:|
| shape | 64,252 | **1.326%** (852 pairs) | 0.035% | **38×** |
| both | 57,317 | **0.803%** (460 pairs) | 0.013% | **63×** |

**When one segment falls in a clump, its neighbour usually does too.** At 1.9%
coverage, chance would put both members of a pair in clumps 0.035% of the time;
the measured rate is 38 times that. The islands are not scattered — they arrive
in runs.

That is independent support for the islands being real, and it is the most
encouraging number in this document. It is also not a vocabulary.

## Step 4 does not run, and the reason is a count

Step 4 was gated on Step 3 finding clumps. It found some; they cover 1.9% of
segments; and their temporal clustering gives **852 mergeable adjacent pairs on
`shape`, 460 on `both`**.

That is the whole input to the Step 4 measurement. Byte-pair merging selected by
**held-out MDL on the per-animal distribution** against a **surrogate merged
under identical rules** needs, at minimum, enough pairs per animal to estimate a
distribution: 852 pairs spread over the 46 animals that participate is about
eighteen each, before any held-out split. This programme's standing rule refuses
a stratum below 20,000 scored keypoint-frames rather than reporting it with a
caveat, and this is three orders of magnitude under that.

The remaining 98% of the stream is a single `unassigned` symbol. Merging in that
stream would merge `unassigned` with `unassigned`, the MDL would select merges of
a symbol meaning "no symbol", and the surrogate control would be decided by the
behaviour of a placeholder.

**So Step 4 is refused on sample size, not on principle**, and the refusal is a
measured one. The alternative — dropping `min_size`, or widening θ until coverage
rose — would be choosing a parameter against an outcome after seeing the outcome,
and both are registered.

**What would unblock Step 4** is not a different clustering. It is more segments
falling within θ of each other, which means either a detector whose boundaries
are more reproducible across animals, or a distance that is not dominated by the
heavy tail these distributions all carry. Both need their own registration.

## What this licenses, and what it does not

**Licenses:** there are groups of segments, found without assuming how many,
that two dwell-matched nulls do not reproduce under the identical graph; on
shape one of them is shared by 46 of 89 animals; and they cover under 2% of the
corpus.

**Does not license:** calling them behaviours, naming them, calling the set a
vocabulary, or building a tokenizer on them. It does not license reading
`twist`, where the null won. And the 2% coverage is not a bound on how much of
behaviour is stereotyped — it is a bound on how much **this detector's segments,
under this distance, at this θ** fall into groups of twenty or more.
