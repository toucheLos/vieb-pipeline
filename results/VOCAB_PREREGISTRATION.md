# Pre-registration — is there a vocabulary, or a continuum?

Written and committed **before any neighbour graph, any component and any clump
count exists**. Step 2's gate passed on all three channel groups
(`results/SEGRECUR.md`), so this runs.

## 1. The question, and why both answers are results

Segments recur across animals more than length-matched random windows cut from
the same signal. That says the units are real; it does not say they are
*discrete*. Two structures produce the same recurrence excess:

* **a vocabulary** — the recurring segments fall into groups with gaps between
  them, and the groups are the entries;
* **a continuum** — segments recur by lying near other segments, with no gaps,
  so there are local neighbourhoods and no global partition.

**Both are reportable and neither is failure.** A continuum verdict is enough for
averaging power without a partition, and it would be the first such claim in
this literature made by an instrument capable of returning the opposite.

## 2. No assumed count, and no clustering that requires one

ExBias's stack returned `n_states = 0` and self-diagnoses why: MiniBatchKMeans
centroids tile at uniform density, the opposite of what a density-based
clusterer needs. It is not the model and nothing here fits `k` anything.

**Clumps are connected components of a neighbour graph.** Each report segment is
linked to its `k = 10` nearest **cross-animal** neighbours whose normalised
distance is below **θ**, and clumps are the connected components with at least
**20** members.

**θ is inherited, not chosen.** It is the same threshold Step 2's statistic
already used: the **null's own quantile** at `NULL_RATE = 0.01` of its
normalised cross-animal nearest-neighbour distances, read out of
`results/seg_recur.json`. No new free parameter enters at this stage.

**Cross-animal only**, for the reason the recurrence statistic is: a segment's
own animal supplies near-copies of itself, and a clump built from those would be
one animal's habit rather than a shared unit.

`k = 10` bounds the graph so a dense region cannot blow it up. A segment with
more than ten neighbours inside θ keeps its ten nearest, which can only ever
**split** a clump, never merge two — so the bound is conservative for the
vocabulary hypothesis.

`min_size = 20` is not a size prior on behaviour. It is the point below which
"present in N of 89 animals" cannot be estimated at all.

## 3. Everything is computed identically on both nulls

A neighbour graph thresholded at a fixed quantile produces components in **any**
point cloud, a Gaussian one included. So every statistic below is computed on
`microstate` and `microstate0` through the identical code, each arm divided by
its own ambient scale, and **clumpiness a null reproduces is not clumpiness**.

`microstate` remains the **weak** arm per `DWELL.md`.

## 4. The three statistics, fixed now

**Component structure**: number of clumps, unassigned fraction, largest-clump
fraction, size distribution.

**Shape of the nearest-neighbour distance distribution**: the bimodality
coefficient with its **excess kurtosis** reported beside it, and the BIC gain of
a two-component over a one-component Gaussian mixture. BC rises on heavy tails
as well as on two peaks — a standard exponential scores 0.58 — and the corpus's
distances are heavy-tailed for reasons `ROUGHNESS.md` documents, so BC alone is
not read.

**Cross-animal participation per clump**: the number of distinct animals, the
animal fraction, and the **top animal's share of the clump's members**. The last
travels with the count because a clump can sit in many animals while one supplies
most of it, and that is not a shared behaviour.

## 5. The verdicts, fixed in advance

| read | `PASS` | `FAIL` |
|---|---|---|
| **clumps** | the corpus exceeds **every** null on **both** the clump count and the two-component BIC gain | it does not — **a continuum**, reported as such |
| **coverage** | the unassigned fraction is at or below 90% | above it: `NOT_A_RESULT`, because a vocabulary covering a tenth of behaviour while appearing to describe all of it is worse than none |
| **participation** | at least one clump is present in ≥ 50% of animals | no clump is — they are artifacts of whichever animals dominate the sample |

Requiring the corpus to beat every null on **both** statistics is deliberate: a
null matching it on either one is enough to withhold the `PASS`.

**Ranking is on the lower bound of an animal-level interval, never on
p-values**, which saturate at 1.28e-16 at n = 89 and would rank on sample size
rather than on effect.

## 6. The unassigned bin is a headline number

Reported in the first table, not in a footnote. **Any method that assigns 100%
of its input is misreporting its coverage**, and the fraction of behaviour a
proposed vocabulary does not describe is the first thing a reader needs.

## 7. What a PASS would and would not license

It would license: there are groups of segments, found without assuming how many,
that the dwell-matched nulls do not reproduce, and some of them are shared across
most animals.

It would **not** license naming them, claiming they are behaviours a human would
recognise, or claiming the vocabulary is complete — the unassigned fraction
bounds that directly. Step 4 may then merge adjacent pairs and charge the result
by MDL; nothing here licenses Step 4's merges being meaningful either.
