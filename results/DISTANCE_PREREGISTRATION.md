# Pre-registration — does the distance make the clumps?

Written and committed **before either metric is computed on the corpus**. It
fixes the two distances, the search, the candidate budget and the reading.

## 1. Why this runs before anything else

Step 3 found 19 clumps the dwell-matched nulls do not reproduce, covering 1.9% of
segments. The distance that found them **time-normalises every segment to 40
points**, so duration is free by construction: two movements of the same shape at
different tempos score as identical. Matched partners differ by a mean factor of
**3.7×** in duration.

So the clumps may be *same shape, any tempo* — a property of the ruler rather
than of the animal. If they are, raising coverage produces more of the same
artifact and Step 4 would tokenize something that is not a behavioural unit.
**Nothing downstream is worth running until this is settled.**

**And the distance evidence that previously bore on this is withdrawn**
(`DEVIATIONS.md` D8): it compared a per-frame RMS against a scale-normalised
560-dimensional norm. Only the duration gap survives, and it is suggestive rather
than decisive. This stage supplies the missing ambient scales.

## 2. The two metrics

| arm | definition | duration |
|---|---|---|
| **`open`** | compare over `m = min(T_a, T_b)`, no warp, normalised per frame | **still free** — a short segment matches a long one's prefix |
| **`union`** | compare over `M = max(T_a, T_b)`, the shorter **held at its last frame**, normalised per `M` | **charged** — unmatched tail counts as mismatch |

`open` is `vieb/seg/embed.py:open_end_distance`, the secondary
`SEGRECUR_PREREGISTRATION.md` registered and never ran. **It is the registered
primary here and the registration is not being rewritten after the fact.**

`union` is registered as the **decisive diagnostic**, because `open` compares
only the shared prefix and therefore leaves duration free too — it tests whether
the *warp* matters, not whether *duration* does. Only `union` can drive the
matched-partner duration ratio toward 1.

Hold-last is the convention `vieb/clean/arms.py:held_array` already uses for
gaps. The padding is not a new invention and it is not zero-padding, which would
charge the tail against an origin the ego space does not have.

**If the two disagree, that disagreement is the finding**, and it is reported as
such rather than resolved by preferring one.

## 3. The search, and the approximation named in advance

Both metrics have a **per-pair extent**, so neither is a fixed-dimension
Euclidean distance and `recur.recurrence.search.search` cannot run on them.
Segment lengths span 16–4,537 frames (median 28); a full exact pass over ~64,000
queries against ~200,000 bank entries is not feasible.

**Retrieve, then rerank exactly:**

1. **Retrieve** in a fixed-prefix space — every segment truncated to **L = 16
   frames**, the identifiability floor, which retains **100%** of segments.
   Rectangular, so `bank.build`, `sort_by_animal` and `se.search` run
   **unmodified** through `embed.stack_segments`.
2. **Rerank** all candidates with the true metric; keep the true nearest.
   **Candidate budget `k = 64`, fixed now.**
3. **Measure the approximation rather than caveat it.** Run the full exact
   per-pair search on a reduced bank of ~4,000 segments and report the fraction
   of true nearest neighbours the candidate stage recovered, in the spirit of
   Q1's `exactness_read`.

This departs from Q1's "exact leave-one-animal-out, no approximate index" **at
the retrieval step only**, and the departure is registered here rather than
discovered later.

## 4. Each arm divided by its own ambient scale

A `randpairs` pass **per metric, per channel group**: the median distance between
**random cross-animal pairs** in that arm, which has no nearest-neighbour
feedback. Without it the comparison measures spread rather than recurrence — and
its absence is exactly what made D8's withdrawn row meaningless.

## 5. What is reported, per metric, per channel group

* **Step 2 excess** against `microstate` and `microstate0`, and the gate against
  the length-matched windowed control. Does `shape`'s **+5.193% [+3.959,
  +6.449]** survive?
* **Step 3 clump count** against the nulls — currently **19 against 2 and 4** —
  and the unassigned fraction, currently 98.1%.
* **Duration ratio between matched partners**, currently **3.7×**.

Animal-level bootstrap, n = 89, 2,000 replicates, paired within animal.
`k_mad = 3.0`, three channel groups, θ inherited as before.

## 6. The gate, fixed in advance

| outcome | verdict | consequence |
|---|---|---|
| excess survives **and** clumps survive **and** the duration ratio falls under `union` | `PASS` | the structure is real and was partly obscured by the warp. Step B runs |
| excess survives, **clumps collapse** | `FAIL` (vocabulary) | the 19 clumps were tempo-invariance. Reported, and the vocabulary claim weakens materially |
| **excess collapses** | `FAIL` | segment recurrence depended on the warp. **Stop and report** — that closes this route with an answer |

"Survives" means the animal-level interval still excludes zero in the same
direction. "Collapses" means it does not. No threshold on how much the point
estimate may move — one fixed now would be arbitrary and one fixed later would be
selected against the outcome.

## 7. What a PASS would not license

It would not license the clumps being behaviours, and it would not restore D8's
withdrawn sentence. It would say the grouping is not an artifact of
time-normalisation. Whether the detector that produced the segments can find what
is there is Step B's question, and it currently isolates **2.3%** of planted
instances.
