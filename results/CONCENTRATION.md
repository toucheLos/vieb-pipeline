# H1 — are tracking errors spread evenly, or concentrated on particular footage?

89 report animals, 1,149 recordings, 22.4M keypoint-frames. Computed on Step 1's
bit-packed masks (`viol|unfiltered|raw|0.1|skull`) — the same arrays
`results/bones.json`'s 2.134% came from, not a second implementation.

## The headline

**Violation rates are 350× more dispersed across recordings than independent
frames at the same rate would produce.** The errors are situation-driven, not
random.

| | |
|---|---:|
| pooled rate | 2.1585% |
| per-recording range | 0.000% – 39.084% |
| **dispersion against binomial** | **349.7×** |
| Gini over recordings | 0.650 |

| worst recordings | carry | uniform would be |
|---|---:|---:|
| 1% | **10.0%** | 1% |
| 5% | 30.5% | 5% |
| 10% | **46.7%** | 10% |
| 25% | 73.5% | 25% |

Half the corpus's tracking failure lives in a tenth of its recordings.

## Where the structure is — and the ranking reverses under a fair statistic

My first reading ranked these by worst-group / best-group ratio, which put
**date** on top at 861.7×. That statistic is not comparable across groupings with
different group counts: date has 168 groups with a median of 35k frames, and 27
of them hold under 20k frames, so its extremes are noise. Re-ranked by
**size-weighted between-group overdispersion**, which is comparable:

| grouping | groups | worst/best | **overdispersion** |
|---|---:|---:|---:|
| **context** | 3 | 6.5× | **28,183** |
| day | 8 | 8.3× | 9,595 |
| date | 168 | 861.7× | 636 |
| animal | 89 | 5.2× | 299 |
| **box** | 3 | 1.0× | **7.7** |

**Context dominates by an order of magnitude; box is essentially flat.**

| context | rate | frames |
|---|---:|---:|
| C | **7.3420%** | 479,962 |
| B | 2.7704% | 2,377,646 |
| A | **1.1265%** | 3,820,880 |

| box | rate |
|---|---:|
| 3 | 2.2002% |
| 1 | 2.1454% |
| 2 | 2.1375% |

**Context and day are completely confounded and must not be read as two
findings.** Context C is 479,962 frames; Day 2 is 479,962 frames. They are the
same recordings — Context C occurs only on Day 2. So there is **one** session
factor here, not two, and nothing in this measurement can say whether it is the
environment, the day, or something that happened once on that day.

The box result is the informative negative: three physical apparatus units,
2.14% / 2.14% / 2.20%. **It is not the camera or the rig.**

## Position in the arena — the clean signal

No arena definition exists in shapeflow's artifacts, so the arena is derived per
recording from the animal's own centroid cloud: median position as the middle,
IQR as the scale, so boxes and camera mounts are comparable. Units are IQR,
unclipped.

| decile (middle → edge) | IQR range | violation rate | n |
|---|---|---:|---:|
| 0 | 0.01–0.24 | **1.2080%** | 2.24M |
| 1 | 0.24–0.35 | 1.3212% | 2.23M |
| 2 | 0.35–0.47 | 1.3828% | 2.24M |
| 3 | 0.47–0.59 | 1.5309% | 2.23M |
| 4 | 0.59–0.73 | 1.6898% | 2.23M |
| 5 | 0.73–0.89 | 1.9211% | 2.24M |
| 6 | 0.89–1.11 | 2.2871% | 2.24M |
| 7 | 1.11–1.46 | 2.5479% | 2.24M |
| 8 | 1.46–2.30 | 2.9646% | 2.23M |
| 9 | **2.30–5.74** | **4.4862%** | 2.24M |

**Monotone across all ten deciles, 3.7× from middle to edge**, on 2.2M frames per
bin. This is the occlusion signature: the animal against a wall is partly hidden,
and the network fails there.

## What this means for the ensemble, and what it cannot mean

**It bounds the ensemble's likely benefit.** An ensemble averages away errors its
members make *independently*. An animal pressed against a wall is occluded for
every member equally; five networks trained on the same footage would all be
guessing at the same hidden keypoint. Errors that concentrate 350× on particular
recordings, rise monotonically toward the arena edge, and cluster on one session
type are the kind an ensemble has least purchase on.

**It cannot decide the question, and this is not a hedge.** Hard footage and a
network that fails on hard footage are indistinguishable from outside — both
produce exactly this pattern. Only across-network variance separates them.

**Two things weaken the ratio itself**, stated so the 350× is not read as more
than it is:

* Violations come in runs — median 1 frame, p75 3 (`RUNLEN.md`) — so frames are
  not independent even under a pure-noise model, and *some* overdispersion is
  guaranteed. 350× is an **upper bound** on structure.
* Recordings differ in length and in how much the animal moved, and neither is
  controlled for here.

## What would change the recommendation

If the dispersion had come back near 1 — per-recording rates within binomial
spread — the worst-recording list would have been the tail of a random process,
the ensemble would have had a great deal to average away, and the case for
building it would be much stronger. That outcome was reachable and did not occur.

The measured answer points the other way: the largest single lever on this corpus
is probably **the footage** — arena position and one session type — not the
network. That is a cheaper thing to fix than retraining, and it is not fixable
retrospectively for data already collected.
