# F1 — the vieb2 diagnostics, recomputed on deduplicated input

Blocking item. Slurm job 74414, `gpu` partition, `g01`, 3× A100, GPU HDBSCAN.
Banked run: `~/vieb2-results/run_20260804_160351`. Deduplicated re-run:
`~/vieb-runs/f1_dedupe`.

## What was wrong

The banked run aligned **4,925 "recordings"** — which is a **file** count, not a
recording count. `~/dlc-training/raw_videos` holds 3,846 `.h5` and 1,079 `.csv`,
and every `.csv` duplicates an `.h5` (csv-only stems: **0**). The current code
says so directly:

```
[vieb2] deduplicated 4925 file(s) -> 3846 recording(s)
        (1079 duplicate export(s) dropped, .h5 preferred)
```

Near-duplicate detection on the banked array agrees: **1,081** surplus
recordings, **3,844** unique implied, with an example pair differing by a maximum
of **4.26e-05** and **zero** bit-identical rows of 6,303 — the signature of one
recording round-tripped through CSV text at limited precision.

| | banked | deduplicated |
|---|---:|---:|
| "recordings" | 4,925 | **3,846** |
| aligned frames | 28,626,107 | **22,355,989** |
| frames entering the comparison | 28,586,707 | **22,325,221** |
| duplicated frames | — | **6,270,118 (21.9%)** |

The arithmetic ties out exactly in three independent places: 28,626,107 −
4,925×**8.0000** = the banked comparison's frame count (8 frames per recording
lost to the delay embed); the deduplicated align reproduces shapeflow's corpus
(22,355,989) to the frame; and the deduplicated comparison lands on 22,325,221,
the predicted value.

This is the failure `vieb_v2/cli.py` warns about in its own comment — "most
damagingly in the stationary measure of the transfer operator, which is literally
an occupancy count". `noise_speed_ratio` and `size_speed_rank_corr` are
occupancy-weighted, and the 1/|v| density argument *is* an argument about
occupancy.

## The result

Same code, same defaults, same GPU backend on both sides. The only change is that
1,079 duplicate exports are dropped.

### pca-HDBSCAN

| | banked (4,925) | dedup (3,846) |
|---|---:|---:|
| `noise_speed` | 93.7329 | 93.5751 |
| `clustered_speed` | 9.6948 | 9.4570 |
| **`noise_speed_ratio`** | **9.668** | **9.895** |
| **`size_speed_rank_corr`** | **+0.0857** | **−0.4643** |
| `noise_frac` | 0.1544 | 0.1621 |
| `largest_state_frac` | 0.8387 | 0.8299 |
| `n_states` | 6 | 7 |

### diffusion-HDBSCAN

| | banked (4,925) | dedup (3,846) |
|---|---:|---:|
| `noise_speed` | 0.6370 | 0.8495 |
| `clustered_speed` | 0.0336 | 0.0424 |
| **`noise_speed_ratio`** | **18.955** | **20.013** |
| **`size_speed_rank_corr`** | **−0.5076** | **−0.4234** |
| `noise_frac` | 0.1078 | 0.0671 |
| `largest_state_frac` | 0.8601 | 0.9038 |
| `n_states` | 37 | 34 |

## Verdict: the headline survives, one number was wrong

**`noise_speed_ratio` is robust.** 9.668 → 9.895 (+2.3%) and 18.955 → 20.013
(+5.6%). Same direction, same order of magnitude, conclusion unchanged. The
density-duration argument — that HDBSCAN's noise label is speed-biased, that
density along a uniformly-sampled trajectory goes as 1/|v| and therefore *is*
dwell time — **stands**. A ratio of means is evidently insensitive to
double-weighting a fifth of the corpus, which is plausible and is now measured
rather than assumed.

**`size_speed_rank_corr` on the pca arm was wrong, and not slightly.**
**+0.0857 → −0.4643.** A sign flip, from "cluster size and cluster speed are
unrelated" to "bigger clusters are substantially slower".

The strongest evidence that the deduplicated value is the correct one is that
**it makes the two arms agree**. Banked, pca said +0.086 and diffusion said
−0.508 — a flat contradiction between two reducers on the same alignment.
Deduplicated, they read −0.464 and −0.423. A rank correlation over ~6–37 clusters
is exactly the kind of statistic a 22% duplication should be able to wreck, and
it did.

**Nothing in any written argument cited the pca value.** The two docstrings that
cite a rank correlation — `vieb/src/vieb/segmenters/ulam.py:6` and
`vieb/vieb_v2/representation/transfer_operator.py:10` — both cite the **diffusion**
figure −0.508, which moves to −0.423: same sign, 17% smaller, argument intact.
So the number that broke was not load-bearing, and the numbers that were
load-bearing did not break.

The corrected value **strengthens** the density-duration case for the pca arm,
where the banked value had appeared to contradict it.

## Numbers to update

| where | from | to |
|---|---|---|
| `vieb/src/vieb/segmenters/ulam.py:5` | `noise_speed_ratio` 9.67 / 18.96 | 9.90 / 20.01 |
| `vieb/src/vieb/segmenters/ulam.py:6` | `size_speed_rank_corr` −0.508 | −0.423 |
| `vieb/vieb_v2/representation/transfer_operator.py:10` | `size_speed_rank_corr` −0.508 | −0.423 |
| `~/vieb2-results/*` ranking JSONs | whole-corpus `noise_speed_ratio` | recompute or mark as pre-dedupe |

## Three confounds in this comparison, none of them fatal

1. **The banked pose directory is not recoverable.** It held 4,925 entries; the
   directory holds 3,846 `.h5` today. This is a re-run, not a clean subset
   experiment, so anything else that changed since August 2026 is in the
   difference. The frame arithmetic tying out three ways argues that nothing else
   did, but it does not prove it.
2. **`cmd_align` wrote no recording names** into the banked `aligned.npz`, so the
   banked artifact cannot be subset to check directly. The current version writes
   `recordings.csv`; that gap is closed going forward.
3. **Both sides ran GPU HDBSCAN**, which is the like-for-like that matters, since
   `cluster.py` documents cuml as "a separate implementation, not a faster copy"
   with agreement verified only on a 3,568×35 embedding and "untested" at project
   scale. This corpus is 22.3M. Backend is held constant rather than trusted.

## Cost

GPU: pca 163.5 s, diffusion 964.4 s. The first attempt ran CPU-only on a login
node and was still inside the pca arm after **six hours** without finishing. The
GPU partition was available the whole time and was not being used.
