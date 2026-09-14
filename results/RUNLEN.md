# G1 — how long is a violating run, and where does the violating mass live?

89 report animals, 1,149 recordings, 298 shards, ε = 0.10. Measured on
`held_array(pose_unfiltered, missing)` — **the array `scripts/disposition.py`
hands the corrector**, before any smoother.

This had never been measured. Every run-length figure this programme published
was a hardcoded literal in `scripts/cleaning_md.py` or prose in a docstring,
backed by no artifact, and taken from **eight recordings**. See
`PROVENANCE_AUDIT.md`.

## The measurement

| arm | runs | violating frames | median | p75 | p90 | max | mass ≤3 | mass ≤7 | **mass >7** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **raw** (corrector's input) | 24,882 | 74,997 | **1** | **3** | 6 | 123 | **38.4%** | 60.3% | **39.7%** |
| `median_0.50` | 6,831 | 44,859 | 3 | 9 | 16 | 123 | 13.0% | 26.1% | 73.9% |
| `viterbi` | 17,145 | 69,073 | 2 | 4 | 9 | 123 | 26.9% | 50.5% | 49.5% |

Animal bootstrap, 89 animals:

| arm | median run | mass in runs ≤3 frames |
|---|---|---|
| raw | 1.47 [1.36, 1.57] | 37.3% [35.5%, 39.2%] |
| `median_0.50` | 3.29 [3.04, 3.57] | 14.1% [13.0%, 15.4%] |
| `viterbi` | 2.08 [2.02, 2.16] | 27.6% [26.3%, 29.0%] |

## The design question, answered

**Violating runs are short by count and long by mass, and that was the claim.**

On the corrector's real input the median run is **1 frame** and p75 is **3** — so
a 3-frame envelope reaches the large majority of *runs*. But those runs carry
only **38.4%** of violating keypoint-frames. **61.6% of the violating mass sits in
runs longer than 3 frames**, and 39.7% in runs longer than 7.

So both operators are needed and neither is redundant: minimal projection is the
right operator for the majority of events, and it reaches under two-fifths of the
error. Correction inside an abstention envelope is what the distribution supports,
which is what Phase D built. **`envelope_read` returns PASS**, and this is the
first measurement that bears on it.

### What was wrong, and it is not what was feared

The concern was that the design had been calibrated against a truncation
artifact. It had not — the registered threshold was calibrated against the raw
distribution ("median violating run is 1–2 frames, p75 = 3"), and the measured
raw distribution is median **1**, p75 **3**. The p75 is exactly right.

What was wrong is that **two of the three numbers were wrong in magnitude**:

| figure | published | measured | |
|---|---:|---:|---|
| raw median run | 2 frames | **1 frame** | shorter |
| post-`median_0.50` median run | **9 frames** | **3 frames** | a third of it |
| mass in runs ≤3 (raw) | 43.7% | **38.4%** | 5.3 points low |

The 9-frame figure is the one that was used to argue abstention should be the
outer envelope. Corpus-wide the post-filter median is **3 frames**, not 9; 9 is
the corpus-wide **p75**, which is what the eight-recording sample appears to have
been reporting. The argument for abstention does not need it: the mass figure
carries it, and the mass figure is measured.

## The truncation mechanism, confirmed directly

`median_0.50` is a 15-frame centred median, so it cannot alter a run of 8 frames
or more and annihilates any run of 7 or fewer. If that is the whole story, the
violating mass **above** the cliff should survive the filter untouched.

It does: the filter preserved a factor of **1.11** of the violating mass in runs
longer than 7 frames, while the median run length rose from 1 to 3.

**Nothing lengthened.** The rise in median run length is the arithmetic of
deleting every run at or below the cliff — the surviving distribution is the
original one truncated from below. `truncation_read` returns PASS. The mechanism
claim on the DLC cleaning page is now confirmed by measurement rather than
argued from the window size.

Two supporting numbers: the filter removes 73% of the *runs* (24,882 → 6,831)
while removing only 40% of the violating *frames* (74,997 → 44,859), which is what
deleting many short runs and no long ones looks like. And `viterbi` sits exactly
where a de-glitcher should — 31% of runs removed, 8% of frames — since it targets
single-frame excursions and declines to touch anything sustained.

## Consequences

* **Phase D's envelope stands**, and now rests on a measurement.
* **The abstention rationale is re-based.** It is justified by 61.6% of violating
  mass sitting outside a 3-frame envelope, not by a 9-frame median that does not
  exist at corpus scale.
* **`inject.PARK_FRAMES = (4, 30)`** was drawn from the post-filter distribution —
  a residual distribution used to model a pre-filter error. The measured raw
  distribution has p90 = 6 and max = 123, so 4–30 frames is not absurd but it is
  not derived from the right population either. **Recorded, not silently
  re-run**: the injection result is pre-registered and banked, and changing its
  corruption model requires its own registration.
* `results/runlen.json` supersedes the retracted table in `cleaning_md.py`.
