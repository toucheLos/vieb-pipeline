# Island look — the scorer failed the positive control

Registered in `results/ISLAND_LOOK_PREREGISTRATION.md`, committed at `9e22ad1`
and amended at `09db47e` (72 → 180 trials) before the builder was run. Source
array `work/ego/raw__bodylen__*.npz`, the F3 `raw` arm; clump 0 of `shape` at
`k_mad = 3.0`; 89 report animals. Digest `198eb14ff258c7f6`.

Scorer: **Claude (this session)**, one rater, 180 trials, contact sheets only
(4 plain frames + 4 skeleton frames per clip, three clips per trial). The key
was not read until `--score` ran.

## The MDE, reported before the accuracy

> **MDE = 0.0984** over 180 trials at 80% power against chance 0.3333, below the
> **0.2000** that would count as a real effect, so a null here is informative.

The design could detect a lift of about **+10 points** over chance. The
registration fixed **+20 points** as the lift that would count. So a null in this
run means *"not as big as +10"*, not *"nothing"* — and that is enough resolution
for the question that was asked.

## The verdict, on the registered primary

**The positive control is at chance. Under §8 of the registration the run is not
interpretable in either arm.**

| arm | trial type | correct | accuracy | animal-clustered interval | verdict |
|---|---|---:|---:|---|---|
| **`unmatched`** | **both** | **34/90** | **37.8%** | **[0.280, 0.467]** | **FAIL** |
| `matched` | both | 35/90 | 38.9% | [0.274, 0.500] | FAIL |

`unmatched` is the arm where the control is a uniform draw and the island's
**0.205×** speed is left in as a free cue. It should have been the easy one. It
was not distinguishable from guessing.

The registration's reading table, fixed in advance:

> *at chance on unmatched → **the instrument is blind.** Neither arm is
> interpretable and the result is about the scorer, not the island.*

That is the row this run landed on. **This test did not settle whether the island
is a behaviour or a metric artifact.** It disqualified the scorer.

## The trial-type split, which is descriptive and is not a claim

The registration required these be reported separately and **never averaged into
one claim**, because they ask different questions. Reported:

| arm | trial type | odd clip is | correct | accuracy | interval |
|---|---|---|---:|---:|---|
| `matched` | `island_pair` | a control | 22/45 | 48.9% | [0.341, 0.625] |
| `matched` | `control_pair` | an island | 13/45 | 28.9% | [0.162, 0.420] |
| `unmatched` | `island_pair` | a control | 22/45 | 48.9% | [0.325, 0.646] |
| `unmatched` | `control_pair` | an island | 12/45 | 26.7% | [0.136, 0.400] |

One cell — `matched / island_pair` — has an interval clearing chance. **It is not
quotable**, for two reasons fixed before the scoring and one found after.

**Fixed before (§8):** the gate is the positive control, and it failed. A cell
that passes inside a blind instrument is not evidence.

**Fixed before (§4, M1):** the two arms perform **identically** (48.9% / 48.9%,
28.9% / 26.7%). The `unmatched` arm hands the scorer an enormous extra cue —
unmatched speed — and performance did not move by a point. A scorer using the
intended cue would have done *better* where the task was easier. This one did not,
which means the cue being used is not the one the design was built around.

**Found after unblinding — the mechanism.** The scorer picked an island clip
**71 times out of 180**, where uniform guessing over the same trials gives **90**.
That is a systematic bias *away* from island clips: they read as the ordinary
clip, and the control read as the odd one. That single directional bias is enough
to generate the entire split by arithmetic — in `island_pair` the odd clip *is*
the control, so the bias scores; in `control_pair` the odd clip is the island, so
the same bias misses. The two are one artifact with opposite signs, not two
results.

It is specifically **not** the signature of grouping-by-similarity. If the island
clips resembled each other, the two controls in a `control_pair` trial should have
resembled each other too, and that trial type should also have run above chance.
It ran below.

## Scorer diagnostics

| | A | B | C |
|---|---:|---:|---:|
| odd position, by construction | 61 | 59 | 60 |
| scorer's responses | 59 | 52 | **69** |
| accuracy when that key was pressed | 19/59 | 22/52 | 28/69 |

Odd position was balanced by construction (`triad.plan_trials`, the
`rng.permutation(np.tile(...))` form) and came out 61/59/60. The scorer drifted
toward **C** (69 against an expected 60) and no position paid better than any
other, so the drift cost nothing but is recorded.

## What the island looks like — description, not evidence

Unblinded after scoring, and labelled as the registration requires: this is a
**description of what is on the screen**, carrying no statistical weight, written
by the same scorer the run just disqualified.

Island clips are, overwhelmingly, **a stationary mouse in contact with the arena
wall or the bar floor**, held for the length of the clip: a compact lateral body
against the grid, or a body wedged into a corner, with the tail laid out and the
keypoint polygon barely deforming across the four sampled frames. The postures
are not identical to each other — some are lateral, some hunched, some vertical
against the bars — and that heterogeneity is the honest part of the description
and is consistent with the chaining verdict.

The control clips drawn from the 98.1% unassigned continuum look, to the eye,
**largely the same kind of thing**, which is the plainest statement of why this
scorer could not do the task: at 4 frames and 240 px, most of this corpus is a
still mouse near a wall.

## What would settle it

The trial set is built, sealed, and reusable — 180 trials, 1,080 clips, one key.
It needs a scorer who **passes the positive control**. The next step is not a new
statistic; it is putting the same sealed trials in front of a human, or several,
and reading `unmatched` first. The clips are published for exactly this at
`carloseckert.com/vieb/models.html`, blind, in key order, with a skeleton toggle.

**One cue the published page does not remove.** Every clip is captioned with its
recording id, and a recording id names its context — `(Context_A)` or
`(Context_B)`. Step C measured the island at **1.275%** occupancy in A against
**0.470%** in B, so the caption carries a weak prior toward the A clip being the
island. It is left in because `check.py` requires an animal and a recording on
every published clip and because both classes are drawn from the same
context-mixed corpus, so the marginal information is small — but it is a cue, it
is recorded here, and a reader who wants the clean version should cover the
caption.

Two changes worth making before a human run, neither of them a re-tune of this
one:

1. **Show motion, not stills.** The scorer here had 8 frames per clip. The mp4s
   carry the whole segment and the island's defining property is measured in
   *speed*. A contact sheet throws that away.
2. **Fewer trials, watched properly.** 180 trials at 3 clips each is 540 clips to
   watch. The MDE at 90 trials is +0.139, still inside the +0.20 that counts.

## Provenance

Every number above is computed on the **`raw`** preprocessing arm
(`held_array(pose_unfiltered, missing)`), the F3 carried arm. No Wiener-arm array
enters this stage. Bootstrap is `recur.boot.animal_interval`, 2000 replicates,
`how="mean"`, clustered on animal — never on trial alone.

Artifacts: `results/island_look.json` (reads, responses, per-trial outcomes),
`results/island_look/key.json` (sealed key), `results/island_look/trial_*.png`
(180 contact sheets), `results/island_look/clips/` (1,080 mp4s),
`results/island_look/manifest.json` (the blind publication manifest —
written beside the clips rather than under `results/behaviour/` as planned, so
the 13 MB is not copied twice).
