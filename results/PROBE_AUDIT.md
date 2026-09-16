# Step 0 — the plant has a weak edge. Step B is not withdrawn, and not clean

Registered in `results/PROBE_AUDIT_PREREGISTRATION.md`, committed alone at
`8d4b1c6` before `vieb/seg/probe_audit.py` or `scripts/probe_audit.py` existed.
Source array `work/ego/raw__bodylen__*.npz`, the F3 **`raw`** arm. `shape` group,
`deriv_sec = 0.133`, `h = 4`, planted duration 0.5 s (15 frames), occupancy 0.1,
planted into `microstate`, `EDGE_TOL = ±2`, 12 animals. Digest
`198eb14ff258c7f6`.

Every setting is inherited from `scripts/detector_sweep.py` **as an attribute**,
never retyped, so this stage shares Step B's probe exactly.

## The anchor held, so the rest is readable

> **PASS** — count-matched random frames score **0.5018 [0.4982, 0.5057]**
> against the 0.5000 a uniform draw has by construction.

This is an arithmetic identity, not a hypothesis. The registration made a
departure from it a `NOT_A_RESULT` for the whole stage, because a percentile
scored against its own distribution that does not average to a half means the
scoring is broken and every other number is a plausible-looking artifact. It is
not broken.

## The result

> **INCONCLUSIVE** — the plant has a **weak** edge.

| population | mean percentile of `D` within its own recording |
|---|---:|
| **random**, count-matched | **0.5018** [0.4982, 0.5057] |
| **planted edges** | **0.5983** [0.5657, 0.6328] |
| **detector firings** | **0.9614** [0.9520, 0.9697] |

**planted − random = +0.0965 [+0.0659, +0.1290]**, animal-clustered, 2000
replicates, **n = 12 animals over 154 recordings**.

The interval excludes zero, so the crossfade **is** visible to an acceleration
criterion — the first row of the registered reading table does not fire and
**Step B's FAIL is not withdrawn.**

But the plant's edge sits at the **60th percentile** of the criterion's own
scalar while the boundaries the detector actually fires on sit at the **96th**.
"Well below the firing interval's lower bound" was fixed in the registration
before any number existed, and 0.5983 is far below 0.9520. So the second row
fires: **the 12.8% is part probe and part criterion, and this design cannot
separate them.**

## The number that makes it concrete

**11,670 of 58,721 planted-edge frames — 19.87% — exceed their own recording's
`mad_threshold(D, K_MAD)`.** Four fifths of the edges the detector is being asked
to find cannot clear the threshold that would let them fire, *whatever* the
detector does downstream.

That 19.87% and Step B's 12.8% isolation are the same order of magnitude. The
registration listed the threshold-crossing share as a reported quantity but did
**not** register a comparison between them, so this is an observation and not a
test: the ceiling the probe imposes is close enough to the score the detector
got that the score is largely a statement about the ceiling.

## What this changes, stated narrowly

**It does not reverse Step B.** The criterion was never shown to find real
boundaries and this says nothing about whether it does.

**It does not withdraw Step B either.** The plant is not invisible. A detector
that found every findable planted edge would have scored better than 12.8%.

**What it does is price the evidence.** Step B's FAIL was read as "the criterion
cannot see the insertion, and no further threshold sweep is indicated"
(`DETECTOR.md`, and `vieb/seg/detector.py:71-79` emits that text on a
`no_edge`-dominant failure). That reading assumed the insertion was a fair
target. It was a target four fifths of which is below the firing threshold by
construction. The conclusion survives in weakened form: the criterion misses most
of a deliberately smooth insertion, which is weaker evidence about behaviour than
it looked.

**So the human annotation is not optional, and it is now the only thing that can
settle this.** That was the brief's argument and this measurement supports it
without proving the stronger claim it hoped for.

## What this does not discharge

The registered **negative control** is still owed.
`DETECTOR_PREREGISTRATION.md:84-90` requires any cell clearing the gate to be run
on `white` and to fail there; no cell cleared, so it never ran
(`DETECTOR.md:104-109`). Nothing here substitutes for it, and it comes due the
moment any future detector passes.

## Provenance

`raw` arm, asserted by name in `seg_recur.load_arm`. No Wiener array enters this
stage — a low-pass filter manufactures exactly the smoothness whose absence was
under test, which would have made the plant look *more* findable, not less.
Bootstrap `recur.boot.animal_interval`, `how="mean"`, 2000 replicates, clustered
on animal. The frame-level interval is stored in
`detail.frame_level_interval_do_not_quote` and licenses nothing.

Artifact: `results/probe_audit.json`.
