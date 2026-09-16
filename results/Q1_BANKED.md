# Q1, banked — and the arm it holds on

Quoted, hashed, **not re-scored**. Q1 was computed in the `recur` repository;
this document and `results/q1_banked.json` are the record this arm stands on, so
that the number is defined in exactly one place and carries its cell.

## The headline

| | |
|---|---|
| cell | `wiener · w = 0.4 s · base · amplitude · d = 192 · split report` |
| excess over VAR(5) | **+1.6391% [+1.2049%, +2.0796%]** |
| occupancy-equivalent | **10.87% [7.69%, 14.22%]**, log-log slope 0.887 |
| planted floor | 0.25% occupancy recovered |
| white control | +0.0024% [−0.0073%, +0.0127%] — flat |
| continuum margin | −0.0187% [−0.0549%, +0.0142%] |
| planted 1% | +0.2081% [+0.1588%, +0.2575%] |
| search exactness | GPU reproduces the float64 CPU reference to 5.62e-06 |
| queries | 3,022,508, animal-level bootstrap at n = 89 |

Also on this cell: +15.08% over `phase`, +27.97% over `white`, +71.63% over
`ou`. 61.45% of queries find a closer within-animal match than cross-animal
match, against a random-pair ambient scale of 16.81.

**Correction to how this has been quoted.** The +1.639% is the **0.4 s** cell.
`q1_verdict.json` records `primary_window_s: 0.4`. The 1.5 s cell (d = 720) is a
different number, +5.854%.

## The part that has not been quoted, at equal prominence

**The same file carries an `unfiltered` cell at the identical window and
dimension, and its excess over VAR(5) is negative:**

| cell | vs VAR(5) | vs phase |
|---|---|---|
| `wiener · w0.4 · d192` | **+1.6391% [+1.2049%, +2.0796%]** | +15.08% |
| `unfiltered · w0.4 · d192` | **−0.1991% [−0.4636%, +0.0418%]** | +9.41% |

The interval spans zero. The cell is **not degenerate** — it ran on 1,954,444
queries and its phase comparison is large — so this is a `FAIL`, not a
`NOT_A_RESULT`.

**Why this matters here.** The segmentation arm must run on `raw`: a low-pass
filter manufactures exactly the smoothness whose breaks the detector looks for,
and Wiener's effect is 5× larger in the twist channels than in the pose block.
So the sentence *"a vocabulary exists to be found"* is evidenced on an arm this
work is not allowed to use.

**And the limit on that claim, stated plainly.** recur's `unfiltered` is bare
`pose_unfiltered` in the 44-dim channel space. F3's `raw` is
`held_array(pose_unfiltered, missing)` in the 17-dim ego space, and
`channels.py:212` drops swap correction for surrogate arms. These are different
arrays in different spaces. The −0.199% is **suggestive, not identical**, and it
is the reason the segment work registers its own comparator rather than reading
against +1.639%.

## What this licenses, and what it does not

**Licenses:** that something recurs across animals above a VAR(5) surrogate, on
the Wiener arm, at 0.4 s, measured with no labeller, against a planted floor
that says 0.25% occupancy would have been visible.

**Does not license:** any statement about the raw arm; any statement at segment
scale; and — because the excess is a rate below a null-set threshold — any claim
about *what* recurs.

## Provenance, and why this is not in the freeze

`recur/results/q1.json` is deliberately **not** added to
`vieb/io/spine.py:CONSUMED_JSON`. Adding it would change the inherited digest
`198eb14ff258c7f6` that every result JSON already written carries, and a digest
that moves because a file was banked stops meaning what it said. The SHA-256 of
each source file is recorded in `results/q1_banked.json` instead, which pins the
same fact without rewriting history.

## The other two passes

**Roughness concentration** — `results/ROUGHNESS.md`, newly banked here.

**The dwell-matched ladder** — already banked, in `results/falsifier.json`'s
eight per-arm `Read`s and written up in `results/DWELL.md`. Re-stated by
reference rather than copied, so those numbers keep one definition:
`microstate_N256` and `microstate0_N256` both `PASS`, at +1.317 [+0.899, +1.728]
and +10.045 [+9.370, +10.710] nats/s. `DWELL.md`'s own caveat stands —
`microstate` must be read weakly, because it preserves one-step visit dynamics
and therefore much of what the comparison measures, making +1.317 a lower bound.
