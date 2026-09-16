# Step A — the clumps are not an artifact of the warp

Registered in `results/DISTANCE_PREREGISTRATION.md`, committed at `2197e42`
before either metric touched the corpus. Source array
`work/ego/raw__bodylen__*.npz`, the F3 `raw` arm. 89 report animals, `k_mad =
3.0`, θ inherited per cell. Digest `198eb14ff258c7f6`.

## The verdict, on the registered primary

**`open` — compare the shared extent, no warp — passes on `shape` and `both`.**

| cell | excess vs `microstate` | under the warp | clumps | nulls' clumps |
|---|---|---|---:|---:|
| **shape / open** | **+3.3911% [+2.5327%, +4.2137%]** | +5.9301% | **13** | 4 |
| both / open | +2.0112% [+1.4516%, +2.5784%] | +4.0483% | 6 | 3 |
| twist / open | −0.0569% [−0.3523%, +0.2211%] | +0.1170% | 1 | 1 |

**The excess survives and the clumps survive.** The time-normalised distance was
inflating the excess by about **1.75×** and the clump count from 13 to 19, but it
was not manufacturing them. The worry that motivated this stage — that the 19
clumps were *same shape, any tempo* — is not what the data says.

`twist` collapses to an interval spanning zero, which is consistent with Step 2,
where twist segments did not recur either.

## The duration ratio fell, and not the way the registration expected

| | matched-partner duration ratio |
|---|---:|
| time-normalised (the warp) | **3.70×** |
| `open` | **2.55×** |
| `union` | 3.10× |

Under the warp, segments called nearest neighbours differed by a factor of 3.7 in
duration. Under `open` that falls to 2.55×. **So part of the tempo-invariance was
the ruler and part of it was not** — a factor of 2.5 in duration between matched
partners survives a distance that does not warp at all.

**`union` did not drive the ratio toward 1**, which the registration anticipated
it might. The mechanism is visible once stated: `union` holds the shorter segment
at its last frame, and in a corpus whose recurrence is dominated by
near-immobility (`BEHAVIOUR.md`: the widest clump runs 3.7× slower than
baseline), **holding a static segment costs almost nothing** against a long
static one. `union` charges duration mismatch only when the longer segment keeps
moving, so on this corpus it is far less of a duration penalty than its
definition suggests.

## `union` is not quotable, and its own check says so

| cell | retrieval recovers the true nearest | verdict |
|---|---:|---|
| shape / **open** | **98.8%** | PASS |
| both / open | 97.2% | PASS |
| twist / open | 91.2% | PASS |
| shape / **union** | **74.0%** | **FAIL** |
| both / union | 74.8% | FAIL |
| twist / union | **50.5%** | FAIL |

The two-stage search retrieves candidates in a 16-frame prefix space and reranks
them under the true metric. For `open` that works — the shared extent of most
pairs *is* near the prefix, so a prefix neighbour is usually the true neighbour.
For `union` it does not: `union` is dominated by what happens **after** the
prefix, and a 16-frame prefix is a poor proxy for a comparison that may run to
4,537 frames.

`recovery_read`'s own words: *"the index is losing the answer, so the excess read
off it is not the excess."* **So `union`'s numbers are reported and not quoted.**
They are listed above for completeness and they license nothing.

That is the measurement the registration promised instead of a caveat — and it is
the reason the registered primary was `open` and not the metric that happened to
look more decisive.

## What the windowed control says now

| cell | segments | length-matched windows |
|---|---|---|
| shape / open | +3.3911% [+2.5327%, +4.2137%] | +0.2734% [−0.0042%, +0.5312%] |
| both / open | +2.0112% [+1.4516%, +2.5784%] | −0.1193% [−0.4032%, +0.1200%] |

The windowed control sits at or below zero under `open`, so the boundaries still
carry the whole of the excess — the same conclusion `SEGRECUR.md` reached under
the warp, reached again under a distance that does not warp.

## Coverage is slightly worse, and the structure is still there

Unassigned rises from 98.14% (warp) to **99.34%** on `shape / open`: 13 clumps
instead of 19, over a slightly sparser graph. **The vocabulary question is
unchanged** — there is clump structure the dwell-matched nulls do not reproduce,
and it covers under one per cent of segments.

## What this licenses

**Licenses:** the grouping found in Step 3 is not an artifact of
time-normalisation. It is smaller than the warp made it look and it is real
against two dwell-matched nulls under a distance that compares unwarped frames.

**Does not license** a vocabulary, does not restore `DEVIATIONS.md` D8's
withdrawn sentence, and does not license any `union` number.

And it does not clear the way to coverage: **Step B**, run next, found that the
detector isolates 12.8% of planted instances against a 20% gate, with no boundary
at all at either edge of two-thirds of them. The metric was not the bottleneck.
The detector is.
