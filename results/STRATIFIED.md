# F2 — the injection benchmark in the fast regime

89 report animals, 298 shards per arm, four pool thresholds. Configuration fixed
in `STRATIFIED_PREREGISTRATION.md`, committed before the sweep ran.

## The headline: Phase F's verdict was an artifact of a slow pool

`median_0.50` won Phase F. It wins only while the pool excludes fast movement.

| `CLEAN_SPIKE_BL` | fast share | trunk viol. in pool | **`median_0.50` net** | **`viterbi` net** | median damage |
|---|---:|---:|---:|---:|---:|
| **0.02** (Phase F) | 0.95% | 0.000% | **−0.0026** | −0.0016 | 0.0031 |
| 0.05 | 2.29% | 0.000% | −0.0007 | −0.0016 | 0.0049 |
| 0.10 | 2.74% | 0.027% | −0.0001 | −0.0016 | 0.0054 |
| **off** | 2.74% | 0.026% | **−0.0000** [−0.0006, +0.0005] | **−0.0016** | 0.0055 |

*fast share = pool frames above that animal's own all-frames p90 centre speed;
10% would be unbiased.*

**`viterbi` is invariant at −0.0016 across every threshold.** `median_0.50` goes
from best arm to **statistically indistinguishable from doing nothing** — its
interval spans zero at `off`.

The mechanism is visible in the last column. The median's **repair stays flat**
(~0.12 at every threshold); its **damage rises monotonically**, 0.0031 → 0.0055.
It is not repairing less as the pool widens. It is breaking more.

## Per stratum, the crossing is clean and monotonic

Net by speed stratum, slowest → fastest, at `off`:

| arm | | | | | | |
|---|---:|---:|---:|---:|---:|---:|
| `median_0.50` | −0.0331 | −0.0023 | −0.0014 | −0.0005 | **+0.0011** | **+0.0054** |
| `viterbi` | 0.0000 | −0.0015 | −0.0016 | −0.0017 | −0.0020 | **−0.0026** |

**In the two fastest strata the median actively makes the data worse than raw**,
while Viterbi reaches its best there. The two arms move in opposite directions
across the whole range and cross between the third and fourth stratum.

`recovery_read` classifies `median_0.50` as **harmed** at 0.10 and at `off`, and
as **mixed by speed** at 0.05 — helps still frames, harms moving ones. That is
the guard built in Phase F for exactly this, firing for the first time.

## The five predictions

| # | prediction | outcome | |
|---|---|---|---|
| 1 | fast share above 5% at `off` | **2.74%**, up from 0.95% | **failed** |
| 2 | trunk contamination at `off` > 2× its value at 0.02 | 0.000% → 0.026% | **unfalsifiable as written** |
| 3 | median's advantage keeps falling with speed, Viterbi's keeps rising | monotonic in both, at every threshold | **held** |
| 4 | they cross in the fastest adequately powered stratum | they cross, and the median goes **positive** | **held** |
| 5 | `median_0.50` remains best pooled at every threshold | beaten at 0.05, 0.10 and `off` | **failed** |

**Prediction 1 failed for the reason the registration named in advance.** The
registration's closing section said continuity was not the binding constraint —
**confidence at 10.55% is** — and that this would measure further into the fast
regime without reaching all of it. Turning continuity fully off raises the fast
share only to 2.74%, and thresholds 0.10 and `off` give identical results, so the
continuity criterion is fully saturated by 0.10 and binds nothing beyond it. The
remaining 3.6× under-representation is the confidence gate, which is not
touchable without giving up the ground truth entirely.

**Prediction 2 was badly specified and I should not have written it that way.**
The contamination proxy at 0.02 is **0.000%**, so "exceeds 2×" is satisfied by
any positive number and cannot fail. What the measurement actually says: trunk
violations inside the pool rise from 0.000% to 0.026% — a real rise, and small
in absolute terms against a 2.40% corpus-wide trunk rate. **The pool stays clean
as it widens**, which is the substantive point and is what makes the rest of the
table trustworthy.

## What this changes

**The Step 4 decision reverses again, and this time on the regime that matters.**
Phase F's registered falsifier put `median_0.50` back on the MDL branch because
its net was negative. That result was measured on a pool where fast frames were
under-represented 5.5×. With even a modest correction to that bias, the median's
net reaches zero and its per-stratum net goes **positive** where the animal moves.

On a fear-conditioning corpus, the fast rare events are the signal. An arm that
is neutral overall and actively harmful on fast frames is not a defensible
default, and the movement-retention argument — that `median_0.50` deletes 86–90%
of median movement — now has a direct measurement behind it rather than an
inference.

**Recommended arms for the MDL branch: `raw`, `viterbi`, `disposition`.**
`median_0.50` is reported, not carried. This is a recommendation, not a freeze;
F3 records the decision with its date and evidence.

## A defect found and fixed mid-phase

Phase F seeded each recording's corruption with
`abs(hash((SEED, tag, rid))) % 2**32`. **Python salts `hash()` on strings per
interpreter**, so despite `SEED = 0` every run drew a different corruption
layout, and the pre-registration's claim that the benchmark was seeded was
false. Three processes given identical inputs returned three different seeds.

Replaced with `blake2b`, which is stable across processes, machines and versions,
and pinned by `TestTheSeedIsActuallyStable` — including a subprocess check that
would catch a salted hash returning.

All four arms in this sweep were re-run under the stable seed. **Phase F's
conclusions are unaffected**: re-running its own threshold moves `repair` by
about 1% relative (0.3635 → 0.3688 on `raw`) and leaves every verdict, every
sign and every ordering intact. The numbers in `injection.json` are now the
reproducible ones.
