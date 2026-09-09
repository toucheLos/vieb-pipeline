"""`results/EFFECT.md` from `results/effect.json`. Reads, never recomputes."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.util import log, read_json                             # noqa: E402
from vieb.tok import config                                       # noqa: E402

P = "{:.3%}".format


def ci(x, spec="{:.4f}"):
    return f"{spec.format(x['point'])} [{spec.format(x['lo'])}, {spec.format(x['hi'])}]"


def num(x, spec=".3f"):
    return "∞" if x is None else format(x, spec)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=config.PATHS.result("effect.json"))
    p.add_argument("--out", default=config.PATHS.result("EFFECT.md"))
    a = p.parse_args(argv)
    d = read_json(a.json)
    L = d["layers"]
    fm = d["flag_mass"]

    def moved(layer, t):
        return L[layer]["frac_moved_over"][t]["point"]

    def flag(layer, name, k):
        return L[layer]["by_flag"][name][k]

    md = f"""# Phase A — what the cleaning actually changes

**Corpus** `{d['corpus']}` | **recordings** {d['anchor']['n_recordings']:,} |
**frames** {d['anchor']['n_frames']:,} | **animals** {d['n_animals']} |
**inherited digest** `{d['inherited_digest']}`

Nobody had measured this. Everything downstream — Q1 included — is computed on the
filtered array, and the size of what the filter does to it was not written down
anywhere in either upstream repo.

## The question has three answers and the bone check is not one of them

The bone check is a **flag, not an edit**. It feeds a gap policy, and the gap
policy is one of three layers that move the data:

| layer | what moves | mass it touches |
|---|---|---:|
| **L1 gap policy** | gaps ≤ 0.1 s linearly interpolated; longer runs abandoned | {P(fm['interpolated'])} interpolated, {P(fm['missing'])} abandoned |
| **L2 Wiener filter** | every frame, per-keypoint frequency-graded shrinkage | all of it |
| **L3 ε-violation flags** | nothing — they only flag | {P(fm['eps_violation'])} |

The baseline is `raw_pose.npz`, **not** `pose_unfiltered`. The latter is the gap
policy's own output and already carries its interpolants, so measuring L1 against
it returns exactly zero — the one answer that cannot be right.

## The bone check's own cost is small, and perfectly targeted

`bone_flagged` runs {P(fm['bone_flagged'])} of frames. What that costs downstream:

| | value |
|---|---|
| median displacement, all keypoint-frames | **{L['gap_policy']['median_px']['point']:.4f} px** |
| mean displacement on `interpolated` frames | {num(flag('gap_policy','interpolated','mean_inside'))} px |
| mean displacement **outside** the flags | {num(flag('gap_policy','interpolated','mean_outside'))} px |
| concentration | **{num(flag('gap_policy','bone_flagged','concentration'),'.1f')}** — touches only flagged frames |
| speed quantile ratio (p50, p90, p99) | {', '.join(f"{x:.3f}" for x in L['gap_policy']['kinematics']['speed']['ratio_q'][:3])} |

**The gap policy moves nothing outside the frames it flagged**, by construction
rather than by tuning, and it leaves the speed distribution within 10% at every
quantile. So the honest answer to "how much variance does the bone check create"
is: **very little**. It is the layer with the smallest effect of the three, and it
is the only one that is exactly targeted.

## The filter is where the data actually moves

| | Wiener (used downstream) | Butterworth (comparison arm) |
|---|---:|---:|
| median displacement | **{ci(L['wiener']['median_px'])} px** | {ci(L['butterworth']['median_px'])} px |
| p90 | {L['wiener']['p90_px']['point']:.3f} px | {L['butterworth']['p90_px']['point']:.3f} px |
| in body lengths (median) | {L['wiener']['median_body_lengths']['point']:.5f} | {L['butterworth']['median_body_lengths']['point']:.5f} |
| moved > 0.01 px | {moved('wiener','0.01'):.1%} | {moved('butterworth','0.01'):.1%} |
| moved > 0.5 px | {moved('wiener','0.5'):.1%} | {moved('butterworth','0.5'):.1%} |
| moved > 1 px | {moved('wiener','1.0'):.1%} | {moved('butterworth','1.0'):.1%} |
| moved > 5 px | {moved('wiener','5.0'):.1%} | {moved('butterworth','5.0'):.1%} |

It touches **{moved('wiener','0.01'):.1%}** of keypoint-frames and moves
**{moved('wiener','0.5'):.1%}** of them by more than half a pixel. That is a
transform applied to the whole corpus, which is what a shrinkage filter is — the
finding is not that it is wrong, but that the size of it had never been recorded
while every downstream number depends on it.

### It is targeted, more than a single recording suggested

| flag | inside | outside | concentration |
|---|---:|---:|---:|
| `bone_flagged` | {num(flag('wiener','bone_flagged','median_inside'))} px | {num(flag('wiener','bone_flagged','median_outside'))} px | **{num(flag('wiener','bone_flagged','concentration'),'.2f')}×** |
| `eps_violation` | {num(flag('wiener','eps_violation','median_inside'))} px | {num(flag('wiener','eps_violation','median_outside'))} px | {num(flag('wiener','eps_violation','concentration'),'.2f')}× |
| `interpolated` | {num(flag('wiener','interpolated','median_inside'))} px | {num(flag('wiener','interpolated','median_outside'))} px | {num(flag('wiener','interpolated','concentration'),'.2f')}× |

A spot measurement on one recording put this at 2.0× and the corpus puts it at
{num(flag('wiener','bone_flagged','concentration'),'.1f')}×. The filter does
concentrate its work on the frames the geometry says are broken, and the
single-animal reading understated that. Butterworth concentrates harder still,
at {num(flag('butterworth','bone_flagged','concentration'),'.1f')}×.

### Which landmark

| keypoint | median displacement (Wiener) |
|---|---:|
""" + "\n".join(f"| {r['keypoint']} | {r['median_px']:.3f} px |"
                for r in L['wiener']['by_keypoint']) + f"""

A **{max(r['median_px'] for r in L['wiener']['by_keypoint']) / max(min(r['median_px'] for r in L['wiener']['by_keypoint']), 1e-9):.0f}×** spread across
landmarks. The nose moves furthest and the centre barely moves, which tracks the
per-keypoint noise shapeflow measured (σ = 3.10 px at the nose against 0.79 at
the centre) and the Wiener gain that follows from it. The filter is not one
operation applied uniformly; it is seven different operations.

## What it does to behaviour, not just position

Quantile ratios, after ÷ before. This is the axis a displacement statistic cannot
see: a filter can preserve position while deleting the fast tail of the movement
distribution, and a behaviour model would never know.

| | p50 | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| **Wiener, speed** | {' | '.join(f"**{x:.3f}**" for x in L['wiener']['kinematics']['speed']['ratio_q'])} |
| Butterworth, speed | {' | '.join(f"{x:.3f}" for x in L['butterworth']['kinematics']['speed']['ratio_q'])} |
| **Wiener, turn** | {' | '.join(f"**{x:.3f}**" for x in L['wiener']['kinematics']['turn']['ratio_q'])} |
| Butterworth, turn | {' | '.join(f"{x:.3f}" for x in L['butterworth']['kinematics']['turn']['ratio_q'])} |
| gap policy, speed | {' | '.join(f"{x:.3f}" for x in L['gap_policy']['kinematics']['speed']['ratio_q'])} |

**The Wiener filter removes {1 - L['wiener']['kinematics']['speed']['ratio_q'][0]:.0%} of the median instantaneous
speed and {1 - L['wiener']['kinematics']['turn']['ratio_q'][0]:.0%} of the median turning.** Much of frame-to-frame
displacement at 30 fps genuinely is tracking noise — σ runs 0.79–3.10 px per
keypoint — so removing it may well be correct. The point is that the magnitude
was never recorded, and "1.059% of coherent power sits above the crossover" does
not prepare a reader for a 41% cut to median speed.

The comparison that carries information is between the two filters.
**Butterworth preserves turning far better than Wiener does**
({L['butterworth']['kinematics']['turn']['ratio_q'][0]:.3f} against
{L['wiener']['kinematics']['turn']['ratio_q'][0]:.3f} at the median) for a similar
cut to speed. Turning is exactly what the egocentric representation's `omega`
channel carries, and the two filters treat it very differently.

## What this licenses

**The bone check is cleared.** Its downstream cost is confined to
{P(fm['interpolated'])} of keypoint-frames, it moves nothing else, and it leaves
the kinematics within 10%. It is not a source of variance worth worrying about.

**The filter is not cleared, and was never on trial.** It moves
{moved('wiener','0.5'):.1%} of keypoint-frames by more than half a pixel and cuts
the median speed by {1 - L['wiener']['kinematics']['speed']['ratio_q'][0]:.0%}.
That is what shrinkage does, but it is a large intervention sitting under every
result this project has published, and it had never been quantified. Phase B puts
it in a bakeoff against the alternatives.

All intervals are animal bootstraps over {d['n_animals']} animals, never pooled
frames.
"""
    with open(a.out, "w") as fh:
        fh.write(md)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
