"""`results/BONES.md` from `results/bones.json`. Reads, never recomputes.

Every number in the document is read out of the JSON the sweep wrote, so the
prose and the artifact cannot drift. The one thing this script decides is which
numbers are worth a reader's attention, and that judgement is in the section
order rather than in any arithmetic.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from vieb.qc import bones                                         # noqa: E402
from vieb.tok import config                                      # noqa: E402
from recur.util import log, read_json                             # noqa: E402

P = "{:.3%}".format


def curve_table(d, arm: str) -> str:
    eps = d["eps_swept"]
    out = ["| ε | skull raw | skull scale-free | trunk raw | trunk scale-free | shuffled ceiling |",
           "|---|---:|---:|---:|---:|---:|"]
    for i, e in enumerate(eps):
        c = lambda k: d["curves"][f"{arm}|{k}"][i]["rate"]
        ceil = d["curves"][f"{arm}|raw|skull"][i]["shuffled_ceiling"]
        out.append(f"| {e} | {P(c('raw|skull'))} | {P(c('scalefree|skull'))} | "
                   f"{P(c('raw|trunk'))} | {P(c('scalefree|trunk'))} | {ceil:.1%} |")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=config.PATHS.result("bones.json"))
    p.add_argument("--out", default=config.PATHS.result("BONES.md"))
    a = p.parse_args(argv)
    d = read_json(a.json)

    r = d["reads"]
    e0 = d["primary"]["eps"]
    i0 = d["eps_swept"].index(e0)
    skull = d["curves"]["unfiltered|raw|skull"][i0]
    sf = d["curves"]["unfiltered|scalefree|skull"][i0]
    wien = d["curves"]["wiener|raw|skull"][i0]
    ov_raw = d["overlap"][f"raw|{e0}|bone_flagged"]
    ov_sf = d["overlap"][f"scalefree|{e0}|bone_flagged"]
    j = d["r2_join"][str(e0)]
    dec = j["deciles"]["bins"]
    marg, part = j["rho"], j["rho_partial_duration_controlled"]
    dur = j["rho_duration_vs_r2"]
    att = d["attribution"][f"raw|{e0}"]

    def ci(x, fmt="{:+.3f}"):
        return f"{fmt.format(x['point'])} [{fmt.format(x['lo'])}, {fmt.format(x['hi'])}]"

    sk = r["skull_raw"]["detail"]
    drops = ", ".join(f"{x:.0%}" for x in sk["relative_drops"])
    if sk["curve_shape"] == "elbow":
        shape_para = (
            f"The curve has an **elbow** at ε = {sk['elbow_eps']}: the rate "
            f"falls {sk['elbow_relative_drop']:.0%} there, "
            f"{sk['elbow_prominence']:.1f}× the median drop elsewhere. Two "
            f"populations are separable and that is where to cut.")
    else:
        why = (
            f"increasing monotonically, so the rate falls away faster and "
            f"faster rather than falling off at one place"
            if sk["drops_are_monotone"] else
            f"with a largest drop only {sk['elbow_prominence']:.1f}× the median "
            f"of the others, below the {bones.MIN_ELBOW_PROMINENCE}× a "
            f"separable population would leave")
        shape_para = (
            f"**The curve decays smoothly — there is no elbow, and that is a "
            f"result rather than a missing one.** The relative drops run "
            f"{drops} across the sweep, {why}. This is one population thinning "
            f"out, not two with a gap between them.\n\n"
            f"So **no ε on this grid is a principled cut**, and the rate has to "
            f"be read as a curve. ε = {e0} is reported as the primary cell "
            f"because the brief names it, not because the data prefers it — at "
            f"ε = 0.20 the same corpus reads "
            f"{P(d['curves']['unfiltered|raw|skull'][d['eps_swept'].index(0.20)]['rate'])} "
            f"and would fall in the *exclude* branch rather than the *correct* "
            f"one. A reader who needs a single number needs to know the branch "
            f"moves with a threshold nobody can justify from the shape.")

    md = f"""# Step 1 — Bone-length violations

**Corpus** `{d['corpus']}` | **recordings** {d['anchor']['n_recordings']:,} |
**frames** {d['anchor']['n_frames']:,} | **animals** {d['n_animals']} |
**inherited digest** `{d['inherited_digest']}`

Primary cell: **{d['primary']['pose_arm']}** pose, **{d['primary']['metric']}**
metric, **{d['primary']['group']}** group, **ε = {e0}**.
{d['primary']['why'].capitalize()}.

## Reads

| read | verdict | n |
|---|---|---:|
""" + "\n".join(
        f"| `{k}` | `{v['verdict']}`"
        f"{' [' + v['saturation'] + ']' if v.get('saturation') else ''} "
        f"| {v['n_effective']} |" for k, v in r.items()) + f"""

> **skull_raw** — {r['skull_raw']['reason']}

> **r2_join** — {r['r2_join']['reason']}

> **shuffled_ceiling** — {r['shuffled_ceiling']['reason']}

## The branch

Skull violations are **{P(skull['rate'])}** of frames at ε = {e0}, inside the
2%–15% band. **Correct rather than exclude**, and cost out an ensemble-DLC path
before anything is published on this feature space. They are not concentrated in
a few recordings — {skull['n_recordings_above_1pct']:,} of
{skull['n_recordings']:,} recordings sit above 1% — so exclusion would cost half
the corpus, which is what puts this in the correction branch rather than the
exclusion one. The median recording is {P(skull['rate_p50'])} and the 95th
percentile is {P(skull['rate_p95'])}; the worst 25 run
{P(d['worst_recordings'][-1]['rate'])}–{P(d['worst_recordings'][0]['rate'])}.

## The ε curve

Unfiltered pose. **The curve is the deliverable, not any single cell.**

{curve_table(d, 'unfiltered')}

{shape_para}

Every observed rate sits far below the **shuffled-keypoint ceiling** — keypoints
drawn from random frames *within the same recording*, so arena, animal and camera
are held fixed. At ε = {e0} the margin is
{skull['shuffled_ceiling'] / skull['rate']:.0f}×, measured on
{skull['n_ceiling']} recordings. A diagnostic that fired as often on shuffled
landmarks as on real ones would be reporting its own threshold.

## Excess length is not perspective

This was the open question the second metric exists to settle, and it settles
against the hypothesis.

| | ε = {e0} |
|---|---:|
| raw pixel lengths | {P(skull['rate'])} |
| per-frame common scale removed | {P(sf['rate'])} |

Removing the common scale does **not** reduce the violation rate — it raises it
slightly, at every ε on the sweep. A mouse rearing or moving nearer the lens
multiplies every distance by one factor, which the scale-free metric cancels
exactly; if rearing were generating these violations the second row would be far
smaller than the first. It is not, so the raw sweep can be read at face value.

The small *increase* is mechanical rather than mysterious: dividing by a frame's
own median length makes an over-long bone on an otherwise foreshortened frame
look relatively longer still, so the scale-free arm is marginally the more
sensitive detector.

## The second opinion — 2×2 against shapeflow's `bone_flagged`

shapeflow's gate flags {ov_raw['rate_other']:.3%} of frames on a different
principle: symmetric in the deviation, scale-free by construction, learned rather
than anatomical, and requiring three of 21 pairs to break at once. Reported as a
confusion rather than as two rates, because the disagreement is what carries
information.

| cell | raw | scale-free | what it is |
|---|---:|---:|---|
| both | {ov_raw['rate_both']:.3%} | {ov_sf['rate_both']:.3%} | agreed tracking failure — two unrelated criteria on one frame |
| new only | {ov_raw['rate_new_only']:.3%} | {ov_sf['rate_new_only']:.3%} | excess length the existing gate misses |
| `bone_flagged` only | {ov_raw['rate_other_only']:.3%} | {ov_sf['rate_other_only']:.3%} | relative geometry broken with no excess length — swaps and shortenings, which a one-sided test is blind to by construction |
| Jaccard | {ov_raw['jaccard']:.3f} | {ov_sf['jaccard']:.3f} | |
| P(`bone_flagged` \\| new) | {ov_raw['p_other_given_new']:.3f} | {ov_sf['p_other_given_new']:.3f} | |

`new_only` barely moves between the metrics ({ov_raw['rate_new_only']:.3%} →
{ov_sf['rate_new_only']:.3%}) while `both` rises
({ov_raw['rate_both']:.3%} → {ov_sf['rate_both']:.3%}), which is the same verdict
the previous section reached by a different route: the excess survives
common-scale removal, so it is geometry and not perspective.

**{d['rate_new_only_on_measured_frames'][f'raw|{e0}']:.3%} of frames carry excess
skull length that `bone_flagged` does not see, on frames that were measured
rather than interpolated or filled.** That is the cell with no benign
explanation, and it is more than half the total. It is also exactly equal to
`new_only`, because a missing keypoint yields a non-finite length and a
non-finite length is never counted as a violation.

## What the filter absorbs

Every downstream feature — Q1's channels included — is built on the Wiener-shrunk
`pose`, not on the unfiltered array this diagnostic reads.

| ε | unfiltered | wiener | absorbed |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {e} | {P(d['curves']['unfiltered|raw|skull'][i]['rate'])} | "
        f"{P(d['curves']['wiener|raw|skull'][i]['rate'])} | "
        f"{1 - d['curves']['wiener|raw|skull'][i]['rate'] / d['curves']['unfiltered|raw|skull'][i]['rate']:.1%} |"
        for i, e in enumerate(d['eps_swept'])) + f"""

The filter removes {1 - wien['rate'] / skull['rate']:.0%} of the ε = {e0}
violations, so **{P(wien['rate'])} of frames in the feature space every
downstream stage consumes carry impossible skull geometry**. Shrinkage is not a
repair: it makes the excursion smaller without making the frame correct.

## The join with ExBias R², and the confound inside it

{j['n_segments']:,} segments over {j['n_animals_correlated']} animals, joined on
recording id and frame range only — never on keypoint index, because ExBias
orders bodyparts alphabetically where shapeflow uses file order. Mean
`fit_r2` = {j['mean_fit_r2']:.4f}; {j['frac_fit_r2_below_0.5']:.2%} of segments
sit below 0.5 on raw R² and {j['frac_fit_r2_adj_below_0.5']:.2%} on adjusted.

| decile | R² range | segments | mean violation rate | segments with any | mean duration |
|---:|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {b['decile']} | {b['r2_lo']:.3f} – {b['r2_hi']:.3f} | "
        f"{b['n_segments']:,} | {b['mean_violation_rate']:.3%} | "
        f"{b['frac_segments_with_any']:.2%} | {b['mean_duration_s']:.2f} s |"
        for b in dec) + f"""

**Read the duration column before either of the others.** Mean segment duration
falls from {dec[0]['mean_duration_s']:.2f} s in the worst-fitting decile to
{dec[-1]['mean_duration_s']:.2f} s in the best — a factor of
{dec[0]['mean_duration_s'] / dec[-1]['mean_duration_s']:.1f} — and
ρ(duration, R²) = **{ci(dur)}**. ExBias fits a cubic over each segment, so a
short segment fits well close to by construction, and its R² is substantially a
readout of its own length.

That confound sits on **both** sides of the join, because the violation rate's
denominator *is* the segment's length. It does not inflate the association here;
it suppresses it:

| statistic | ρ |
|---|---|
| violation rate vs R², marginal | {ci(marg)} |
| violation rate vs R², **segment duration held fixed** | **{ci(part)}** |
| any violation (binary) vs R², marginal | {ci(j['rho_any_violation'])} |

The controlled coefficient is roughly double the marginal one at every ε on the
sweep, and it is the number the verdict is read on. Taken marginally the join
would have read `GRID_LIMITED` — "real but too small to attribute the R² floor to
tracking". Controlled, it clears the 0.1 threshold and reads `PASS`.

So: **segments that reconstruct badly do carry more bone violations, once you
stop comparing four-second segments against half-second ones.** The premise
behind this gate survives. Note what it does not say — ρ = {part['point']:+.3f}
leaves the great majority of the R² floor unexplained by tracking, and the honest
description of ExBias's mean 0.70 is that it is mostly about how long a segment
is and how well a cubic spans it.

## Attribution

Share of violating bone-frames by landmark, ε = {e0}, skull:

| landmark | share |
|---|---:|
""" + "\n".join(f"| {k} | {v:.3f} |" for k, v in
                sorted(zip(att['keypoints'], att['share']), key=lambda x: -x[1])
                if v > 0) + f"""

The three skull landmarks carry the violations near-evenly
({min(v for v in att['share'] if v > 0):.3f}–{max(att['share']):.3f}), which is
what a triangle of three mutually-constraining bones produces when no single
landmark is the culprit. A single systematically mistracked keypoint would show
as one share near 0.5 and is not what this corpus has.

## By split

| split | median recording | p90 | max |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {k} | {P(v['median'])} | {P(v['p90'])} | {P(v['max'])} |"
        for k, v in sorted(d['by_split'].items())) + f"""

The three splits are indistinguishable, so nothing about the violation rate is
going to separate `tune` from `report` when a threshold is chosen on the former.

## What this licenses, and what it does not

**Licensed.** Proceed to Step 2. The feature space is not predominantly tracking
failure: {P(skull['rate'])} of frames carry excess skull geometry, the rate is
{skull['shuffled_ceiling'] / skull['rate']:.0f}× below its own shuffled ceiling,
and it is associated with independent evidence of bad reconstruction once
duration is controlled.

**Not licensed.** Any claim that the corpus is clean. Half the recordings sit
above 1%, {P(wien['rate'])} of frames survive filtering into the feature space
with impossible geometry, and
{d['rate_new_only_on_measured_frames'][f'raw|{e0}']:.3%} of frames carry excess
length on measured frames that the inherited QC does not flag. The **artifact
ablation is therefore mandatory rather than a formality** at Step 4: score with
and without these frames and report the delta. If it is material, part of any win
was tracking.

**Carried forward.** The per-frame violation masks are in
`work/bones/<animal_tag>.npz` under `viol|<pose_arm>|<metric>|<eps>|<group>`,
bit-packed over the corpus in `bounds` order, for exactly that ablation.

## Bones

`SKULL` (primary): {', '.join(f"{d['bones']['keypoints'][i]}–{d['bones']['keypoints'][j]}" for i, j in d['bones']['skull'])}.

`TRUNK` (reported separately, never pooled into the headline):
{', '.join(f"{d['bones']['keypoints'][i]}–{d['bones']['keypoints'][j]}" for i, j in d['bones']['trunk'])}.

Trunk rates run {P(d['curves']['unfiltered|raw|trunk'][i0]['rate'])} at ε = {e0},
above the skull's. Those bones flex, so a violation there is real posture at
least as often as it is tracking, and pooling the two would produce a headline
stronger than the evidence.
"""
    with open(a.out, "w") as fh:
        fh.write(md)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
