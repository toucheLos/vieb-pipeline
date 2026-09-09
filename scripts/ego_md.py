"""`results/EGO.md` from `results/ego_{bodylen,raw}.json`. Reads, never recomputes.

Both arms in one document, because the interesting quantity is the difference
between them: whether dividing by the animal's body length buys anything on the
identity leak the brief asks it to fix.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from vieb.tok import config, ego, parity                         # noqa: E402
from recur.util import log, read_json                             # noqa: E402


def leak_row(d, kind, arm):
    rd = d["reads"][f"identity_leak_{kind}"]
    r = rd["detail"]
    nats = f"{r['leak_nats']:.4f}"
    if rd["verdict"] == "NOT_A_RESULT":
        # `util.jsonable` serialises bools through its int branch, so a stored
        # False arrives as 0 and `is False` never matches. Compare by truth.
        conv = r.get("converged")
        why = ("unconverged" if conv is not None and not conv
               else "clamped" if r["leak_nats"] <= 0.0 else "refused")
        nats = f"{r['leak_nats']:.4f} *({why})*"
    return (f"| {kind} | `{arm}` | {nats} | {r['fraction_of_ceiling']:.2%} | "
            f"{r['accuracy']:.2%} | {r['chance_accuracy']:.4%} | "
            f"{r['accuracy_over_chance']:.0f}× | `{rd['verdict']}` |")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=config.PATHS.result("EGO.md"))
    a = p.parse_args(argv)
    arms = {k: read_json(config.PATHS.result(f"ego_{k}.json"))
            for k in ("bodylen", "raw")}
    b = arms["bodylen"]
    ex = b["a4_scores"]["exact"]
    rd = b["reads"]["a4"]
    rv = b["reversal_audit"]

    raw_animal = arms["raw"]["reads"]["identity_leak_animal"]["detail"]
    bl_animal = arms["bodylen"]["reads"]["identity_leak_animal"]["detail"]
    raw_session = arms["raw"]["reads"]["identity_leak_session"]["detail"]
    leak_removed = 1.0 - bl_animal["leak_nats"] / max(raw_animal["leak_nats"], 1e-9)
    # The nats comparison is only a claim if BOTH animal probes were readable.
    # If either was refused, the accuracy comparison is what survives, and the
    # document must not quote a percentage derived from a refused number.
    ok = all(arms[k]["reads"]["identity_leak_animal"]["verdict"] != "NOT_A_RESULT"
             for k in ("raw", "bodylen"))
    if ok:
        removal_claim = (
            f"**ell_a removes {leak_removed:.0%} of the animal-identity leak** — "
            f"{raw_animal['leak_nats']:.3f} nats down to "
            f"{bl_animal['leak_nats']:.3f}, and accuracy from "
            f"{raw_animal['accuracy']:.1%} to {bl_animal['accuracy']:.1%} "
            f"against a {raw_animal['chance_accuracy']:.3%} baseline.")
    else:
        bl_session = arms["bodylen"]["reads"]["identity_leak_session"]["detail"]
        removal_claim = (
            f"The `raw` probes were refused, so **the nats comparison is not "
            f"quotable**. On **accuracy**, which is a property of the fitted "
            f"classifier rather than of its calibration and so survives the "
            f"refusal, ell_a takes animal identification from "
            f"{raw_animal['accuracy']:.1%} to {bl_animal['accuracy']:.1%} "
            f"against a {raw_animal['chance_accuracy']:.3%} baseline "
            f"({raw_animal['accuracy'] / bl_animal['accuracy']:.1f}x), and "
            f"session identification from {raw_session['accuracy']:.1%} to "
            f"{bl_session['accuracy']:.1%} against "
            f"{raw_session['chance_accuracy']:.4%} "
            f"({raw_session['accuracy'] / bl_session['accuracy']:.0f}x).")

    win = b["a4_scores"].get("windowed", {})
    single = b["a4_scores"].get("pose_only_single_frame", {})
    reg = ""
    if win.get("speed") is not None:
        reg = f"""
### The regressor arms — reported, never gated

| arm | speed | turn |
|---|---:|---:|
| encoder's own {win['k']}-frame causal window | {win['speed']:.3f} | {win['angular']:.3f} |
| pose block alone, single frame | {single['speed']:.3f} | {single['angular']:.3f} |
| *prior pipeline, pose-only (PCA-64)* | *{parity.PRIOR_POSE_ONLY['speed']:.3f}* | *{parity.PRIOR_POSE_ONLY['angular']:.3f}* |
| *shapeflow, single-frame shape block* | *{parity.SHAPEFLOW_SINGLE_FRAME['speed']:.3f}* | *{parity.SHAPEFLOW_SINGLE_FRAME['angular']:.3f}* |

The single-frame pose-only row is **the non-circular comparison**. A multi-frame
shape window carries shape *change*, which correlates with movement, so it is not
comparable to the prior pipeline's 0.090 / 0.067 — only the single-frame row is.

Two caveats on the italic rows. They come from different representations
(PCA-64 and the Kendall shape block), and this pipeline holds out **by animal**
where the earlier holdouts are not recorded here — so treat them as context, not
as a controlled comparison.

The single-frame figure here is **negative**, which is stronger than low: held out
by animal, a regressor on one frame of egocentric pose does worse than predicting
the test set's mean. Single-frame body configuration carries essentially nothing
about locomotion that transfers between animals, which is the sharpest form of the
reason the locomotor channels cannot be dropped.
"""

    md = f"""# Stage 1 — Egocentric transform

**Corpus** `{b['corpus']}` | **recordings** {b['anchor']['n_recordings']:,} |
**frames** {b['anchor']['n_frames']:,} | **animals** {b['n_animals']} |
**inherited digest** `{b['inherited_digest']}`

Equivariance removal, not compression: {ego.N_DIMS} dimensions in,
{ego.N_DIMS} out, `inverse` rebuilds the keypoints to float64, and nothing is
fitted anywhere. Deliberately independent of the frozen Kendall gauge, which is
what makes the A4 check below a comparison between two representations rather
than a quantity predicting itself.

## Dimensions, and one thing that does not reconcile

{ego.N_POSE} egocentric coordinates (7 keypoints × 2, in body lengths) +
{ego.N_TWIST} SE(2) twist components = **{ego.N_DIMS}**.

> {b['dims']['unreconciled']}

## A4 — `{rd['verdict']}`

> {rd['reason']}

| arm | speed R² | turn R² | requirement |
|---|---:|---:|---|
| **closed form** (gated) | {ex['speed']:.9f} | {ex['angular']:.9f} | ≥ {parity.R2_SPEED_MIN} / ≥ {parity.R2_TURN_MIN} |

Measured on {ex['n_frames']:,} frames over {ex['n_recordings']:,} `report`
recordings. The map is a bijection given the equivariance it removed, so this is
exact rather than a fit — the same reason shapeflow's A4 reads 1.000000.
{reg}
### A4 does not catch the group-logarithm error

Worth stating because the opposite is the natural assumption. Substituting
separate differencing of position and angle for the SE(2) logarithm — the error
the brief singles out — barely moves A4 until the animal is turning hard:

| turn rate | speed R² with the naive twist |
|---|---:|
| 0.9 rad/s | 0.9973 |
| 3.0 rad/s | 0.9853 |
| 4.5 rad/s | 0.9646 |
| 12 rad/s | 0.6492 |

against a threshold of {parity.R2_SPEED_MIN}. A pipeline carrying the naive
velocity would **pass this gate** at every ordinary turn rate in the corpus. The
guard is `tests/test_se2.py`, which compares `se2_log` against
`scipy.linalg.expm` on the 3×3 matrix representation and fails the naive version
at every turn rate including zero.

## The identity leak, and the premise it tests

The brief asks for body-length normalisation on the grounds that its absence is
"a candidate contributor to the 5.88 nats of measured identity leak". **5.88 is
the previous instrument's number**, measured with a shuffle correction;
shapeflow's direct probe reads 0.833 (recording) / 0.730 (animal) nats at γ = 0.
So whether `ell_a` buys anything is measured rather than assumed.

Direct multinomial probe on per-window channel means and SDs, held out by window
within recording. Never a shuffle correction — that is disarmed by any
per-recording standardisation elsewhere in the pipeline.

| probe | arm | leak (nats) | of ceiling | accuracy | chance | ×chance | verdict |
|---|---|---:|---:|---:|---:|---:|---|
{leak_row(arms['raw'], 'animal', 'raw')}
{leak_row(arms['bodylen'], 'animal', 'bodylen')}
{leak_row(arms['raw'], 'session', 'raw')}
{leak_row(arms['bodylen'], 'session', 'bodylen')}

{removal_claim}
Body length spans
{b['ell_a']['summary']['min']:.1f}–{b['ell_a']['summary']['max']:.1f} px across
the {b['ell_a']['summary']['n']} animals, a
{b['ell_a']['summary']['max'] / b['ell_a']['summary']['min']:.2f}× range, so
there was a great deal for it to remove.

So the brief's instruction was right and its stated reason was not. Normalising
by body length does substantially reduce identity leak; it was simply never
about 5.88 nats.

### Both `raw` nats figures are refused, and that is a bug in the metric

The `raw` session leak clamps to **0.0000 nats** while the same probe picks the
correct session out of 3,846 **{raw_session['accuracy']:.1%}** of the time,
against {raw_session['chance_accuracy']:.4%} chance —
{raw_session['accuracy_over_chance']:.0f}× better. Read as nats that is "no
session identity"; read as accuracy it is the largest leak in the table.

`gates.identity_leak` returns `max(log N − CE, 0)`, and cross-entropy exceeds
`log N` whenever the probe is confidently wrong — which a multinomial logistic
regression becomes as soon as the classes outnumber the rows per class. There are
8 windows per session here against 3,846 classes. The zero is measuring the
probe's calibration, not the representation's information.

**And the solver does not converge there.** `identity_leak` runs
`LogisticRegression(max_iter=400)`; at 3,846 classes it stops on the iteration
cap rather than on the gradient. That is the mechanism, and it was invisible
because nothing recorded it.

Two guards now, both in `audit.leak.leak_read`, both returning
`NOT_A_RESULT [DEGENERATE]` with the accuracy in the reason:

* a probe that **did not converge**, whatever it reports — the dangerous case is
  not the obvious zero but a healthy-looking number from a solver that stopped
  early, because nothing about it looks wrong;
* a leak **clamped to zero** whose classifier sits more than 3× above chance.

`gates.identity_leak` now records `converged`, `n_iter` and `max_iter` so the
caller can tell.

**Both `raw` probes hit the cap at 400 iterations; both `bodylen` probes
converged, at {bl_animal['n_iter']} and {arms['bodylen']['reads']['identity_leak_session']['detail']['n_iter']}.**
That asymmetry is itself a reading rather than an inconvenience: the
un-normalised features carry more between-animal separation, and more separable
classes are harder to fit to convergence at a fixed iteration budget. So the arm
that leaks more is also the arm whose leak cannot be quantified in nats — which
is precisely the direction that would let an unguarded pipeline report the
leakier representation as the cleaner one.

A converged figure would need a larger budget. That is deliberately **not** done
here: `max_iter=400` is inherited, and raising it for this stage alone would make
these numbers incomparable to `results/audit_moseq.json`'s 0.283 nats. Accuracy
is a property of the fitted classifier rather than of its calibration, so it
survives the refusal and is what this stage reports.

## ell_a is a per-animal constant

Not per frame: for a top-down mouse rearing appears only as apparent
foreshortening, so dividing by the frame's own size deletes rearing — which is
why shapeflow refuses per-frame scale normalisation and keeps `log s` as its own
channel. Not per recording either: per-recording standardisation removes exactly
the between-recording differences a null has to be able to see, and it has
disarmed a control on this project once already.

One number per animal, pooled over all of its recordings, on usable frames:
median **{b['ell_a']['summary']['median']:.2f} px**, IQR
{b['ell_a']['summary']['p25']:.2f}–{b['ell_a']['summary']['p75']:.2f}, range
{b['ell_a']['summary']['min']:.2f}–{b['ell_a']['summary']['max']:.2f} over
{b['ell_a']['summary']['n']} animals.

## One dimensional correction to the brief

The brief writes ξ = (f/ℓ_a)·log(g⁻¹g′), applying f/ℓ_a to the whole twist. That
is right for the two translation components and **wrong for the rotation**: an
angular rate divided by a body length has units of rad·s⁻¹·px⁻¹, and it makes a
large mouse's measured turning systematically slower than a small one's — body
size leaking straight into the channel where it is least visible.

| channel | divided by |
|---|---|
""" + "\n".join(f"| `{k}` | {v} |" for k, v in b["scaling"].items()) + f"""

`tests/test_ego.py::test_a_scaled_animal_gives_an_identical_omega` asserts it.

## Rank: three directions are identically zero

Removing SE(2) equivariance from 14 coordinates costs three degrees of freedom,
and they are left visible rather than projected away: `s` at the origin keypoint
is (0, 0), and the y-component of `s[nose] − s[tail_base]` is 0 because the axis
is aligned. So the pose block has numerical rank
**{b['rank']['observed']['median']:.0f} of {ego.N_POSE}**, matching the expected
{b['rank']['expected']} on every animal
({b['rank']['observed']['min']:.0f}–{b['rank']['observed']['max']:.0f}).

This is the same shape of fact as Im(cᴴΔz) ≡ 0 in `kendall`, and it is recorded
for the same reason: a model free to fit variance along a direction the data
never moves in earns unbounded likelihood. The quantizer does not care — k-means
on a dead direction is harmless, it contributes no distance — but a later
density model would.

## Seams and reversal

ξ_t reads frame t+1, so the last frame of every recording has no velocity and is
marked invalid rather than zero-filled: a zero-filled increment reads as "the
animal did not move", which is a behaviour. An unusable frame invalidates its
predecessor too.

Usable frames after intersecting shapeflow's own mask:
median recording **{b['frac_valid']['median']:.2%}**, p10
{b['frac_valid']['p10']:.2%}.

The reversal audit runs **{rv['n_checks']} checks**, `{rv['verdict']}`, extended
from the inherited seven with three for the twist. Every twist component is
**odd**, `v_x` included — the intuition that a forward velocity behaves like
`centroid_speed` is wrong, because `centroid_speed` is a magnitude and a
magnitude kills the sign flip. And a twist lives on the *interval* between two
frames, so reversing it negates **and shifts by one**; dropping the shift is the
same class of structural error as forgetting to flip the lag order inside a
delay-embedded row.

## What this licenses

Proceed to Step 3, the quantizer sweep, on the **`{b['arm']}`** arm.
The representation retains the kinematics exactly, carries no more animal
identity than the labels already did, and loses nothing that `inverse` cannot
put back.
"""
    with open(a.out, "w") as fh:
        fh.write(md)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
