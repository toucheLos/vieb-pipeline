# Stage 1 — Egocentric transform

## Step 1R — re-run on the carried pose arm (post-freeze)

The original run consumed `clean["pose"]`, which `shapeflow/results/clean.json`
records as `filter.default = wiener`. `F3_PREPROCESSING_FREEZE.md` SS1 carries
`raw`, `viterbi` and `disposition` onto the MDL branch and lists Wiener as *not
benchmarkable, therefore not carried*. The stage this feeds measures memory
depth, and a low-pass filter manufactures exactly that — so it was re-run.

| | value |
|---|---|
| pose arm | `raw` = `held_array(pose_unfiltered, missing)` |
| scale arm | `bodylen` |
| A4, closed form | speed **1.000000**, angular **1.000000** on 6,403,616 frames |
| reversal audit | **PASS**, 10 checks (7 inherited + 3 twist), 0 failed |
| rank | 11 of 14 on every animal |
| identity leak | animal **PASS**, session **PASS** |
| inherited digest | `198eb14ff258c7f6`, `preprocessing_freeze: F3` |
| result | `results/ego_raw_bodylen.json` |

Two predictions made in advance were **wrong** and are corrected here rather than
quietly dropped. `frac_valid` was expected to fall: it does not move at all
(0.95957 on both arms for animal 103), because validity comes from shapeflow's
`usable` mask, which is arm-independent. And `held_array` was expected to leave
NaN where the filter had interpolated: it does not — it interpolates, which is
what the gap policy fed the filter in the first place. `ell_a` did move, 110.20
to 111.73 px on that animal.

What did change is concentrated where it matters: the median absolute difference
between the two arms' channels is **0.0037** in the pose block and **0.0179** in
the twist block — the filter's effect is five times larger in the velocity
channels, which is where temporal structure would be manufactured.

One regression was found and fixed on the way: `scripts/ego.py` imported
`recur.geom.reversal` rather than `vieb.tok.reversal`, so the audit had been
running the inherited **seven** checks with none of the three SE(2) twist checks
— and still reporting PASS, because seven passing checks do pass. The script now
refuses to proceed unless the composed audit ran.

---

## The scale-arm comparison below is pre-freeze

Everything from here down is the `bodylen`-against-`unitlen` comparison on the
**Wiener** array, kept because its question is about `ell_a` and is settled. It
is not a description of the coordinates Step 2 consumes.

---


**Corpus** `luna` | **recordings** 3,846 |
**frames** 22,355,989 | **animals** 298 |
**inherited digest** `fe62fec33da4485a`

Equivariance removal, not compression: 17 dimensions in,
17 out, `inverse` rebuilds the keypoints to float64, and nothing is
fitted anywhere. Deliberately independent of the frozen Kendall gauge, which is
what makes the A4 check below a comparison between two representations rather
than a quantity predicting itself.

## Dimensions, and one thing that does not reconcile

14 egocentric coordinates (7 keypoints × 2, in body lengths) +
3 SE(2) twist components = **17**.

> the brief says '28 dims stay 28 dims'; at 7 keypoints the two formulas give 14 + 3 = 17 and 28 is not reconstructible from them. Recorded rather than papered over with invented channels

## A4 — `PASS`

> the egocentric representation reconstructs raw centroid speed at R^2 = 1.000000 (requirement >= 0.98) and raw angular velocity at R^2 = 1.000000 (requirement >= 0.94), in closed form from (s, xi, ell_a) rather than through a regressor. The map is a bijection given the equivariance it removed, so this is exact and not a fit. A gradient-boosted regressor on the encoder's own 5-frame causal window, held out by animal, recovers 0.893 and 0.799. The pose block alone at a single frame falls to -0.185 and -0.065, against the prior pipeline's 0.090 / 0.067 and shapeflow's 0.175 / 0.065 -- which is the non-circular comparison, and is why the locomotor channels cannot be dropped.

| arm | speed R² | turn R² | requirement |
|---|---:|---:|---|
| **closed form** (gated) | 1.000000000 | 1.000000000 | ≥ 0.98 / ≥ 0.94 |

Measured on 6,403,616 frames over 1,149 `report`
recordings. The map is a bijection given the equivariance it removed, so this is
exact rather than a fit — the same reason shapeflow's A4 reads 1.000000.

### The regressor arms — reported, never gated

| arm | speed | turn |
|---|---:|---:|
| encoder's own 5-frame causal window | 0.893 | 0.799 |
| pose block alone, single frame | -0.185 | -0.065 |
| *prior pipeline, pose-only (PCA-64)* | *0.090* | *0.067* |
| *shapeflow, single-frame shape block* | *0.175* | *0.065* |

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

### A4 does not catch the group-logarithm error

Worth stating because the opposite is the natural assumption. Substituting
separate differencing of position and angle for the SE(2) logarithm — the error
the brief singles out — barely moves A4 until the animal is turning hard:

> **UNSOURCED.** The four R² values that stood here (0.9973, 0.9853, 0.9646,
> 0.6492 at 0.9, 3.0, 4.5 and 12 rad/s) are hardcoded in this generator and trace
> to no artifact. Reconstructing the obvious procedure -- synthetic mouse at each
> turn rate, exact SE(2) twist for the labels, `ego.transform(..., naive=True)`
> scored by `parity.exact_scores` -- returns 1.0000 at every rate, so whatever
> produced them did something else and the code does not record what. See
> `results/PROVENANCE_AUDIT.md`.

The claim itself is **not** in doubt -- `tests/test_se2.py` compares `se2_log`
against `scipy.linalg.expm` and fails the naive version at every turn rate
including zero, and it passes. What is withdrawn is the four numbers and the
"passes this gate at every ordinary turn rate" framing built on them, against a
threshold of 0.98. The
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
| animal | `raw` | 1.3228 *(unconverged)* | 23.22% | 15.00% | 0.3356% | 45× | `NOT_A_RESULT` |
| animal | `bodylen` | 0.4999 | 8.77% | 6.87% | 0.3356% | 20× | `PASS` |
| session | `raw` | 0.0000 *(unconverged)* | 0.00% | 17.87% | 0.0260% | 687× | `NOT_A_RESULT` |
| session | `bodylen` | 0.4088 | 4.95% | 1.42% | 0.0260% | 55× | `PASS` |

The `raw` probes were refused, so **the nats comparison is not quotable**. On **accuracy**, which is a property of the fitted classifier rather than of its calibration and so survives the refusal, ell_a takes animal identification from 15.0% to 6.9% against a 0.336% baseline (2.2x), and session identification from 17.9% to 1.4% against 0.0260% (13x).
Body length spans
89.5–171.7 px across
the 298 animals, a
1.92× range, so
there was a great deal for it to remove.

So the brief's instruction was right and its stated reason was not. Normalising
by body length does substantially reduce identity leak; it was simply never
about 5.88 nats.

### Both `raw` nats figures are refused, and that is a bug in the metric

The `raw` session leak clamps to **0.0000 nats** while the same probe picks the
correct session out of 3,846 **17.9%** of the time,
against 0.0260% chance —
687× better. Read as nats that is "no
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
converged, at 142 and 111.**
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
median **113.72 px**, IQR
107.55–120.73, range
89.49–171.70 over
298 animals.

## One dimensional correction to the brief

The brief writes ξ = (f/ℓ_a)·log(g⁻¹g′), applying f/ℓ_a to the whole twist. That
is right for the two translation components and **wrong for the rotation**: an
angular rate divided by a body length has units of rad·s⁻¹·px⁻¹, and it makes a
large mouse's measured turning systematically slower than a small one's — body
size leaking straight into the channel where it is least visible.

| channel | divided by |
|---|---|
| `omega` | fps |
| `v_x` | fps / ell_a |
| `v_y` | fps / ell_a |

`tests/test_ego.py::test_a_scaled_animal_gives_an_identical_omega` asserts it.

## Rank: three directions are identically zero

Removing SE(2) equivariance from 14 coordinates costs three degrees of freedom,
and they are left visible rather than projected away: `s` at the origin keypoint
is (0, 0), and the y-component of `s[nose] − s[tail_base]` is 0 because the axis
is aligned. So the pose block has numerical rank
**11 of 14**, matching the expected
11 on every animal
(11–11).

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
median recording **98.40%**, p10
88.98%.

The reversal audit runs **10 checks**, `PASS`, extended
from the inherited seven with three for the twist. Every twist component is
**odd**, `v_x` included — the intuition that a forward velocity behaves like
`centroid_speed` is wrong, because `centroid_speed` is a magnitude and a
magnitude kills the sign flip. And a twist lives on the *interval* between two
frames, so reversing it negates **and shifts by one**; dropping the shift is the
same class of structural error as forgetting to flip the lag order inside a
delay-embedded row.

## What this licenses, and what happened next

The representation retains the kinematics exactly, carries no more animal
identity than the labels already did, and loses nothing that `inverse` cannot
put back. On that basis the quantizer sweep ran, on the `raw`/`bodylen` arm
above.

**It stopped.** All eight alphabets — two arms by four sizes — returned a median
run of one frame and were retired by the pre-registered run-length condition.
Occupancy was healthy in every one of them; the failure is on the time axis, not
in the alphabet. `results/ALPHABET.md` has the sweep, the diagnosis, and the
measurement that Wiener doubles the median run.

Nothing in this document is withdrawn by that. A4 is a bijection test and holds
at 1.000000; the tokenization built on top of it is the part that did not.
