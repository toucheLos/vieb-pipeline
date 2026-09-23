# The 3–8 Hz grooming gate: refused, not failed

Registered in `GROOMING_PREREGISTRATION.md`, committed alone, then amended alone
(**Amendment 1** / `DEVIATIONS.md` **D20**) before any verdict existed.
`vieb/pixel/head.py`, `scripts/grooming_gate.py`, `jobs/grooming.slurm`.
Digest **`198eb14ff258c7f6`**. Sample: the 300 already-registered `fit`
recordings, 30 animals, 10 per box.

**Two independent refusals.** The detector's **precision is 0%** — 0 of 30
candidate clips are grooming, scored blind, against a registered 50% bar — and
separately, `NOT_A_RESULT` at **every one of the 54 arms** — 27 registered
grid points × 2 signals. No arm reached the registered minimum of **500**
candidate windows. **The 3–8 Hz question was never tested.**

> **This is a refusal, not a negative result, and the distinction decides what
> may be written.** §10 fixed a form of words for the case where the gate *does
> not pass*. **It is not invoked**, because the gate did not run: `peak_excess`
> was never compared between arms. Claiming that fine limb behaviour is
> "unrecoverable by motion-energy features at this resolution" would be
> asserting the outcome of a test that was refused for want of data.

---

## 1. What actually failed: the detector, not the spectrum

§2 asks for windows that are simultaneously in the **bottom 25% of keypoint
speed** and the **top 25% of head-region motion**. Those two sets barely
intersect, because head-region pixel motion is very largely **body** motion.

**Within recording**, median over all 300 recordings, 2 s windows:

| signal | disc 0.4 bl | 0.6 bl | 0.8 bl |
|---|---|---|---|
| `energy` — the disc in **image** coordinates, as §1 registered it | +0.900 | **+0.917** | +0.924 |
| `energy_ego` — **stabilised** on the animal (Amendment 1) | +0.826 | **+0.815** | +0.808 |
| `energy_ego_hip` — the hindquarter control | +0.876 | +0.858 | +0.841 |

Pearson correlation with keypoint speed. Spearman sits near +0.88–0.93 for all
three.

**The trend across radius reverses between the two signals, and that is the
cleanest evidence the stabilisation does what it claims.** In image coordinates
a *bigger* disc catches *more* of the scene sliding beneath a translating animal,
so the coupling **rises** (+0.900 → +0.924). Stabilised, a bigger disc averages
over more of the animal and the coupling **falls** (+0.826 → +0.808). A measure
of body-relative motion must behave the second way; §1's did not, which is D20.

**Stabilisation helped, and not enough.**

| | candidates at the headline arm | best of the 27 arms |
|---|---|---|
| `energy` | 51 | 164 |
| `energy_ego` | **96** | **347** |
| **registered minimum** | **500** | **500** |

Stabilising roughly doubled the yield and the best arm reaches **69%** of the
minimum — so this is not off by orders of magnitude, and it is still a refusal.
The verdict is unanimous: **27 of 27 arms `NOT_A_RESULT` for each signal**, so
nothing here is a knife-edge that a different grid point would have flipped.

## 2. The candidates are a real outlier population — of jittery tracking

Under a Gaussian copula at the observed rank correlation, the two conditions
would co-occur essentially never:

| dependence ρ | P(bottom 25% speed **and** top 25% head motion) |
|---|---|
| 0 (independent) | 6.25% |
| 0.70 | 0.54% |
| 0.80 | 0.16% |
| **0.91 (observed)** | **0.01%** |
| **observed yield** | **0.20% – 0.42%** |

**The detector finds 20–40× more of these windows than the bulk dependence
allows.** They are a genuine outlier population, not the tail of the
speed–motion relationship — and that is exactly why they were worth cutting into
clips rather than discarding.

> **Correction, on this document's own first draft.** Before the panel was
> scored, this section was headed *"the candidates are not noise"*. **§3 and D21
> falsify that reading.** The statistics were right — there is a population here
> that the bulk dependence does not explain — but the population is **windows
> with 3.4× the skull jitter**, not windows with behaviour in them. An excess
> over a smooth copula says *something structured is here*; it says nothing
> about whether that something is the animal or the tracker. The original wording
> let the first be read as the second, and that was my error.

So the honest form of this section is: the detector selects a real, structured,
non-random subpopulation, and **§3 identifies it as still animals with noisy
keypoints**. 3–8 Hz remains untested either way.

## 3. The clips were watched, and the detector's precision is 0%

§3's panel was scored blind: **0 of 30 candidate clips are grooming**, against a
registered bar of **50%**. None of the 15 speed-matched controls was called
grooming either, so the panel was not scored indiscriminately.

**The detector does not detect grooming.** §9.4 therefore forbids reading the
spectral gate as an answer about grooming — moot here, since §6 had already
refused it for want of candidates, but the two refusals are independent and both
stand.

> The verdict was given as a **blanket judgement over the whole panel**, not
> clip by clip, and `work/grooming/confirmation.json` records it that way. A
> file holding 45 independent answers would overstate what was observed.

### What it is firing on instead — D21

The rater's reading was that DLC keypoints move in micro-increments and that this
alone produces the detection. **The skull triangle is rigid**, so within-window
variance in its three bone lengths is tracking noise and cannot be behaviour.
Selection used speed and pixel energy and never saw a bone length; the control is
matched on speed. The measurement was free to come out flat:

| | candidates | speed-matched controls |
|---|---|---|
| skull jitter, body lengths | **0.03126 [0.01679, 0.04484]** | **0.00908 [0.00533, 0.01297]** |

**3.44×, non-overlapping**, 26 animals, 96 pairs — about **3.2 px** of
fluctuation in a physically constant distance. Skull jitter predicts head-region
energy at **+0.276** with speed held fixed (+0.774 raw).

**The mechanism, end to end.** A genuinely still animal has low keypoint speed
and passes the first criterion. Its skull keypoints still jitter; the egocentric
warp is driven by that pose, so the crop rotates and translates frame to frame,
which reads as pixel motion everywhere inside it — and the window passes the
second criterion. **The detector selects still animals with noisy tracking.**

**§4's control said so before the eye did.** The stabilised hip signal tracks
speed at +0.876 → +0.841 against the head's +0.826 → +0.808: a disc where no
grooming can occur behaves like the disc where it would. It was added because
registering on a noisy pose injects motion, and it was describing the actual
failure.

## 4. The clips, and why they were worth cutting anyway

`results/grooming/` — **45 clips: 30 candidates over 19 animals and 15
speed-matched controls**, shuffled, under opaque ids, key held back, with
`score.html` and a README.

**This was §3's fix, and it did its job.** It answered the question no statistic
in this repo can — **what are these windows?** — and the answer falsified the
detector, which is the outcome a blind panel with controls in it exists to make
reachable. The detector selects a still body with head-region
pixel motion; whether that is grooming, sniffing, chewing, a twitch or a
tracking artefact is a claim about content. Per §9.6, if the confirmed content
is some other held-still-with-head-motion behaviour, **that** is what gets named.

One blinding leak is measured and published rather than left to be found:
candidate clips encode larger (median **29,486** bytes against **24,024**),
because they contain more motion. It is invisible in `score.html` and
recoverable from a directory listing.

## 5. The hindquarter control never got to speak, and it matters that it exists

Amendment 1 added it because **registering on a noisy pose injects motion**: one
pixel of keypoint jitter shifts the whole crop and reads as motion everywhere. A
3–8 Hz excess appearing equally at the hips would be registration noise, and the
gate is wired to fail on that whatever the head shows.

It was never reached, because no arm produced enough candidates. But the numbers
already argue it would have been binding: the stabilised **hip** signal
(+0.876 → +0.841 with speed) behaves almost exactly like the stabilised **head**
signal (+0.826 → +0.808). **A disc where no grooming happens looks like the disc
where it would.** Any future version of this gate has to clear that control
before its head-disc number means anything.

## 6. What would change the answer

In descending order of leverage, and none of them is a re-run of this:

1. **A higher frame rate.** At 30 fps, 8 Hz is **3.75 samples per cycle** — §0
   said so before anything ran. This is the ask already in
   `results/outbox/LUNA_EMAIL.md`.
2. **A paw-visible view.** The skeleton has no forelimb point, so the behaviour
   is invisible to keypoints by construction; a side camera is the only
   structural fix.
3. **Registration that does not depend on keypoints — now the first fix, not the
   fourth.** D21 shows the failure runs pose → warp → apparent motion, so the
   pathway has to be cut rather than damped. `segment-anything` is installed and
   pinned in both repos, off the critical path for exactly this; image-domain
   phase correlation on the animal's box is cheaper and uses no keypoints at
   all.
4. **Smoothing or cleaning the pose**, with its limit stated in advance:
   selection is a **within-recording percentile** on both criteria, so a uniform
   reduction in jitter moves the values and the threshold together and need not
   change which windows are selected. It helps only insofar as it removes jitter
   *preferentially* from the jittery windows — an empirical question, not a
   given.
5. **A detector that is not a conjunction of two correlated thresholds.** At
   ρ ≈ 0.81 even after stabilisation, "still **and** moving" is close to
   self-contradictory. Selecting on the **residual** of head motion after
   regressing out speed would be the obvious replacement — and it is **not
   tried here**, because choosing a selection rule after watching this one fail
   is what §9.4 forbids. It belongs in its own registration, with D21 as its
   stated motivation.

## 7. Provenance

| number | source |
|---|---|
| within-recording correlations | all 300 recordings, 2 s windows, median over recordings |
| candidate counts | `results/grooming_gate.json`, 27 arms × 2 signals |
| Gaussian copula expectations | bivariate normal orthant probability at the observed ρ |
| 500-candidate minimum, 20-animal minimum, 50% confirmation bar | `GROOMING_PREREGISTRATION.md` §6, registered before any scan |
| `peak_excess` calibration (+0.06 null, exact rescaling-invariance, slope sensitivity) | `DEVIATIONS.md` D19, synthetic spectra, before any video was read |
| incumbent: nothing above 2 Hz clears on keypoints | `DYNAMICS.md` |

**Nothing here is described as shock versus no-shock.**
