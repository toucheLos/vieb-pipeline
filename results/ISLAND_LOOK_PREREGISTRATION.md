# Pre-registration — does the island look like one thing?

Written and committed **before any trial is built and before any clip is cut**.
It fixes the trials, the controls, the scorer's task, the sample size and the
reading.

## 1. Why looking, and why it has to be this careful

Every statement about the island so far is a statistic, and they pull in
opposite directions. It is 361 segments across 46 of 89 animals; it runs
**3.7× slower** than the same animals' other segments; its occupancy **moves
with context** (1.275% in A against 0.470% in B, pair-flip p = 0.0050); and it
is a **single-linkage chain** whose typical pair sits **2.1×** the linking
distance apart. A real behavioural state and a path through a distance metric
both produce that pattern, and no further statistic in this programme separates
them.

Looking separates them. But **the scorer is the least reliable instrument
here**, and that is documented rather than assumed: `ADJUDICATION.md` records
two frames read as "rearing against a wall" that a contact sheet showed were
teleports, and `METHODS_FINDINGS.md` M7 records this same scorer reading a
clump as "pressed against the wall" when it measured **closer to the centre**,
paired −1.92 [−2.95, −0.83]. So the design assumes an unreliable eye.

## 2. Three cues that would decide the test if left alone

| cue | measured | consequence if unmatched |
|---|---|---|
| **speed** | the island runs at **0.205×** the mean segment speed in its own animals | the scorer sorts on "is it moving" and learns nothing new |
| **duration** | island median 1.30 s against the corpus's 0.93 s; p90 9.8 s against 4.7 s | `partition.py:20-36` records a rater scoring **+0.183** — a third of a real effect — purely by calling the six longest clips "same" |
| **animal and scene** | one animal supplies **56 of 361** island segments | two clips of one mouse in one box resemble each other because it is one mouse in one box |

**All three clips in a trial come from three distinct animals and three distinct
recordings.** This is why the control is *not* drawn from the island member's own
recording, which would otherwise have been the tighter nuisance control: in a
triad it places two clips from one scene into the trial and points the scorer at
the **wrong** answer systematically. Matching is therefore corpus-wide, and
background varies across all three clips regardless of class, carrying no class
information.

## 3. The trials

**72 triads**, balanced by construction:

* **36 `2 island + 1 control`** — the odd clip is the control. Answering
  requires island members to resemble **each other**. This is the arm that tests
  coherence.
* **36 `2 control + 1 island`** — the odd clip is the island. Answering requires
  only that the island differs from a control.

They are different questions and are **reported separately**, never averaged
into one claim.

**Odd-one position is balanced by construction**, not drawn i.i.d., copying
`forced_choice.plan_trials`. **Trial ids are salted hashes of (condition, index)
only** — never of the answer — so the key cannot be recovered by re-running the
builder.

## 4. Two control arms, and why the weaker one is required

* **`matched`** — the nearest available non-island segment on joint
  (log duration, log speed), searched over the **whole** non-island pool at once.
  `partition.take_other` records why the pool is not narrowed first: *"a target
  of 20 frames can be matched against a label whose clips are all 180, and the
  match fails while appearing to have been made."* The achieved |log| gaps are
  reported per trial.
* **`unmatched`** — a uniform draw from non-island segments. **This is the
  positive control and it is not optional.** Speed alone should make it easy, so
  if the scorer cannot beat chance even here, the scorer is blind and the matched
  arm's null means nothing. `METHODS_FINDINGS.md` M1: *every probe needs a null
  it is known to fail* — here, a condition it is known to pass.

Both controls are drawn from segments with `label != 0`, which is 98.1% of the
corpus and is overwhelmingly the unassigned continuum.

## 5. The media, and the two cues the rendering itself could introduce

**Equal frame count within every trial**, trimmed to the shortest member — the
rule `instruments.hstack3` states as *"if the odd one is the short one, a rater
picks it by length and the experiment measures a stopwatch."*

**Both overlay variants are rendered** — plain and with the pose skeleton — and
**scored together as one judgement per trial**, not as two arms. The codebase
disagrees with itself here on purpose: `montage.cut_one` keeps the overlay off
for raters because *"a green stick figure … advertises exactly which frames the
tracker got wrong"*, while `adjudicate.py` draws it because the question there
was tracking. Here the scorer sees both and answers once.

**`flagged=False` always.** The red border is the leak `adjudicate.py` disables.

**Cutting is gated** on `results/compare_gate.json` — availability PASS and
alignment PASS — and refuses otherwise, as `compare_clips.py` does.
`behaviour_clips.py` skipped this gate; this does not.

## 6. Blinding, including the part that is not a file

Filenames are shuffled indices. The key is written and **not read until scores
are supplied**.

**The build step's stdout is part of the blind.** The scorer and the builder are
the same process here, so anything printed in trial order — which position is
odd, per-arm counts as trials are made — leaks the answer into the scorer's
context as surely as reading the key would. The builder prints totals only.

## 7. The statistic, and the power stated in advance

Accuracy per arm against **chance = 1/3**, with an **animal-clustered** interval
beside the binomial one. `percall.py` records why the pooled binomial alone was
abandoned: batch 1 returned 43.1% at p = 0.025 with a label-clustered CI of
[0.320, 0.546] that **includes chance**.

**The MDE is computed at the realised n and reported before the accuracy is
read**, and it is reported whether or not it passes — the pattern
`journeys.py` and `learning_curve.py` both use. At 72 trials this design detects
roughly a **+17-point** lift over chance at 80% power. It cannot detect +5. **A
null here means "not this big", never "nothing"**, and the write-up will say so
in those words.

## 8. The reading, fixed now

| unmatched arm | matched arm | reading |
|---|---|---|
| above chance | **above chance** | island segments resemble each other beyond speed and duration. The statistics found a behaviour and the remaining problem is coverage |
| above chance | **at chance** | the island is "slow and long" and nothing further. The chaining verdict stands and the continuum reading strengthens |
| **at chance** | either | **the instrument is blind.** Neither arm is interpretable; the result is about the scorer, not about the island |

The third row is the one that makes this worth running. Without the unmatched
arm, "the island is not coherent" and "the scorer cannot see anything in a
320-pixel grayscale crop" have identical output and opposite meanings.

## 9. What a PASS would and would not license

It would license: island segments are visually distinguishable as a group from
segments matched to them on speed and duration, to a blind scorer.

It would **not** license naming the behaviour, and it would **not** overturn the
chaining verdict — a chain whose members share a visible property is still a
chain. It would not license the island being one behaviour rather than one
**state**, which is what `BEHAVIOUR.md` already argues from the speed contrast.

The unblinded description written afterwards is **description, not evidence**,
and will be labelled as such wherever it appears.

---

# Amendment — the trial count, raised before any trial was built

**Nothing had been built when this was written.** No trial existed, no clip had
been cut, and no response had been given.

## What was wrong

§7 fixed **72 trials** and said the design detects "roughly a +17-point lift".
That figure is the **naive binomial** MDE. It is not the statistic §7 also
commits the verdict to: *"an **animal-clustered** interval beside the binomial
one"*, with `percall.py`'s reason quoted — batch 1 returned 43.1% at binomial
p = 0.025 with a clustered CI of [0.320, 0.546] that **includes chance**.

So the registration named one power and gated on another. Measured by simulating
the analysis as it will actually run:

| trials per arm | accuracy needed for the clustered interval to clear chance |
|---:|---:|
| 36 (what 72 total gives) | **0.65** |
| 60 | 0.55 |
| **90** | **0.50** |
| 120 | 0.50 |

At the registered size the test could only have found an effect of **+32 points**
— nearly double chance — and a null would have meant almost nothing.

## The amendment

**180 trials, 90 per arm**, 45 per (arm × trial type).

* **The per-arm cell (n = 90) is what the verdict reads.** It detects a
  +17-point lift, which is what §7 intended.
* **The trial-type split (n = 45) is descriptive** and is reported with its
  interval, not gated. At 45 it would need ~0.60, and saying so is better than
  quietly reading it as though it were powered.

Everything else in the registration stands unchanged: the arms, the matching,
the three-distinct-animals rule, the blinding, and the reading table.

## Why this is an amendment and not a tuning

The change is to the **sample size**, made **before any data existed**, in the
direction that makes a null harder to claim rather than easier. It cannot have
been selected against an outcome because there is no outcome yet. The failure it
corrects is mine: quoting a power figure from a statistic the verdict does not
use.
