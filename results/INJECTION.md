# Phase F — the injection benchmark

89 report animals, 298 shards, ε = 0.10, seed 0. Configuration fixed in
`INJECTION_PREREGISTRATION.md`, committed before the corpus run. **The bakeoff
verdict is not restated and Q1 is not re-scored.**

Corruption with known truth is injected into frames every instrument in this repo
calls clean; the arm is then run over the whole recording and asked to put them
back. Two numbers in body lengths — **repair** on corrupted keypoint-frames,
**damage** on uncorrupted ones — and **net**, the change in total error against
truth over every scored keypoint-frame.

## The headline: I was wrong about the smoothers

| arm | repair | damage | **net** | 95% interval |
|---|---:|---:|---:|---|
| `raw` | 0.3635 | **0.0000** | +0.0000 | [+0.0000, +0.0000] |
| `median_0.50` | **0.1122** | 0.0031 | **−0.0026** | [−0.0029, −0.0024] |
| `viterbi` | 0.2894 | 0.0000 | −0.0017 | [−0.0017, −0.0016] |
| `disposition` | 0.3035 | 0.0000 | −0.0013 | [−0.0014, −0.0013] |

**Prediction 3 said `median_0.50`'s net would be positive — that it would leave
the data further from the truth than it found it. It is negative, it is the most
negative of any arm, and it is negative in every speed stratum.**

The pre-registration named the consequence in advance: *"if `median_0.50`'s net
is negative, dropping the smoothers from the Step 4 MDL branch is wrong and that
decision is reversed."* It is reversed. **`median_0.50` goes back on the MDL
branch**, and the arms carried to Step 4 are `raw`, `viterbi`, `median_0.50`,
`disposition`.

The movement-retention evidence that prompted the recommendation is unchanged and
still true — `median_0.50` deletes 86–90% of median movement. What this benchmark
shows is that the movement it deletes is, on this pool, *more wrong than right*:
it repairs **95.4%** of teleport error against Viterbi's 35.3%, and the damage it
does in exchange is 0.0031 body lengths, an order of magnitude smaller than the
error it removes.

## The six predictions

| # | prediction | outcome | |
|---|---|---|---|
| 1 | `raw`: damage exactly 0, repair exactly the injected magnitude | 0.0000 [0, 0] | **held** |
| 2 | `viterbi` damage under 0.01 bl | **0.00002** [0.00001, 0.00002] | **held**, by 500× |
| 3 | `median_0.50`'s net is positive | **−0.0026** [−0.0029, −0.0024] | **failed** |
| 4 | `viterbi` repairs teleports and not parks | teleports 35.3%, parks **−0.0%** | **held** |
| 5 | no arm repairs parks (under 25%) | 10.5% / −0.0% / 0.0% | **held** |
| 6 | `disposition` damage under 0.002 bl | **0.00004** | **held**, by 50× |

### What each arm actually fixes

Fraction of each corruption's error removed:

| arm | teleport | swap | **park** |
|---|---:|---:|---:|
| `median_0.50` | **95.4%** | 75.4% | 10.5% |
| `viterbi` | 35.3% | 7.4% | −0.0% |
| `disposition` | 30.9% | −0.0% | 0.0% |

**Prediction 5 held and it matters more than prediction 3 failing.** No arm
repairs a sustained park — a landmark held off the animal for 4–30 frames. The
best any of them manages is 10.5%.

This is Phase E's failed prediction 2, rebuilt on an instrument that can actually
falsify it. That version used a smoothness statistic, and a 15-frame median
smooths *across* a 9-frame park, so it scored as a repair. Here the park has a
known true position and smoothing across it earns nothing. **The claim in
`CLEANING.md` — that what a temporal filter leaves behind needs an anatomical
prior or better detections, not another filter — now has direct support rather
than a bad test.**

`disposition` is exactly what it was built to be: it repairs 30.9% of teleports,
does nothing to parks or swaps by construction, and its damage is 0.00004 body
lengths. It touches 0.5% of frames and breaks nothing.

## Net by speed stratum, and why the trends matter more than the totals

| arm | slowest | | | | | fastest |
|---|---:|---:|---:|---:|---:|---:|
| `median_0.50` | **−0.0118** | −0.0037 | −0.0032 | −0.0030 | −0.0026 | **−0.0024** |
| `viterbi` | 0.0000 | −0.0016 | −0.0017 | −0.0018 | −0.0019 | **−0.0022** |
| `disposition` | −0.0014 | −0.0013 | −0.0014 | −0.0016 | −0.0015 | −0.0016 |

**The two trends run in opposite directions.** `median_0.50`'s benefit falls by
5× from the slowest stratum to the fastest; `viterbi`'s rises from exactly zero —
it reassigns nothing in the slowest stratum — to its maximum in the fastest. The
gap between them narrows from unbounded to 1.4×.

That is a trend and not a verdict, and it must not be read as one. The registered
falsifier fired and the decision it governs has been reversed. But the trend is
the reason the next section is not a footnote.

## The limitation that bounds every number above

**The pool is 4.45% of keypoint-frames, not the 23.4% the feasibility check
suggested**, and the binding constraint is DLC confidence: only **10.55%** of
frames have every keypoint above 0.60 on the report split, against 98.9% passing
the bone check and 37.3% passing the continuity check. The feasibility estimate
came from 12 tune recordings and did not generalise.

And the pool is slow. Against the corpus, on 20 report recordings:

| | corpus | pool | ratio |
|---|---:|---:|---:|
| median centre speed | 0.1795 | 0.1040 | 0.58× |
| p90 | 0.8415 | 0.4245 | 0.50× |
| p99 | 2.2613 | 1.0034 | 0.44× |
| mean | 0.3513 | 0.1756 | 0.50× |

**Only 1.81% of pool frames exceed the corpus's p90 speed, where 10% would be
unbiased — fast frames are under-represented 5.5×.** This was pre-registered as
the main threat, and it is worse than the feasibility check implied.

So the benchmark is decisive about the regime it covers and silent about the one
where the dispute actually lives. `median_0.50`'s benefit is concentrated in slow
frames, its advantage shrinks monotonically as speed rises, and the corpus's fast
frames are roughly twice as fast as the fastest frames in this pool. **This
phase cannot say what happens out there.** Held-out MDL, with all four arms and
the entropy normalisation, is where that gets settled.

## Two arms could not be benchmarked, and that is a limitation not an omission

`wiener` — the incumbent, the array every existing result in this programme was
built on — and `butterworth` are read off disk from shapeflow and were never
reimplemented here. A benchmark has to apply the arm to an array it has just
corrupted, and these cannot be applied to anything. Writing a fresh Wiener filter
and calling it "the incumbent" would attribute a benchmark score to a filter that
is not the one that produced the corpus.

The pre-registration listed `wiener` among the arms. It was dropped when the run
failed on it, and that is recorded here rather than quietly edited out.

## The real ceiling

Every arm here is post-hoc repair of a detector. Two independent instruments now
agree the detector is wrong on roughly 4% of frames — the continuity residual
flags 3.80%, the bone check 1.12%, and only 8.2% of those overlap — and the one
error class none of these arms can touch is the sustained park, which is what a
detector locking onto the wrong body part produces.

**No filter beats better detections.** Retraining DLC and re-inferring 3,846
videos remains the expensive branch and remains costed out, but it is the honest
answer to "keep the data as well as possible", and the 10.5% park repair ceiling
measured here is the price of not doing it.
