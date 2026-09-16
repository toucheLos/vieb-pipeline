# Roughness concentration — the piecewise-smooth axiom, with no detector

The only measurement in this programme of *"behaviour is piecewise smooth"* made
with **no detector, no threshold and no sweep**. It came out of the diagnostic
that followed Step 2's separability failure (`SEGMENTATION.md`, `DEVIATIONS.md`
D7) and had no `Read` until now.

Source array: **`work/ego/raw__bodylen__*.npz`**, the F3-carried `raw` arm. 0.4 s
windows, 200 per animal, all **298** animals — the probe is a check on how a
null was constructed rather than an effect estimate, so it is not scored on a
held-out split, and that is stated rather than buried.

## The measurement

Per-channel `log` mean-squared first difference of each window, averaged over the
17 ego channels:

| arm | mean log MSD | SD across windows |
|---|---:|---:|
| **corpus** | **−7.67** | **2.88** |
| `phase` | −5.45 | 1.34 |
| `var5` | −5.45 | 1.35 |
| `ou` | −3.04 | 1.39 |
| `white` | −3.35 | 0.95 |

Against `phase`: **2.22 nats lower** — a factor of **9.2** in mean squared
increment — in **15 of 17 channels**, with **2.15×** the window-to-window spread,
in the same 15.

## Why `phase` is the null this is read against

Phase randomisation preserves the power spectrum **exactly**, and therefore the
mean squared increment exactly. It cannot preserve the mean of the *log*, which
is lower precisely when the same total power is concentrated in fewer windows.

No other null in the set has that property: `ou` and `white` differ in scale, so
a gap against them would be a statement about amplitude. The comparison is only
informative because the thing held fixed is the thing the naive statistic would
have measured.

## What it says

Two numbers, one fact. A lower typical window energy **and** a larger spread mean
the corpus's roughness is **concentrated**: smooth stretches punctuated by rare
rough moments. That is the piecewise-smooth axiom, visible directly in the data,
before any boundary has been detected.

The verdict is on the conjunction and on the per-channel count, not on the means:
a mean over 17 channels can be dragged by one outlier, and either half of the
signature alone has an innocent explanation — a uniformly smoother null gives the
first without the second, a null noisier in a few windows gives the second
without the first.

## What it does not license

**It locates no boundary and licenses no boundary count.** It says the roughness
is unevenly distributed in time; it does not say where, how many, or whether the
rough moments are behavioural transitions rather than tracking artifacts.

It also cannot be turned into a gate. That is the lesson of D7: a probe sharp
enough to see this concentration will separate the corpus from **any**
boundary-free null, which is why the separability precondition was unsatisfiable
and why this measurement is banked as a description rather than as a test.

## Where it came from

The number exists only because a negative control was run. The first separability
probe returned i.i.d. white noise as inseparable from a mouse trajectory at
AUC 0.496 — chance — which proved the instrument was blind: a logistic regression
is linear, and smooth-versus-rough lives in the second moment of the increments.
Adding the roughness features made `white` fail as it must, made all four nulls
fail, and produced this table as the diagnostic of why.

`work/tok/seg_validate/_separability_diagnose.json`, hashed into
`results/roughness.json`.
