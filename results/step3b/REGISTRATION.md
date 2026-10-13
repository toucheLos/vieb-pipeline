# Registration — STEP 3B: pose-ensemble pilot, as selected by STEP 1's O2

Committed **alone**, before any 3B code exists. Step 1's registered outcome
was **O2 → 3B** (37df6d6), and the human said to continue through the plan.

## The gate is `NOT_A_RESULT` by construction, decided here before any data

3B's target is groom's **breakdown noise** in Luna jitter units: the jitter
Luna's input would have to reach. Step 1 found grooming **never recoverable**
by any instrument at any level, **including CalMS21's native noise
(0.633× Luna)**. Its breakdown is the bound "< native"; no level exists.

The prompt's gate (3B.5) reads: "PASS only if the upper CI bound of jitter is
below the target". With no target, the gate cannot be evaluated. So:

> **3B's gate is `NOT_A_RESULT`**, with the reason "groom's breakdown noise
> does not exist: groom is not recoverable even from keypoints cleaner than
> Luna's (Step 1)". **No full-corpus run will be registered from this pilot.**

This is raised under the prompt's §7 ("a registered threshold you believe is
wrong: say why; do not change it"). It is not overridden.

**What 3B can still measure, descriptively.** These are 3B items 3–4, run if
the human supplies labels:
- the jitter an ensemble achieves on Luna;
- its error against human clicks.

Those numbers are useful whatever the target, because they say how far Luna's
tracking can be pushed at all. They are reported **as measurements, with no
PASS/FAIL**.

## 1. Pilot set (3B.1): registered, run now

**10 Luna recordings, tune split only** (`recur/splits`, 783 recordings).
Stratified by day × context, one recording per stratum:

| day | 0 | 1 | 2 | 3 | 3 | 4 | 5 | 6 | 7 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|
| context | A | A (No_Shock tag) | C | A | B | A | B | A | A | B |

- Within each stratum, a recording is drawn with `random.Random(0)` over the
  sorted recording IDs, **from 10 distinct animals**. An already-used animal is
  skipped in favour of the next draw.

## 2. Frame selection (3B.2): registered, run now. **Anatomy only, never behaviour**

- Per pilot recording: the frames where every keypoint is usable, taken from
  `work/repaired/<animal>.npz` (REPAIR's output), then:
  1. the egocentric pose `ego.egocentric(pose, ell_a)` (14-D);
  2. **k-means with k = 20** (seed 0, 10 restarts);
  3. per cluster, the frame nearest its centroid.
- That gives 200 frames. **No behaviour label, freeze score, MoSeq syllable or
  STATES state is read.** Pose diversity is the only criterion.
- **Shared subset:** the first 5 frames per recording, in cluster order (50
  frames). Both labellers label these, which gives the human click-noise
  ceiling.
- **Package:** `work/step3b/label_package/` holds
  - PNGs of the full frame, with no overlay;
  - `manifest.csv` (recording, frame, cluster, shared flag);
  - a DLC-format `CollectedData` template for the 7 Luna keypoints (tail tip
    excluded).

## 3. Stop for the human (3B items 2–6)

Labelling needs **two human labellers** (G12). The DLC project's
`labeled-data` folders are empty. Once both labellers' files are committed,
the following are registered **before** any ensemble is trained:
- the 5-member ensemble (DLC / Lightning Pose);
- EKS;
- held-out click error.

Every metric will be reported **per context**, as 3B.6 requires.

## Prohibitions

- No report- or fit-split data.
- No behaviour signal in frame selection.
- No full-corpus run.
- No change to Step 1 or its outcome.
