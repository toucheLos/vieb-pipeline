# Controls behavioural segmentation needs: how standard evidence for behavioural states is manufactured by the pipeline

*Working title; the authors decide. Draft rendered from `paper1/draft.src.md`: every number below is filled from Table S1 (`paper1/provenance.csv`) by `paper1/render.py`, and `paper1/check.py` fails the build if any other number appears.*

**Authors:** [to be decided — see GAPS.md]

## Abstract

Unsupervised segmentation tools turn pose trajectories into an inventory of behavioural states, and the inventory is then used as evidence that behaviour is organised into discrete states. The evidence usually offered is of five kinds: dwell times that depart from a geometric distribution, sequences that beat a Markov null (a "grammar"), recurrence of the same segments across animals, irreversibility of the dynamics, and agreement between methods. We asked whether each can be produced by the pipeline alone. On a corpus of 3,846 overhead recordings of mice (22,355,989 frames), we ran the identical pipeline — the same fitted labeller, the same preprocessing, the same estimator — on surrogate inputs that contain no behaviour. Keypoint-MoSeq at its fitted stickiness rejected geometric dwell in 298 of 298 animals, and in 100% of animals when fed white noise. The same model found a surviving "grammar" in white noise (21 of 69 n-grams). A cross-animal recurrence excess of +1.639% over a vector-autoregressive null existed only after Wiener filtering; on unfiltered pose it was −0.199% [−0.464%, +0.042%]. A plug-in entropy-production estimator assigned white noise more irreversibility (1.22 nats) than a planted, driven system carried (0.19). Representations leaked recording identity that a white-noise control cannot see, and 2.82% of frames sat in syllables that track tracker failure rather than posture. Each failure is caught by a control costing one extra pipeline run. We give the controls as a table and the rules that follow from them. We make no claim about whether mouse behaviour is discrete.

## Introduction

Pose estimation has made it routine to record the trajectories of animal body parts at video frame rate, and a family of unsupervised tools now turns those trajectories into a sequence of discrete labels: Keypoint-MoSeq (Weinreb et al., 2024), B-SOiD (Hsu and Yttri, 2021), VAME (Luxem et al., 2022) and others. The output is an inventory of states — "syllables", "motifs", "clusters" — with their durations and transitions. These tools are useful, and the best of them are carefully built and documented.

Every such tool returns an inventory. None of them can return "there are no states here": a hidden Markov model fits a sequence of regimes to any input, and a clustering algorithm partitions any point cloud. The tools also disagree with one another. A recent comparison on the same data found that B-SOiD, BFA, VAME and Keypoint-MoSeq returned different numbers of clusters and different typical motif durations, while the authors judged their overall performance comparable (Mlost et al., 2025). If the inventory is not evidence by itself, the evidence for discrete behavioural states has to come from properties of the inventory that a continuum would not produce.

Five such properties are commonly offered. **Dwell times** that are not geometric are read as evidence of states with their own persistence. **Sequences** that occur more often than a first-order Markov chain predicts are read as a grammar. **Recurrence** of the same segment across animals is read as a shared vocabulary. **Irreversibility** of the inferred dynamics is read as evidence of driven, structured behaviour. **Agreement** between methods built on different principles is read as convergent validity.

Each of these is a statistic computed at the end of a pipeline: video, keypoints, preprocessing, a representation, a labeller, a statistic (Figure 1). Each stage can impose structure. A sticky prior imposes persistence; a smoothing filter imposes temporal correlation; a delay embedding imposes a direction of travel; a per-recording standardisation imprints recording identity. The question this paper asks is narrow: **can each piece of evidence be produced by the pipeline alone, and is there a cheap control that would show it?**

The answer, for every piece of evidence we tested, is that it can, and that there is. The controls are not new in kind. Surrogate data have a long history in nonlinear time-series analysis (Theiler et al., 1992), and preprocessing a null identically is standard practice in other fields. What we show is that they are not run in this field's standard workflow, and what happens when they are.

The work was done on one corpus, with one labeller audited in depth, by a project that made several of the errors described here and retracted them. We report the retractions alongside the controls that forced them, because they are the clearest evidence that the controls work. This is a methods paper. It does not claim that mouse behaviour is or is not organised into discrete states; on this corpus the question is not settled by any of the evidence below, once the controls are run.

## Dataset and pipeline

**Recordings.** The corpus is a contextual fear-conditioning experiment from the Luna laboratory: 3,846 recordings, 22,355,989 frames at 30 frames per second, filmed from overhead with an unobstructed view. Each animal was recorded in Context A on the first two days, in Context C on the third, and in Context A and Context B on the following days. We refer to contexts only by these names; the protocol's assignment of shock to contexts is not part of this analysis.

**Keypoints.** DeepLabCut (Mathis et al., 2018) tracked 7 keypoints per frame: the nose, both ears, a body centre, both hips and the tail base. Tracking is imperfect, and how it is imperfect matters for every control below. The skull triangle (nose and both ears) is rigid, so any change in its bone lengths is tracking error. Skull bones stretch more than 10% beyond their reference length on 2.1% of frames, and the violation rate rises from 1.2% at the centre of the arena to 4.5% at the wall. Measured on the skull, the per-keypoint jitter is 0.0618 body lengths.

**Splits.** Animals, not frames or recordings, were split into a tuning set (60 animals, 770 recordings), a fitting set (149, 1,927) and a held-out report set (89, 1,149). Every interval in this paper resamples animals.

**Labeller.** Keypoint-MoSeq was used as published, at the settings of the project's labeller of record, including its stickiness κ = 10⁶. It returned 57 syllables on the corpus. We audited it in depth because it is the most widely used and best documented of these tools, with an explicit noise model and published validations against behavioural ground truth; the findings below concern how its outputs are used as evidence, not whether it is a good segmenter.

**Record keeping.** Analyses were registered before they ran, with the decision rule and the meaning of a null written down; where a registration came after a result, the record says so. Every result is a structured verdict ("Read") with a scored object, a reason and an effective sample size in named units, and every number in this paper is traced in Table S1 to a committed result file and, where indexed, to a record in the project's atlas with its status.

## Results

### Finding 1. A fitted stickiness prior manufactures non-geometric dwell

**The evidence as usually reported.** If behaviour is a sequence of states with their own persistence, the time spent in each state should not be geometrically distributed, as it would be for a memoryless switch. Rejecting geometric dwell, per animal, is therefore offered as evidence of states.

**What we found first.** Keypoint-MoSeq's labels rejected geometric dwell in 298 of 298 animals. The median ratio of dwell variance to squared mean was 5.10, far from the value near one a geometric distribution gives. The project reported this as evidence of discrete states.

**The control.** We generated three surrogate inputs per recording — an Ornstein–Uhlenbeck continuum, a vector-autoregressive (VAR) process fitted to each recording's own pose, and white noise at each coordinate's mean and variance — and passed each through the same fitted model, with the same stickiness.

**What it shows.** Every surrogate rejected geometric dwell in every animal (298, 298 and 298 of 298; Figure 2). The corpus's dispersion (5.10) sat inside the surrogates' range: OU 2.22, VAR 7.41, white noise 13.23. White noise even produced the corpus's median dwell, 12 frames against 12, from 32 syllables. The non-geometric dwell is a property of the sticky prior, which is designed to produce persistent states and does so on any input. The project retracted the claim; Table S1 lists the retracted record beside the retraction.

**Rule 1.** Before reporting dwell structure as evidence of states, run the fitted labeller, at its fitted hyperparameters, on surrogates that have no states, and report the corpus beside them. A dwell statistic that the surrogates reproduce is a statement about the prior.

### Finding 2. The same model finds a grammar in white noise

**The evidence.** Sequences of states that occur more often than a first-order Markov chain predicts are read as higher-order structure — a behavioural grammar. The usual report is the number of n-grams that survive a false-discovery correction.

**What we found first.** On the report animals, 1,491 of 1,585 tested n-grams of two to four syllables survived Benjamini–Hochberg correction at q = 0.05 against a per-animal first-order Markov null. An earlier version of this table (1,553 of 1,638), computed on a label stream that was later replaced, is superseded and appears here only to record the change.

**The control.** The same test, on the labels the same fitted model assigns to each surrogate. A positive control — the corpus pose sent through the surrogate machinery unchanged — reproduced the corpus table exactly (1,491 of 1,585), so the arms differ only in their input.

**What it shows.** Every arm has a grammar (Figure 3). OU: 814 of 1,331. VAR: 425 of 573. White noise: 21 of 69. The white-noise survivors are not marginal: their median effect, 0.893 in log ratio of observed to expected counts, is the largest of any arm. Inspecting the white-noise labels shows why. Fed noise, the model does not produce a poor repertoire; it settles into a near two-state alternation, and the test duly finds that the alternation is not first-order Markov. The "grammar" is the labeller ringing.

These grammar reads are marked provisional in the project's record, pending a restricted re-run; the comparison between arms is internal to one pipeline and does not depend on that re-run.

**Rule 2.** An n-gram survival count is not evidence of grammar without the same count on surrogates through the same fitted model. Report effects, not survival counts, and report them for every arm.

### Finding 3. Preprocessing manufactures cross-animal recurrence

**The evidence.** If animals share a vocabulary, short segments of one animal's behaviour should have close matches in other animals more often than a structureless null predicts.

**What we found first.** At a 0.4 s window and full dimension (192), the rate at which a segment's nearest cross-animal neighbour fell inside a fixed threshold exceeded a vector-autoregressive null by +1.639% [+1.205%, +2.080%] (89 report animals, 1,954,444 query windows). The machinery's own control passed: white noise against a second draw of white noise gave +0.0024% [−0.0073%, +0.0127%].

**The control.** The pose had been Wiener-filtered before the statistic was computed. The null, a VAR(5) fitted to each recording's own filtered pose and re-simulated with Gaussian innovations, saw the filter's output only through a linear fit; it never passed through the filter. We recomputed the same statistic, at the same window and dimension, on the unfiltered pose.

**What it shows.** Without the filter the excess is −0.199% [−0.464%, +0.042%], an interval that contains zero (Figure 4). The same corpus and statistic, differing only in a preprocessing step, gives a positive result in one arm and a null in the other. Against a phase-randomised null both arms show large excesses (+15.076% filtered, +9.413% unfiltered), which says that phase randomisation is too weak a null for this statistic, not that the unfiltered pose recurs: a null that preserves only the power spectrum is beaten by any process with nonlinear structure. The recurrence result is therefore always stated with its arm, and the unfiltered cell is shown beside it.

**Rule 3.** Any transformation applied to the data — smoothing, interpolation, resampling, standardisation — must be applied inside every null, after the null is generated, not merely fitted to the transformed data. A result that depends on the arm is a result about the arm.

### Finding 4. An entropy-production estimator manufactures irreversibility from white noise

**The evidence.** Time-irreversible dynamics — a nonzero entropy production — are read as evidence that behaviour is driven through a structured cycle of states rather than fluctuating around equilibrium.

**The estimator.** Pose was embedded with a delay of 12 frames, reduced to 64 components, partitioned into 1,000 regions, and the entropy production was computed from the transition counts between regions at lag τ — a plug-in estimator of a standard kind. Consecutive delay windows share all but one of their frames.

**The control.** Two inputs whose answer is known. A driven three-state ring, rendered as pose at the corpus's sample size, has a known irreversibility. White-noise pose has none.

**What it shows.** At τ = 0.2 s the estimator assigns white noise an entropy production of 1.22 nats, while the driven ring's own irreversibility is 0.19 nats (Figure 5a). The ring, estimated through the same pipeline, reads 1.47 against 1.45 for its own phase-randomised surrogates, which preserve every second-order statistic and carry no irreversibility: the estimator cannot see an irreversibility planted to be seen. The guard that tests for this returned FAIL. Most of the estimate comes from the embedding. On white noise, the estimate rises with the delay order, from 0.22 nats at K = 1 to 0.89 at K = 24 (Figure 5b): overlapping windows make the embedded trajectory advance in one direction whatever it is fed. At τ = 0.3 s, white noise (0.241 nats) clears the detailed-balance sampling floor (0.040) several-fold — so a detailed-balance null, which holds the operator's own counts fixed, cannot catch an asymmetry the embedding put there before the counts were taken.

**What we did not test.** We did not test a cross-fitted estimator. Non-overlapping windows alone (K = 1) did not remove the bias: white noise still read 0.22 nats. We therefore state the rule in the form we can support.

**Rule 4.** Report every irreversibility estimate beside the same estimator's value on white noise, and on a planted system of known irreversibility at the same sample size. If the estimator does not separate the planted system from a reversible surrogate of it, its value on the data is not a measurement.

### Finding 5. Representations and labels carry recording identity, and a white-noise control cannot see it

**The evidence at risk.** Many downstream analyses — context effects, learning curves, individual differences — compare labels or representations across recordings. If those labels encode which recording they came from, a "context effect" can be a recording effect.

**The control.** A direct probe: a classifier that predicts the recording (or the animal) from the representation, held out by window within each recording, scored in nats of identity recovered against the ceiling log N. Not a shuffle correction: per-recording standardisation anywhere in the pipeline disarms a shuffle, because it makes shuffled and unshuffled data look alike.

**What it shows** (Supplementary Figure S1). Body-length-normalised egocentric pose windows recover 0.41 nats of recording identity, against a ceiling of 8.25 nats — a classifier that is right 1.4% of the time where chance is 0.026% — and 0.50 nats of animal identity against 5.70. The same representation on unfiltered pose recovers 0.49 and 0.54 nats. A learned encoder's state, before adding a gradient-reversal identity head at the smallest strength that met the leak limit on the tuning split, recovered 0.83 nats of recording identity on the report split (ceiling 7.05) and 0.73 of animal identity (ceiling 4.49); after it, 0.32 and 0.31. Keypoint-MoSeq's labels carry 0.28 nats of animal identity, 5.0% of the 5.70-nat ceiling. Each value is reported with its own ceiling, because the ceilings differ by split and by target, and fractions of different ceilings do not compare.

The probe has its own failure mode. A multinomial classifier over thousands of classes can be confidently wrong, pushing cross-entropy above log N and clamping the leak to zero. The label probe's recording-level read did exactly this: zero nats, from a classifier right 0.083% of the time against 0.026% chance. The project's probe now refuses any zero that arrives with accuracy above 3.0 times chance; the zero is not reported as an absence of leakage.

**Why the usual control cannot detect it.** A white-noise control asks whether the pipeline manufactures a result from structureless input. White noise has no recording identity to leak, so any statistic that exploits recording identity passes the white-noise control untouched. Only a probe that asks for identity directly, held out within recording, can see it.

**Rule 5.** Probe identity directly — recording and animal — on every representation and label set that is compared across recordings, held out by window within recording, and report the leak in nats against its own ceiling. Do not infer its absence from a passed white-noise control.

### Finding 6. Some syllables track the tracker, and posture separates them from real behaviour that breaks the tracker

**The evidence at risk.** A published state inventory is treated as an inventory of behaviour. Some of its states may instead be states of the tracker.

**The naive control, and why it overstates.** Enrichment of tracker-quality flags in a syllable is the obvious test. It overstates contamination, because real behaviours break trackers: rearing occludes keypoints and darting blurs them. An unstratified version of this test attributed 17.6% of frames to artifact-enriched syllables; that figure is superseded.

**The control.** Condition on what the animal is doing. Within each of fifteen strata of speed (quintiles) and body extension (terciles), compare each syllable's flag rate with the stratum's base rate. An artifact syllable is one enriched at least 2.0-fold in every stratum where it has at least 200 frames: the tracker fails whatever the animal is doing. An occlusion-linked syllable is enriched only in some strata — where the posture is one that breaks a tracker.

**What it shows** (Figure 6). 12 syllables, carrying 2.82% of frames, are artifact by this rule. Another 9, carrying 18.89%, are occlusion-linked: they are real behaviours that the tracker struggles with, and removing them would remove behaviour. The remaining 25 syllables (78.22% of frames) show no consistent enrichment. The difference between the unstratified and stratified answers is the difference between calling a fifth of the inventory contaminated and calling a few percent of it contaminated.

**Rule 6.** Triage every syllable against the tracker's quality flags, stratified by posture, before interpreting the inventory. Report artifact and occlusion-linked syllables separately; they need different treatment.

## What the instruments could and could not have seen

A null result is informative only if the instrument could have detected the effect. We report detection floors measured on the corpus, so that a reader can tell an absence from a blind spot.

**Boundaries.** A boundary detector based on breaks in acceleration was calibrated by planting breaks of known amplitude into real pose. At 16 times the skull jitter σ̄ (0.0618 body lengths each, so about one body length in total), it recovered a second-order break 54.2% [48.5%, 59.9%] of the time against 7.4% chance, over 7,356 planted breaks in 60 tuning animals; at one σ̄ it recovered 6.2% (Supplementary Figure S4). Transitions subtler than roughly a body length of acceleration change are below this instrument's floor.

**Frequency content.** Keypoint jitter is not white: it is correlated in time and partly shared across keypoints. Against a null built from the measured noise spectrum, the persistence of the dominant autoregressive pole separated corpus windows from noise (88.7% of corpus windows above the null's 95th percentile), while the pole's frequency did not (2.1%). Band power separated weakly above 2.0 Hz: 58.7% of windows in the band from 0.5 to 2.0 Hz exceeded the null's percentile, against 24.5%, 38.3% and 27.3% in the bands up to 4.0, 6.0 and 15.0 Hz, and in those faster bands the corpus's median relative power lay below the noise's (Supplementary Figure S5). Claims about fast keypoint dynamics — rhythmic grooming, tremor — need a better tracker than this one, and a claim that such dynamics are absent from this corpus would be a claim about the tracker.

**Recurrence.** The recurrence statistic of Finding 3 was calibrated by planting a stereotyped template at known occupancies against the same null; the floor is reported with that result in the project's record and is not repeated here.

## Recommended controls

**Table 1.** Controls for evidence of behavioural states. Each control re-runs part of an existing pipeline; none needs new data.

| evidence | control | what failing looks like | cost |
|---|---|---|---|
| non-geometric dwell | the fitted labeller, at its fitted hyperparameters, on white-noise, VAR and OU surrogates of the same recordings | the surrogates reject geometric dwell as often as the data, with dispersion in the same range | one labelling run per surrogate |
| n-gram "grammar" | the same n-gram test on the surrogates' labels from the same fitted model; a positive control through the surrogate machinery | surrogates have surviving n-grams with effects as large as the data's | the same runs as above |
| cross-animal recurrence | every preprocessing step applied inside every null, after generation; the unfiltered arm reported beside the filtered | the result changes sign or loses significance between arms | one extra arm |
| irreversibility | the estimator on white noise and on a planted driven system at the same sample size | white noise gives a nonzero value comparable to the data; the planted system is not separated from a reversible surrogate of it | two synthetic inputs |
| comparisons across recordings | a direct identity probe (recording and animal), held out by window within recording, in nats against log N; refuse clamped zeros | identity is recoverable; a white-noise control does not detect it | one probe per representation |
| state inventory | flag enrichment per syllable, stratified by speed and extension | enrichment in every stratum (artifact) versus only in some (occlusion-linked) | one pass over existing flags |
| any null result | a planted-signal detection floor at the corpus's own noise level | the planted effect is not recovered at the size claimed to be absent | one planting run |
| any interval | resample the independent unit (animal), not frames | the interval excludes its own point estimate, or narrows when more frames are added | none |

## Discussion

**What these findings have in common.** Each piece of evidence is a statistic at the end of a pipeline, and each was produced, in whole or in part, by a stage upstream of it: a prior (dwell, grammar), a filter (recurrence), an embedding (irreversibility), a standardisation (identity), a tracker (artifact syllables). In each case the remedy was the same: run the identical pipeline on an input where the answer is known. This is cheap. Every control in Table 1 reuses code the analysis already has.

**Agreement between methods is not independent evidence.** A related argument, which we did not test with numbers we can report here, concerns convergent validity. Density-based clustering, diffusion maps with a symmetric kernel and reversibilised transfer operators all compute functionals of the same stationary density of the data. If that density is dominated by a single mode, all three return a dominant state for the same structural reason, and their agreement says nothing about whether the state is real. The project's runs that bear on this were made outside version control and are not reported; a registered re-run is listed in the gaps.

**Frame mass and event mass.** Inventories are usually summarised by the share of frames in each state. Where durations vary, that summary misleads. The syllable holding the most frames held 41.7% of them but only 10.1% of runs; another held 7.0% of frames and 17.7% of runs (Supplementary Figure S2; these reads are provisional in the project's record). A claim about how often animals do something is a claim about events.

**A probe that cannot fail.** A linear separability probe on the raw frames of short windows rated white-noise surrogates as indistinguishable from mice (AUC 0.496; phase 0.504, VAR 0.501, OU 0.498; Supplementary Figure S3). The difference between a smooth trajectory and white noise lies in the second moment of its increments, which a linear model on raw frames cannot form, so the probe could not have failed. With one roughness feature added — the log mean-squared first difference of each channel — white noise separated at AUC 0.999. Every probe needs a negative control it is known to fail.

**Scope.** These results come from one species, one laboratory, one camera, one tracker and one labeller audited in depth. What generalises is the controls, which are defined by the pipeline and not by the data. What does not generalise is the magnitudes: another corpus, tracker or labeller will give other numbers, and some controls may pass. We tested Keypoint-MoSeq's dwell and grammar outputs; we did not test B-SOiD, VAME or other tools, and nothing here says how they behave. We used a sticky AR-HMM at the stickiness its users fitted; at a lower stickiness the dwell control may come out differently, which is exactly why it should be run at the fitted value.

**What we do not claim.** None of these findings says that mouse behaviour is a continuum. They say that the evidence usually offered for discreteness does not distinguish the two on this corpus once the pipeline is controlled for. Whether behaviour here is organised into states remains open; answering it needs evidence that survives the controls.

**Our own practice.** Several findings began as positive results in this project: 298 of 298 animals with non-geometric dwell, a surviving grammar, a contamination figure of 17.6%, an irreversible operator. Each was retracted or superseded when the corresponding control was run, and Table S1 lists the retracted records beside their replacements. We report them because they show the controls doing their job on careful, registered work — not because the errors were unusual.

## Methods

**Surrogates.** All surrogates were generated per recording, never across a recording boundary, and passed through the identical downstream pipeline, including its own fits. *White noise:* each keypoint coordinate drawn independently from a Gaussian at that recording's mean and standard deviation. *Phase randomisation:* the multichannel Fourier transform of the pose with a common random phase added per frequency, preserving every auto- and cross-spectrum. *VAR:* a vector autoregression of order 5 fitted to the recording's own pose and re-simulated with Gaussian innovations. *Ornstein–Uhlenbeck continuum:* a low-dimensional OU process mapped linearly onto the pose coordinates at the recording's mean and scale — smooth, aperiodic, and with no states.

**Dwell.** Per animal, all run durations were tested against the best-fitting geometric distribution with an Anderson–Darling-type statistic whose null was simulated at the fitted parameter; an animal rejects at p < 0.05. Animals are counted, never pooled. Because the test pools an animal's labels, a mixture of label-specific geometric distributions also rejects; the surrogates carry the same pooling, which is why the comparison that matters is with them. Dispersion is the median over animals of the variance of dwell divided by its squared mean.

**Grammar.** Labels were collapsed to runs; abstained or missing stretches split sequences rather than joining them. For each n-gram of two to four labels occurring at least 50 times pooled across the corpus, observed counts were compared per animal with those expected under that animal's first-order Markov chain, by a one-sided Wilcoxon test across animals, with Benjamini–Hochberg correction at q = 0.05. Effects are the median log ratio of observed to expected counts.

**Recurrence.** For each window of 0.4 s in a report animal, the nearest window in any other animal was found by exact search at full dimension (192); the statistic is the excess rate of neighbours inside a threshold set on the null's own distances, after dividing each arm by its own median distance between random cross-animal pairs. Intervals are paired animal bootstraps (2,000 replicates).

**Irreversibility.** Delay embedding of order K over aligned pose, principal-component reduction, k-means partition, transition counts at lag τ, and the plug-in entropy production of the resulting operator without reversibilisation. The driven ring is a three-state cycle with a mean dwell of 14.3 frames, rendered as pose with tracking-scale noise at the corpus's sample size; its surrogates are phase randomisations of the ring's pose, which keep every second-order statistic and are reversible.

**Identity.** Multinomial logistic regression from window features to recording (or animal) identity, trained and tested on disjoint windows of every recording, scored as log N minus held-out cross-entropy, in nats, with accuracy and chance reported beside it. A probe that does not converge, or that clamps to zero at more than 3.0 times chance accuracy, is refused.

**Tracker triage.** Each frame's speed quintile and body-extension tercile (the nose-to-tail-base distance divided by the recording's median) define fifteen strata. A syllable's enrichment in a stratum is its flag rate divided by the stratum's base rate (overall 4.56% of frames flagged).

**Detection floors.** Breaks of known order and amplitude, in multiples of the measured skull jitter, were planted into tuning-set pose and recovered by the registered detector; recall intervals are animal bootstraps.

**Statistics and registration.** Intervals are percentile bootstraps over animals (95%). No interval in this paper resamples frames: frames within an animal are not independent, and a frame-level bootstrap gives intervals far too narrow. Each analysis was preregistered in the project repositories before it ran, with its decision rule; deviations were registered before the result they affected was known. Every claim is recorded as a Read (verdict, scored object, reason, effective sample size with its unit), and refusals — results the rule declined to report — are listed with the same prominence as results.

## Data and code availability

[To be decided by the authors: which recordings, pose files and code can be released, and under what terms. See GAPS.md.]

## Figure legends

**Figure 1.** The pipeline, and where each control attaches. A: surrogate pose through every later stage. B: the same smoothing inside every null. C: a direct identity probe held out within recording. D: the fitted labeller on surrogates, and triage of syllables against tracking flags. E: the estimator's value on white noise.

**Figure 2.** Dwell. Left: median dwell dispersion (variance over squared mean) under the same fitted Keypoint-MoSeq model, for the mouse corpus and three surrogates. Right: the share of animals rejecting geometric dwell.

**Figure 3.** Grammar. Left: n-grams surviving FDR (filled) out of those tested (outline), per arm. Right: the effects of the surviving n-grams. Provisional reads.

**Figure 4.** Recurrence. The same cross-animal recurrence statistic on Wiener-filtered and unfiltered pose, against a VAR null (left) and a phase-randomised null (right); points are excess rates, bars are animal-bootstrap intervals.

**Figure 5.** Irreversibility. (a) Entropy production by lag for white noise and for a driven ring (estimated, its phase-randomised surrogates, and its true value). (b) Entropy production on white-noise pose as the delay-embedding order grows.

**Figure 6.** Tracker triage. Log enrichment of tracking flags per syllable (rows) and posture stratum (columns), for artifact syllables (above the line) and occlusion-linked syllables (below). Blank cells are strata where the syllable has too few frames to score.

**Supplementary Figure S1.** Identity recoverable by a direct probe, per representation and target, with each probe's own log N ceiling.

**Supplementary Figure S2.** Frame mass against event mass per syllable.

**Supplementary Figure S3.** A linear separability probe, mouse against each surrogate, on raw window frames and with one roughness feature added.

**Supplementary Figure S4.** Recall of planted acceleration breaks by amplitude and order.

**Supplementary Figure S5.** Separation of corpus windows from a tracking-noise null, by spectral descriptor.

**Supplementary Table S1.** Provenance of every number: value, interval, unit, scored object, effective sample size and its unit, repository, path, JSON pointer, commit, atlas record and status (`paper1/provenance.csv`).

## References

Hsu, A.I., and Yttri, E.A. (2021). B-SOiD, an open-source unsupervised algorithm for identification and fast prediction of behaviors. Nature Communications 12, 5188.

Luxem, K., Mocellin, P., Fuhrmann, F., Kürsch, J., Miller, S.R., Palop, J.J., Remy, S., and Bauer, P. (2022). Identifying behavioral structure from deep variational embeddings of animal motion. Communications Biology 5, 1267.

Mathis, A., Mamidanna, P., Cury, K.M., Abe, T., Murthy, V.N., Mathis, M.W., and Bethge, M. (2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nature Neuroscience 21, 1281–1289.

Mlost, J., Dawli, R., Liu, X., Costa, A.R., and Pollak Dorocic, I. (2025). Evaluation of unsupervised learning algorithms for the classification of behavior from pose estimation data. Patterns 6, 101237.

Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. Physica D 58, 77–94.

Weinreb, C., Pearl, J.E., Lin, S., Osman, M.A.M., Zhang, L., Annapragada, S., Conlin, E., Hoffmann, R., Makowska, S., Gillis, W.F., et al. (2024). Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. Nature Methods 21, 1329–1339.
