# Paper 1 — gaps, unsupported claims, and questions for the authors

The draft, `draft.md`, is rendered from `draft.src.md`. Every number in it comes from Table S1
(`provenance.csv`), and `check.py` and `check_figures.py` confirm that by script. This file lists
everything the draft needed and could not source, every claim I am not sure is supported, and
the decisions that belong to the authors.

## A. Numbers the prompt asked for that have no committed source (not used, or used without the number)

| # | item | status | what the draft does |
|---|---|---|---|
| A1 | "Frame-level bootstrap gives intervals ~19× too narrow" | **No measurement exists.** The figure is a docstring assertion (`recur/recur/boot.py` L3, and copies in shapeflow and the preregistrations). vieb-pipeline `CONTEXT.md` measures 1.07× for one analysis; the "1.19×" in `SEGMENTATION.md`/`LADDER.md` is a different quantity. | States the principle (animal is the unit) without a number. A registered measurement would be needed. |
| A2 | Density blind spot: "96–99% one mode across three representations" | Only two values exist: pca-HDBSCAN 0.992 and diffusion-HDBSCAN 0.964, both in `vieb2-results`, which has **no git**. The transfer-operator runs are in `vieb-runs`, also **no git**. A third representation's value was not found. | **Demoted to the Discussion as an argument with no numbers** (user decision). A registered re-run under git is needed before it can be a finding. |
| A3 | Off-manifold null: "matched ambient scale to 2%, sat 24× further" | Prose only, recur `LEDGER.md` entry 55 (now committed, atlas `recur.ledger.55` LIVE, **no value field**). 24.4× and 7.3× appear in prose; no JSON stores them. | **Omitted.** It would need a stored read. |
| A4 | Grammar: "the model collapses to a two-state limit cycle" (97.7% of runs; 95.7% of pairs) | Prose only (`METHODS_FINDINGS.md`, ledger 48). No JSON stores either percentage. The white-noise grammar JSON stores `alphabet_size` = 53, which a reader may think contradicts "two-state". | Stated **qualitatively** ("a near two-state alternation"), with no percentages. |
| A5 | Grammar: per-arm median effects (corpus 0.489, OU 0.327, VAR 0.644) | Not stored as fields. Only the white-noise value has an atlas record with a value (`claim.methods.grammar_white_effect`, 0.893, LIVE; it rests on a prose table). | Quotes only the white-noise effect. Figure 3 plots the full stored effect distributions for every arm. |
| A6 | Irreversibility: a cross-fitted estimator | **Does not exist** anywhere. Non-overlapping windows (K = 1) alone still read about 0.22 on white noise. | Finding 4 is the problem only, with a "what we did not test" paragraph (user decision). |
| A7 | Identity: the older 5.88-nat figure | No JSON and no atlas record. Its origin is reversibility-tests A2 (InfoNCE, global sampler), which is not a leak probe. The nearest stored values (5.8745, 5.8876) do not round to 5.88. | Not quoted, as the prompt requires. |
| A8 | Ceiling "log 3846 = 8.26" | ln 3846 = 8.2548, so it rounds to **8.25**. The stored `ceiling_nats` gives 8.25. | Uses the stored ceiling (8.25), never 8.26. |
| A9 | Detection floor "~0.99 body lengths" | Derived: 16 × σ̄ (0.0618). PLANT.md's sentence "Both orders cross 50% at 16 σ̄" conflicts with its own table (order 1 is already 77.4% at 16 σ̄). | Quotes the stored order-2 recall at 16 σ̄ and σ̄, and says "about one body length" in words. |
| A10 | "Keypoint content above ~2 Hz does not clear the noise floor" | The registered read `dynamics_floor/reads/descriptor_floor` is **NOT_A_RESULT**. The band fractions above 2 Hz are 24.5–38.3% above the null's p95, not about 5%. DYNAMICS.md's "nothing above 2 Hz clears" rests on its own reading of these. | Reports the stored per-band fractions and medians without the "nothing clears" sentence. **Authors should check this paragraph against DYNAMICS.md.** |
| A11 | Irreversibility: the earlier "6–41× above detailed balance" claim | Prose only (`SURROGATE_TEST.md`). The atlas still carries `reversibility-tests.results/summary/verdict` as **LIVE** ("strongly irreversible"), although the committed ring guard FAILs. The atlas has no superseded_by link between them. | Mentioned only as "an irreversible operator" in "Our own practice", with no number. **The atlas link should be added.** |
| A12 | Main text length | About 4,000 words from Introduction to Discussion, against the prompt's 5,000–7,000. | Not padded. Expanding means more words per finding, not more numbers. |

## B. Claims I am not sure are supported (please check)

1. **Dataset protocol sentence.** "Context A on the first two days, Context C on the third, Context A and Context B on the following days" comes from `recur/labels.py`'s design table (prose, with no numbers in the draft). The CFC/CFD session types are not stated, because they are not recorded.
2. **Finding 3, the mechanism.** The draft says the VAR null "never passed through the filter". `scripts/channels.py` generates the surrogate from the arm's own (filtered) pose, which supports this. The draft does **not** claim that filtering the null would remove the excess, because that has not been run. Rule 3 is a recommendation, not a demonstrated fix.
3. **Finding 3, the phase null.** "Phase randomisation is too weak a null for this statistic" is an interpretation of the unfiltered arm beating it by +9.4%. It is reasonable, but nothing tests it.
4. **Finding 4.** The guard ring's surrogates are **phase-randomised** surrogates of the ring's pose (SURROGATE_TEST.md), not time reversals. The draft now says so. Its claim that they "carry no irreversibility" rests on phase randomisation producing a stationary Gaussian, hence reversible, process.
5. **Finding 5.** "A learned encoder" is shapeflow's G3 model, h_t, with a gradient-reversal identity head (γ). The values come from two files at different γ. The encoder itself is outside Paper 1's scope (it was disqualified elsewhere); it appears only as a representation that leaks identity. Consider whether to keep it.
6. **Finding 5, the session clamp.** recur's committed `audit_moseq.json` session read (PASS, 0.0 nats) predates the clamp guard (`ebf7acc`). The draft reports that zero as refused, but **the committed JSON still says PASS**. A registered re-run of the audit would make the record agree with the text.
7. **Finding 6.** "Tracker-quality flags" are the spine's per-frame tracking flags. Name them precisely (identity swap, bone violation, low confidence?) before submission.
8. **Grammar and frame/event-mass reads are PROVISIONAL** in the atlas ("the four-gate restricted re-run"). The draft says so in the text. If that re-run changes the counts, Finding 2 needs re-rendering, and the rendering is automatic.
9. **Dwell.** `recur.results/dwell_moseq/read` is RETRACTED, and the draft cites it **as** the retracted result. There is no replacement read for MoSeq dwell. The nonparametric labeller's matched-coverage arm (corpus 100% against OU 47% and white noise 62%) was left out, because its VAR arm also rejects 100%, which makes it a muddy contrast.
10. **"Keypoint-MoSeq was used as published"**: it ran unmodified, at the labeller of record's settings. `~/moseq` is **not under git**, so the training protocol is pinned only through its restatement in vieb-pipeline `scripts/moseq_kpms.py`.

## C. Provenance housekeeping done for this draft

All of these are commits with no recomputation.
- **recur:**
  - `c60e7cf`: grammar code and every grammar result, including the surrogate arms;
  - `04e0ab0`: montage, with event mass;
  - `ebf7acc`: the leak clamp guard;
  - `0546917`: PROGRESS, INVENTORY, LEDGER, METHODS_FINDINGS, NONPARAMETRIC.
  The committed tree passes 190 tests in a clean worktree. The microstate-null work, the render media and the forced-choice deletions remain uncommitted.
- **vieb-pipeline:** `dce4fff` and `39ce0a2` promote the two separability reads from gitignored `work/` into `results/`, byte for byte.
- **Atlas** (`~/vieb-atlas`, **not a git repo**): re-indexed twice (1,719 records, 0 problems). The previous index is kept as `index.backup-20260927.json`. **The atlas itself has no version control**, so S1's atlas statuses can change if it is re-indexed. Each S1 row also carries the repo commit, which is the durable reference.

## D. Questions for the authors (the prompt's §8)

1. **Authorship and order.** Are Luna and Hassan co-authors? **Does the Luna lab approve publishing this corpus?**
2. **Release.** Which data (videos, pose, labels) and code can be released, and under what licence? `Data and code availability` is a placeholder.
3. **Venue.** bioRxiv first, then eLife Tools & Resources, PLOS Computational Biology or a Nature Methods Analysis?
4. **Context-to-shock mapping.** It is not recorded in `labels.py` or any metadata (`results/step4/PROTOCOL.md`), so the draft uses "Context A" and "Context B" only. May it be stated, and from what source?
5. **Finding 5 (irreversibility).** Its records **do** have git provenance (reversibility-tests `3d185c4`). The repo-without-git worry applies only to the density numbers (A2). It is included as the problem only, as decided.
6. **Title.** The working title is kept verbatim from the prompt.

## E. Revision of 2026-10-17 (the Paper 1 revision prompt)

| # | item | finding | what the draft does |
|---|---|---|---|
| E1 | **The 11 syllables not triaged** (12 + 9 + 25 = 46 of 57) | All are excluded by recur `recur/audit/triage.py` `MIN_STRATUM = 200` (L40, L96–108). Eight have fewer than 200 frames with a valid stratum (36, 39, 47, 50, 73, 77, 87, 98). Three (40, 82, 84) have no single stratum holding 200 frames. **Their frame mass is in no committed result.** It exists only in `work/labels/moseq.npz`: 2,380 frames, 0.0106%, not the ≈0.07% the prompt estimated. `triage_moseq.json /read/detail/frame_mass` sums to 0.999378; the remainder mixes these labels with the frames that have no stratum. | States the rule and the counts (57 labels, 46 scored = `len(per_label)`), with no mass |
| E2 | **298 vs 295 animals** | 298 is `animal_tag`, which keeps the cohort twins apart (772/772H, 773/773H, 797/797H); the splits and every Paper 1 analysis use it (`splits.json /animal_id_rule = "animal_tag"`). 295 is `animal_key` (twins merged), stored as `/n_animals_if_cohort_merged`. **The only analysis reporting 295 is shapeflow `results/dynamics.json`, through a bug:** `scripts/dynamics.py:239` `row.update(lab.parse(rid))` overwrites the per-row `animal` with the merged key. That pools each twin pair, and the twins sit in different splits. Separately, `labels.py:87, 104–106` says splitting uses the merged key, which contradicts `splits.py:18–35`. | Unchanged: 298 is right for every analysis in the paper. **The shapeflow dynamics bug and the docstring contradiction are flagged here and not fixed** (outside Paper 1) |
| E3 | **"Nothing above ~2 Hz clears the tracking-noise floor"** (`MOUSE_PROGRAM_PROMPT.md` §0) | **Superseded by the stored values.** 24.5–38.3% of windows in the bands above 2 Hz exceed the noise null's 95th percentile, against 5% expected under the null. That is partial separation, not absence. | Frequency paragraph rewritten to report the stored fractions and medians. The "needs a better tracker" sentence is removed |
| E4 | **Wiltschko 2015** | The prompt said their shuffles "test the data, not the labeller". They **also** ran a synthetic, non-modular dataset through the AR-HMM ("failed to identify any well-separated modules"). | The draft says their control is "of the same kind" as Weinreb's, beside raw-data shuffles |
| E5 | **The 400 ms target** | Verified in Weinreb 2024's Methods (κ "adjusted … to ensure a median state duration of 400 ms"). Our labeller of record's median dwell is 12 frames at 30 fps, which is on that target. | The literature number is an S1 entry (`lit.kpms_target_ms`, by DOI and quote) |
| E6 | **Optional analyses (approved 2026-10-20)** | **Synthetic null: run.** On KPMS's own LDS null the labeller collapses: 5–7 syllables in use, the top two holding 60–64% of frames, median runs of thousands of frames. Only 16–26 of 60 animals have enough runs to test, and 31–65% of those reject geometric dwell. Under the registered rule this is reading 1 by the seed median (62.5%), but the starved test is the real finding, and the draft says so. **Practice audit:** the citing-works list (150) and coding sheet are in `paper1/audit/`. **Coding needs two human coders.** | Finding 1 updated with the synthetic result. Audit awaits coders |
| E7 | **Dataset sentence** | Revised after Part B. It now names the two phases with their day ranges (S1 `proto.*` from `results/step4/protocol.json`) and the per-day contexts, cites the naming schema, and states no shock mapping. **The opening still says "a contextual fear-conditioning experiment"**, which is the program's description; the names do not expand CFC/CFD. Confirm with the lab. | Revised |
| E8 | **The lab's `Animal_IDs.xlsx`**, found in the raw-video directory | 298 rows map 1:1 onto the corpus animals. The cohort mixes sex (156 F / 142 M), age (mostly 18–24 months, some young), genotype (Het3, APOE2/3/4, BL6x129) and drug treatment (TMZ, DQP-1105, CIQ, vehicle; most blank). No analysis has used these. The splits did not stratify on them. | One sentence, **with no numbers**: the animals differ in sex, age, genotype and treatment, and none is used. **Whether and how to describe the cohorts is the authors' and the lab's decision.** |
| E9 | **The jitter level σ̄ = 0.0618 BL** (skull estimator), quoted in the dataset and detection-floor paragraphs | A Step 4 pre-run diagnostic finds the stillest real frames steadier than a motionless pose plus this noise, and steadier even than shapeflow's native noise (`results/step4/hsmm_floor_diagnostic.json`). The skull estimator may absorb real head movement, or tracking noise may be lower on still frames. | Unchanged; the authors should decide whether to qualify σ̄ as "measured over all frames" |

