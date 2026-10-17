# Registration (optional; NOT approved to run): which controls do Keypoint-MoSeq application papers report?

**Status:** written for the authors' decision (Paper 1 revision prompt, A4).
**Nothing here has been run.**

## Question

The draft now says that no review or tool documentation we examined recommends
the fitted-labeller surrogate control or identical preprocessing of the null, and
that no survey has quantified how often such controls are reported. This audit
would quantify it.

## Sample (fixed before any search)

- **Population:** peer-reviewed papers citing Weinreb et al. 2024
  (doi:10.1038/s41592-024-02318-2) that **apply** Keypoint-MoSeq to new data.
  Methods-only papers, reviews and preprints are excluded.
- **Frame:** the citing list from Crossref / OpenCitations on the day the audit
  starts. The date is recorded.
- **N = 40** papers, drawn uniformly at random (seed 0) from the eligible list.
  If fewer than 40 are eligible, all of them are used.

## Coding (each paper, from main text, methods and supplement)

Each item is coded present or absent, with the quote:
1. the fitted labeller run on surrogate or synthetic input;
2. preprocessing applied identically inside a null;
3. dwell or duration distributions offered as evidence of states;
4. a sequence or grammar test, and its null (on labels, or through the labeller);
5. an identity or recording-leak probe;
6. tracker-artifact triage of syllables;
7. the unit of resampling for intervals (animal, session, frame, or not stated).

**Two coders**, blind to each other. Agreement is reported as Cohen's κ per item, and
disagreements are resolved by discussion and reported.

## Reporting

Per item: the count and share of papers, with a binomial 95% interval. Nothing is
reported per paper by name in the main text; the coded table goes in a supplement.

## Cost

Human coding time, about 40 papers × 20–30 minutes × 2 coders. No compute.
