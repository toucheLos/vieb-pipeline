# KPMS practice audit: materials for the two coders

**Registration:** `paper1/registrations/KPMS_PRACTICE_AUDIT.md` (67e7dda). Approved
to start on 2026-10-20.

## Files

- **`citing_works.json`:** the works citing Weinreb et al. 2024
  (doi:10.1038/s41592-024-02318-2), from the OpenCitations index v2 on the retrieval
  date recorded inside, with Crossref metadata. There are **150 works**: 70
  journal-article, 70 posted-content (preprints), 4 proceedings, 3 book chapters, and
  3 with no Crossref record.
- **`coding_sheet.csv`:** one row per work per coder, with the registered items.

## Screening order (the registered "uniform random draw of 40 eligible")

The works are in a **seeded random order** (`screen_order`; `random.Random(0)`
shuffle of the works sorted by DOI). Screen down the list and code each work that is
**eligible**, until 40 eligible works are coded. Taking the first 40 eligible works
in a uniformly random order is a uniform random sample of the eligible set. If fewer
than 40 are eligible, all of them are coded.

**Eligible** means a peer-reviewed paper that **applies** Keypoint-MoSeq to new data.
Excluded:
- preprints (`posted-content`);
- reviews;
- methods or tool papers that only compare against it;
- papers that only cite it.

Record the reason for every exclusion in column `eligible_application`.

## Coding (each coder alone, blind to the other)

Items 1–7 as registered. Code each as present / absent / unclear, with a quote and
its location (main text, methods or supplement). Agreement is reported as Cohen's κ
per item; disagreements are resolved by discussion, and the resolutions are reported.

**Nothing has been coded.** The coders are the authors' choice.
