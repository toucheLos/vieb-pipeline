# Paper 1 — controls behavioural segmentation needs

Build (from the repo root, after `source env.sh`):

    python3 paper1/sources.py                 # Table S1 -> provenance.csv, values.json (committed bytes only)
    python3 paper1/figures/make_figures.py    # figures from committed JSON; numbers from S1
    python3 paper1/check_figures.py           # no unsourced number on any figure
    python3 paper1/render.py                  # draft.src.md -> draft.md
    python3 paper1/check.py                   # no unsourced number in the draft

Edit `draft.src.md`, never `draft.md`. Every number is a `{{key}}` token from `sources.py`.
Open questions and gaps: `GAPS.md`.
