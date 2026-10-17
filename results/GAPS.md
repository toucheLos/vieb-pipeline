# GAPS — things the four-step programme needs that are not committed, or that the plan does not register

Each row is for the human to decide. Nothing here has been computed.

| # | gap | needed by | status |
|---|---|---|---|
| G1 | **`PAPER1_PROMPT.md` does not exist** (searched vieb-pipeline, recur, vieb-atlas, vieb, personal_site, and `~` to depth 2) | Step 2: "the controlling document for this step" | **Step 2 not started** |
| G2 | **Shock times, session types, shock-paired context** | Step 4 §4.0 | **Step 4 stopped**: `step4/PROTOCOL.md` |
| G3 | **The "certified encoder"**: no code in any repo; it was disqualified (AMI 0.432), not certified | Step 1 §1.4 ("if runnable") | excluded from Step 1 |
| G4 | **A fitted κ for Keypoint-MoSeq** never existed; κ = 1e6 is the labeller of record's untuned value | Step 1 §1.4, §1.8 ("MoSeq κ: fitted / one decade lower") | Step 1 uses 1e6; the ablation's "fitted" arm means 1e6 |
| G5 | **H.264 bits/frame 1.32×** has no committed computation | §0 confound list | quoted as text only |
| G6 | **Luna dropout run-length distribution** | Step 1 §1.3 | measured inside Step 1 as a registered quantity |
| G7 | **Per-keypoint jitter in BL from the skull estimator**: the estimator sees only nose and ears, so a per-keypoint profile needs a second source (shapeflow's `sigma_hat_px`, imported) | Step 1 §1.2–1.3 | registered in Step 1: shapeflow ratios scaled to the skull σ̄ |
| G8 | **Rearing is untestable in CalMS21**: "whiterearing" is the intruder's behaviour, in 3 sequences | Step 1 §1.7 decision rule | decided with the user: grooming alone decides O3 |
| G9 | **`~/moseq` is not under version control** | provenance of the MoSeq protocol | restated in `scripts/moseq_kpms.py` |
| G10 | recur working copy: many uncommitted edits besides Ledger 35, including JOURNEYS (the 73% figure's working copy) and an untracked microstate null | provenance | listed in `INVENTORY.md`; left for the user |
| G11 | **Wiener smoothing on non-Luna data**: only shapeflow's stored Luna arrays exist; `vieb/clean/arms.py` refuses a second implementation | Step 1 §1.8 smoothing ablation | that variant is `NOT_A_RESULT` |
| G12 | **Human keypoint labels for the 3B pilot**: `~/dlc-training/trained_dlc/labeled-data/*` folders exist but are empty; anatomy-only labelling needs two human labellers | Step 3B items 2–5 | labelling package prepared; stops for the human |
| G13 | **3B's target**: groom is never recoverable on CalMS21, even at native (0.633× Luna), so "groom's breakdown noise" does not exist | Step 3B gate | the 3B gate is `NOT_A_RESULT` by construction (`step3b/REGISTRATION.md`) |
| G14 | **The tracking-noise level used as "Luna's level" may overstate noise on still frames.** A motionless pose plus measured noise moves more than the stillest real frames: the real pose-change p5 is 0.010 BL/s, against medians of 0.129 (shapeflow native noise) and 0.362 (Step 1 operator) (`results/step4/hsmm_floor_diagnostic.json`, a pre-run diagnostic) | Step 1's degradation level; Paper 1's detection-floor paragraph; Step 4's freeze emission | **Not acted on.** If confirmed, Step 1's 1.0× was harsher than Luna, so its conclusions are conservative. Step 4's emission needs an amended anchor (human decision) |
| G15 | **The Step 3B pilot pool** used the merged twin key for the split lookup (731 tune recordings instead of 770). All 10 selected are tune by tag | Step 3B pilot | Noted; the pilot stands as committed |

