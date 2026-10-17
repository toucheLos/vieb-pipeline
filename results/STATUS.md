# STATUS: what is done, what is blocked, and on whom (2026-10-17)

## Done and committed (branch `noisefloor`, pushed; recur has no remote)

| item | where | commits |
|---|---|---|
| Paper 1 revision A1–A5 | `paper1/` (`draft.md`, `provenance.csv`, `references_check.md`, `GAPS.md` §E). check.py and check_figures.py report 0 unsourced numbers | b00dcdf, 60e39fa |
| Optional analyses: **registered only, not run** | `paper1/registrations/KPMS_SYNTHETIC_DWELL.md`, `KPMS_PRACTICE_AUDIT.md` | 82b7489, 67e7dda |
| Protocol record from the file names | `results/step4/PROTOCOL.md`, `protocol.json`; recur `labels.py` + `tests/test_labels.py` | ca68426; recur 994c935 |
| Step 4 machinery registration (C1–C4) | `results/step4/REGISTRATION.md` | 7d26641 |
| HSMM core: tests pass, mypy --strict clean | `vieb/hsmm/core.py`, `tests/test_hsmm.py` | 7cd6fe6 |
| Mouse emissions + tune driver | `vieb/hsmm/mouse.py`, `scripts/hsmm_tune.py` | 4c8b255 |

## Blocked

| what | blocked on | who |
|---|---|---|
| **Step 4 C3–C4 fits and controls** (the HSMM on the tune split) | **The registered freeze emission does not fit real stillness.** A motionless pose plus measured tracking noise moves *more* than the stillest real frames: half the real frames lie below the noise surrogate's own 5th percentile (`results/step4/hsmm_floor_diagnostic.json`). A decision on how to anchor the freeze emission is needed, registered as an amendment. | **human** |
| Step 4 C5 (learning fits on fit) | shock timing and count, which Context-A sessions on days 3–7 deliver shock, session types (CFC/CFD meaning), within-day order and counterbalancing | **the lab** |
| Step 3 branch (3B as registered vs 3A for grooming) | a post-result decision; the follow-up runs point to 3A | **human** |
| Step 3B ensemble | two human labellers for the 200-frame package (`work/step3b/label_package/`) | **human / lab** |
| Paper 1: authorship and lab approval to publish the corpus; data and code release; venue; whether to state the shock mapping; whether to run the two optional analyses; how to describe the cohort covariates | | **human / lab** |
| `recur` remote | its commits (Paper 1's sources, labels) exist only on this machine | **human** |

## Found on the way (needs a decision, not acted on)

1. **The lab's `Animal_IDs.xlsx`** sits in the raw-video directory and maps 1:1 onto the 298 corpus animals. It records sex, age, genotype and drug treatment, none of which any analysis has used, and the splits did not stratify on them. Aggregates only are recorded (`PROTOCOL.md`).
2. **The skull-estimator noise level may overstate tracking noise on still frames** (the same diagnostic). Step 1's "Luna level" (0.0618 BL) and Paper 1's detection-floor paragraph use it. If it is high, Step 1's degradations were harsher than Luna, so its conclusions are conservative. Flagged in `GAPS.md` G14 and `paper1/GAPS.md` E9.
3. **Step 3B pilot pool.** `step3b_package.py` looked up splits by the merged twin key, so its pool held 731 tune recordings instead of 770. All 10 selected recordings are tune by tag, so the pilot is valid as committed. Noted, not regenerated.
4. **shapeflow `dynamics.json` reports 295 animals** through a key-overwrite bug, and `labels.py`'s docstring contradicts `splits.py` on which key splits (`paper1/GAPS.md` E2).
