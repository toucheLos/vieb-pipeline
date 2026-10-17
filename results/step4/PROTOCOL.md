# STEP 4 §4.0: protocol facts, as the file names state them

**Source:** the file-naming schema,
`<date>_Box_<box>_<phase>_Day_<day>_(<context>)_<animal>`. Every raw video in
`~/dlc-training/raw_videos` carries this form, with `.mp4`.
- **Parser:** recur `recur/labels.py`, now with `schedule`, `majority_schedule` and
  `deviants` (994c935).
- **Tests:** recur `tests/test_labels.py`. Every corpus id and every raw `.mp4` name
  parses, and the design below holds.
- **Record:** `results/step4/protocol.json`, from `scripts/protocol_record.py`.

**Only what the names state is recorded here.** Nothing is inferred from behaviour or
video.

## What the names state

| day | phase string | context(s) | explicit tag | recordings |
|---|---|---|---|---|
| 0 | `CFC` | Context A | — | 298 |
| 1 | `CFC` | Context A | **`No_Shock`** | 298 |
| 2 | `CFC` | Context C | — | 298 |
| 3–6 | `CFD` | Context A **and** Context B, **on the same date** | — | 298 each |
| 7 | `CFD` | Context A and Context B, same date | — | 284 each |

- **Size:** 3,846 recordings, 298 animals by tag, or 295 if the cohort twins are
  merged (772/772H, 773/773H, 797/797H).
- **Dates:** every CFD animal-day has both of its contexts on one date (1,476 of
  1,476).
- **Schedules:** 284 animals follow the majority schedule. The **14 deviants are
  exactly the animals with no Day 7**. There are no other differences: no extra
  sessions and no missing days 0–6.
- **The raw-video directory** holds the 3,846 corpus videos and 11,085 DLC outputs
  for those videos. Two other files are the lab's `Animal_IDs.xlsx` and its macOS
  `._` companion; see below.

### Shock, as named

- **Day 1 is the only session with a shock tag**, `No_Shock`, on all 298 animals.
- **That Day 0 delivered shock is implied** by the Day-1 tag, not stated anywhere.
  It is recorded as **implied**.
- **CFC and CFD are recorded as literal strings.** What they stand for is not stated
  in the names; the lab should confirm it.

### Within-day order of A and B: **not recoverable from the files**

- Video containers carry a `creation_time` tag, but in a seeded sample of 100 videos
  **0/100 match the date in the name**. File mtimes also match **0/100**.
- Both are copy or re-encode times: a 2024 recording carries a 2026 tag.
- **There is no reliable within-day timestamp.** The order is a gap for the lab.

## A lab document found in the video directory: `Animal_IDs.xlsx`

The file has two sheets: "Animal_IDs (no FreezeFrame)", with **298 rows that map
one-to-one onto the 298 corpus animals**, and "Animal_IDs", with 331 rows. Its
columns are animal ID, treatment, sex, age and genotype. sha256
`3c2de660…` (`protocol.json /lab_animal_sheet`).

**Aggregates only are recorded.** The per-animal rows are the lab's data and are not
copied into this repository. Values are as written, so spelling variants are not
merged.

| | counts over the 298 corpus animals |
|---|---|
| sex | Female 156, Male 142 |
| age | 18–24 Mo 197, Aged 35, 3–6 Mo 34, 4–8 Mo 28, Young 4 |
| genotype | Het3 121, APOE4 58, APOE3 53, APOE2 45, BL6x129 16 (+2 "Bl6x129"), blank 3 |
| treatment | blank 196, Vehicle 34 (+6 "Veh"), TMZ 34, DQP-1105 13 (+5 "DQP"), CIQ 7, Gcamp 10e11 3 |

**This matters beyond Step 4.** The corpus is not one homogeneous cohort: it mixes
sex, age, genotype and drug treatment.
- No analysis so far has used these covariates.
- The splits were drawn by animal **without** stratifying on them.
- The sheet gives **no drug timing or dose**, and **nothing about shock**.

How to use it is a decision for the human and the lab (`STATUS.md`).

## Still missing: only the lab can supply these

1. **Whether Context A sessions on Days 3–7 deliver shock.**
2. **Shock count and timing**, in seconds into the session, for every shocked session.
3. **Whether the within-day order of A and B was counterbalanced**, and the order
   itself.
4. **What CFC and CFD stand for**, and the session type of each day (conditioning,
   test, extinction).
5. **Drug timing and dose** for the treated animals, and whether treatment interacts
   with the protocol.

Until items 1, 2 and 4 are supplied, the Step 4 learning fits (C5) cannot run. **Shock
times are never inferred from startle or any other signal in the video.**
