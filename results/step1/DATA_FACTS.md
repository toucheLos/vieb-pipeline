# STEP 1 §1.0 — CalMS21 data facts, read from the files before registration

Source: Caltech Data record `s0vdx-0k302`, downloaded on 2026-10-12 to
`~/data/calms21/` (SHA-256 in `~/data/calms21/SHA256SUMS`). Converted by
`scripts/calms_convert.py` into per-sequence npz, with **pose and labels in
separate stores**. Counts below come from `~/data/calms21/npz/facts.json`;
quotations come from the dataset's `readme.md`.

## Facts

| fact | value | source |
|---|---|---|
| **Licence** | **CC BY 4.0** (research use permitted, with attribution) | Caltech Data record page |
| **Frame rate** | **30 Hz.** The files carry no frame-rate field. The paper (Sun et al. 2021, arXiv 2104.02710) states that poses were "recorded … at 30Hz" | paper |
| Resampling | **none**: 30 Hz = Luna's 30 fps | — |
| Keypoints | 7 per mouse, ordered **(nose, left ear, right ear, neck, left hip, right hip, tail base)**, top view, pixels in a 1024 × 570 image | readme: "There are 7 tracked body parts, ordered (nose, left ear, right ear, neck, left hip, right hip, tail base)" |
| Array layout | `keypoints` (T, mouse, xy, keypoint); `scores` (T, mouse, keypoint) in [0, 1]; no NaNs | files |
| Resident | **mouse 0** = the resident (black) mouse | readme: "mouse 0 is the resident (black) mouse and mouse 1 is the intruder (white) mouse" |
| Whose behaviour | the resident's, **"unless specifically noted"** | readme: "annotations refer specifically to behaviors initiated by the black mouse, the 'resident'" |
| Tracker | MARS (stacked hourglass); confidence is MARS's, not DLC's likelihood | readme |
| Animal identities | **not provided.** Train and test are split by animal; within a split, identity is unknown | readme: "Training and test sets are split by animal identity"; paper: identities "not tracked in the dataset" |
| Median resident nose–tail_base | about 184–199 px (six sequences inspected) | files |
| Confidence (resident + intruder) | median ≈ 0.72; 5th percentile ≈ 0.42–0.52 | files |

## Sequences and labels

**Task 1** (annotator 0; vocabulary attack 0, investigation 1, mount 2, other 3):

| split | sequences | frames | attack bouts (seqs) | investigation bouts (seqs) | mount bouts (seqs) |
|---|---|---|---|---|---|
| train | 70 | 507,738 | 194 (18) | 1,811 (69) | 282 (41) |
| test | 19 | 262,107 | 258 (**5**) | 1,068 (19) | 195 (14) |

**Task 2** (annotators 1–5, the same vocabulary). Each annotator has 6 train sequences, plus 13, 6, 4, 4 and 20 test sequences.
- **Overlap: only annotators 3 and 4 labelled the same videos.** These are 10 sequences (6 train + 4 test), with byte-identical keypoint arrays.
- No other pair of annotators shares a video.

**Task 3** (binary per behaviour: other 0, behaviour 1):

| behaviour | annotator | train seqs | test seqs | bouts (train + test) |
|---|---|---|---|---|
| approach | 0 | 3 | 25 | 208 |
| disengaged | 2 | 2 | 1 | 87 |
| **groom** | 6 | **2** | **13** | **184**, all 15 sequences contain groom |
| intromission | 0 | 1 | 10 | 37 |
| mount_attempt | 0 | 4 | 12 | 33 |
| sniff_face | 0 | 3 | 47 | 432 |
| **whiterearing** | 2 | **2** | **1** | 203 |

**Shared videos.** By keypoint hash, the labelled files hold **168 unique videos**, 78 of them in more than one file:
- every Task 3 video except disengaged and whiterearing is also a Task 1 video, including **9 of the 15 groom sequences**;
- Task 2's annotators 3 and 4 share their 10.

The unlabelled set's overlap with these 168 is recorded in `npz/facts_unlabeled.json`.

## Consequences for the registration (decided with the user before registering)

1. **"White rearing" is the intruder's rearing.** It is the "specifically noted" exception (white = the intruder). It exists in 3 sequences, so the resident-only arm cannot score it, and it cannot meet the ≥ 10-unit support rule. **Rearing is untestable in CalMS21, and grooming alone is the Luna-relevant positive** that decides O3.
2. **Grooming:** 15 sequences, split **5 mapping / 10 evaluation** by a seeded draw, so the support rule (≥ 30 bouts over ≥ 10 sequences) can be met.
3. **The unit of independence is the sequence.** Within a split, a resident may recur across sequences undetectably, so **every Read states that its intervals may be too narrow.**
4. **Ceiling.** The inter-annotator agreement of annotators 3 and 4 on their 10 shared videos serves for attack, investigation and mount. **Groom has no second annotator, so its ceiling is 1** (the user's decision).
5. **Neck fills `center`'s slot** in the primary arm, as the prompt specifies. Luna's centre is a mid-body point and the neck is not, so the ego origin moves forward. This is what the 6-keypoint sensitivity arm (deferred) exists to check.
6. **Body length** follows `ego.body_length` (nose → tail_base), with its median taken per **sequence**, because there are no animal identities.
