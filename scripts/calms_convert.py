"""CalMS21 JSON -> per-sequence npz, with pose and labels in separate stores.

    python3 scripts/calms_convert.py            # all three tasks
    # writes $CALMS/npz/pose/<task>/<group>/<seq>.npz   (keypoints, scores)
    #        $CALMS/npz/labels/<task>/<group>/<seq>.npz (annotations, vocab)
    #        $CALMS/npz/facts.json                     (DATA_FACTS inputs)

The label store is read by `vieb.calms.score` and nothing else (STEP 1 §1.4).
This converter touches labels only to move them into that store and to count
them for DATA_FACTS; no choice downstream depends on what it reads.

Task 2 overlap is checked by hashing each sequence's keypoint array: two
annotators labelled the same video iff their keypoints are byte-identical.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys

import numpy as np

CALMS = os.environ.get("CALMS", os.path.expanduser("~/data/calms21"))
FILES = {
    "task1": ["task1_classic_classification/calms21_task1_train.json",
              "task1_classic_classification/calms21_task1_test.json"],
    "task2": ["task2_annotation_styles/calms21_task2_train.json",
              "task2_annotation_styles/calms21_task2_test.json"],
    "task3": ["task3_new_behaviors/calms21_task3_train.json",
              "task3_new_behaviors/calms21_task3_test.json"],
}


def bouts(a: np.ndarray, value: int) -> int:
    on = np.diff(np.r_[0, (a == value).astype(np.int8), 0])
    return int((on == 1).sum())


def safe(name: str) -> str:
    return name.replace("/", "__")


def unlabeled() -> int:
    """The unlabelled set: pose only, plus its overlap with every labelled video."""
    import glob
    out = os.path.join(CALMS, "npz")
    labelled = {str(np.load(p)["sha1"]) for p in
                glob.glob(os.path.join(out, "pose", "task*", "*", "*.npz"))}
    pdir = os.path.join(out, "pose", "unlabeled", "unlabeled_videos")
    os.makedirs(pdir, exist_ok=True)
    seqs, overlap = [], []
    for part in range(1, 5):
        path = f"unlabeled_videos/calms21_unlabeled_videos_part{part}.json"
        print(f"loading {path}", flush=True)
        with open(os.path.join(CALMS, path)) as fh:
            data = json.load(fh)
        for _group, recs in data.items():
            for name, rec in recs.items():
                kp = np.asarray(rec["keypoints"], dtype=np.float32)
                sc = np.asarray(rec["scores"], dtype=np.float32)
                h = hashlib.sha1(kp.tobytes()).hexdigest()[:16]
                seqs.append({"seq": name, "frames": int(kp.shape[0]), "sha1": h})
                if h in labelled:
                    overlap.append(name)
                np.savez_compressed(os.path.join(pdir, f"{safe(name)}.npz"),
                                    keypoints=kp, scores=sc, seq=name,
                                    split="unlabeled", sha1=h)
        del data
        gc.collect()
    with open(os.path.join(out, "facts_unlabeled.json"), "w") as fh:
        json.dump({"n_sequences": len(seqs),
                   "frames": int(sum(s["frames"] for s in seqs)),
                   "overlap_with_labelled": overlap, "sequences": seqs}, fh, indent=1)
    print(f"unlabeled: {len(seqs)} sequences, {len(overlap)} shared with labelled",
          flush=True)
    return 0


def main() -> int:
    if "--unlabeled" in sys.argv:
        return unlabeled()
    out = os.path.join(CALMS, "npz")
    facts: dict = {"tasks": {}, "hash_index": {}}
    for task, paths in FILES.items():
        tf: dict = {"files": paths, "groups": {}}
        for path in paths:
            split = "train" if path.endswith("_train.json") else "test"
            print(f"loading {path}", flush=True)
            with open(os.path.join(CALMS, path)) as fh:
                data = json.load(fh)
            for group, seqs in data.items():
                g = tf["groups"].setdefault(group, {"splits": {}})
                gs = g["splits"].setdefault(split, {
                    "n_sequences": 0, "frames": 0, "label_frames": {},
                    "label_bouts": {}, "label_sequences": {}})
                for name, rec in seqs.items():
                    kp = np.asarray(rec["keypoints"], dtype=np.float32)
                    sc = (np.asarray(rec["scores"], dtype=np.float32)
                          if "scores" in rec else None)
                    an = np.asarray(rec["annotations"], dtype=np.int16)
                    meta = rec.get("metadata", {})
                    vocab = meta.get("vocab")
                    if "keys" not in g:
                        g["keys"] = sorted(rec.keys())
                        g["metadata"] = {k: v for k, v in meta.items()}
                        g["keypoints_shape_example"] = list(kp.shape)
                        g["scores_shape_example"] = (list(sc.shape)
                                                     if sc is not None else None)
                    assert an.shape[0] == kp.shape[0], name
                    h = hashlib.sha1(kp.tobytes()).hexdigest()[:16]
                    facts["hash_index"].setdefault(h, []).append(
                        {"task": task, "group": group, "split": split,
                         "seq": name, "frames": int(kp.shape[0])})
                    gs["n_sequences"] += 1
                    gs["frames"] += int(kp.shape[0])
                    names = (vocab if isinstance(vocab, dict)
                             else {str(v): v for v in np.unique(an).tolist()})
                    for lab, idx in names.items():
                        idx = int(idx)
                        n = int((an == idx).sum())
                        gs["label_frames"][lab] = gs["label_frames"].get(lab, 0) + n
                        gs["label_bouts"][lab] = gs["label_bouts"].get(lab, 0) + bouts(an, idx)
                        gs["label_sequences"][lab] = (gs["label_sequences"].get(lab, 0)
                                                      + int(n > 0))
                    pdir = os.path.join(out, "pose", task, safe(group))
                    ldir = os.path.join(out, "labels", task, safe(group))
                    os.makedirs(pdir, exist_ok=True)
                    os.makedirs(ldir, exist_ok=True)
                    stem = safe(name)
                    np.savez_compressed(
                        os.path.join(pdir, f"{stem}.npz"), keypoints=kp,
                        scores=sc if sc is not None else np.zeros(0, np.float32),
                        seq=name, split=split, sha1=h)
                    np.savez_compressed(
                        os.path.join(ldir, f"{stem}.npz"), annotations=an,
                        vocab=json.dumps(vocab), seq=name, split=split)
            del data
            gc.collect()
        facts["tasks"][task] = tf
    shared = {h: v for h, v in facts["hash_index"].items() if len(v) > 1}
    facts["shared_videos"] = shared
    facts["n_unique_videos"] = len(facts["hash_index"])
    del facts["hash_index"]
    with open(os.path.join(out, "facts.json"), "w") as fh:
        json.dump(facts, fh, indent=1)
    print(f"unique videos {facts['n_unique_videos']}, "
          f"shared {len(shared)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
