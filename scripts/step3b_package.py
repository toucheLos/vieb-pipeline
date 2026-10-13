"""STEP 3B.1-3B.2: the pilot set and the anatomy-only labelling package.

    python3 scripts/step3b_package.py      # -> work/step3b/label_package/, results/step3b/pilot.json

Registered in `results/step3b/REGISTRATION.md`. Frames are chosen for **pose
diversity only**: k-means on the egocentric repaired pose. No behaviour signal
(freeze score, syllable, state, label) is read anywhere in this script.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                    # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.tok import config, ego                                    # noqa: E402

OUT = os.path.join(config.REPO, "work", "step3b", "label_package")
RES = os.path.join(config.REPO, "results", "step3b", "pilot.json")
#: (day, context letter): REGISTRATION §1, in order.
STRATA = ((0, "A"), (1, "A"), (2, "C"), (3, "A"), (3, "B"), (4, "A"), (5, "B"), (6, "A"),
          (7, "A"), (7, "B"))
K = 20
N_SHARED = 5
SEED = 0
KEYPOINTS = ("left_ear", "right_ear", "nose", "center", "left_hip", "right_hip", "tail_base")
SCORERS = ("labeller1", "labeller2")


def pilot() -> list[str]:
    sp = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    tune = sorted(r for r in spine.recording_ids()
                  if sp.get(lab.parse(r)["animal"]) == "tune")
    rng = random.Random(SEED)
    used: set[str] = set()
    out = []
    for day, ctx in STRATA:
        cand = [r for r in tune if lab.parse(r)["day"] == day
                and lab.parse(r)["context_letter"] == ctx]
        rng.shuffle(cand)
        pick = next(r for r in cand if lab.parse(r)["animal"] not in used)
        used.add(lab.parse(pick)["animal"])
        out.append(pick)
    return out


def frames_for(rid: str) -> list[tuple[int, int]]:
    """``[(cluster, frame)]``: the frame nearest each k-means centroid, cluster order."""
    from sklearn.cluster import KMeans
    tag = lab.animal_tag(rid)
    z = np.load(os.path.join(config.REPO, "work", "repaired", f"{tag}.npz"))
    ids = [str(r) for r in z["recording_ids"]]
    b = z["bounds"]
    allp = np.asarray(z["pose"], dtype=np.float64)
    ok_all = np.isfinite(allp).all(axis=(1, 2))
    ell = float(np.median(ego.body_length(allp[ok_all])))      # the animal's ell_a
    i = ids.index(rid)
    pose = allp[b[i]:b[i + 1]]
    ok = np.flatnonzero(np.isfinite(pose).all(axis=(1, 2)))
    X = ego.egocentric(pose[ok], ell)
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
    out = []
    for c in range(K):
        mem = np.flatnonzero(km.labels_ == c)
        d = np.linalg.norm(X[mem] - km.cluster_centers_[c], axis=1)
        out.append((c, int(ok[mem[np.argmin(d)]])))
    return out


def template(path: str, scorer: str, images: list[str]) -> None:
    """A DLC CollectedData CSV: three header rows, NaN coordinates to fill."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer"] + [scorer] * (2 * len(KEYPOINTS)))
        w.writerow(["bodyparts"] + [k for k in KEYPOINTS for _ in "xy"])
        w.writerow(["coords"] + ["x", "y"] * len(KEYPOINTS))
        for im in images:
            w.writerow([im] + [""] * (2 * len(KEYPOINTS)))


def main() -> int:
    import cv2
    os.makedirs(OUT, exist_ok=True)
    rids = pilot()
    rows, imgs = [], {s: [] for s in SCORERS}
    for rid in rids:
        sel = frames_for(rid)
        d = os.path.join(OUT, "labeled-data", rid)
        os.makedirs(d, exist_ok=True)
        cap = cv2.VideoCapture(vid.video_path(rid))
        for rank, (c, f) in enumerate(sel):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            assert ok, (rid, f)
            name = f"img{f:06d}.png"
            cv2.imwrite(os.path.join(d, name), fr)
            rel = f"labeled-data/{rid}/{name}"
            shared = rank < N_SHARED
            rows.append({"recording_id": rid, "frame": f, "cluster": c, "shared": shared,
                         "image": rel})
            imgs["labeller1"].append(rel)
            if shared:
                imgs["labeller2"].append(rel)
        cap.release()
        log(f"  {rid}: {len(sel)} frames")
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in SCORERS:
        template(os.path.join(OUT, f"CollectedData_{s}.csv"), s, imgs[s])
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "# STEP 3B labelling package (anatomy only)\n\n"
            f"{len(rows)} frames from {len(rids)} tune-split recordings, chosen for pose "
            "diversity (k-means on egocentric pose); no behaviour was looked at.\n\n"
            "- **labeller1** labels every frame in `CollectedData_labeller1.csv`.\n"
            f"- **labeller2** labels the {N_SHARED * len(rids)} shared frames in "
            "`CollectedData_labeller2.csv`, without seeing labeller1's clicks.\n"
            "- Keypoints, in pixels of the full frame: " + ", ".join(KEYPOINTS) + ".\n"
            "  `center` is the body centre (mid-back between shoulders and hips), as in the "
            "original DLC project. Leave a keypoint blank if it is not visible.\n")
    write_json({**provenance.header(anchors.LUNA, stage="step3b_package",
                                    unverified="labels pending (two human labellers)"),
                "registration": "results/step3b/REGISTRATION.md", "pilot": rids,
                "strata": [list(s) for s in STRATA], "k": K, "n_frames": len(rows),
                "n_shared": sum(r["shared"] for r in rows),
                "frames": rows}, RES)
    log(f"package: {len(rows)} frames, {sum(r['shared'] for r in rows)} shared -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
