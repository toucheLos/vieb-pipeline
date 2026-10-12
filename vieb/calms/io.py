"""CalMS21 pose, partitions and Luna geometry (REGISTRATION §1). No labels here.

The pose store (`$CALMS/npz/pose/...`, written by `scripts/calms_convert.py`)
holds keypoints and MARS scores only. This module decides partitions from
**which file a video sits in**, never from what its annotations say: the groom
partition is "the videos in the groom file", which is a fact about the dataset's
layout, not a label value.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Any

import numpy as np
import numpy.typing as npt

from vieb.tok import ego

F64 = npt.NDArray[np.float64]

CALMS = os.environ.get("CALMS", os.path.expanduser("~/data/calms21"))
POSE = os.path.join(CALMS, "npz", "pose")

#: CalMS21 order (nose, left ear, right ear, neck, left hip, right hip, tail base)
#: -> Luna order (left_ear, right_ear, nose, center, left_hip, right_hip,
#: tail_base), with the neck in `center`'s slot (REGISTRATION §1).
TO_LUNA: tuple[int, ...] = (1, 2, 0, 3, 4, 5, 6)
RESIDENT = 0
#: Luna's median ell_a in pixels (`results/ego_bodylen.json`).
LUNA_ELL_PX = 113.71871081670267
#: The size of Luna's MoSeq fit set, in frames (`work/moseq/sets.json` fit).
FIT_FRAMES = 573_872
#: Groom sequences drawn for the map side (D-S2).
N_MAP_GROOM = 5
SEED = 0

PARTITIONS = ("fit", "map_t1", "eval_t1", "map_g", "eval_g", "ceiling")


def _files(*parts: str) -> list[str]:
    return sorted(glob.glob(os.path.join(POSE, *parts, "*.npz")))


def _sha(path: str) -> str:
    with np.load(path, allow_pickle=False) as z:
        return str(z["sha1"])


def _frames(path: str) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(z["keypoints"].shape[0])


def partitions() -> dict[str, Any]:
    """Every partition as sorted video hashes, plus where each video's pose lives.

    Deterministic: sorted file names, `random.Random(SEED)`.
    """
    t1 = {s: [_sha(p) for p in _files("task1", "annotator-id_0") if f"__{s}__" in p]
          for s in ("train", "test")}
    groom_files = _files("task3", "groom")
    groom = sorted({_sha(p) for p in groom_files})
    a3 = {_sha(p) for p in _files("task2", "annotator-id_3")}
    a4 = {_sha(p) for p in _files("task2", "annotator-id_4")}
    ceiling = sorted(a3 & a4)

    rng = random.Random(SEED)
    map_g = sorted(rng.sample(groom, N_MAP_GROOM))
    eval_g = sorted(set(groom) - set(map_g))
    map_t1, eval_t1 = set(t1["train"]), set(t1["test"])
    # No video on both a map and an eval side; a collision leaves the map side.
    dropped = {"map_t1": sorted(map_t1 & (set(eval_g) | eval_t1)),
               "map_g": sorted(set(map_g) & (eval_t1 | set(eval_g)))}
    map_t1 -= set(dropped["map_t1"])
    map_g = sorted(set(map_g) - set(dropped["map_g"]))

    where: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(POSE, "task*", "*", "*.npz"))):
        where.setdefault(_sha(p), p)
    labelled = set(where)

    unl = _files("unlabeled", "unlabeled_videos")
    rng_f = random.Random(SEED)
    order = list(unl)
    rng_f.shuffle(order)
    fit, n = [], 0
    for p in order:
        h = _sha(p)
        if h in labelled or h in where:
            continue
        where[h] = p
        fit.append(h)
        n += _frames(p)
        if n >= FIT_FRAMES:
            break
    parts: dict[str, list[str]] = {
        "fit": sorted(fit), "map_t1": sorted(map_t1), "eval_t1": sorted(eval_t1),
        "map_g": map_g, "eval_g": eval_g, "ceiling": ceiling}
    for a in ("map_t1", "map_g"):
        for b in ("eval_t1", "eval_g"):
            assert not set(parts[a]) & set(parts[b]), (a, b)
    used = set().union(*(set(v) for v in parts.values()))
    return {**parts, "dropped_for_collision": dropped, "fit_frames": n,
            "where": {h: where[h] for h in sorted(used)}}


def scored(parts: dict[str, Any]) -> list[str]:
    """Every labelled video an instrument must label (map, eval, ceiling)."""
    return sorted(set().union(*(set(parts[k]) for k in PARTITIONS if k != "fit")))


def resident(path: str) -> tuple[F64, F64, float]:
    """``(pose (T, 7, 2) on Luna's pixel scale, scores (T, 7), ell_seq px)``.

    Luna keypoint order, the neck in `center`'s slot, each sequence scaled so
    its median nose-to-tail_base length equals Luna's median ell_a.
    """
    with np.load(path, allow_pickle=False) as z:
        kp = np.asarray(z["keypoints"], dtype=np.float64)[:, RESIDENT]   # (T, 2, 7)
        sc = np.asarray(z["scores"], dtype=np.float64)[:, RESIDENT]      # (T, 7)
    pose = np.transpose(kp, (0, 2, 1))[:, list(TO_LUNA)]                 # (T, 7, 2)
    scores = sc[:, list(TO_LUNA)]
    ell = float(np.median(ego.body_length(pose)))
    assert np.isfinite(ell) and ell > 0, path
    return pose * (LUNA_ELL_PX / ell), scores, ell


def save(parts: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(parts, fh, indent=1)


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        out: dict[str, Any] = json.load(fh)
    return out
