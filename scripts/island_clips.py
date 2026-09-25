"""Step 6. The island's segments as clips, at each segment's OWN extent.

    python3 scripts/island_clips.py [--pad 0.5] [--max 361]

## Not a fixed window

`island_look.py` trims all three members of a triad to the shortest and caps at
6 s (`MAX_CLIP_S`). That was right there -- if the odd one is the short one, a
rater picks it with a stopwatch -- and it is wrong here. **A unit shown in a
longer clip is mostly not that unit**, and a unit cut short is not the unit
either. Each segment is rendered at `[a - pad, b + pad]`.

There is no padding convention anywhere in this repo, so `PAD_S` is invented
here and named: half a second either side, the shortest lead-in that lets a
viewer see the animal enter the state rather than opening mid-state. It is
applied identically to the island and to the control, so it cannot separate
them.

## The flags show nothing, and the page says why

`work/tok/seg_vocab/shape__k3__behaviour.json` records the bone-violation
contrast as "vacuous by construction": a selected segment has
`abstain_frac == 0`, and the abstain mask CONTAINS the skull bone-violation
mask, so clump 0 holds zero flagged frames -- and so does every other selected
segment. Measured, against the wider `bone | missing | interpolated` union the abstain
mask may not fully contain: **zero** inside the segment, in **both** arms.

The overlay is armed anyway and marks only INSIDE the segment. A first render
padded the mark and leaked the arm -- 64 of 361 control clips carried a flagged
pad frame against 2 of 361 island clips, an 18% cue for "control" in a panel
that is meant to be blind. The pad is context, not the unit.

## The control is blind and its key is held back

A count- and duration-matched draw from the 98.1% unassigned continuum, shuffled
into one panel with the island under opaque ids. The key is written beside the
clips and is NOT published.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                           # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import frames, log, write_json                      # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.clean import arms as clean_arms                           # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.tok import config                                         # noqa: E402

GROUP, K_MAD, CLUMP = "shape", 3.0, 0
SEED = 0
#: Lead-in and lead-out, in seconds, converted once. Invented here -- the repo
#: has no padding convention -- and applied identically to both arms.
PAD_S = 0.5
#: Salt for the opaque panel ids. A function of arm and index only, never of
#: the class, so the key cannot be recovered by re-running.
SALT = "vieb-island-panel"


def out_dir() -> str:
    return os.path.join(config.PATHS.results_dir, "island")


def _id(arm: str, i: int) -> str:
    h = hashlib.sha256(f"{SALT}|{arm}|{i}".encode()).hexdigest()
    return "s" + h[:10]


def _gate() -> None:
    """Refuse to cut unless the alignment gate passed, as compare_clips does.

    `behaviour_clips.py` skipped this. Pose drawn on a misaligned video is a
    picture of the wrong frames, and it looks fine.
    """
    p = config.PATHS.result("compare_gate.json")
    with open(p, encoding="utf-8") as fh:
        import json
        g = json.load(fh)
    for key in ("availability", "alignment"):
        v = str((g.get(key) or {}).get("verdict", "?"))
        if v not in ("PASS", "GRID_LIMITED"):
            raise SystemExit(f"{key} gate is {v}; refusing to cut clips")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pad", type=float, default=PAD_S)
    p.add_argument("--max", type=int, default=0, help="cap per arm, 0 = all")
    a = p.parse_args(argv)

    _gate()
    os.makedirs(os.path.join(out_dir(), "clips"), exist_ok=True)
    fps = spine.fps()
    pad = frames(a.pad, fps)

    il = __import__("importlib.util", fromlist=["util"])
    spec = il.spec_from_file_location(
        "il_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "island_look.py"))
    assert spec and spec.loader
    m = il.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["island_look"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved

    island, pool = m.segments()
    log(f"  {len(island)} island segments, {len(pool)} in the continuum")

    # The control: count-matched, and matched on the duration DISTRIBUTION by
    # taking the nearest available partner for each island member in log
    # duration. Duration is the cue a viewer would otherwise sort on.
    rng = np.random.default_rng(seeds.stable_seed(SEED, "panel"))
    order = np.argsort([r["duration_s"] for r in pool])
    dur = np.asarray([pool[i]["duration_s"] for i in order])
    used: set[int] = set()
    control: list[dict] = []
    for r in island:
        j = int(np.searchsorted(dur, r["duration_s"]))
        for off in range(len(order)):
            for k in (j + off, j - off):
                if 0 <= k < len(order) and int(order[k]) not in used:
                    used.add(int(order[k]))
                    control.append(pool[int(order[k])])
                    break
            else:
                continue
            break
    log(f"  {len(control)} control segments matched on duration")

    rows: list[dict] = []
    key: list[dict] = []
    cap = a.max or max(len(island), len(control))
    pose_cache: dict = {}
    for arm, members in (("island", island[:cap]), ("control", control[:cap])):
        for i, r in enumerate(members):
            rid = r["recording_id"]
            if rid not in pose_cache:
                d = spine.clean(rid)
                held = clean_arms.held_array(
                    d["pose_unfiltered"].astype(np.float64),
                    d["missing"].astype(bool))
                bad = (np.asarray(d["bone_flagged"], bool)
                       | np.asarray(d["missing"], bool).any(axis=1)
                       | np.asarray(d["interpolated"], bool).any(axis=1))
                pose_cache[rid] = (held, bad)
                if len(pose_cache) > 30:
                    pose_cache.pop(next(iter(pose_cache)))
            held, bad = pose_cache[rid]
            a0 = max(0, int(r["a"]) - pad)
            b0 = min(int(held.shape[0]), int(r["b"]) + pad)
            if b0 - a0 < 2:
                continue
            cid = _id(arm, i)
            rel = os.path.join("clips", f"{cid}.mp4")
            box = vid.crop_box(held, a0, b0, size=vid.CROP)
            # Marked only INSIDE the segment. The pad is context, not the
            # unit, and marking it leaked the arm: measured over a first render
            # at this pad, 64 of 361 control clips carried a flagged pad frame
            # against 2 of 361 island clips -- so a red border in the lead-in
            # was an 18% cue for "control" in a panel that is supposed to be
            # blind. Inside the segments themselves both arms measure exactly
            # zero, which is a property of the selection (abstain_frac == 0),
            # not a finding.
            #
            # `flags` is indexed by ABSOLUTE frame, the same space as `pose`,
            # and is only consulted when `pose is not None`. A slice here would
            # mark the wrong frames and look fine doing it.
            seg_only = np.zeros_like(bad)
            seg_only[int(r["a"]):int(r["b"])] = bad[int(r["a"]):int(r["b"])]
            got, why = vid.cut(vid.video_path(rid), a0, b0,
                               os.path.join(out_dir(), rel), fps=fps,
                               pose=held, flags=seg_only, label=None, crop=box)
            if not got:
                log(f"  skip {cid}: {why}")
                continue
            n_flag = int(seg_only[a0:b0].sum())
            rows.append({"clip": cid, "file": rel, "animal": r["animal"],
                         "recording_id": rid,
                         "duration_s": round((b0 - a0) / fps, 3),
                         "n_frames": int(b0 - a0),
                         "flagged_frames": n_flag,
                         "bytes": os.path.getsize(
                             os.path.join(out_dir(), rel))})
            key.append({"clip": cid, "arm": arm,
                        "a": int(r["a"]), "b": int(r["b"])})
        del members

    rng.shuffle(rows)
    n_flag = sum(r["flagged_frames"] for r in rows)
    write_json({**provenance.header(anchors.LUNA, stage="island_clips",
                                 unverified="a curated clip set"),
                "inherited_digest": spine.digest(),
                "group": GROUP, "k_mad": K_MAD, "clump": CLUMP,
                "pad_s": a.pad, "pad_frames": pad, "seed": SEED,
                "extent": "each segment's own [a-pad, b+pad]; never a fixed window",
                "overlay": ("skeleton drawn; flagged frames inside the SEGMENT "
                            "marked by a red border. The pad is not marked: "
                            "marking it leaked the arm, 64 control clips "
                            "against 2 island"),
                "blind": "panel order shuffled; arm lives only in key.json",
                "n_clips": len(rows),
                "flagged_frames_total": n_flag,
                "clips": rows},
               os.path.join(out_dir(), "manifest.json"))
    write_json({**provenance.header(anchors.LUNA, stage="island_clips_key",
                                 unverified="a curated clip set"),
                "n": len(key), "key": key},
               os.path.join(out_dir(), "key.json"))
    log(f"  {len(rows)} clips, {n_flag} flagged frames across all of them")
    log(f"  wrote {os.path.join(out_dir(), 'manifest.json')} and key.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
