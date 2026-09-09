"""Phase C. Before/after clips: what the cleaning does, where a viewer can see it.

    python3 scripts/compare_clips.py --check     # the gates, first
    python3 scripts/compare_clips.py --render

Three sets, and the middle one is what keeps the reel honest:

  worst     the highest violation-rate recordings
  random    a seeded stratified sample across the violation distribution --
            a reel of only worst cases overstates how bad the corpus is
  bakeoff   the same segments under several cleaning arms, so the methods are
            compared against each other and not each against raw

The violation masks are read from `work/bones/<animal>.npz`, bit-packed at Step 1
over that animal's concatenated frames. Recomputing them here would risk a second
implementation disagreeing with the published rate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab                           # noqa: E402
from recur.render import video as vid                              # noqa: E402
from recur.util import log, read_json, write_json                  # noqa: E402
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.render import compare                                    # noqa: E402
from vieb.tok import config                                        # noqa: E402

EPS = 0.10
CELL = f"viol|unfiltered|raw|{EPS}|skull"
N_WORST = 6
N_RANDOM = 12
N_BAKEOFF = 4
WINDOWS_PER_RECORDING = 3
SEED = 0


def violation_mask(recording_id: str) -> np.ndarray:
    """The Step 1 skull mask for one recording, unpacked from its animal shard."""
    tag = lab.animal_tag(recording_id)
    with np.load(config.PATHS.bones_shard(tag), allow_pickle=False) as z:
        ids = [str(v) for v in z["recording_ids"]]
        r = ids.index(str(recording_id))
        lo, hi = int(z["bounds"][r]), int(z["bounds"][r + 1])
        total = int(z["bounds"][-1])
        return np.unpackbits(z[CELL])[:total][lo:hi].astype(bool)


def pick(args) -> dict:
    """The three sets, by name -> [(recording_id, rate)]."""
    d = read_json(config.PATHS.result("bones.json"))
    worst = d["worst_recordings"]
    rng = np.random.default_rng(SEED)

    chosen: dict = {"worst": [(w["recording_id"], w["rate"])
                              for w in worst[:N_WORST]]}

    # Stratified across the violation-rate distribution, not uniform: a uniform
    # sample of 3,846 recordings is a sample of the median and shows nothing.
    ids = spine.recording_ids()
    by_rate = {w["recording_id"]: w["rate"] for w in worst}
    pool = [r for r in ids if r not in by_rate]
    rng.shuffle(pool)
    chosen["random"] = [(r, float("nan")) for r in pool[:N_RANDOM]]
    chosen["bakeoff"] = chosen["worst"][:N_BAKEOFF]
    return chosen


def gates(args) -> int:
    ids = spine.recording_ids()
    obj = {"dataset": "luna", "arm": "compare", "split": "all"}
    av = vid.availability_read(ids, scored_object=obj, n_effective=len(ids))
    log(av.line())
    bounds = []
    off = 0
    for rid in ids:
        n = int(spine.clean(rid)["pose"].shape[0])
        bounds.append(off)
        off += n
    bounds.append(off)
    al = vid.alignment_read(ids, np.asarray(bounds), scored_object=obj,
                            n_effective=len(ids), sample=120, seed=SEED)
    log(al.line())
    write_json({"availability": av.to_dict(), "alignment": al.to_dict()},
               config.PATHS.result("compare_gate.json"))
    return 0 if al.verdict in ("PASS", "GRID_LIMITED") else 1


def render(args) -> int:
    gate_path = config.PATHS.result("compare_gate.json")
    if not os.path.exists(gate_path):
        raise SystemExit("run --check first: no clip is cut against an "
                         "unverified pose/video alignment")
    gate = read_json(gate_path)
    if gate["alignment"]["verdict"] not in ("PASS", "GRID_LIMITED"):
        raise SystemExit(f"alignment gate is {gate['alignment']['verdict']}")

    out_dir = os.path.join(config.PATHS.results_dir, "compare", "clips")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    sets = pick(args)
    manifest: list = []

    for kind, picks in sets.items():
        for rid, rate in picks:
            d = spine.clean(rid)
            unfiltered = d["pose_unfiltered"].astype(np.float64)
            wiener = d["pose"].astype(np.float64)
            mask = violation_mask(rid)
            wins = compare.windows(mask, fps=fps)
            if not wins:
                # No violation to centre on: take one window from the middle, so
                # the random set is not silently reduced to its dirty members.
                mid = mask.shape[0] // 2
                wins = [(mid, min(mid + int(4 * fps), mask.shape[0]))]
            wins = wins[:WINDOWS_PER_RECORDING if kind != "random" else 1]

            for w, (a, b) in enumerate(wins):
                if kind == "bakeoff":
                    held = clean_arms.held_array(unfiltered, d["missing"].astype(bool))
                    panes = [("raw", unfiltered), ("wiener", wiener),
                             ("median 0.50s",
                              clean_arms.apply("median_0.50", held, d["conf"], fps))]
                else:
                    panes = [("raw", unfiltered), ("wiener", wiener)]
                name = f"{kind}_{lab.animal_tag(rid)}_{w:02d}.mp4"
                path = os.path.join(out_dir, name)
                ok, why = compare.compare(
                    vid.video_path(rid), a, b, path, fps=fps, panes=panes,
                    crop_from=unfiltered, flags=mask)
                if not ok:
                    log(f"  SKIP {name}: {why}")
                    continue
                manifest.append({
                    "kind": kind, "file": os.path.join("clips", name),
                    "recording_id": rid, "animal": lab.animal_tag(rid),
                    "a": int(a), "b": int(b),
                    "duration_s": round((b - a) / fps, 2),
                    "panes": [p[0] for p in panes],
                    "violation_rate": None if not np.isfinite(rate) else rate,
                    "violations_in_clip": float(mask[a:b].mean()),
                    "bytes": os.path.getsize(path),
                })
                log(f"  {name}  {(b - a) / fps:.1f}s  "
                    f"{os.path.getsize(path) / 1024:.0f} KB")

    doc = {
        **anchors.header(anchors.LUNA, stage="compare",
                         unverified="a curated clip set, not a corpus statistic"),
        "inherited_digest": spine.digest(),
        "eps": EPS, "seed": SEED,
        "n_clips": len(manifest),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "sets": {k: len(v) for k, v in sets.items()},
        "note": ("Left pane is the pose after the gap policy and before the "
                 "filter; right pane is the filtered array every downstream "
                 "stage consumes. Same frames, same crop box, same red outline "
                 "on violating frames -- the only difference between panes is "
                 "the skeleton."),
        "clips": manifest,
    }
    write_json(doc, os.path.join(config.PATHS.results_dir, "compare",
                                 "manifest.json"))
    log(f"{len(manifest)} clips, {doc['total_bytes'] / 1e6:.1f} MB")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--render", action="store_true")
    a = p.parse_args(argv)
    if a.check:
        return gates(a)
    if a.render:
        return render(a)
    raise SystemExit("pass --check or --render")


if __name__ == "__main__":
    raise SystemExit(main())
