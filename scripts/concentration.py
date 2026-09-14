"""H1. Are tracking errors spread evenly, or concentrated on particular footage?

    python3 scripts/concentration.py --write-grid
    sbatch --array=0-29%15 jobs/concentration.slurm
    python3 scripts/concentration.py --combine --split report

The cheap half of the ensemble decision. An ensemble averages away errors its
members make independently; errors driven by the situation are the ones every
member makes, and averaging cannot touch them. So the dispersion of violation
rates bounds how much an ensemble could help -- it does not decide it, and every
read says so.

Reuses the Step 1 masks rather than recomputing violations: `work/bones/<tag>.npz`
holds them bit-packed per cell, which is the same array `results/bones.json`'s
2.134% was computed from. A second implementation could disagree with the
published rate, which is the thing to avoid here above all.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                   # noqa: E402
from recur.util import log, peak_rss_gb, write_json                # noqa: E402
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.qc import concentration as cc                            # noqa: E402
from vieb.tok import config                                        # noqa: E402

EPS = 0.10
#: The published Step 1 cell: unfiltered input, raw metric, eps 0.10, skull
#: bones. `results/bones.json`'s 2.134% is this array.
CELL = f"viol|unfiltered|raw|{EPS}|skull"
CENTRE = 3


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def session_of(rid: str) -> dict:
    """Box, day and context from the recording id, for the group decomposition.

    Parsed rather than looked up: the ids are structured
    `<date>_Box_<n>_CF?_Day_<n>_(Context_<X>)_<animal>` and no artifact carries
    the decomposition separately.
    """
    box = re.search(r"Box_(\d+)", rid)
    day = re.search(r"Day_(\d+)", rid)
    ctx = re.search(r"Context_([A-Z])", rid)
    date = re.match(r"(\d{8})", os.path.basename(rid))
    return {"box": box.group(1) if box else "?",
            "day": day.group(1) if day else "?",
            "context": ctx.group(1) if ctx else "?",
            "date": date.group(1) if date else "?"}


def violation_mask(rid: str) -> np.ndarray:
    """Step 1's skull mask for one recording, unpacked from its animal shard."""
    tag = lab.animal_tag(rid)
    with np.load(config.PATHS.bones_shard(tag), allow_pickle=False) as z:
        ids = [str(v) for v in z["recording_ids"]]
        r = ids.index(str(rid))
        lo, hi = int(z["bounds"][r]), int(z["bounds"][r + 1])
        total = int(z["bounds"][-1])
        return np.unpackbits(z[CELL])[:total][lo:hi].astype(bool)


def shard(args, tag: str) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "concentration")
    os.makedirs(out_dir, exist_ok=True)
    mine = animals_of(spine.recording_ids())[tag]

    rows: list = []
    edge_hist: dict = {}
    for rid in mine:
        mask = violation_mask(rid)
        d = spine.clean(rid)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        if held.shape[0] != mask.shape[0]:
            log(f"  SKIP {rid}: {held.shape[0]} frames vs {mask.shape[0]} mask")
            continue
        edge = cc.edgeness(held[:, CENTRE])
        prof = cc.profile(edge, mask)
        for b in prof.get("bins", []):
            cell = edge_hist.setdefault(int(b["bin"]), {"n": 0, "k": 0,
                                                        "lo": [], "hi": []})
            cell["n"] += int(b["n"])
            cell["k"] += int(round(b["rate"] * b["n"])) if b["n"] else 0
            cell["lo"].append(b["lo"])
            cell["hi"].append(b["hi"])
        rows.append({"recording_id": rid, "animal": tag,
                     "n_frames": int(mask.shape[0]),
                     "n_violating": int(mask.sum()),
                     "rate": float(mask.mean()),
                     **session_of(rid)})
        log(f"[{tag}] {rid[-26:]}  {mask.mean():.4%}  ({int(mask.sum()):,} frames)")

    write_json({"animal": tag, "eps": EPS, "cell": CELL,
                "edge_bins": {str(k): {"n": v["n"], "k": v["k"],
                                       "lo": float(np.median(v["lo"])),
                                       "hi": float(np.median(v["hi"]))}
                              for k, v in sorted(edge_hist.items())},
                "inherited_digest": spine.digest(), "rows": rows},
               os.path.join(out_dir, f"{tag}.json"))
    log(f"[{tag}] peak_rss={peak_rss_gb():.2f} GB")
    return 0


def combine(args) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "concentration")
    shards = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    edge: dict = {}
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        rows += doc["rows"]
        for b, cell in doc.get("edge_bins", {}).items():
            acc = edge.setdefault(int(b), {"n": 0, "k": 0, "lo": [], "hi": []})
            acc["n"] += cell["n"]
            acc["k"] += cell["k"]
            acc["lo"].append(cell["lo"])
            acc["hi"].append(cell["hi"])
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    sel = [r for r in rows if of.get(r["animal"]) == args.split]
    log(f"{len(shards)} shards, {len(sel)} recordings on {args.split}")

    counts = [r["n_violating"] for r in sel]
    sizes = [r["n_frames"] for r in sel]
    disp = cc.dispersion(counts, sizes)
    lor = cc.lorenz(counts)

    groups = {}
    for key in ("animal", "box", "day", "context", "date"):
        k_by: dict = {}
        n_by: dict = {}
        for r in sel:
            k_by[r[key]] = k_by.get(r[key], 0) + r["n_violating"]
            n_by[r[key]] = n_by.get(r[key], 0) + r["n_frames"]
        groups[key] = cc.grouped(k_by, n_by)

    edge_profile = [
        {"bin": b, "n": v["n"],
         "rate": float(v["k"] / v["n"]) if v["n"] else float("nan"),
         "lo_iqr": float(np.median(v["lo"])), "hi_iqr": float(np.median(v["hi"]))}
        for b, v in sorted(edge.items())]

    obj = {"dataset": "luna", "arm": "concentration", "split": args.split,
           "eps": EPS, "cell": CELL}
    rd = cc.concentration_read(disp, lor, scored_object=obj,
                               n_effective=len({r["animal"] for r in sel}))
    doc = {
        **anchors.header(anchors.LUNA, stage="concentration",
                         unverified="scored on one split; the anchor counts all"),
        "inherited_digest": spine.digest(),
        "preprocessing_freeze": "F3",
        "split": args.split, "eps": EPS, "cell": CELL,
        "reads": {"concentration": rd.to_dict()},
        "dispersion": disp, "lorenz": lor,
        "by_group": groups,
        "edge_profile": edge_profile,
        "edgeness_note": (
            "no arena definition exists in shapeflow's artifacts, so the arena "
            "is derived per recording from the animal's own centroid cloud: "
            "median position as the middle, IQR as the scale. Units are IQR, "
            "unclipped, because the tail is the part in question"),
        "caveat": cc.BOTH_HYPOTHESES,
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(rd.line())
    log(f"  dispersion {disp['dispersion']:.1f}x binomial  "
        f"(pooled rate {disp['pooled_rate']:.4%}, "
        f"range {disp['rate_min']:.3%}-{disp['rate_max']:.3%})")
    for s, cell in lor["shares"].items():
        log(f"  {s:12s} carries {cell['share_of_violations']:.1%} "
            f"(uniform {cell['uniform_baseline']:.0%})")
    log(f"  gini {lor['gini']:.3f}")
    for k, g in groups.items():
        if g.get("n_groups"):
            log(f"  by {k:8s} worst/best {g['worst_over_best']:.1f}x "
                f"over {g['n_groups']} groups")
    if edge_profile:
        log("  edge-ness profile (arena middle -> edge): " + " ".join(
            f"{b['rate']:.3%}" for b in edge_profile))
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("concentration")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--split", default="report")
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("concentration.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("concentration"), encoding="utf-8") as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "concentration")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.json")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
