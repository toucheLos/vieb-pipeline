"""Step 1. The boundary statistic on the carried arm, swept.

    python3 scripts/breaks.py --write-grid
    sbatch --array=0-29%12 jobs/breaks.slurm --shard
    python3 scripts/breaks.py --combine

Emits the continuous acceleration-mismatch statistic `D` and, at a **reference**
threshold, a segment table. It does not decide whether the boundaries are real:
that needs the surrogate families, and a detector fires on noise. Step 2 turns
`D` into boundaries at a null-calibrated threshold.

## The source array is asserted, not assumed

Every shard is checked for `pose_arm == "raw"` and `scale_arm == "bodylen"`. The
provenance defect that consumed an earlier run was `scripts/ego.py:66` reading
`clean["pose"]` while `clean.json` records `filter.default = wiener`, an arm
`F3_PREPROCESSING_FREEZE.md` does not carry. It matters more here than anywhere:
**a low-pass filter manufactures exactly the smoothness whose breaks this
detects**, and Wiener's effect is five times larger in the twist channels (0.0179)
than in the pose block (0.0037).

## Standardisation, and why it is the frozen one

`D` is a norm across channels, so unstandardised it would be a statement about
which channel has the largest variance. The divisor is `basis.json`'s `sd_used` —
segment-balanced, corpus-pooled, fitted on tune, with the SE(2) dead columns
floored — reused rather than refitted so the detector lives in the same space
every other stage does.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                  # noqa: E402
from recur.util import describe, frames, log, peak_rss_gb         # noqa: E402
from recur.util import write_json                                 # noqa: E402
from vieb.io import spine                                         # noqa: E402
from vieb.seg import breaks as bk                                 # noqa: E402
from vieb.tok import config                                       # noqa: E402

POSE_ARM, SCALE_ARM = "raw", "bodylen"

#: Derivative half-window, in SECONDS. 0.133 is ExBias's run value and the
#: middle of the sweep; a config never holds a frame count.
DERIV_SWEEP: tuple[float, ...] = (0.067, 0.133, 0.267)
DEGREE_SWEEP: tuple[int, ...] = (2, 3)
#: The cell whose `D` is persisted for auditing, and the one Step 2 calibrates.
PRIMARY = ("both", 0.133, 3)


def out_dir() -> str:
    return os.path.join(config.PATHS.tok_dir, "breaks")


def ego_shard(tag: str) -> str:
    return os.path.join(config.PATHS.ego_dir,
                        f"{POSE_ARM}__{SCALE_ARM}__{tag}.npz")


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def load_animal(tag: str) -> dict:
    """The ego shard, with its pose arm asserted by name."""
    with np.load(ego_shard(tag), allow_pickle=False) as z:
        got_pose = str(z["pose_arm"]) if "pose_arm" in z.files else "<absent>"
        got_scale = str(z["scale_arm"]) if "scale_arm" in z.files else "<absent>"
        if got_pose != POSE_ARM or got_scale != SCALE_ARM:
            raise SystemExit(
                f"[{tag}] shard says pose_arm={got_pose!r} "
                f"scale_arm={got_scale!r}, expected {POSE_ARM!r}/{SCALE_ARM!r}. "
                f"A low-pass filter manufactures the smoothness this stage "
                f"detects breaks in, so the arm is asserted rather than assumed")
        out = {"X": z["X"], "valid": z["valid"].astype(bool),
               "bounds": z["bounds"].astype(np.int64),
               "recording_ids": [str(v) for v in z["recording_ids"]]}
    p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
    with np.load(p, allow_pickle=False) as z:
        out["abstain"] = z["abstain"].astype(bool) | ~out["valid"]
    return out


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def shard(args, tag: str) -> int:
    os.makedirs(out_dir(), exist_ok=True)
    fps = spine.fps()
    a = load_animal(tag)
    sd = basis_sd()
    xs = np.asarray(a["X"], dtype=np.float64) / sd[None, :]
    bounds, ab = a["bounds"], a["abstain"]
    n_frames = int(xs.shape[0])

    cells: list = []
    primary_d = np.zeros(n_frames, dtype=np.float32)
    # Boundary sets at the primary window, per channel group, for the agreement
    # measurement below. Kept in memory: a few thousand int64 per animal.
    peak_sets: dict = {}
    for group, idx in bk.CHANNEL_GROUPS.items():
        sub = xs[:, list(idx)]
        for deriv_sec in DERIV_SWEEP:
            h = max(2, frames(deriv_sec, fps))
            guard = bk.guard_frames(fps, deriv_sec=deriv_sec)
            for degree in DEGREE_SWEEP:
                min_gap = bk.min_segment_frames(degree, guard)
                rows, peaks_n = [], 0
                for r in range(bounds.shape[0] - 1):
                    lo, hi = int(bounds[r]), int(bounds[r + 1])
                    d = bk.discontinuity(sub[lo:hi], h)
                    thr = bk.mad_threshold(d, bk.K_MAD)
                    pk = bk.boundaries(d, thr, min_gap=min_gap,
                                       blocked=ab[lo:hi])
                    peaks_n += int(pk.size)
                    rows += bk.segment_table(
                        xs, pk + lo, lo=lo, hi=hi, abstain=ab, guard=guard,
                        degree=degree, fps=fps)
                    if (group, deriv_sec, degree) == PRIMARY:
                        primary_d[lo:hi] = d.astype(np.float32)
                    if (deriv_sec, degree) == (PRIMARY[1], PRIMARY[2]):
                        peak_sets.setdefault(group, []).append(pk + lo)
                cells.append(_summarise(rows, peaks_n, n_frames, fps,
                                        group=group, deriv_sec=deriv_sec,
                                        degree=degree, h=h, guard=guard))
    agree = _agreement({g: np.concatenate(v) if v else np.zeros(0, np.int64)
                        for g, v in peak_sets.items()})
    write_json({"animal": tag, "n_frames": n_frames,
                "group_agreement": agree,
                "n_recordings": int(bounds.shape[0] - 1),
                "pose_arm": POSE_ARM, "scale_arm": SCALE_ARM,
                "abstain_frac": float(ab.mean()),
                "inherited_digest": spine.digest(), "cells": cells},
               os.path.join(out_dir(), f"{tag}.json"))
    np.savez_compressed(os.path.join(out_dir(), f"{tag}_D.npz"),
                        D=primary_d, bounds=bounds, abstain=ab,
                        cell=np.array("|".join(str(v) for v in PRIMARY)),
                        inherited_digest=np.array(spine.digest()))
    best = [c for c in cells if (c["group"], c["deriv_sec"], c["degree"])
            == PRIMARY][0]
    log(f"[{tag}] primary: {best['n_peaks']:,} peaks, "
        f"{best['n_segments']:,} segments ({best['n_segments_clean']:,} clean), "
        f"{best['boundary_rate_per_s']:.3f} b/s, median "
        f"{best['median_duration_s']:.3f}s / clean "
        f"{best['median_duration_clean_s']:.3f}s, adj R2 "
        f"{best['mean_fit_r2_adj']:.4f}, peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _agreement(sets: dict, tol: int = 2) -> dict:
    """Do the channel groups flag the same FRAMES, or just the same rate?

    Two detectors can agree on how often they fire and disagree completely on
    when. `tol` frames of slack, because the detector's own localisation
    uncertainty is the derivative half-window and demanding exact equality
    would measure rounding.
    """
    out: dict = {}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = np.sort(sets[a]), np.sort(sets[b])
            if x.size == 0 or y.size == 0:
                out[f"{a}|{b}"] = {"jaccard": float("nan"), "n_a": int(x.size),
                                   "n_b": int(y.size)}
                continue
            j = np.searchsorted(y, x)
            near = np.zeros(x.size, dtype=bool)
            for off in (-1, 0):
                k = np.clip(j + off, 0, y.size - 1)
                near |= np.abs(y[k] - x) <= tol
            hit = int(near.sum())
            out[f"{a}|{b}"] = {
                "jaccard": float(hit / (x.size + y.size - hit)),
                "frac_a_matched": float(hit / x.size),
                "n_a": int(x.size), "n_b": int(y.size), "tol_frames": int(tol)}
    return out


def _summarise(rows, n_peaks: int, n_frames: int, fps: float, **kw) -> dict:
    """Pooled per animal, with clean segments reported apart from all segments.

    The distinction is not cosmetic. Abstain splits the stream, and abstain on
    this corpus is **fragmented** -- 7.03% of frames in short runs -- so it
    creates far more segment boundaries than the detector does: measured on one
    animal, 1,080 detected peaks against 2,532 segments. A median duration over
    all segments is therefore mostly a statement about tracking failure. The
    clean subset -- segments containing no abstained frame -- is what the
    detector actually delimits, and the two are reported side by side rather
    than one standing in for the other.
    """
    dur = np.array([r["duration_s"] for r in rows], dtype=np.float64)
    fit = np.array([r["fit_r2_adj"] for r in rows], dtype=np.float64)
    raw = np.array([r["fit_r2"] for r in rows], dtype=np.float64)
    clean = np.array([r["abstain_frac"] == 0.0 for r in rows], dtype=bool)
    ok = np.isfinite(fit)
    secs = n_frames / float(fps)
    return {
        "n_segments_clean": int(clean.sum()),
        "median_duration_clean_s": (float(np.median(dur[clean]))
                                    if clean.any() else float("nan")),
        "duration_clean_s": describe(dur[clean]) if clean.any() else {},
        "mean_fit_r2_adj_clean": (float(np.nanmean(fit[clean & ok]))
                                  if (clean & ok).any() else float("nan")),
        **kw,
        "n_segments": int(len(rows)), "n_peaks": int(n_peaks),
        "boundary_rate_per_s": float(n_peaks / secs) if secs > 0 else float("nan"),
        "median_duration_s": float(np.median(dur)) if dur.size else float("nan"),
        "duration_s": describe(dur) if dur.size else {},
        "frac_fittable": float(ok.mean()) if ok.size else float("nan"),
        "mean_fit_r2": float(np.nanmean(raw)) if ok.any() else float("nan"),
        "mean_fit_r2_adj": float(np.nanmean(fit)) if ok.any() else float("nan"),
        "frac_adj_below_0.5": (float((fit[ok] < 0.5).mean())
                               if ok.any() else float("nan")),
        "n_frames": int(n_frames),
    }


def combine(args) -> int:
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    shards = sorted(glob.glob(os.path.join(out_dir(), "*.json")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir()}")
    per_cell: dict = {}
    agree_acc: dict = {}
    n_animals = 0
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        if of.get(doc["animal"]) != args.split:
            continue
        n_animals += 1
        for k, v in doc.get("group_agreement", {}).items():
            if np.isfinite(v.get("jaccard", np.nan)):
                agree_acc.setdefault(k, []).append(float(v["jaccard"]))
        for c in doc["cells"]:
            key = (c["group"], c["deriv_sec"], c["degree"])
            acc = per_cell.setdefault(key, {"n_segments": 0, "n_peaks": 0,
                                            "n_frames": 0, "dur": [],
                                            "adj": [], "raw": [], "fit": [],
                                            "n_clean": 0, "dur_clean": [],
                                            "adj_clean": []})
            acc["n_segments"] += int(c["n_segments"])
            acc["n_peaks"] += int(c["n_peaks"])
            acc["n_frames"] += int(c["n_frames"])
            acc["dur"].append(float(c["median_duration_s"]))
            acc["adj"].append(float(c["mean_fit_r2_adj"]))
            acc["raw"].append(float(c["mean_fit_r2"]))
            acc["fit"].append(float(c["frac_fittable"]))
            acc["n_clean"] += int(c["n_segments_clean"])
            acc["dur_clean"].append(float(c["median_duration_clean_s"]))
            acc["adj_clean"].append(float(c["mean_fit_r2_adj_clean"]))

    fps = spine.fps()
    cells, reads = [], {}
    for (group, deriv_sec, degree), acc in sorted(per_cell.items()):
        secs = acc["n_frames"] / fps
        summ = {
            "group": group, "deriv_sec": deriv_sec, "degree": degree,
            "n_segments": acc["n_segments"], "n_peaks": acc["n_peaks"],
            "boundary_rate_per_s": acc["n_peaks"] / secs if secs else float("nan"),
            "median_duration_s": float(np.nanmedian(acc["dur"])),
            "frac_fittable": float(np.nanmean(acc["fit"])),
            "mean_fit_r2": float(np.nanmean(acc["raw"])),
            "mean_fit_r2_adj": float(np.nanmean(acc["adj"])),
            "n_segments_clean": acc["n_clean"],
            "median_duration_clean_s": float(np.nanmedian(acc["dur_clean"])),
            "mean_fit_r2_adj_clean": float(np.nanmean(acc["adj_clean"])),
            "n_frames": acc["n_frames"],
        }
        cells.append(summ)
        obj = {"dataset": "luna", "arm": "breaks", "split": args.split,
               "pose_arm": POSE_ARM, "group": group,
               "deriv_sec": deriv_sec, "degree": degree}
        rd = bk.breaks_read(summ, scored_object=obj, n_effective=n_animals)
        reads[f"{group}_d{deriv_sec}_p{degree}"] = rd.to_dict()
        log(f"{group:6s} deriv={deriv_sec:.3f}s degree={degree}  "
            f"{summ['n_segments']:>9,} seg  "
            f"{summ['boundary_rate_per_s']:.3f} b/s  "
            f"median {summ['median_duration_s']:.3f}s "
            f"(clean {summ['median_duration_clean_s']:.3f}s) "
            f"adjR2 {summ['mean_fit_r2_adj']:.4f}")

    doc = {**anchors.header(anchors.LUNA, stage="seg_breaks",
                            unverified=f"{args.split} split only; the anchor "
                                       f"counts the whole corpus"),
           "inherited_digest": spine.digest(), "preprocessing_freeze": "F3",
           "pose_arm": POSE_ARM, "scale_arm": SCALE_ARM,
           "split": args.split, "n_animals": n_animals,
           "primary": {"group": PRIMARY[0], "deriv_sec": PRIMARY[1],
                       "degree": PRIMARY[2]},
           "reference_threshold": {"k_mad": bk.K_MAD,
                                   "note": ("a REFERENCE only -- it carries no "
                                            "stated false-positive rate. Step 2 "
                                            "calibrates against the surrogate "
                                            "families")},
           "group_agreement": {k: {"mean_jaccard": float(np.mean(v)),
                                   "n_animals": len(v)}
                               for k, v in sorted(agree_acc.items())},
           "group_agreement_note": (
               "Jaccard between the boundary SETS of two channel groups at the "
               "primary window, +/-2 frames. Two detectors can agree on how "
               "often they fire and disagree completely on when"),
           "cells": cells, "reads": reads, "peak_rss_gb": peak_rss_gb()}
    write_json(doc, args.out)
    for k, v in sorted(agree_acc.items()):
        log(f"  group agreement {k:14s} mean Jaccard {float(np.mean(v)):.3f} "
            f"over {len(v)} animals")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("breaks")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals, "
        f"{len(bk.CHANNEL_GROUPS) * len(DERIV_SWEEP) * len(DEGREE_SWEEP)} "
        f"cells each")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--shard", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--split", default="report")
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("breaks.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("breaks"), encoding="utf-8") as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir(), f"{tag}.json")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
