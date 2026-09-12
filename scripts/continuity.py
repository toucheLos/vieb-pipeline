"""Phase E1. Does a frame agree with the frames either side of it?

    python3 scripts/continuity.py --write-grid
    sbatch --array=0-29%15 jobs/continuity.slurm
    python3 scripts/continuity.py --combine --split report

Five arms -- the cheap branch that carries through to MDL, plus Phase D's
corrector. Configuration fixed in `results/CONTINUITY_PREREGISTRATION.md`,
committed before this ran on the corpus.

**This does not restate the bakeoff.** `results/cleaning.json` chose on three
axes and is committed; continuity is a fourth view reported in its own file.
`hf_retained` is carried alongside every residual because a filter can make a
recording perfectly continuous by deleting all of the movement.
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

from recur import anchors, boot, labels as lab, splits             # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json      # noqa: E402
from vieb.clean import arms as clean_arms, score                   # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.qc import bones, continuity as ct, effect                # noqa: E402
from vieb.tok import config, ego                                   # noqa: E402

EPS = 0.10
#: The cheap branch plus the corrector. `disposition` is not a `clean_arms` arm:
#: it is read from Phase D's shards, because recomputing it here would be a
#: second implementation of a committed result.
ARMS = ("raw", "wiener", "median_0.50", "viterbi", "disposition")
KEYPOINTS = anchors.LUNA.keypoints


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def _disposition_dir() -> str:
    return os.path.join(os.path.dirname(config.PATHS.bones_dir), "disposition")


def shard(args, tag: str) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "continuity")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    f_c = float(spine.json_("sf_calibration").get("f_c_hz", score.DEFAULT_F_C_HZ))
    mine = animals_of(spine.recording_ids())[tag]
    pairs = bones.pair_indices(anchors.LUNA.n_kept_keypoints)

    # Phase D's output and its donor choices, read from the shard rather than
    # recomputed. The donor is what makes the held-out side well defined.
    dis_path = os.path.join(_disposition_dir(), f"{tag}.npz")
    with np.load(dis_path, allow_pickle=False) as z:
        if "donor" not in z.files:
            raise SystemExit(f"{dis_path} predates the donor array; re-run "
                             f"scripts/disposition.py --force")
        dis_ids = [str(v) for v in z["recording_ids"]]
        dis_bounds = z["bounds"]
        dis_pose = z["pose"].astype(np.float64)
        dis_donor = z["donor"].astype(np.int64)

    inputs, per_arm = [], {a: [] for a in ARMS}
    donors, flags = [], []
    for rid in mine:
        d = spine.clean(rid)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        conf = d["conf"].astype(np.float64)
        inputs.append(held)
        per_arm["raw"].append(held)
        per_arm["wiener"].append(d["pose"].astype(np.float64))
        per_arm["median_0.50"].append(
            clean_arms.apply("median_0.50", held, conf, fps))
        per_arm["viterbi"].append(clean_arms.apply("viterbi", held, conf, fps))
        k = dis_ids.index(str(rid))
        lo, hi = int(dis_bounds[k]), int(dis_bounds[k + 1])
        per_arm["disposition"].append(dis_pose[lo:hi])
        donors.append(dis_donor[lo:hi])
        # The bone check's own mask, for the 2x2. Same cell as Step 1.
        keep = bones.rigid_pairs(bones.log_lengths(held, pairs), pairs)
        L = bones.metric_lengths(held, bones.SKULL, "raw", pairs=pairs, keep=keep)
        l_hat = np.array([bones.reference_length(L[:, m], EPS)["l_hat"]
                          for m in range(len(bones.SKULL))])
        flags.append(bones.frame_mask(bones.violations(L, l_hat, EPS)))

    ell = ego.ell_a(inputs)
    rows: list = []
    for arm in ARMS:
        spike_all, step_all, hf, n_frames = [], [], [], []
        two_sided_all, overlaps = [], []
        for i, rid in enumerate(mine):
            p = per_arm[arm][i]
            if p.shape[0] != inputs[i].shape[0]:
                log(f"  SKIP {rid} on {arm}: length mismatch")
                continue
            rm, rp = ct.residuals(p, ell)
            spike, step = ct.decompose(rm, rp)
            two_sided_all.append(spike)
            # Phase D moves the suspect TO the donor-side prediction, so its
            # residual there is near zero by construction. Score the other side.
            scored = (ct.held_out_side(donors[i], rm, rp)
                      if arm == "disposition" else spike)
            spike_all.append(scored)
            step_all.append(step)
            hf.append(score.high_frequency_retention(inputs[i], p, fps,
                                                     f_c=f_c)["retained"])
            n_frames.append(p.shape[0])
            with np.errstate(invalid="ignore"):
                # Rows that are all-NaN are the first and last frame of the
                # recording, which have no neighbour on one side. `ok` drops
                # them; the warning numpy raises for them is not information.
                fs = np.where(np.isfinite(scored).any(axis=1),
                              np.nanmax(np.where(np.isfinite(scored), scored,
                                                 -np.inf), axis=1), np.nan)
            ok = np.isfinite(fs)
            overlaps.append(ct.overlap_2x2(ok & (fs > ct.SPIKE_BODY_LENGTHS),
                                           flags[i] & ok))
        if not spike_all:
            continue
        # For the disposition this is the DONOR side -- the one the corrector
        # moved the suspect onto, and therefore circular. Recorded beside the
        # held-out number so prediction 3 can be checked rather than asserted.
        two_cat = np.concatenate(two_sided_all)
        fin_two = two_cat[np.isfinite(two_cat)]
        spike_cat = np.concatenate(spike_all)
        step_cat = np.concatenate(step_all)
        fin_s = spike_cat[np.isfinite(spike_cat)]
        fin_t = step_cat[np.isfinite(step_cat)]
        rows.append({
            "arm": arm, "animal": tag, "n_frames": int(sum(n_frames)),
            "spike": float(np.median(fin_s)) if fin_s.size else float("nan"),
            "spike_p99": float(np.percentile(fin_s, 99)) if fin_s.size else float("nan"),
            "step": float(np.median(fin_t)) if fin_t.size else float("nan"),
            "step_p99": float(np.percentile(fin_t, 99)) if fin_t.size else float("nan"),
            "frac_spiking": (float(np.mean(fin_s > ct.SPIKE_BODY_LENGTHS))
                             if fin_s.size else float("nan")),
            "frac_stepping": (float(np.mean(fin_t > ct.SPIKE_BODY_LENGTHS))
                              if fin_t.size else float("nan")),
            "spike_two_sided_do_not_cite_for_disposition": (
                float(np.median(fin_two)) if fin_two.size else float("nan")),
            "frac_spiking_two_sided": (
                float(np.mean(fin_two > ct.SPIKE_BODY_LENGTHS))
                if fin_two.size else float("nan")),
            "hf_retained": float(np.nanmean(hf)),
            "per_keypoint": ct.summarize(spike_cat, names=KEYPOINTS)["per_keypoint"],
            "overlap": {k: int(sum(o[k] for o in overlaps))
                        for k in ("spike_and_bone", "spike_only", "bone_only",
                                  "neither", "n_frames")},
            "ell_px": float(ell),
        })
        log(f"[{tag}] {arm:12s} spike {rows[-1]['spike']:.4f} "
            f"(p99 {rows[-1]['spike_p99']:.4f})  step {rows[-1]['step']:.4f}  "
            f"hf {rows[-1]['hf_retained']:.3f}  "
            f"spiking {rows[-1]['frac_spiking']:.3%}")

    write_json({"animal": tag, "eps": EPS, "arms": list(ARMS),
                "threshold_body_lengths": ct.SPIKE_BODY_LENGTHS,
                "inherited_digest": spine.digest(), "rows": rows},
               os.path.join(out_dir, f"{tag}.json"))
    log(f"[{tag}] peak_rss={peak_rss_gb():.2f} GB")
    return 0


def combine(args) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "continuity")
    shards = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            rows += json.load(fh)["rows"]
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    sel = [r for r in rows if of.get(r["animal"]) == args.split]
    log(f"{len(shards)} shards, {len(sel)} arm-animals on {args.split}")

    obj = {"dataset": "luna", "arm": "continuity", "split": args.split,
           "eps": EPS, "threshold_bl": ct.SPIKE_BODY_LENGTHS}
    out_rows, intervals = [], {}
    for arm in ARMS:
        mine = [r for r in sel if r["arm"] == arm]
        if not mine:
            continue
        an = [r["animal"] for r in mine]
        w = [r["n_frames"] for r in mine]
        ci = lambda k: boot.animal_interval([r[k] for r in mine], an,
                                            weights=w, how="wmean")
        cells = {k: int(sum(r["overlap"][k] for r in mine))
                 for k in ("spike_and_bone", "spike_only", "bone_only",
                           "neither", "n_frames")}
        union = cells["spike_and_bone"] + cells["spike_only"] + cells["bone_only"]
        out_rows.append({
            "arm": arm,
            "spike": float(np.average([r["spike"] for r in mine], weights=w)),
            "spike_p99": float(np.average([r["spike_p99"] for r in mine], weights=w)),
            "step": float(np.average([r["step"] for r in mine], weights=w)),
            "step_p99": float(np.average([r["step_p99"] for r in mine], weights=w)),
            "frac_spiking": float(np.average([r["frac_spiking"] for r in mine],
                                             weights=w)),
            "frac_spiking_two_sided": float(np.average(
                [r["frac_spiking_two_sided"] for r in mine], weights=w)),
            "spike_two_sided": float(np.average(
                [r["spike_two_sided_do_not_cite_for_disposition"] for r in mine],
                weights=w)),
            "frac_stepping": float(np.average([r["frac_stepping"] for r in mine],
                                              weights=w)),
            "hf_retained": float(np.average([r["hf_retained"] for r in mine],
                                            weights=w)),
            "per_keypoint": _pool_keypoints(mine),
            "overlap_2x2": {**cells,
                            "jaccard": (float(cells["spike_and_bone"] / union)
                                        if union else float("nan"))},
            "n_animals": len({r["animal"] for r in mine}),
        })
        intervals[arm] = {"spike": ci("spike"), "step": ci("step"),
                          "hf_retained": ci("hf_retained"),
                          "frac_spiking": ci("frac_spiking")}

    rd = ct.continuity_read(out_rows, intervals, scored_object=obj,
                            n_effective=len({r["animal"] for r in sel}))
    doc = {
        **anchors.header(anchors.LUNA, stage="continuity",
                         unverified="scored on one split; the anchor counts all"),
        "inherited_digest": spine.digest(),
        "preregistration": "results/CONTINUITY_PREREGISTRATION.md",
        "reads": {"continuity": rd.to_dict()},
        "split": args.split, "eps": EPS,
        "threshold_body_lengths": ct.SPIKE_BODY_LENGTHS,
        "arms": out_rows, "intervals": intervals,
        "bakeoff_not_restated": (
            "results/cleaning.json chose on three axes and is unchanged. This is "
            "a fourth view reported beside it, not folded into it"),
        "disposition_scored_on": (
            "the held-out neighbour -- the one donor_frame did not pick. The "
            "corrector moves the suspect TO the donor-side prediction, so its "
            "residual there is near zero by construction"),
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(rd.line())
    for r in out_rows:
        log(f"  {r['arm']:12s} spike {r['spike']:.4f}  step {r['step']:.4f}  "
            f"hf {r['hf_retained']:.3f}  spiking {r['frac_spiking']:.3%}  "
            f"jaccard {r['overlap_2x2']['jaccard']:.3f}")
    log(f"wrote {args.out}")
    return 0


def _pool_keypoints(rows) -> list:
    """Frame-weighted mean of each keypoint's median spike, across animals."""
    out = []
    for k, name in enumerate(KEYPOINTS):
        vals, wts = [], []
        for r in rows:
            cell = r["per_keypoint"][k]
            if cell.get("median") is not None and np.isfinite(cell.get("median", np.nan)):
                vals.append(float(cell["median"]))
                wts.append(int(cell.get("n", 0)))
        out.append({"keypoint": name,
                    "median_spike": (float(np.average(vals, weights=wts))
                                     if vals and sum(wts) else float("nan")),
                    "n": int(sum(wts))})
    return out


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("continuity")
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
        a.out = a.out or config.PATHS.result("continuity.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("continuity"), encoding="utf-8") as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "continuity")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.json")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
