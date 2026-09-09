"""Phase B. The cleaning bakeoff: every arm on three axes that can disagree.

    python3 scripts/cleaning.py --write-grid
    sbatch --array=0-29%15 jobs/cleaning.slurm
    python3 scripts/cleaning.py --combine

Every arm receives the **identical** input -- shapeflow's hold-for-filtering
array, which is what its Wiener filter actually consumed. `wiener` and
`butterworth` are read off disk rather than recomputed, so a second
implementation cannot silently disagree with the arrays every existing result
was built on.
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
from vieb.qc import bones, effect                                  # noqa: E402
from vieb.tok import config, ego                                   # noqa: E402

EPS = 0.10


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def shard(args, tag: str) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "cleaning")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    f_c = float(spine.json_("sf_calibration").get("f_c_hz", score.DEFAULT_F_C_HZ))
    mine = animals_of(spine.recording_ids())[tag]
    pairs = bones.pair_indices(anchors.LUNA.n_kept_keypoints)
    names = sorted(clean_arms.ARMS) + sorted(clean_arms.STORED_ARMS)

    # ell_a, and the cleaned arrays, held in memory one animal at a time.
    raw_poses, cleaned = [], {n: [] for n in names}
    inputs, flags = [], []
    for rid in mine:
        d = spine.clean(rid)
        missing = d["missing"].astype(bool)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64), missing)
        inputs.append(held)
        raw_poses.append(d["pose"].astype(np.float64))
        flags.append({"bone_flagged": d["bone_flagged"].astype(bool),
                      "interpolated": d["interpolated"].astype(bool),
                      "missing": missing})
        for n in clean_arms.ARMS:
            cleaned[n].append(clean_arms.apply(n, held, d["conf"], fps))
        for n, key in clean_arms.STORED_ARMS.items():
            cleaned[n].append(d[key].astype(np.float64))
    ell = ego.ell_a(raw_poses)
    del raw_poses

    rows = []
    for a, name in enumerate(names):
        # The reference length is refitted PER ARM: an arm that shrinks every
        # distance uniformly would otherwise post a lower violation rate for
        # having made the animal smaller.
        stacked = np.concatenate(cleaned[name])
        keep = bones.rigid_pairs(bones.log_lengths(stacked, pairs), pairs)
        pooled = bones.metric_lengths(stacked, bones.SKULL, "raw",
                                      pairs=pairs, keep=keep)
        l_hat = np.array([bones.reference_length(pooled[:, m], EPS)["l_hat"]
                          for m in range(len(bones.SKULL))])
        del stacked, pooled

        viol, disp_px, disp_mean, disp_bl, ret, conc = [], [], [], [], [], []
        n_frames = []
        for i, rid in enumerate(mine):
            out = cleaned[name][i]
            lengths = bones.metric_lengths(out, bones.SKULL, "raw",
                                           pairs=pairs, keep=keep)
            m = bones.frame_mask(bones.violations(lengths, l_hat, EPS))
            viol.append(float(m.mean()))
            d = effect.displacement(inputs[i], out)
            disp_px.append(float(np.median(d)))
            # The MEAN as well as the median: a targeted arm moves under half
            # its frames, so its median displacement is exactly 0 and the axis
            # goes blind between arms that differ by a factor of ten.
            disp_mean.append(float(d.mean()))
            disp_bl.append(float(np.median(d) / ell) if ell else float("nan"))
            conc.append(effect.by_flag(d, flags[i])["bone_flagged"]["concentration"])
            ret.append(score.high_frequency_retention(inputs[i], out, fps,
                                                      f_c=f_c)["retained"])
            n_frames.append(int(out.shape[0]))
        rows.append({
            "arm": name, "animal": tag, "ell_a": ell,
            "n_recordings": len(mine), "n_frames": int(sum(n_frames)),
            "violation_rate": float(np.average(viol, weights=n_frames)),
            "distortion_px": float(np.average(disp_px, weights=n_frames)),
            "distortion_mean_px": float(np.average(disp_mean, weights=n_frames)),
            "distortion_body_lengths": float(np.average(disp_bl, weights=n_frames)),
            "hf_retained": float(np.nanmean(ret)),
            "concentration": (float(np.nanmedian(conc))
                              if np.isfinite(conc).any() else float("nan")),
        })
        log(f"[{tag}] {a + 1}/{len(names)} {name:14s} "
            f"viol {rows[-1]['violation_rate']:.4%}  "
            f"disp {rows[-1]['distortion_px']:.3f}/{rows[-1]['distortion_mean_px']:.3f} px  "
            f"hf {rows[-1]['hf_retained']:.3f}")

    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"),
                        rows_json=np.array(json.dumps(rows)),
                        inherited_digest=np.array(spine.digest()))
    log(f"[{tag}] peak_rss={peak_rss_gb():.2f} GB")
    return 0


def combine(args) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "cleaning")
    shards = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            rows += json.loads(str(z["rows_json"]))
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    log(f"{len(shards)} shards, {len(rows)} (arm, animal) cells")

    names = sorted({r["arm"] for r in rows})
    per_arm: dict = {}
    for name in names:
        sel = [r for r in rows if r["arm"] == name
               and of.get(r["animal"]) == args.split]
        if not sel:
            continue
        animals = [r["animal"] for r in sel]
        w = [r["n_frames"] for r in sel]
        per_arm[name] = {
            "n_animals": len(sel),
            "violation_rate": boot.animal_interval(
                [r["violation_rate"] for r in sel], animals, weights=w, how="wmean"),
            "distortion_px": boot.animal_interval(
                [r["distortion_px"] for r in sel], animals, weights=w, how="wmean"),
            "distortion_mean_px": boot.animal_interval(
                [r["distortion_mean_px"] for r in sel], animals, weights=w,
                how="wmean"),
            "distortion_body_lengths": boot.animal_interval(
                [r["distortion_body_lengths"] for r in sel], animals, weights=w,
                how="wmean"),
            "hf_retained": boot.animal_interval(
                [r["hf_retained"] for r in sel], animals, weights=w, how="wmean"),
            "concentration": describe([r["concentration"] for r in sel]),
        }
    flat = {n: {k: v["point"] for k, v in m.items() if isinstance(v, dict)
                and "point" in v} for n, m in per_arm.items()}
    ranked = score.rank(flat)
    rd = score.bakeoff_read(
        ranked, scored_object={"dataset": "luna", "arm": "cleaning",
                               "split": args.split, "eps": EPS},
        n_effective=max((m["n_animals"] for m in per_arm.values()), default=1))

    doc = {
        **anchors.header(anchors.LUNA, stage="cleaning",
                         unverified=("scored on one split at a time; the anchor "
                                     "counts the whole corpus")),
        "inherited_digest": spine.digest(),
        "reads": {"bakeoff": rd.to_dict()},
        "split": args.split, "eps": EPS,
        "common_input": ("shapeflow's hold-for-filtering array, which is what "
                         "its Wiener filter consumed. Every arm gets the same "
                         "one; wiener and butterworth are read off disk rather "
                         "than recomputed"),
        "arms": per_arm,
        "ranked": ranked,
        "viterbi": {
            "note": ("Anipose's implementation, loaded from the installed "
                     "package's own source; its import chain (aniposelib, numba) "
                     "is calibration code the filter does not use"),
        },
        "unavailable": {
            "confidence_filter": ("inert -- shapeflow's threshold estimator "
                                  "refuses for all 7 keypoints; its gate masks "
                                  "0.000% of keypoint-frames"),
            "anipose_viterbi": ("CORRECTION: this was recorded as unavailable "
                                "on the grounds that all 3,080 _full.pickle "
                                "files carry one detection per bodypart-frame. "
                                "The count is right, the conclusion was wrong -- "
                                "viterbi_path builds its candidate set from the "
                                "previous n_back frames as well as the current "
                                "one, so it runs on single-detection input. It "
                                "is now a scored arm, not an excluded one"),
            "movement_package": ("not installed -- resolves to ~70 transitive "
                                 "packages into a venv shared with recur. The "
                                 "savgol and median semantics are matched to it "
                                 "in vieb/clean/arms.py instead"),
        },
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(rd.line())
    for r in ranked:
        log(f"  {r['arm']:14s} viol {r['violation_rate']:.4%} "
            f"({r['violation_reduction']:+.1%})  "
            f"disp {r['distortion_px']:.3f}/{r['distortion_mean_px']:.3f} px  "
            f"hf {r['hf_retained']:.3f}  "
            f"eff {r['reduction_per_px'] * 100:.1f}%/px")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("cleaning")
    with open(path, "w") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--split", default="report",
                   choices=("tune", "fit", "report", "all"))
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result(f"cleaning_{a.split}.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("cleaning")) as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "cleaning")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
