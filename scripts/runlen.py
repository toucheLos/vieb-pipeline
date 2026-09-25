"""G1. Measure the violating run-length distribution. Nothing ever had.

    python3 scripts/runlen.py --write-grid
    sbatch --array=0-29%15 jobs/runlen.slurm
    python3 scripts/runlen.py --combine --split report

Every run-length figure in this repo's published documents -- 2 frames, 9 frames,
70.4%, 96.1%, 43.7%, 31.3% -- was a hardcoded literal in a Markdown generator or
prose in a docstring, backed by no artifact. This computes the quantity, on the
array the corrector actually receives, corpus-wide rather than on eight
recordings, and records it.

Three arms, so the truncation mechanism is visible rather than argued: `raw` is
the corrector's input, `median_0.50` is the arm whose 7-frame cliff should delete
the short runs and leave the rest untouched, and `viterbi` is the de-glitcher.
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
from vieb import provenance                                         # noqa: E402
from recur.util import log, peak_rss_gb, write_json                # noqa: E402
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.qc import bones, runlen                                  # noqa: E402
from vieb.tok import config                                        # noqa: E402

EPS = 0.10
#: `raw` is the array `scripts/disposition.py` hands the corrector -- the gap
#: policy's output, before any smoother. That is the distribution the envelope
#: should have been calibrated against.
ARMS = ("raw", "median_0.50", "viterbi")
PAIRS = bones.pair_indices(anchors.LUNA.n_kept_keypoints)


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def shard(args, tag: str) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "runlen")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    mine = animals_of(spine.recording_ids())[tag]

    rows: list = []
    for rid in mine:
        d = spine.clean(rid)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        conf = d["conf"].astype(np.float64)
        # One reference per recording, from the corrector's input, so every arm
        # is judged against the same yardstick rather than against its own.
        keep = bones.rigid_pairs(bones.log_lengths(held, PAIRS), PAIRS)
        lengths = bones.metric_lengths(held, bones.SKULL, "raw", pairs=PAIRS,
                                       keep=keep)
        l_hat = np.array([bones.reference_length(lengths[:, m], EPS)["l_hat"]
                          for m in range(len(bones.SKULL))])

        per_arm: dict = {}
        for arm in ARMS:
            pose = (held if arm == "raw"
                    else clean_arms.apply(arm, held, conf, fps))
            viol = bones.violations(
                bones.metric_lengths(pose, bones.SKULL, "raw", pairs=PAIRS,
                                     keep=keep), l_hat, EPS)
            mask = bones.frame_mask(viol)
            per_arm[arm] = runlen.run_lengths(mask).tolist()
        rows.append({"recording_id": rid, "animal": tag,
                     "n_frames": int(held.shape[0]), "lengths": per_arm})
        log(f"[{tag}] {rid[-26:]}  " + "  ".join(
            f"{a} n={len(per_arm[a])} med="
            f"{np.median(per_arm[a]) if per_arm[a] else 0:.0f}" for a in ARMS))

    write_json({"animal": tag, "eps": EPS, "arms": list(ARMS),
                "buckets": list(runlen.BUCKETS),
                "input": "held_array(pose_unfiltered, missing) -- the array "
                         "scripts/disposition.py hands the corrector",
                "inherited_digest": spine.digest(), "rows": rows},
               os.path.join(out_dir, f"{tag}.json"))
    log(f"[{tag}] peak_rss={peak_rss_gb():.2f} GB")
    return 0


def combine(args) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "runlen")
    shards = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            rows += json.load(fh)["rows"]
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    sel = [r for r in rows if of.get(r["animal"]) == args.split]
    log(f"{len(shards)} shards, {len(sel)} recordings on {args.split}")

    dists, intervals = {}, {}
    for arm in ARMS:
        pooled = np.concatenate(
            [np.asarray(r["lengths"][arm], dtype=np.int64) for r in sel]
            or [np.zeros(0, dtype=np.int64)])
        dists[arm] = runlen.distribution(pooled)
        # Per-animal, so the interval is over animals and not over runs.
        by_animal: dict = {}
        for r in sel:
            by_animal.setdefault(r["animal"], []).extend(r["lengths"][arm])
        tags = sorted(by_animal)
        med = [float(np.median(by_animal[t])) if by_animal[t] else float("nan")
               for t in tags]
        mass = []
        for t in tags:
            v = np.asarray(by_animal[t], dtype=np.int64)
            mass.append(float(v[v <= 3].sum() / v.sum()) if v.sum() else float("nan"))
        ok = [i for i, m in enumerate(med) if np.isfinite(m)]
        intervals[arm] = {
            "median_run": boot.animal_interval([med[i] for i in ok],
                                               [tags[i] for i in ok]),
            "frac_mass_le_3": boot.animal_interval([mass[i] for i in ok],
                                                   [tags[i] for i in ok]),
            "n_animals": len(ok),
        }

    obj = {"dataset": "luna", "arm": "runlen", "split": args.split, "eps": EPS}
    n_eff = len({r["animal"] for r in sel})
    env = runlen.envelope_read(dists["raw"], scored_object={**obj, "input": "raw"},
                               n_effective=n_eff, max_correct=3)
    trunc = runlen.truncation_read(dists["raw"], dists["median_0.50"],
                                   scored_object={**obj, "compare": "median_0.50"},
                                   n_effective=n_eff, cliff=7)

    doc = {
        **provenance.header(anchors.LUNA, stage="runlen",
                         unverified="scored on one split; the anchor counts all"),
        "inherited_digest": spine.digest(),
        "split": args.split, "eps": EPS, "buckets": list(runlen.BUCKETS),
        "input": ("held_array(pose_unfiltered, missing) -- the array "
                  "scripts/disposition.py hands the corrector, before any "
                  "smoother"),
        "reads": {"envelope": env.to_dict(), "truncation": trunc.to_dict()},
        "distributions": dists, "intervals": intervals,
        "supersedes": ("the hardcoded table in scripts/cleaning_md.py, which "
                       "reported 2 and 9 frames on the eight worst recordings "
                       "and was backed by no artifact"),
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(env.line())
    log(trunc.line())
    for arm in ARMS:
        d = dists[arm]
        log(f"  {arm:12s} runs {d['n_runs']:>7,}  median {d['median_run']:.0f}"
            f"  p75 {d['p75_run']:.0f}  p90 {d['p90_run']:.0f}"
            f"  mass<=3 {d['frac_mass_le_3']:.1%}"
            f"  mass<=7 {d['frac_mass_le_7']:.1%}"
            f"  mass>7 {d['frac_mass_gt_7']:.1%}")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("runlen")
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
        a.out = a.out or config.PATHS.result("runlen.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("runlen"), encoding="utf-8") as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "runlen")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.json")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
