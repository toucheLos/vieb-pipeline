"""Stage 1. The egocentric transform over the corpus, and the A4 parity gate.

    python3 scripts/ego.py --write-grid
    sbatch --array=0-29%12 jobs/ego.slurm --arm bodylen
    sbatch --array=0-29%12 jobs/ego.slurm --arm raw
    python3 scripts/ego.py --combine --arm bodylen
    python3 scripts/ego.py --combine --arm raw

Sharded by **animal**, because `ell_a` is the animal's own median body length
pooled over all of its recordings. Not per recording -- per-recording
standardisation is what silently disarmed a control on this project once already.

Two arms, because the brief's premise about the identity leak is wrong. It asks
for body-length normalisation on the grounds that its absence is "a candidate
contributor to the 5.88 nats of measured identity leak"; 5.88 is the PREVIOUS
instrument's number, measured with a shuffle correction, and shapeflow's direct
probe reads 0.833 / 0.730 nats at gamma = 0. So whether `ell_a` buys anything is
measured here rather than assumed: `bodylen` applies it, `raw` sets it to 1, and
both go through the identical leak probe.
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
from recur.audit import leak                                      # noqa: E402
from recur.geom import represent as rep, reversal                 # noqa: E402
from vieb.io import spine                                        # noqa: E402
from recur.qc import swap                                         # noqa: E402
from vieb.tok import config, ego, parity                         # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json     # noqa: E402

ARMS = ("bodylen", "raw")
#: Contiguous windows per recording for the leak probe. Matches `scripts/audit.py`
#: so the two numbers are on the same footing.
N_WINDOWS = 8
MIN_WINDOW = 50


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def shard(args, tag: str) -> int:
    paths = config.PATHS
    os.makedirs(paths.ego_dir, exist_ok=True)
    fps = spine.fps()
    mine = animals_of(spine.recording_ids())[tag]

    poses, usable, lengths = [], [], []
    for rid in mine:
        d = spine.clean(rid)
        r = spine.representation(rid)
        poses.append(d["pose"].astype(np.float64))
        usable.append(r["usable"].astype(bool))
        lengths.append(int(poses[-1].shape[0]))

    # One number per animal, over every recording it has, on usable frames only.
    ell = 1.0 if args.arm == "raw" else ego.ell_a(poses, usable)
    if not np.isfinite(ell) or ell <= 0:
        raise SystemExit(f"[{tag}] ell_a = {ell!r}; refusing to divide by it")

    xs, vs, rows = [], [], []
    for i, rid in enumerate(mine):
        x, valid = ego.transform(poses[i], ell, fps, usable=usable[i])
        speed, omega = rep.raw_kinematics(poses[i], fps, axis=ego.AXIS,
                                          causal=True)
        rows.append({
            "recording_id": rid, "animal": tag, "n_frames": lengths[i],
            "frac_valid": float(valid.mean()),
            "exact": parity.exact_scores(x, ell, fps, speed, omega, valid=valid),
            "channel_sd": np.nanstd(np.where(valid[:, None], x, np.nan),
                                    axis=0).tolist(),
        })
        xs.append(x.astype(np.float32))
        vs.append(valid)

    X = np.concatenate(xs)
    valid = np.concatenate(vs)
    bounds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez_compressed(
        os.path.join(paths.ego_dir, f"{args.arm}__{tag}.npz"),
        X=X, valid=valid, bounds=bounds, recording_ids=np.array(mine),
        ell_a=np.array(ell), arm=np.array(args.arm),
        channels=np.array(ego.CHANNELS),
        rank=np.array(json.dumps(ego.rank_read(X[valid][:200000]))),
        rows_json=np.array(json.dumps(rows)),
        inherited_digest=np.array(spine.digest()))
    log(f"[{tag}] ell_a={ell:.3f} valid={valid.mean():.4f} "
        f"peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _window_rows(X, valid, bounds, rids, fps):
    """One probe row per contiguous window: per-channel mean and SD.

    A representation is continuous, so the probe sees summary statistics rather
    than the histogram a labeller's probe would use. Held out **by window within
    recording**, which is what asks whether the representation carries session
    identity rather than whether two windows of one session resemble each other.
    """
    feats, animals, sessions = [], [], []
    for r in range(len(rids)):
        lo, hi = int(bounds[r]), int(bounds[r + 1])
        edges = np.linspace(lo, hi, N_WINDOWS + 1).astype(int)
        for w in range(N_WINDOWS):
            a, b = edges[w], edges[w + 1]
            m = valid[a:b]
            if int(m.sum()) < MIN_WINDOW:
                continue
            blk = X[a:b][m].astype(np.float64)
            feats.append(np.concatenate([blk.mean(axis=0), blk.std(axis=0)]))
            animals.append(lab.animal_tag(str(rids[r])))
            sessions.append(str(rids[r]))
    return (np.asarray(feats), animals, sessions)


def combine(args) -> int:
    paths = config.PATHS
    anchor = anchors.LUNA
    shards = sorted(glob.glob(os.path.join(paths.ego_dir, f"{args.arm}__*.npz")))
    if not shards:
        raise SystemExit(f"no {args.arm} shards in {paths.ego_dir}")

    rows, ells, ranks = [], {}, []
    feats, animals, sessions = [], [], []
    rng = np.random.default_rng(0)
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            rows += json.loads(str(z["rows_json"]))
            tag = os.path.basename(path).split("__", 1)[1][:-4]
            ells[tag] = float(z["ell_a"])
            ranks.append(json.loads(str(z["rank"])))
            f, a, s = _window_rows(z["X"], z["valid"], z["bounds"],
                                   z["recording_ids"], spine.fps())
            if len(f):
                feats.append(f)
                animals += a
                sessions += s
    feats = np.concatenate(feats) if feats else np.zeros((0, 2 * ego.N_DIMS))
    log(f"{len(shards)} shards, {len(rows)} recordings, {feats.shape[0]} windows")

    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    obj = {"dataset": "luna", "arm": f"ego_{args.arm}", "split": "all",
           "dim": ego.N_DIMS}

    # ---- A4, on the report split -----------------------------------------
    report = [r for r in rows if of.get(r["animal"]) == "report"]
    n_frames = np.array([r["exact"]["n_frames"] for r in report], dtype=np.float64)
    scores = {"exact": {
        "speed": float(np.average([r["exact"]["speed"] for r in report],
                                  weights=n_frames)),
        "angular": float(np.average([r["exact"]["angular"] for r in report],
                                    weights=n_frames)),
        "n_frames": int(n_frames.sum()),
        "n_recordings": len(report)}}

    if args.regressor:
        sub = rng.permutation(len(shards))[:args.regressor_shards]
        xs, vs, sp, om, gr = [], [], [], [], []
        for i in sub:
            with np.load(shards[i], allow_pickle=False) as z:
                b, rids = z["bounds"], z["recording_ids"]
                for r in range(len(rids)):
                    lo, hi = int(b[r]), int(b[r + 1])
                    pose = spine.clean(str(rids[r]))["pose"].astype(np.float64)
                    s_, o_ = rep.raw_kinematics(pose, spine.fps(),
                                                axis=ego.AXIS, causal=True)
                    xs.append(z["X"][lo:hi].astype(np.float64))
                    vs.append(z["valid"][lo:hi])
                    sp.append(s_)
                    om.append(o_)
                    gr += [lab.animal_tag(str(rids[r]))] * (hi - lo)
        X = np.concatenate(xs)
        V, S, O = np.concatenate(vs), np.concatenate(sp), np.concatenate(om)
        scores["windowed"] = parity.windowed_scores(X, S, O, gr, k=5, valid=V)
        scores["pose_only_single_frame"] = parity.windowed_scores(
            X, S, O, gr, k=1, valid=V, pose_only=True)

    reads = {"a4": parity.a4_read(scores, scored_object=obj,
                                  n_effective=len({r["animal"] for r in report}))}

    # ---- the identity leak, the whole reason there are two arms ----------
    for kind, ids in (("animal", animals), ("session", sessions)):
        reads[f"identity_leak_{kind}"] = leak.leak_read(
            feats, ids, kind=kind, scored_object={**obj, "probe": "window"},
            n_effective=len(set(ids)))

    rv = reversal.audit(fps=spine.fps())
    doc = {
        **anchors.header(anchor, stage=f"ego_{args.arm}", observed={
            "n_recordings": len(rows),
            "n_frames": sum(r["n_frames"] for r in rows),
            "fps": spine.fps()}),
        "inherited_digest": spine.digest(),
        "reads": {k: v.to_dict() for k, v in reads.items()},
        "arm": args.arm,
        "dims": {"total": ego.N_DIMS, "pose": ego.N_POSE, "twist": ego.N_TWIST,
                 "channels": list(ego.CHANNELS),
                 "unreconciled": ("the brief says '28 dims stay 28 dims'; at 7 "
                                  "keypoints the two formulas give 14 + 3 = 17 "
                                  "and 28 is not reconstructible from them. "
                                  "Recorded rather than papered over with "
                                  "invented channels")},
        "scaling": ego.SCALING,
        "scaling_note": ("the brief applies f/ell_a to the whole twist; that is "
                         "dimensionally wrong for omega and makes a large "
                         "mouse's turning read systematically slower, so "
                         "rotation takes f alone"),
        "rank": {"expected": ego.POSE_RANK,
                 "observed": describe([r["rank"] for r in ranks if r.get("rank")]),
                 "why": ("SE(2) removal costs 3 of 14 pose dimensions: two to "
                         "the origin keypoint and one to the aligned axis")},
        "ell_a": {"per_animal": ells, "summary": describe(list(ells.values()))},
        "frac_valid": describe([r["frac_valid"] for r in rows]),
        "a4_scores": scores,
        "reversal_audit": rv,
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    for name, rd in reads.items():
        log(f"{name}: {rd.line()}")
    log(f"reversal audit: {rv['verdict']} ({rv['n_checks']} checks)")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("ego")
    with open(path, "w") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, default="bodylen")
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--regressor", action="store_true",
                   help="run the reported (never gated) regressor arms")
    p.add_argument("--regressor-shards", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result(f"ego_{a.arm}.json")
        return combine(a)

    swap.check_order(anchors.LUNA.keypoints)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        path = config.PATHS.grid("ego")
        with open(path) as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")

    for tag in tags:
        out = os.path.join(config.PATHS.ego_dir, f"{a.arm}__{tag}.npz")
        if not a.force and os.path.exists(out):
            log(f"[{tag}] {a.arm} shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
